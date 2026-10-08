"""LightGBM training with entity-grouped K-fold (out-of-fold predictions).

Folds are assigned per S1 entity so that every candidate row of an entity is
in the same fold: out-of-fold predictions are then honest inputs for the
stage-2 context features and for tuning the decision rule. Rows of entities
not used for fitting (hold-out, or beyond ``max_train_s1``) are scored by the
average of the fold models, exactly like the test set.

Feature matrices never have to fit in RAM: they live on disk as float32 .npy
parts (``PartsMatrix``) and are streamed into LightGBM's binned dataset, which
takes about one byte per value. Each finished fold (model + its predictions)
is checkpointed, so an interrupted training resumes at the next fold.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# On-disk feature matrix
# --------------------------------------------------------------------------
class PartsMatrix:
    """Row-wise concatenation of float32 .npy parts (memory-mapped), plus
    optional extra in-memory columns appended on the right."""

    def __init__(self, paths: list[Path], extra: np.ndarray | None = None):
        self.parts = [np.load(p, mmap_mode="r") for p in paths]
        self.bounds = np.cumsum([0] + [len(p) for p in self.parts])
        self.n = int(self.bounds[-1])
        self.extra = extra
        if extra is not None:
            assert len(extra) == self.n, (len(extra), self.n)
        self.ncol = self.parts[0].shape[1] + (0 if extra is None else extra.shape[1])

    def __len__(self) -> int:
        return self.n

    def rows(self, a: int, b: int) -> np.ndarray:
        """Contiguous rows [a, b) as an in-memory float32 matrix."""
        out, s, i = [], a, int(np.searchsorted(self.bounds, a, side="right") - 1)
        while s < b:
            hi = min(b, self.bounds[i + 1])
            out.append(self.parts[i][s - self.bounds[i]:hi - self.bounds[i]])
            s, i = hi, i + 1
        M = np.concatenate(out) if len(out) != 1 else np.array(out[0])
        if self.extra is not None:
            M = np.hstack([M, self.extra[a:b]])
        return np.ascontiguousarray(M, dtype=np.float32)

    def take(self, idx: np.ndarray) -> np.ndarray:
        """Rows at sorted positions ``idx`` as an in-memory float32 matrix."""
        idx = np.asarray(idx, dtype=np.int64)
        part_of = np.searchsorted(self.bounds, idx, side="right") - 1
        out = [self.parts[i][idx[part_of == i] - self.bounds[i]] for i in np.unique(part_of)]
        M = np.concatenate(out) if out else np.zeros((0, self.parts[0].shape[1]), np.float32)
        if self.extra is not None:
            M = np.hstack([M, self.extra[idx]])
        return np.ascontiguousarray(M, dtype=np.float32)


class _PartSeq(lgb.Sequence):
    """Rows [a, b) of a PartsMatrix as a LightGBM Sequence (streamed construction)."""
    batch_size = 65_536

    def __init__(self, M: PartsMatrix, a: int, b: int):
        self.M, self.a, self.b = M, a, b

    def __len__(self) -> int:
        return self.b - self.a

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            assert step == 1
            return self.M.rows(self.a + start, self.a + stop)
        if isinstance(idx, (int, np.integer)):
            # single rows feed LightGBM's bin-boundary sampling, which requires float64
            # (the same float32 values, widened - the bins are unchanged)
            i = int(idx) + (len(self) if idx < 0 else 0)
            return self.M.rows(self.a + i, self.a + i + 1)[0].astype(np.float64)
        return self.M.take(self.a + np.asarray(idx))


def _sequences(M: PartsMatrix, n: int) -> list[_PartSeq]:
    """Sequences covering the first n rows, split at part boundaries."""
    edges = sorted({0, n, *[int(b) for b in M.bounds if 0 < b < n]})
    return [_PartSeq(M, a, b) for a, b in zip(edges[:-1], edges[1:])]


def _predict_rows(booster, M: PartsMatrix, idx: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    out = np.zeros(len(idx), dtype=np.float64)
    for s in range(0, len(idx), chunk):
        out[s:s + chunk] = booster.predict(M.take(idx[s:s + chunk]))
    return out


# --------------------------------------------------------------------------
# Grouped K-fold with per-fold checkpoints
# --------------------------------------------------------------------------
def train_oof(M: PartsMatrix, cols: list, y: np.ndarray, fold: np.ndarray, n_fit: int,
              params: dict, n_folds: int, rounds: int, early_stopping: int, seed: int,
              stage, tag: str):
    """Grouped K-fold LightGBM.

    M      feature matrix whose FIRST ``n_fit`` rows are the fitting rows (their
           ``fold`` ids are used); the remaining rows (hold-out) are only scored.
    stage  checkpoint Stage: ``<tag>_fold<k>.txt`` plus the fold's predictions
           are saved after every fold, and restored instead of retrained.
    Returns (models, predictions): out-of-fold for the first n_fit rows,
    fold-average for the rest.
    """
    pred = np.zeros(M.n, dtype=np.float64)
    fold_fit = fold[:n_fit]
    rest = np.arange(n_fit, M.n)
    models, imp, ds = [], np.zeros(len(cols)), None
    for k in range(n_folds):
        f_model, f_va, f_rest = f"{tag}_fold{k}.txt", f"{tag}_oof{k}.npy", f"{tag}_rest{k}.npy"
        va = np.flatnonzero(fold_fit == k)
        if all(stage.has(f) for f in (f_model, f_va, f_rest)):
            booster = lgb.Booster(model_file=str(stage.fetch(f_model)))
            p_va, p_rest = stage.load_npy(f_va), stage.load_npy(f_rest)
            log.info("%s fold %d restored from checkpoint", tag, k)
        else:
            if ds is None:
                ds = lgb.Dataset(_sequences(M, n_fit), label=y[:n_fit].astype(np.float32),
                                 feature_name=cols, free_raw_data=True,
                                 params={"max_bin": params.get("max_bin", 255), "verbose": -1})
                ds.construct()
            tr = np.flatnonzero(fold_fit != k)
            d_tr, d_va = ds.subset(tr), ds.subset(va)
            booster = lgb.train(
                dict(params, seed=seed + k), d_tr, num_boost_round=rounds,
                valid_sets=[d_va], valid_names=["valid"],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                           lgb.log_evaluation(250)],
            )
            del d_tr, d_va
            log.info("%s fold %d: best_iter=%d valid=%s", tag, k, booster.best_iteration,
                     dict(booster.best_score["valid"]))
            p_va = _predict_rows(booster, M, va)
            p_rest = _predict_rows(booster, M, rest)
            # predictions first, model last: the model file marks the fold complete
            stage.save_npy(f_va, p_va)
            stage.save_npy(f_rest, p_rest)
            stage.save_model(f_model, booster, num_iteration=booster.best_iteration)
        pred[va] = p_va
        pred[n_fit:] += p_rest / n_folds
        imp += booster.feature_importance("gain")
        models.append(booster)
    del ds
    stage.save_bytes(f"{tag}_importance.csv", pd.Series(imp / n_folds, index=cols)
                     .sort_values(ascending=False).to_csv(header=["gain"]).encode("utf-8"))
    return models, pred


def load_models(out_dir: Path, tag: str, n_folds: int):
    return [lgb.Booster(model_file=str(out_dir / f"{tag}_fold{k}.txt")) for k in range(n_folds)]


def predict_avg(models, Xm, chunk: int = 2_000_000) -> np.ndarray:
    """Fold-average prediction for an in-memory array or a PartsMatrix."""
    out = np.zeros(len(Xm), dtype=np.float64)
    for s in range(0, len(Xm), chunk):
        e = min(s + chunk, len(Xm))
        Xc = Xm.rows(s, e) if isinstance(Xm, PartsMatrix) else np.asarray(Xm[s:e], dtype=np.float32)
        out[s:e] = np.mean([m.predict(Xc) for m in models], axis=0)
    return out
