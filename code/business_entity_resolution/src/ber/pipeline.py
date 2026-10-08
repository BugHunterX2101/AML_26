"""End-to-end orchestration as resumable steps (used by run.py and the notebook).

TRAIN (dataset/train only)                                        checkpoint stage
  train_load        read ground truth + sources (or restore frames)
  train_normalize   static normalisation + locality phrases          static
  train_split       hold-out / dense / fit / ranker entity split,     (cheap, recomputed)
                    K folds
  train_aliases     aliases + noise tokens from training labels only  prep
  train_finalize    derived views, genericness stats, locality vocab  prep
  train_dense       bi-encoder fine-tuned on the dense entities'      dense  (per part)
                    true pairs (GPU), embeddings of every record
  train_blocking    candidate union per (country, source) block,      block  (per block)
                    pruner trained on the ranker entities' union
                    rows, then every block pruned
  train_stage1      features for fit + hold-out rows as on-disk       stage1 (per part / fold)
                    parts, grouped-OOF LightGBM, fold-average p1
                    for every other candidate (competitors)
  train_cross_encoder  optional mDeBERTa on uncertain pairs (GPU)     ce     (per half)
  train_stage2      context features + grouped-OOF LightGBM           stage2 (per fold)
  train_stage3      context recomputed from p2 + grouped-OOF LightGBM stage3 (per fold)
  train_decision    final stage (2 or 3) and decision rule chosen on  decision
                    OOF, honest hold-out score

PREDICT (any split, e.g. test): the same transforms with the frozen
artefacts, checkpointed in one stage per (split, artefacts).

Checkpoints (checkpoint.py): each stage's hash covers the data, the settings
the stage depends on and its upstream stages. Every completed file is uploaded
to S3 immediately, so a resumed session - on the same instance or a new one -
restores the finished work and continues from the first missing file.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import cross_encoder as ce
from . import dense
from .aliases import learn_aliases
from .blocking import COS_COLS, VIA_SPARSE, block_keys, block_union, fit_ranker, prune_candidates
from .checkpoint import Checkpoints, fingerprint, link_or_copy
from .config import Config
from .decision import decide, macro_fbeta, tune_decision
from .features import add_record_stats, context_features, group_features, stage1_matrix, stage3_context
from .io import read_ground_truth, read_split, write_id_lists
from .models import PartsMatrix, load_models, predict_avg, train_oof
from .normalize import finalize_frame, join_phrases, learn_phrases, normalize_frame

log = logging.getLogger(__name__)

CKPT_VERSION = 2          # bump when a code change invalidates earlier checkpoints
PART_ROWS = 2_000_000     # candidate rows per on-disk feature part (checkpoint granularity)
EMB_ROWS = 2_000_000      # records per on-disk embedding part (checkpoint granularity)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log.info("[start] %s", self.name)

    def __exit__(self, *a):
        log.info("[done]  %s in %.1fs", self.name, time.time() - self.t)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def static_normalize(raw: dict, n_jobs: int, phrase_min_count: int):
    S1 = normalize_frame(raw["s1"], n_jobs)
    parts = []
    for src in (2, 3):
        f = normalize_frame(raw[f"s{src}"], n_jobs)
        f["src"] = np.int8(src)
        parts.append(f)
    X = pd.concat(parts, ignore_index=True)
    phrases = learn_phrases(S1, phrase_min_count)
    join_phrases(S1, phrases, n_jobs)
    join_phrases(X, phrases, n_jobs)
    log.info("joined %d multi-word locality phrases", len(phrases))
    return S1, X


def finalize(S1s, Xs, name_alias, addr_alias, noise):
    S1 = finalize_frame(S1s, name_alias, addr_alias, noise)
    X = finalize_frame(Xs, name_alias, addr_alias, noise)
    X["src"] = Xs["src"].values.astype(np.int8)
    return S1, X


def _spans(n: int) -> list[tuple[int, int]]:
    return [(a, min(a + PART_ROWS, n)) for a in range(0, n, PART_ROWS)]


def _no(d: dict, *keys) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def _publish(art: Path, stage, rels: list[str]) -> None:
    """Expose checkpoint files of a stage in the artefacts folder."""
    for rel, src in zip(rels, stage.fetch_many(rels)):
        link_or_copy(src, art / Path(rel).name)


def _publish_dir(art: Path, stage, rel_dir: str) -> None:
    root = stage.fetch_dir(rel_dir)
    for f in root.rglob("*"):
        if f.is_file():
            link_or_copy(f, art / rel_dir / f.relative_to(root))


def _release_frames(st) -> None:
    """Free the normalised text frames (the largest fixed memory) while LightGBM
    trains; they are reloaded from the prep checkpoint afterwards."""
    st.S1 = st.X = None
    gc.collect()


def _reload_frames(st) -> None:
    if st.S1 is None:
        prep = st.stg["prep"]
        st.S1, st.X = prep.load_frame("s1_final.parquet"), prep.load_frame("x_final.parquet")


def _ce_texts(frame: pd.DataFrame, idx: np.ndarray) -> list[str]:
    return ce.pair_text(frame["raw_name"].values[idx], frame["raw_addr"].values[idx])


def _ce_score(model_dirs, cand, S1, X, rows, cfg) -> np.ndarray:
    ta = _ce_texts(S1, cand["s1_idx"].values[rows])
    tb = _ce_texts(X, cand["x_idx"].values[rows])
    return np.mean([ce.score_pairs(d, ta, tb, cfg) for d in model_dirs], axis=0)


def _embed_parts(stg, name: str, model_dir: Path, texts: list[str], dc) -> np.ndarray:
    """Embeddings of ``texts``, encoded and checkpointed EMB_ROWS at a time."""
    spans = [(a, min(a + EMB_ROWS, len(texts))) for a in range(0, len(texts), EMB_ROWS)]
    parts = []
    for i, (a, b) in enumerate(spans):
        rel = f"{name}/part_{i:04d}.npy"
        if not stg.has(rel):
            stg.save_npy(rel, dense.encode(model_dir, texts[a:b], dc))
            log.info("embedding part %s saved (%d / %d records)", rel, b, len(texts))
        parts.append(stg.load_npy(rel))
    return np.concatenate(parts) if parts else np.zeros((0, 1), np.float16)


def _require_embeddings(cfg: Config, emb, step: str) -> None:
    """A block built without the embeddings would silently get cos_dense = 0 while the
    models expect the dense view - refuse instead of producing skewed candidates."""
    if cfg.dense.enabled and emb is None:
        raise RuntimeError(f"the dense view is enabled but the record embeddings are not loaded: "
                           f"run {step} before building candidates (in the notebook: the cell before this one)")


def _ce_usable(cfg: Config) -> bool:
    if not cfg.cross_encoder.enabled:
        return False
    if not ce.gpu_available():
        log.warning("cross-encoder requested but no CUDA GPU (or torch) found - skipping it")
        return False
    return True


def write_outputs(out_dir: Path, S1: pd.DataFrame, X: pd.DataFrame, cand: pd.DataFrame,
                  selected: np.ndarray, p: np.ndarray) -> dict:
    s1_ids = np.asarray(S1["entity_id"].values, dtype=object)
    x_ids = np.asarray(X["entity_id"].values, dtype=object)
    c = cand[["s1_idx", "x_idx", "block_score"]].copy()
    c["p"] = p
    c["sel"] = selected
    cands = (c.sort_values(["s1_idx", "block_score"], ascending=[True, False])
             .groupby("s1_idx")["x_idx"].apply(lambda v: list(x_ids[v.values])))
    matches = (c[c["sel"]].sort_values(["s1_idx", "p"], ascending=[True, False])
               .groupby("s1_idx")["x_idx"].apply(lambda v: list(x_ids[v.values])))
    cand_map = {s1_ids[k]: v for k, v in cands.items()}
    match_map = {s1_ids[k]: v for k, v in matches.items()}
    write_id_lists(Path(out_dir) / "candidate_pairs.tsv",
                   ("source1_entity_id", "candidate_entity_ids"), s1_ids, cand_map)
    write_id_lists(Path(out_dir) / "matching_results.tsv",
                   ("source1_entity_id", "matched_entity_ids"), s1_ids, match_map)
    summary = {"s1_entities": int(len(s1_ids)),
               "s1_with_matches": int(len(match_map)),
               "matched_ids": int(sum(len(v) for v in match_map.values())),
               "candidate_ids": int(len(cand))}
    log.info("outputs: %s", summary)
    return summary


# --------------------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------------------
def _train_files(data_dir: Path) -> dict:
    files = [data_dir / "train" / f"train_source{k}.tsv" for k in (1, 2, 3)]
    files.append(data_dir / "train" / "train_ground_truth.tsv")
    return {f.name: f.stat().st_size for f in files if f.exists()}


def stage_fingerprints(cfg: Config, data_dir: Path) -> dict:
    """What each training stage depends on (speed / memory settings left out).
    The cross-encoder, stage-2 and decision stages are added as they run."""
    mc = cfg.model
    static = {"v": CKPT_VERSION, "data": _train_files(Path(data_dir)), "geo_min_count": cfg.geo_min_count}
    prep = {"static": fingerprint(static), "seed": mc.seed, "holdout_frac": mc.holdout_frac,
            "alias_min_count": cfg.alias_min_count}
    dns = {"prep": fingerprint(prep), "enabled": cfg.dense.enabled}
    if cfg.dense.enabled:
        dns.update(dense=_no(asdict(cfg.dense), "infer_batch_size", "enabled"), dense_frac=mc.dense_frac)
    block = {"prep": fingerprint(prep), "dense": fingerprint(dns),
             "blocking": _no(asdict(cfg.blocking), "chunk_rows"), "ranker_s1": mc.ranker_s1}
    stage1 = {"block": fingerprint(block), "max_train_s1": mc.max_train_s1, "n_folds": mc.n_folds,
              "lgb": _no(mc.lgb_stage1, "num_threads"), "rounds": mc.num_boost_round,
              "early_stopping": mc.early_stopping}
    return {"static": static, "prep": prep, "dense": dns, "block": block, "stage1": stage1}


def train_init(data_dir, work_dir, cfg: Config, use_cache: bool = True, ckpt_uri: str | None = None,
               s3_region: str | None = None, s3_endpoint: str | None = None,
               legacy_work_dir=None) -> SimpleNamespace:
    """Training state. ``ckpt_uri`` (s3://...) mirrors every checkpoint to S3;
    ``use_cache=False`` recomputes everything. ``legacy_work_dir`` lets a run of
    the earlier local-cache layout donate its normalised frames when they were
    built from the same data and settings."""
    if cfg.dense.enabled and not ce.gpu_available():
        # training the encoder and embedding ~12M records needs a GPU; decided before the
        # checkpoint keys are computed, so the keys always describe what actually runs
        log.warning("dense view requested but no CUDA GPU (or torch) found - blocking without it")
        cfg.dense.enabled = False
    st = SimpleNamespace(cfg=cfg, data_dir=Path(data_dir), work=Path(work_dir), use_cache=use_cache,
                         metrics={}, S1=None, X=None, S1s=None, Xs=None, raw=None, emb=None)
    st.art = st.work / "artifacts"
    if st.art.exists():                       # rebuilt from the checkpoints of this configuration
        shutil.rmtree(st.art)
    st.art.mkdir(parents=True)
    st.ck = Checkpoints(st.work / "ckpt", ckpt_uri, s3_region, s3_endpoint, enabled=use_cache)
    st.stg = {k: st.ck.stage(k, fp) for k, fp in stage_fingerprints(cfg, st.data_dir).items()}
    cfg.save(st.art / "config.json")
    if legacy_work_dir is not None and use_cache:
        _adopt_legacy_cache(st, Path(legacy_work_dir))
    log.info("checkpoints: %s", st.ck.s3_uri or st.ck.root)
    for name, stg in st.stg.items():
        log.info("  %-8s %s  %s", name, stg.key, "complete" if stg.done else "to do")
    return st


def _adopt_legacy_cache(st, legacy: Path) -> None:
    """Reuse the normalised frames of a run made with the earlier local-cache layout."""
    prep = st.stg["prep"]
    fp_path = legacy / "cache" / "fingerprint.json"
    files = {"s1_final.parquet": legacy / "cache" / "s1_final.parquet",
             "x_final.parquet": legacy / "cache" / "x_final.parquet",
             "geo.pkl": legacy / "cache" / "geo.pkl",
             "aliases.json": legacy / "artifacts" / "aliases.json"}
    if prep.done or not fp_path.exists() or not all(p.exists() for p in files.values()):
        return
    mc = st.cfg.model
    try:                                      # an unreadable / other-layout fingerprint: just don't reuse
        old = json.loads(fp_path.read_text())
        same = (old.get("data") == _train_files(st.data_dir) and old["model"]["seed"] == mc.seed
                and old["model"]["holdout_frac"] == mc.holdout_frac
                and old.get("alias_min_count") == st.cfg.alias_min_count
                and old.get("geo_min_count") == st.cfg.geo_min_count)
    except (ValueError, KeyError, TypeError, AttributeError):
        same = False
    if not same:
        log.info("earlier cache in %s was built with other data or settings - not reused", legacy)
        return
    log.info("reusing the normalised frames of the earlier run in %s", legacy)
    for rel, src in files.items():
        link_or_copy(src, prep.path(rel))
        prep.commit(rel)
    prep.finish({"adopted_from": str(legacy)})


def train_load(st) -> None:
    with Timer("read training data"):
        st.gt = read_ground_truth(st.data_dir / "train" / "train_ground_truth.tsv")
        prep, static = st.stg["prep"], st.stg["static"]
        if prep.done:
            st.S1 = prep.load_frame("s1_final.parquet")
            st.X = prep.load_frame("x_final.parquet")
            st.geo = prep.load_pickle("geo.pkl")
            _publish(st.art, prep, ["aliases.json"])
            log.info("restored normalised frames from checkpoint (%d S1, %d S2+S3)", len(st.S1), len(st.X))
        elif static.done:
            st.S1s = static.load_frame("s1_static.parquet")
            st.Xs = static.load_frame("x_static.parquet")
            log.info("restored statically normalised frames from checkpoint")
        else:
            st.raw = read_split(st.data_dir, "train")


def train_normalize(st) -> None:
    if st.S1 is not None or st.S1s is not None:
        return
    with Timer("static normalisation"):
        st.S1s, st.Xs = static_normalize(st.raw, st.cfg.n_jobs, st.cfg.geo_min_count)
        st.raw = None
        stg = st.stg["static"]
        stg.save_frame("s1_static.parquet", st.S1s)
        stg.save_frame("x_static.parquet", st.Xs)
        stg.finish()


def train_split(st) -> None:
    mc = st.cfg.model
    S1 = st.S1 if st.S1 is not None else st.S1s
    X = st.X if st.X is not None else st.Xs
    n1 = len(S1)
    rng = np.random.default_rng(mc.seed)
    perm = rng.permutation(n1)
    n_hold = int(round(mc.holdout_frac * n1))
    st.is_hold = np.zeros(n1, bool)
    st.is_hold[perm[:n_hold]] = True
    train_pos = perm[n_hold:]
    # Dense entities: their true pairs train the bi-encoder and nothing else is fitted on them
    # (their embedding cosines are optimistic). Their candidates are still generated and
    # scored as competitors, exactly like the test set's.
    n_dense = int(round(mc.dense_frac * len(train_pos))) if st.cfg.dense.enabled else 0
    st.is_dense = np.zeros(n1, bool)
    st.is_dense[train_pos[:n_dense]] = True
    train_pos = train_pos[n_dense:]
    n_fit = min(len(train_pos), mc.max_train_s1 or len(train_pos))
    fit_pos = train_pos[:n_fit]
    # Ranker entities: the tail of the permutation - independent of max_train_s1
    # (so the blocking checkpoint survives a profile change) and disjoint from
    # the fit entities whenever enough entities are left over.
    ranker_pos = train_pos[-mc.ranker_s1:] if mc.ranker_s1 > 0 else train_pos[:0]
    st.is_fit = np.zeros(n1, bool)
    st.is_fit[fit_pos] = True
    st.is_ranker = np.zeros(n1, bool)
    st.is_ranker[ranker_pos] = True
    st.fold_s1 = np.full(n1, -1, np.int64)
    st.fold_s1[fit_pos] = np.arange(len(fit_pos)) % mc.n_folds

    s1_index = pd.Index(np.asarray(S1["entity_id"].values, dtype=object))
    x_index = pd.Index(np.asarray(X["entity_id"].values, dtype=object))
    st.gt_s1 = np.asarray([s for s, lst in st.gt.items() for _ in lst], dtype=object)
    st.gt_x = np.asarray([x for _, lst in st.gt.items() for x in lst], dtype=object)
    gi = s1_index.get_indexer(st.gt_s1)
    gx = x_index.get_indexer(st.gt_x)
    st.gt_ok = (gi >= 0) & (gx >= 0)
    st.gt_gi = gi
    st.owner = np.full(len(X), -1, np.int64)
    st.owner[gx[st.gt_ok]] = gi[st.gt_ok]
    st.n_gt = np.bincount(gi[gi >= 0], minlength=n1).astype(np.int64)
    st.metrics.update(n_s1=int(n1), n_x=int(len(X)), singleton_rate=float((st.n_gt == 0).mean()),
                      n_fit_s1=int(st.is_fit.sum()), n_holdout_s1=int(st.is_hold.sum()),
                      n_ranker_s1=int(st.is_ranker.sum()), n_dense_s1=int(st.is_dense.sum()))
    log.info("entities: fit=%d hold-out=%d ranker=%d dense=%d (of %d)", st.is_fit.sum(),
             st.is_hold.sum(), st.is_ranker.sum(), st.is_dense.sum(), n1)


def train_aliases(st) -> None:
    if st.S1 is not None:
        return
    with Timer("learn aliases"):
        keep = st.gt_ok & ~st.is_hold[np.clip(st.gt_gi, 0, None)]
        pairs = pd.DataFrame({"s1_id": st.gt_s1[keep], "x_id": st.gt_x[keep]})
        st.name_alias, st.addr_alias, st.noise = learn_aliases(
            st.S1s, st.Xs, pairs, st.cfg.alias_min_count, seed=st.cfg.model.seed)
        (st.art / "aliases.json").write_text(json.dumps(
            {"name_alias": st.name_alias, "addr_alias": st.addr_alias,
             "noise_tokens": sorted(st.noise)}, indent=1))


def train_finalize(st) -> None:
    if st.S1 is not None:
        return
    with Timer("finalise normalisation"):
        st.S1, st.X = finalize(st.S1s, st.Xs, st.name_alias, st.addr_alias, st.noise)
        st.S1s = st.Xs = None
        st.geo = add_record_stats(st.S1, st.X, st.cfg.geo_min_count)
        prep = st.stg["prep"]
        prep.save_frame("s1_final.parquet", st.S1)
        prep.save_frame("x_final.parquet", st.X)
        prep.save_pickle("geo.pkl", st.geo)
        link_or_copy(st.art / "aliases.json", prep.path("aliases.json"))
        prep.commit("aliases.json")
        prep.finish()


def train_dense(st) -> None:
    """Fine-tune the bi-encoder on the dense entities' true pairs and embed every record.

    The embeddings are only needed to build the candidate union: when the
    blocking stage, or every block's union, is already complete they are not
    loaded at all.
    """
    dc, mc, stg = st.cfg.dense, st.cfg.model, st.stg["dense"]
    st.emb = None
    if not dc.enabled:
        return
    if not stg.has("dense_encoder.ok"):
        xs = np.flatnonzero(st.owner >= 0)
        xs = xs[st.is_dense[st.owner[xs]]]
        if len(xs) > dc.max_train_pairs:
            xs = np.sort(np.random.default_rng(mc.seed).choice(xs, dc.max_train_pairs, replace=False))
        s1s = st.owner[xs]
        with Timer(f"dense encoder training ({len(xs):,} true pairs of {st.is_dense.sum():,} entities)"):
            dense.train_encoder(dense.record_text(st.S1, s1s), dense.record_text(st.X, xs), s1s,
                                st.S1["country"].values[s1s], dc, stg.path("dense_encoder"), mc.seed)
        stg.commit("dense_encoder")
        stg.save_json("dense_encoder.ok", {"pairs": int(len(xs))})   # marks the encoder complete
    enc = stg.fetch_dir("dense_encoder")
    blk = st.stg["block"]
    if not blk.done and not all(blk.has(f"union/{t}.parquet") for *_, t in block_keys(st.S1, st.X)):
        with Timer("dense embeddings of every record"):
            st.emb = (_embed_parts(stg, "emb_s1", enc, dense.record_text(st.S1), dc),
                      _embed_parts(stg, "emb_x", enc, dense.record_text(st.X), dc))
        if not stg.done:
            stg.finish({"dim": int(st.emb[0].shape[1])})
    _publish_dir(st.art, stg, "dense_encoder")


def _read_cand(stg, tags: list[str]) -> pd.DataFrame:
    paths = stg.fetch_many([f"cand/{t}.parquet" for t in tags])
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def train_blocking(st) -> None:
    cfg, mc, stg = st.cfg, st.cfg.model, st.stg["block"]
    if stg.done:
        st.cand = _read_cand(stg, stg.load_json("blocks.json"))
        st.metrics.update(stg.load_json("blocking_metrics.json"))
        _publish(st.art, stg, ["block_ranker.txt"])
        log.info("restored %d candidate pairs from checkpoint", len(st.cand))
        return
    keys = block_keys(st.S1, st.X)
    tags = [t for *_, t in keys]
    eval_mask = st.is_hold if st.is_hold.any() else st.is_fit

    # 1) unpruned candidate union of every block (one checkpoint per block)
    with Timer(f"candidate union for all S1 entities ({len(keys)} blocks)"):
        for country, src, tag in keys:
            if stg.has(f"union/{tag}.parquet"):
                log.info("block %s / S%d restored from checkpoint", country, src)
                continue
            _require_embeddings(cfg, st.emb, "train_dense")
            stg.save_frame(f"union/{tag}.parquet",
                           block_union(st.S1, st.X, country, src, cfg.blocking, cfg.n_jobs, emb=st.emb))
    st.emb = None                             # embeddings are only used to build the union
    gc.collect()

    # 2) pruner trained on the union rows of the ranker entities: exactly the
    #    candidate distribution it will prune
    with Timer("blocking pruner"):
        if stg.has("block_ranker.txt"):
            ranker = lgb.Booster(model_file=str(stg.fetch("block_ranker.txt")))
        else:
            parts = []
            for tag in tags:
                u = stg.load_frame(f"union/{tag}.parquet")
                parts.append(u[st.is_ranker[u["s1_idx"].values]])
            union = pd.concat(parts, ignore_index=True)
            del parts
            yu = st.owner[union["x_idx"].values] == union["s1_idx"].values
            log.info("pruner training pairs: %d (%d true)", len(union), int(yu.sum()))
            ranker = fit_ranker(union, st.X, yu, seed=mc.seed)
            stg.save_model("block_ranker.txt", ranker)
            del union, yu

    # 3) prune every block (one checkpoint per block)
    with Timer("prune the candidate union of every block"):
        for country, src, tag in keys:
            if stg.has(f"cand/{tag}.json"):
                continue
            u = stg.load_frame(f"union/{tag}.parquet")
            hit = (st.owner[u["x_idx"].values] == u["s1_idx"].values) & eval_mask[u["s1_idx"].values]
            sparse = (u["via"].values & VIA_SPARSE) > 0
            info = {"union_pairs": int(len(u)), "union_pairs_sparse": int(sparse.sum()),
                    "union_true_eval": int(hit.sum()), "union_true_eval_dense_only": int((hit & ~sparse).sum())}
            block = prune_candidates(u, st.X, ranker, cfg.blocking.keep_per_source)
            del u, hit
            stg.save_frame(f"cand/{tag}.parquet", block)
            stg.save_json(f"cand/{tag}.json", info)     # written last: marks the block complete
            log.info("pruned %s / S%d: %d union -> %d kept", country, src, info["union_pairs"], len(block))

    st.cand = _read_cand(stg, tags)
    stats = [stg.load_json(f"cand/{t}.json") for t in tags]
    y = st.owner[st.cand["x_idx"].values] == st.cand["s1_idx"].values
    tot = st.n_gt[eval_mask].sum()
    n_union_true = sum(s["union_true_eval"] for s in stats)
    bm = {"union_pairs": int(sum(s["union_pairs"] for s in stats)), "cand_pairs": int(len(st.cand)),
          "cand_per_s1": float(len(st.cand) / len(st.S1)),
          "union_recall_eval": float(n_union_true / max(tot, 1)),
          # what the sparse views alone reach (the difference is the dense view's contribution)
          "union_recall_eval_sparse_only": float(
              (n_union_true - sum(s.get("union_true_eval_dense_only", 0) for s in stats)) / max(tot, 1)),
          "cand_recall_eval": float((y & eval_mask[st.cand["s1_idx"].values]).sum() / max(tot, 1))}
    st.metrics.update(bm)
    log.info("blocking: %s", bm)
    stg.save_json("blocks.json", tags)
    stg.save_json("blocking_metrics.json", bm)
    stg.finish(bm)
    for tag in tags:                          # the unions stay on S3; free the local disk
        stg.drop_local(f"union/{tag}.parquet")
    _publish(st.art, stg, ["block_ranker.txt"])


def train_stage1(st) -> None:
    cfg, mc, stg = st.cfg, st.cfg.model, st.stg["stage1"]
    cand = st.cand
    s1c = cand["s1_idx"].values
    st.y = (st.owner[cand["x_idx"].values] == s1c).astype(np.int8)
    fit_rows = np.flatnonzero(st.is_fit[s1c])
    hold_rows = np.flatnonzero(st.is_hold[s1c])
    st.model_rows = np.concatenate([fit_rows, hold_rows])   # fit rows FIRST
    st.n_fit_rows = len(fit_rows)
    st.fold_m = st.fold_s1[s1c[st.model_rows]]
    spans = _spans(len(st.model_rows))
    st.f1_parts = [f"F1/part_{i:04d}.npy" for i in range(len(spans))]
    arts = [f"stage1_fold{k}.txt" for k in range(mc.n_folds)] + ["stage1_cols.json", "stage1_importance.csv"]
    if stg.done:
        st.p1 = stg.load_npy("p1.npy")
        st.f1_cols = stg.load_json("stage1_cols.json")
        _publish(st.art, stg, arts)
        log.info("restored stage-1 predictions from checkpoint")
        return

    grp = None
    with Timer(f"stage-1 features for {len(st.model_rows):,} fit + hold-out rows ({len(spans)} parts)"):
        for rel, (a, b) in zip(st.f1_parts, spans):
            if stg.has(rel):
                continue
            if grp is None:
                grp = group_features(cand)
            cols = stg.load_json("stage1_cols.json") if stg.has("stage1_cols.json") else None
            F = stage1_matrix(cand, st.model_rows[a:b], st.S1, st.X, st.geo, grp, cfg.n_jobs, cols)
            if cols is None:
                stg.save_json("stage1_cols.json", list(F.columns))
            stg.save_npy(rel, F.to_numpy(np.float32))
            del F
            log.info("stage-1 feature part %s saved (%d / %d rows)", rel, b, len(st.model_rows))
    st.f1_cols = stg.load_json("stage1_cols.json")

    grp = None
    _release_frames(st)
    M = PartsMatrix(stg.fetch_many(st.f1_parts))
    with Timer("stage-1 LightGBM (grouped 5-fold OOF)"):
        models, p1m = train_oof(M, st.f1_cols, st.y[st.model_rows], st.fold_m, st.n_fit_rows,
                                mc.lgb_stage1, mc.n_folds, mc.num_boost_round,
                                mc.early_stopping, mc.seed, stg, "stage1")
    del M
    _reload_frames(st)
    st.p1 = np.full(len(cand), np.nan)
    st.p1[st.model_rows] = p1m
    other = np.flatnonzero(np.isnan(st.p1))
    if len(other):
        with Timer(f"stage-1 scoring of {len(other):,} competitor rows"):
            for i, (a, b) in enumerate(_spans(len(other))):
                rel = f"p1_other/part_{i:04d}.npy"
                if not stg.has(rel):
                    if grp is None:
                        grp = group_features(cand)
                    Fo = stage1_matrix(cand, other[a:b], st.S1, st.X, st.geo, grp, cfg.n_jobs, st.f1_cols)
                    stg.save_npy(rel, predict_avg(models, Fo.to_numpy(np.float32)))
                    del Fo
                st.p1[other[a:b]] = stg.load_npy(rel)
    stg.save_npy("p1.npy", st.p1)
    stg.finish()
    _publish(st.art, stg, arts)


def train_cross_encoder(st) -> None:
    ccfg, mc = st.cfg.cross_encoder, st.cfg.model
    st.ce_enabled = _ce_usable(st.cfg)
    fp = {"stage1": st.stg["stage1"].hash, "enabled": st.ce_enabled}
    if st.ce_enabled:
        fp["ce"] = _no(asdict(ccfg), "infer_batch_size", "enabled", "num_workers")   # speed-only settings
    stg = st.stg["ce"] = st.ck.stage("ce", fp)
    st.ce_m = np.full(len(st.model_rows), np.nan, dtype=np.float32)
    if not st.ce_enabled:
        return
    if stg.done:
        st.ce_m = stg.load_npy("ce_m.npy")
        for h in (0, 1):
            _publish_dir(st.art, stg, f"ce_{h}")
        log.info("restored cross-encoder scores and models from checkpoint")
        return
    with Timer("cross-encoder (2-fold over fit entities)"):
        rows_all = st.model_rows
        p1m = st.p1[rows_all]
        y_m = st.y[rows_all]
        band = (p1m >= ccfg.band_low) & (p1m <= ccfg.band_high)
        is_fit_row = np.arange(len(rows_all)) < st.n_fit_rows
        half = np.where(is_fit_row, st.fold_m % 2, -1)
        dirs = []
        for h in (0, 1):
            d_rel, s_rel = f"ce_{h}", f"ce_scores_{h}.npy"
            sc = np.flatnonzero(is_fit_row & (half == 1 - h) & band)
            if stg.has(f"{d_rel}.ok"):
                d = stg.fetch_dir(d_rel)
                log.info("cross-encoder half %d restored from checkpoint", h)
            else:
                tr = is_fit_row & (half == h)
                pick = ce.sample_training_pairs(np.flatnonzero(tr & band), np.flatnonzero(tr & ~band),
                                                y_m, ccfg.max_train_pairs, mc.seed + h)
                d = ce.train_cross_encoder(_ce_texts(st.S1, st.cand["s1_idx"].values[rows_all[pick]]),
                                           _ce_texts(st.X, st.cand["x_idx"].values[rows_all[pick]]),
                                           y_m[pick].astype(np.float32), ccfg, stg.path(d_rel), mc.seed + h)
                stg.commit(d_rel)
                stg.save_npy(s_rel, _ce_score([d], st.cand, st.S1, st.X, rows_all[sc], ccfg))
                stg.save_json(f"{d_rel}.ok", {"pairs": int(len(pick))})   # marks the half complete
            st.ce_m[sc] = stg.load_npy(s_rel)
            dirs.append(d)
        sc = np.flatnonzero(~is_fit_row & band)
        if not stg.has("ce_hold.npy"):
            stg.save_npy("ce_hold.npy", _ce_score(dirs, st.cand, st.S1, st.X, rows_all[sc], ccfg))
        st.ce_m[sc] = stg.load_npy("ce_hold.npy")
    stg.save_npy("ce_m.npy", st.ce_m)
    stg.finish()
    for h in (0, 1):
        _publish_dir(st.art, stg, f"ce_{h}")


def train_stage2(st) -> None:
    mc = st.cfg.model
    fp = {"ce": st.stg["ce"].hash, "n_folds": mc.n_folds, "lgb": _no(mc.lgb_stage2, "num_threads"),
          "rounds": mc.num_boost_round, "early_stopping": mc.early_stopping, "n_anchors": mc.n_anchors}
    stg = st.stg["stage2"] = st.ck.stage("stage2", fp)
    arts = [f"stage2_fold{k}.txt" for k in range(mc.n_folds)] + ["stage2_importance.csv"]
    if stg.done:
        st.p2m = stg.load_npy("p2m.npy")
        st.f2_cols = stg.load_json("stage2_cols.json")
        _publish(st.art, stg, arts)
        log.info("restored stage-2 predictions from checkpoint")
        return
    if stg.has("ctx.npy") and stg.has("ctx_cols.json"):
        C, ctx_cols = stg.load_npy("ctx.npy"), stg.load_json("ctx_cols.json")
    else:
        with Timer("context features"):
            ctx = context_features(st.cand, st.p1, st.X, mc.n_anchors, rows=st.model_rows)
            ctx["ce"] = st.ce_m
            ctx_cols, C = list(ctx.columns), ctx.to_numpy(np.float32)
            del ctx
            stg.save_npy("ctx.npy", C)
            stg.save_json("ctx_cols.json", ctx_cols)
    st.f2_cols = st.f1_cols + ctx_cols
    _release_frames(st)
    M = PartsMatrix(st.stg["stage1"].fetch_many(st.f1_parts), extra=C)
    with Timer("stage-2 LightGBM (grouped 5-fold OOF)"):
        _, st.p2m = train_oof(M, st.f2_cols, st.y[st.model_rows], st.fold_m, st.n_fit_rows,
                              mc.lgb_stage2, mc.n_folds, mc.num_boost_round,
                              mc.early_stopping, mc.seed + 100, stg, "stage2")
    del M, C
    _reload_frames(st)
    stg.save_npy("p2m.npy", st.p2m)
    stg.save_json("stage2_cols.json", st.f2_cols)
    stg.finish()
    _publish(st.art, stg, arts)
    if not mc.stage3:                         # stage 3 reads the stage-1 parts again
        _drop_stage1_parts(st)
    stg.drop_local("ctx.npy")


def _drop_stage1_parts(st) -> None:
    for rel in st.f1_parts:                   # stay on S3; free the local disk for inference
        st.stg["stage1"].drop_local(rel)


def _competitor_p2(st, stg) -> tuple[np.ndarray, np.ndarray]:
    """Stage-2 probabilities of the competitor-only rows, computed as at inference.

    Rows of entities outside fit / hold-out (the dense slice, entities beyond
    max_train_s1) get no p2 from the grouped OOF. At inference every row has the
    fold-average p2, so they get exactly that here: their stage-1 features, the
    stage-2 context from p1 (with the cross-encoder on their uncertain band) and the
    fold-average stage-2 models. Otherwise stage 3 would see p1 for these competitors
    in training and p2 at inference. Checkpointed per feature part.
    """
    cfg, mc = st.cfg, st.cfg.model
    in_model = np.zeros(len(st.cand), bool)
    in_model[st.model_rows] = True
    other = np.flatnonzero(~in_model)
    if len(other) == 0:
        return other, np.zeros(0)
    if stg.has("p2_other.npy"):
        return other, stg.load_npy("p2_other.npy")
    ce_o = np.full(len(other), np.nan, dtype=np.float32)
    if st.ce_enabled:
        ccfg = cfg.cross_encoder
        band = np.flatnonzero((st.p1[other] >= ccfg.band_low) & (st.p1[other] <= ccfg.band_high))
        if not stg.has("ce_other.npy"):
            with Timer(f"cross-encoder scoring of {len(band):,} competitor rows"):
                dirs = [st.stg["ce"].fetch_dir(f"ce_{h}") for h in (0, 1)]
                stg.save_npy("ce_other.npy", _ce_score(dirs, st.cand, st.S1, st.X, other[band], ccfg))
        ce_o[band] = stg.load_npy("ce_other.npy")
    with Timer(f"stage-2 scoring of {len(other):,} competitor rows"):
        ctx = context_features(st.cand, st.p1, st.X, mc.n_anchors, rows=other)
        ctx["ce"] = ce_o
        assert st.f1_cols + list(ctx.columns) == st.f2_cols, "feature order mismatch"
        C = ctx.to_numpy(np.float32)
        del ctx
        spans = _spans(len(other))
        rels = [f"F1o/part_{i:04d}.npy" for i in range(len(spans))]
        grp = None
        for rel, (a, b) in zip(rels, spans):
            if stg.has(rel):
                continue
            if grp is None:
                grp = group_features(st.cand)
            F = stage1_matrix(st.cand, other[a:b], st.S1, st.X, st.geo, grp, cfg.n_jobs, st.f1_cols)
            stg.save_npy(rel, F.to_numpy(np.float32))
            del F
        del grp
        p2o = predict_avg(load_models(st.art, "stage2", mc.n_folds), PartsMatrix(stg.fetch_many(rels), extra=C))
    stg.save_npy("p2_other.npy", p2o)
    for rel in rels:                          # stay on S3; free the local disk
        stg.drop_local(rel)
    return other, p2o


def train_stage3(st) -> None:
    """Collective refinement: stage-1 features + the context recomputed from p2.

    p2 is out-of-fold for the fit rows and the fold average for the hold-out rows;
    competitor-only rows (the dense slice's entities, or entities beyond
    max_train_s1) get the fold-average p2 too (_competitor_p2), so every row
    carries p2 in the context, exactly as at inference.
    """
    mc = st.cfg.model
    st.p3m, st.f3_cols = None, None
    if not mc.stage3:
        return
    fp = {"stage2": st.stg["stage2"].hash, "n_folds": mc.n_folds, "lgb": _no(mc.lgb_stage3, "num_threads"),
          "rounds": mc.num_boost_round, "early_stopping": mc.early_stopping, "n_anchors": mc.n_anchors,
          "competitors": "p2"}
    stg = st.stg["stage3"] = st.ck.stage("stage3", fp)
    arts = [f"stage3_fold{k}.txt" for k in range(mc.n_folds)] + ["stage3_importance.csv"]
    if stg.done:
        st.p3m = stg.load_npy("p3m.npy")
        st.f3_cols = stg.load_json("stage3_cols.json")
        _publish(st.art, stg, arts)
        log.info("restored stage-3 predictions from checkpoint")
        return
    if stg.has("ctx.npy") and stg.has("ctx_cols.json"):
        C, ctx_cols = stg.load_npy("ctx.npy"), stg.load_json("ctx_cols.json")
    else:
        p_all = st.p1.copy()
        p_all[st.model_rows] = st.p2m
        other, p2o = _competitor_p2(st, stg)
        p_all[other] = p2o
        with Timer("stage-3 context features (from p2)"):
            ctx = stage3_context(st.cand, p_all, st.X, mc.n_anchors, rows=st.model_rows)
            ctx["ce"] = st.ce_m
            ctx_cols, C = list(ctx.columns), ctx.to_numpy(np.float32)
            del ctx, p_all
            stg.save_npy("ctx.npy", C)
            stg.save_json("ctx_cols.json", ctx_cols)
    st.f3_cols = st.f1_cols + ctx_cols
    _release_frames(st)
    M = PartsMatrix(st.stg["stage1"].fetch_many(st.f1_parts), extra=C)
    with Timer("stage-3 LightGBM (grouped 5-fold OOF)"):
        _, st.p3m = train_oof(M, st.f3_cols, st.y[st.model_rows], st.fold_m, st.n_fit_rows,
                              mc.lgb_stage3, mc.n_folds, mc.num_boost_round,
                              mc.early_stopping, mc.seed + 200, stg, "stage3")
    del M, C
    _reload_frames(st)
    stg.save_npy("p3m.npy", st.p3m)
    stg.save_json("stage3_cols.json", st.f3_cols)
    stg.finish()
    _publish(st.art, stg, arts)
    _drop_stage1_parts(st)
    stg.drop_local("ctx.npy")


def train_decision(st) -> dict:
    """Tune the decision rule on out-of-fold predictions of each final-stage
    candidate (stage 2, and stage 3 when trained), keep the best one, then score
    the hold-out once with it."""
    dc = st.cfg.decision
    has3 = getattr(st, "p3m", None) is not None
    stg = st.stg["decision"] = st.ck.stage("decision", {
        "stage2": st.stg["stage2"].hash, "stage3": st.stg["stage3"].hash if has3 else None,
        "decision": asdict(dc)})
    probs = {2: st.p2m, **({3: st.p3m} if has3 else {})}
    if stg.done:
        st.metrics.update(stg.load_json("metrics.json"))
        meta = stg.load_json("meta.json")
        st.decision, st.final_stage = meta["decision"], meta.get("final_stage", 2)
        st.pfin = probs[st.final_stage]
        st.decision_grid = [tuple(r) for r in stg.load_json("grid.json")]
        if stg.has("holdout_sel.npy"):
            st.holdout_sel = stg.load_npy("holdout_sel.npy")
        _publish(st.art, stg, ["meta.json", "metrics.json"])
        log.info("restored the decision rule and metrics from checkpoint")
        return st.metrics
    s1m = st.cand["s1_idx"].values[st.model_rows]
    xm = st.cand["x_idx"].values[st.model_rows]
    ym = st.y[st.model_rows]
    nf = st.n_fit_rows
    fit_ents = np.flatnonzero(st.is_fit)
    hold_ents = np.flatnonzero(st.is_hold)
    tuned = {}
    with Timer("decision tuning on out-of-fold predictions"):
        codes = np.searchsorted(fit_ents, s1m[:nf])
        for k, p in probs.items():
            tuned[k] = tune_decision(codes, xm[:nf], p[:nf], ym[:nf], st.n_gt[fit_ents], dc)
            st.metrics[f"oof_macro_f05_stage{k}"] = tuned[k][1]
    st.final_stage = max(tuned, key=lambda k: tuned[k][1])     # chosen on out-of-fold data only
    params, oof, grid = tuned[st.final_stage]
    st.decision, st.decision_grid, st.pfin = params, grid, probs[st.final_stage]
    st.metrics.update(oof_macro_f05=oof, decision=params, final_stage=st.final_stage)
    log.info("final stage: %d (out-of-fold macro F0.5 %s)", st.final_stage,
             {k: round(v[1], 5) for k, v in tuned.items()})
    if len(hold_ents):
        codes = np.searchsorted(hold_ents, s1m[nf:])
        n_gt_h = st.n_gt[hold_ents]
        for k, p in probs.items():            # reported for information; not used for any choice
            s = decide(codes, xm[nf:], p[nf:], tuned[k][0], dc.beta, dc.max_items)
            st.metrics[f"holdout_macro_f05_stage{k}"] = macro_fbeta(codes, ym[nf:], s, n_gt_h, dc.beta)
        sel = decide(codes, xm[nf:], st.pfin[nf:], params, dc.beta, dc.max_items)
        tp = int((ym[nf:] * sel).sum())
        st.holdout_sel = sel
        st.metrics.update(
            holdout_macro_f05=macro_fbeta(codes, ym[nf:], sel, n_gt_h, dc.beta),
            holdout_macro_f05_stage1_t05=macro_fbeta(codes, ym[nf:], st.p1[st.model_rows][nf:] >= 0.5,
                                                     n_gt_h, dc.beta),
            holdout_macro_f05_oracle=macro_fbeta(codes, ym[nf:], ym[nf:].astype(bool), n_gt_h, dc.beta),
            holdout_pair_precision=tp / max(int(sel.sum()), 1),
            holdout_pair_recall=tp / max(int(n_gt_h.sum()), 1))
        stg.save_npy("holdout_sel.npy", sel)
    meta = {"f1_cols": st.f1_cols, "f2_cols": st.f2_cols, "f3_cols": st.f3_cols if has3 else None,
            "final_stage": st.final_stage, "decision": st.decision,
            "ce_enabled": bool(st.ce_enabled), "n_folds": st.cfg.model.n_folds}
    stg.save_json("meta.json", meta)
    stg.save_json("metrics.json", st.metrics)
    stg.save_json("grid.json", [[s, p] for s, p in st.decision_grid])
    stg.finish()
    _publish(st.art, stg, ["meta.json", "metrics.json"])
    log.info("METRICS %s", json.dumps(st.metrics, default=float))
    return st.metrics


TRAIN_STEPS = [train_load, train_normalize, train_split, train_aliases, train_finalize, train_dense,
               train_blocking, train_stage1, train_cross_encoder, train_stage2, train_stage3,
               train_decision]


def train(data_dir, work_dir, cfg: Config, use_cache: bool = True, **ckpt) -> dict:
    st = train_init(data_dir, work_dir, cfg, use_cache, **ckpt)
    for step in TRAIN_STEPS:
        step(st)
    return st.metrics


# --------------------------------------------------------------------------
# PREDICT
# --------------------------------------------------------------------------
def _artifacts_id(art: Path) -> str:
    """Identity of a set of trained artefacts (content of small files, size of large ones)."""
    h = hashlib.sha1()
    for f in sorted(p for p in art.rglob("*") if p.is_file() and p.name != "metrics.json"):
        h.update(f.relative_to(art).as_posix().encode())
        size = f.stat().st_size
        h.update(f.read_bytes() if size < 64 * 2**20 else str(size).encode())
    return h.hexdigest()[:16]


def predict_init(data_dir, split: str, work_dir, out_dir, use_cache: bool = True,
                 ckpt_uri: str | None = None, s3_region: str | None = None,
                 s3_endpoint: str | None = None) -> SimpleNamespace:
    art = Path(work_dir) / "artifacts"
    ps = SimpleNamespace(data_dir=Path(data_dir), split=split, work=Path(work_dir),
                         out=Path(out_dir), art=art, cfg=Config.load(art / "config.json"),
                         meta=json.loads((art / "meta.json").read_text()),
                         aliases=json.loads((art / "aliases.json").read_text()))
    files = [ps.data_dir / split / f"{split}_source{k}.tsv" for k in (1, 2, 3)]
    ps.ck = Checkpoints(ps.work / "ckpt", ckpt_uri, s3_region, s3_endpoint, enabled=use_cache)
    ps.stg = ps.ck.stage(f"predict-{split}", {"v": CKPT_VERSION, "split": split,
                                               "data": {f.name: f.stat().st_size for f in files},
                                               "artifacts": _artifacts_id(art)})
    log.info("inference checkpoints: %s%s", ps.ck.s3_uri or f"{ps.ck.root}/", ps.stg.key)
    return ps


def predict_prepare(ps) -> None:
    cfg, al, stg = ps.cfg, ps.aliases, ps.stg
    if all(stg.has(f) for f in ("s1.parquet", "x.parquet", "geo.pkl")):
        ps.S1, ps.X = stg.load_frame("s1.parquet"), stg.load_frame("x.parquet")
        ps.geo = stg.load_pickle("geo.pkl")
        log.info("restored normalised %s frames from checkpoint", ps.split)
        return
    with Timer(f"read + normalise {ps.split}"):
        raw = read_split(ps.data_dir, ps.split)
        S1s, Xs = static_normalize(raw, cfg.n_jobs, cfg.geo_min_count)
        del raw
        ps.S1, ps.X = finalize(S1s, Xs, al["name_alias"], al["addr_alias"], set(al["noise_tokens"]))
        del S1s, Xs
        ps.geo = add_record_stats(ps.S1, ps.X, cfg.geo_min_count)
        stg.save_pickle("geo.pkl", ps.geo)
        stg.save_frame("x.parquet", ps.X)
        stg.save_frame("s1.parquet", ps.S1)


def predict_dense(ps) -> None:
    """Embed every record with the trained bi-encoder (only for blocks still to build)."""
    dc, stg = ps.cfg.dense, ps.stg
    ps.emb = None
    if not dc.enabled or all(stg.has(f"cand/{t}.json") for *_, t in block_keys(ps.S1, ps.X)):
        return
    if not ce.gpu_available():
        log.warning("no CUDA GPU: embedding %d records on the CPU will be slow", len(ps.S1) + len(ps.X))
    enc = ps.art / "dense_encoder"
    with Timer(f"dense embeddings of the {ps.split} records"):
        ps.emb = (_embed_parts(stg, "emb_s1", enc, dense.record_text(ps.S1), dc),
                  _embed_parts(stg, "emb_x", enc, dense.record_text(ps.X), dc))


def predict_blocking(ps) -> None:
    stg, bcfg = ps.stg, ps.cfg.blocking
    ranker = lgb.Booster(model_file=str(ps.art / "block_ranker.txt"))
    keys = block_keys(ps.S1, ps.X)
    tags = [t for *_, t in keys]
    with Timer("candidate generation"):
        for country, src, tag in keys:
            if stg.has(f"cand/{tag}.json"):
                continue
            emb = getattr(ps, "emb", None)
            _require_embeddings(ps.cfg, emb, "predict_dense")
            u = block_union(ps.S1, ps.X, country, src, bcfg, ps.cfg.n_jobs, emb=emb)
            n_union = len(u)
            block = prune_candidates(u, ps.X, ranker, bcfg.keep_per_source)
            del u
            stg.save_frame(f"cand/{tag}.parquet", block)
            stg.save_json(f"cand/{tag}.json", {"union_pairs": n_union})   # marks the block complete
        ps.emb = None
        if tags:
            ps.cand = _read_cand(stg, tags)
        else:
            ps.cand = pd.DataFrame({"s1_idx": np.zeros(0, np.int32), "x_idx": np.zeros(0, np.int32),
                                    "src": np.zeros(0, np.int8), "block_score": np.zeros(0, np.float32),
                                    **{c: np.zeros(0, np.float32) for c in COS_COLS},
                                    "via": np.zeros(0, np.uint16)})
        n_union = sum(stg.load_json(f"cand/{t}.json")["union_pairs"] for t in tags)
        log.info("union %d -> %d candidate pairs (%.1f per S1)", n_union,
                 len(ps.cand), len(ps.cand) / max(len(ps.S1), 1))


def predict_stage1(ps) -> None:
    cfg, meta, stg = ps.cfg, ps.meta, ps.stg
    cols = meta["f1_cols"]
    spans = _spans(len(ps.cand))
    ps.f1_parts = [f"F1/part_{i:04d}.npy" for i in range(len(spans))]
    p1_parts = [f"p1/part_{i:04d}.npy" for i in range(len(spans))]
    models, grp = load_models(ps.art, "stage1", meta["n_folds"]), None
    with Timer(f"stage-1 features + scoring ({len(ps.cand):,} rows, {len(spans)} parts on disk)"):
        for f_rel, p_rel, (a, b) in zip(ps.f1_parts, p1_parts, spans):
            if stg.has(p_rel):
                continue
            if grp is None:
                grp = group_features(ps.cand)
            Fn = stage1_matrix(ps.cand, np.arange(a, b), ps.S1, ps.X, ps.geo, grp, cfg.n_jobs,
                               cols).to_numpy(np.float32)
            stg.save_npy(f_rel, Fn)
            stg.save_npy(p_rel, predict_avg(models, Fn))   # written last: marks the part complete
            del Fn
    ps.p1 = np.concatenate([np.load(p) for p in stg.fetch_many(p1_parts)]) if spans else np.zeros(0)


def predict_cross_encoder(ps) -> None:
    ccfg = ps.cfg.cross_encoder
    ps.ce = np.full(len(ps.cand), np.nan, dtype=np.float32)
    if not ps.meta["ce_enabled"]:
        return
    if ps.stg.has("ce.npy"):
        ps.ce = ps.stg.load_npy("ce.npy")
        return
    if not ce.gpu_available():
        raise RuntimeError("These artefacts were trained with the cross-encoder; a CUDA GPU "
                           "is required at inference (or retrain with --no-cross-encoder).")
    with Timer("cross-encoder scoring"):
        rows = np.flatnonzero((ps.p1 >= ccfg.band_low) & (ps.p1 <= ccfg.band_high))
        ps.ce[rows] = _ce_score([ps.art / "ce_0", ps.art / "ce_1"], ps.cand, ps.S1, ps.X, rows, ccfg)
    ps.stg.save_npy("ce.npy", ps.ce)


def predict_stage2(ps) -> None:
    meta, stg = ps.meta, ps.stg
    if stg.has("p2.npy"):
        ps.p2 = stg.load_npy("p2.npy")
        return
    models = load_models(ps.art, "stage2", meta["n_folds"])
    with Timer("context features"):
        ctx = context_features(ps.cand, ps.p1, ps.X, ps.cfg.model.n_anchors)
        ctx["ce"] = ps.ce
        assert meta["f1_cols"] + list(ctx.columns) == meta["f2_cols"], "feature order mismatch"
        C = ctx.to_numpy(np.float32)
        del ctx
    with Timer("stage-2 scoring"):
        ps.p2 = (predict_avg(models, PartsMatrix(stg.fetch_many(ps.f1_parts), extra=C))
                 if len(ps.cand) else np.zeros(0))
    stg.save_npy("p2.npy", ps.p2)


def predict_stage3(ps) -> None:
    """Stage-3 scoring when the decision step chose stage 3; ps.pfin is the final probability."""
    meta, stg = ps.meta, ps.stg
    if meta.get("final_stage", 2) != 3:
        ps.pfin = ps.p2
        return
    if stg.has("p3.npy"):
        ps.pfin = stg.load_npy("p3.npy")
        return
    models = load_models(ps.art, "stage3", meta["n_folds"])
    with Timer("stage-3 context features (from p2)"):
        ctx = stage3_context(ps.cand, ps.p2, ps.X, ps.cfg.model.n_anchors)
        ctx["ce"] = ps.ce
        assert meta["f1_cols"] + list(ctx.columns) == meta["f3_cols"], "feature order mismatch"
        C = ctx.to_numpy(np.float32)
        del ctx
    with Timer("stage-3 scoring"):
        ps.pfin = (predict_avg(models, PartsMatrix(stg.fetch_many(ps.f1_parts), extra=C))
                   if len(ps.cand) else np.zeros(0))
    stg.save_npy("p3.npy", ps.pfin)


def predict_write(ps) -> dict:
    dc = ps.cfg.decision
    p = getattr(ps, "pfin", None)
    p = ps.p2 if p is None else p
    sel = decide(ps.cand["s1_idx"].values, ps.cand["x_idx"].values, p, ps.meta["decision"],
                 dc.beta, dc.max_items)
    ps.selected = sel
    summary = write_outputs(ps.out, ps.S1, ps.X, ps.cand, sel, p)
    ps.stg.finish(summary)
    for rel in getattr(ps, "f1_parts", []):   # stay on S3; free the local disk
        ps.stg.drop_local(rel)
    return summary


PREDICT_STEPS = [predict_prepare, predict_dense, predict_blocking, predict_stage1, predict_cross_encoder,
                 predict_stage2, predict_stage3, predict_write]


def predict(data_dir, split: str, work_dir, out_dir, use_cache: bool = True, **ckpt) -> dict:
    ps = predict_init(data_dir, split, work_dir, out_dir, use_cache, **ckpt)
    out = None
    for step in PREDICT_STEPS:
        out = step(ps)
    return out
