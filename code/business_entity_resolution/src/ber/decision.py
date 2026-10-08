"""From pair probabilities to per-entity match lists, optimised for macro F0.5.

The metric is F_beta computed per S1 entity and averaged, with singletons
scoring 1 for an empty prediction and 0 otherwise. For one entity with
independent match probabilities p_1 >= p_2 >= ... the optimal prediction is
always a top-k prefix, and its expected F_beta can be computed exactly:

  E[F | k] = sum_{a,b} P(TP=a | top-k) * P(FN=b | rest) * (1+b^2) a / (b^2 (a+b) + k)
  E[F | 0] = P(no true match at all)

(Poisson-binomial distributions via DP). We choose k* = argmax_k E[F | k]
per entity - this handles singletons, multi-match entities and ambiguous
cases in one principled rule instead of a single global threshold. A
temperature on the logits (fitted on out-of-fold predictions) corrects
calibration. A global-threshold rule and an "each S2/S3 record belongs to at
most one S1" exclusivity step are evaluated as alternatives; the best rule on
out-of-fold data is kept.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------
def macro_fbeta(s1_codes: np.ndarray, label: np.ndarray, selected: np.ndarray,
                n_gt: np.ndarray, beta: float = 0.5) -> float:
    """Vectorised macro F_beta.

    s1_codes : int array (rows) - dense code of the S1 entity of each candidate row
    label    : 0/1 per row (is the candidate a true match)
    selected : bool per row (is it predicted)
    n_gt     : per-entity number of true matches (length = number of entities,
               including entities without any candidate row)
    """
    n = len(n_gt)
    tp = np.bincount(s1_codes, weights=(label * selected).astype(np.float64), minlength=n)
    npred = np.bincount(s1_codes, weights=selected.astype(np.float64), minlength=n)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(n_gt > 0, tp / n_gt, 0.0)
        f = np.where(tp > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    f = np.where((n_gt == 0) & (npred == 0), 1.0, f)
    return float(f.mean())


# --------------------------------------------------------------------------
# Expected-F optimal selection
# --------------------------------------------------------------------------
def _weights(m: int, beta: float) -> np.ndarray:
    b2 = beta * beta
    W = np.zeros((m + 1, m + 1, m + 1))
    for k in range(1, m + 1):
        for a in range(1, k + 1):
            for b in range(0, m - k + 1):
                W[k, a, b] = (1 + b2) * a / (b2 * (a + b) + k)
    return W


def _poisson_binomial_prefix(P: np.ndarray) -> np.ndarray:
    n, m = P.shape
    out = np.zeros((n, m + 1, m + 1))
    out[:, 0, 0] = 1.0
    for k in range(1, m + 1):
        p = P[:, k - 1][:, None]
        out[:, k, :] = out[:, k - 1, :] * (1 - p)
        out[:, k, 1:] += out[:, k - 1, :-1] * p
    return out


def _expected_best_k(P: np.ndarray, W: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    n, m = P.shape
    best = np.zeros(n, dtype=np.int64)
    for s in range(0, n, chunk):
        Pc = P[s:s + chunk]
        pre = _poisson_binomial_prefix(Pc)
        suf = _poisson_binomial_prefix(Pc[:, ::-1])[:, ::-1, :]  # suf[:, k] = items k..m-1
        # E[n, k] = sum_{a,b} pre[n,k,a] * W[k,a,b] * suf[n,k,b], as m+1 small matrix products
        E = np.empty((len(Pc), m + 1))
        for k in range(m + 1):
            E[:, k] = ((pre[:, k, :] @ W[k]) * suf[:, k, :]).sum(axis=1)
        E[:, 0] = suf[:, 0, 0]
        best[s:s + chunk] = E.argmax(axis=1)
    return best


def _temper(p: np.ndarray, T: float) -> np.ndarray:
    if T == 1.0:
        return p
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / T
    return 1.0 / (1.0 + np.exp(-z))


def _groups(s1_codes: np.ndarray, p: np.ndarray, max_items: int):
    """Per-entity ranking of rows by probability. A temperature is monotone, so
    the same ranking serves every temperature of the grid."""
    order = np.lexsort((-p, s1_codes))
    codes_sorted = s1_codes[order]
    first = np.r_[True, codes_sorted[1:] != codes_sorted[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - start
    grp = np.cumsum(first) - 1
    n_grp = int(grp[-1] + 1) if len(grp) else 0
    return order, rank, grp, n_grp, rank < max_items


def select_expected_f(s1_codes: np.ndarray, p: np.ndarray, beta: float, max_items: int,
                      temperature: float = 1.0, groups=None, W=None) -> np.ndarray:
    order, rank, grp, n_grp, keep = groups if groups is not None else _groups(s1_codes, p, max_items)
    P = np.zeros((n_grp, max_items))
    P[grp[keep], rank[keep]] = _temper(p[order][keep], temperature)
    kstar = _expected_best_k(P, W if W is not None else _weights(max_items, beta))
    selected = np.zeros(len(p), dtype=bool)
    selected[order] = rank < kstar[grp]
    return selected


def apply_exclusivity(x_codes: np.ndarray, p: np.ndarray, selected: np.ndarray,
                      p_order: np.ndarray | None = None) -> np.ndarray:
    """Each S2/S3 record may belong to at most one S1 entity: keep the most probable."""
    if not selected.any():
        return selected
    if p_order is None:
        p_order = np.argsort(-p, kind="stable")
    idx = p_order[selected[p_order]]                      # selected rows, most probable first
    _, first = np.unique(x_codes[idx], return_index=True)
    out = np.zeros_like(selected)
    out[idx[first]] = True
    return out


def decide(s1_codes, x_codes, p, params: dict, beta: float, max_items: int) -> np.ndarray:
    if params["rule"] == "expf":
        sel = select_expected_f(s1_codes, p, beta, max_items, params["temperature"])
    else:
        sel = p >= params["threshold"]
    if params["exclusive"]:
        sel = apply_exclusivity(x_codes, p, sel)
    return sel


def tune_decision(s1_codes, x_codes, p, label, n_gt, cfg) -> tuple[dict, float, list]:
    """Grid-search the decision rule on out-of-fold predictions.

    The per-entity ranking, the F-weights and the probability order are
    computed once and shared by every rule of the grid; each selection is
    scored with and without exclusivity.
    """
    groups = _groups(s1_codes, p, cfg.max_items)
    W = _weights(cfg.max_items, cfg.beta)
    p_order = np.argsort(-p, kind="stable")
    rules = [({"rule": "expf", "temperature": float(T)},
              lambda T=T: select_expected_f(s1_codes, p, cfg.beta, cfg.max_items, T, groups, W))
             for T in cfg.temperatures]
    rules += [({"rule": "threshold", "threshold": float(t)}, lambda t=t: p >= t) for t in cfg.thresholds]
    results = []
    for base, select in rules:
        sel = select()
        for excl in (False, True):
            s = apply_exclusivity(x_codes, p, sel, p_order) if excl else sel
            results.append((macro_fbeta(s1_codes, label, s, n_gt, cfg.beta), {**base, "exclusive": excl}))
    results.sort(key=lambda r: -r[0])
    for score, params in results[:5]:
        log.info("decision candidate %.5f %s", score, params)
    return results[0][1], results[0][0], [(s, p_) for s, p_ in results]
