"""Pairwise and contextual features for candidate (S1, S2/S3) pairs.

Stage-1 features describe one pair in isolation (name / address similarity on
several normalised views, house-number agreement, locality agreement, name
genericness, blocking scores). Stage-2 context features describe the pair
relative to its competitors: the other candidates of the same S1 entity, the
other S1 entities claiming the same S2/S3 record, and the agreement of the
record with the S1 entity's most probable matches ("anchors").

All features are country-agnostic similarities / counts; no country one-hot.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein, Prefix
from rapidfuzz.process import cpdist

from .blocking import COS_COLS, VIA_DENSE, VIA_SPARSE, n_via

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Record-level statistics (computed on the corpus being matched)
# --------------------------------------------------------------------------
def add_record_stats(S1: pd.DataFrame, X: pd.DataFrame, geo_min_count: int) -> set:
    """Adds genericness columns in place; returns the locality vocabulary."""
    both = pd.concat([S1[["country", "name_core"]], X[["country", "name_core"]]], ignore_index=True)
    key = both["country"] + "|" + both["name_core"]
    cnt = key.value_counts()
    S1["name_dup"] = np.log1p((S1["country"] + "|" + S1["name_core"]).map(cnt).fillna(0).values).astype(np.float32)
    X["name_dup"] = np.log1p((X["country"] + "|" + X["name_core"]).map(cnt).fillna(0).values).astype(np.float32)

    toks = both.assign(t=both["name_core"].str.split()).explode("t").dropna(subset=["t"])
    tdf = (toks["country"] + "|" + toks["t"]).value_counts()
    for frame in (S1, X):
        ex = frame[["country", "name_core"]].assign(t=frame["name_core"].str.split()).explode("t")
        d = (ex["country"] + "|" + ex["t"].fillna("")).map(tdf).fillna(1.0)
        g = d.groupby(level=0)
        frame["name_tok_mindf"] = np.log1p(g.min().reindex(frame.index).fillna(1.0).values).astype(np.float32)
        frame["name_tok_maxdf"] = np.log1p(g.max().reindex(frame.index).fillna(1.0).values).astype(np.float32)

    # Locality vocabulary: digit-free address components that recur in S1.
    comps = Counter()
    for country, cs in zip(S1["country"].values, S1["addr_comps"].values):
        for c in cs.split("|") if cs else ():
            if c and not any(ch.isdigit() for ch in c):
                comps[(country, c)] += 1
    geo = {k for k, v in comps.items() if v >= geo_min_count}
    log.info("locality vocabulary: %d components", len(geo))
    return geo


# --------------------------------------------------------------------------
# Python-loop features (numbers / components), parallel over chunks
# --------------------------------------------------------------------------
def _num_comp_chunk(n1, n2, c1, c2, ctry, geo):
    out = np.zeros((len(n1), 15), dtype=np.float32)
    for i in range(len(n1)):
        a = n1[i].split()
        b = n2[i].split()
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        uni = len(sa | sb)
        best = 0.0
        if a and b:
            for x in sa:
                for y in sb:
                    if len(x) >= 2 or len(y) >= 2:
                        s = Levenshtein.normalized_similarity(x, y)
                        if s > best:
                            best = s
        out[i, 0] = len(sa)
        out[i, 1] = len(sb)
        out[i, 2] = inter
        out[i, 3] = inter / uni if uni else np.nan
        out[i, 4] = float(a[0] == b[0]) if a and b else np.nan
        out[i, 5] = float(a[0] in sb) if a and b else np.nan
        out[i, 6] = best if a and b else np.nan
        big_a = {x for x in sa if len(x) >= 3}
        out[i, 7] = float(bool(big_a & sb)) if big_a and b else np.nan

        ca = [c for c in c1[i].split("|") if c]
        cb = [c for c in c2[i].split("|") if c]
        ga = {c for c in ca if (ctry[i], c) in geo}
        gb = {c for c in cb if (ctry[i], c) in geo}
        gi = len(ga & gb)
        out[i, 8] = len(ga)
        out[i, 9] = len(gb)
        out[i, 10] = gi
        out[i, 11] = float(gi == 0) if ga and gb else np.nan
        sca, scb = set(ca), set(cb)
        cu = len(sca | scb)
        out[i, 12] = len(sca & scb) / cu if cu else np.nan
        out[i, 13] = float(ga <= gb or gb <= ga) if ga and gb else np.nan
        if a and b and len(a[0]) <= 9 and len(b[0]) <= 9:
            out[i, 14] = np.log1p(abs(int(a[0]) - int(b[0])))
        else:
            out[i, 14] = np.nan
    return out


_NUM_COMP_COLS = ["num_n1", "num_n2", "num_inter", "num_jacc", "num_first_eq",
                  "num_first_in", "num_best_sim", "num_big_overlap", "geo_n1",
                  "geo_n2", "geo_inter", "geo_conflict", "comp_jacc", "geo_subset",
                  "num_first_logdiff"]


def _street_part(comps: np.ndarray, ctry: np.ndarray, geo: set) -> list[str]:
    out = []
    for cs, c in zip(comps, ctry):
        out.append(" ".join(p for p in cs.split("|") if p and (c, p) not in geo) if cs else "")
    return out


def _sim(q, c, scorer, **kw) -> np.ndarray:
    return cpdist(q, c, scorer=scorer, workers=-1, dtype=np.float32, **kw)


# --------------------------------------------------------------------------
# Stage-1 feature matrix
# --------------------------------------------------------------------------
def pair_features(cand: pd.DataFrame, S1: pd.DataFrame, X: pd.DataFrame, geo: set,
                  n_jobs: int = -1, chunk: int = 2_000_000) -> pd.DataFrame:
    parts = []
    for s in range(0, len(cand), chunk):
        parts.append(_pair_features_chunk(cand.iloc[s:s + chunk], S1, X, geo, n_jobs))
        log.info("features: %d / %d pairs", min(s + chunk, len(cand)), len(cand))
    feats = pd.concat(parts, ignore_index=True)
    feats.index = cand.index
    return feats


def _pair_features_chunk(cand, S1, X, geo, n_jobs) -> pd.DataFrame:
    ia = cand["s1_idx"].values
    ib = cand["x_idx"].values
    g1 = lambda col: S1[col].values[ia]  # noqa: E731
    g2 = lambda col: X[col].values[ib]  # noqa: E731
    f = {}
    for c in COS_COLS + ["block_score"]:
        f[c] = cand[c].values.astype(np.float32)
    f["src"] = cand["src"].values.astype(np.float32)
    # how many retrieval lists found the pair (agreement of independent views)
    f["via_n_sparse"] = n_via(cand["via"].values, VIA_SPARSE).astype(np.float32)
    f["via_n_dense"] = n_via(cand["via"].values, VIA_DENSE).astype(np.float32)

    a_full, b_full = list(g1("name_full")), list(g2("name_full"))
    a_core, b_core = list(g1("name_core")), list(g2("name_core"))
    a_ns, b_ns = list(g1("name_ns")), list(g2("name_ns"))
    a_sk, b_sk = list(g1("name_skel")), list(g2("name_skel"))
    b_alt = list(g2("name_alt"))

    f["nm_full_ratio"] = _sim(a_full, b_full, fuzz.ratio) / 100
    f["nm_full_tset"] = _sim(a_full, b_full, fuzz.token_set_ratio) / 100
    f["nm_full_partial"] = _sim(a_full, b_full, fuzz.partial_ratio) / 100
    f["nm_core_ratio"] = _sim(a_core, b_core, fuzz.ratio) / 100
    f["nm_core_tset"] = _sim(a_core, b_core, fuzz.token_set_ratio) / 100
    f["nm_core_tsort"] = _sim(a_core, b_core, fuzz.token_sort_ratio) / 100
    f["nm_core_jw"] = _sim(a_core, b_core, JaroWinkler.normalized_similarity)
    f["nm_ns_lev"] = _sim(a_ns, b_ns, Levenshtein.normalized_similarity)
    f["nm_ns_partial"] = _sim(a_ns, b_ns, fuzz.partial_ratio) / 100
    f["nm_ns_prefix"] = _sim(a_ns, b_ns, Prefix.normalized_similarity)
    f["nm_ns_eq"] = (np.array(a_ns, dtype=object) == np.array(b_ns, dtype=object)).astype(np.float32)
    f["nm_skel_ratio"] = _sim(a_sk, b_sk, fuzz.ratio) / 100
    f["nm_skel_tset"] = _sim(a_sk, b_sk, fuzz.token_set_ratio) / 100
    f["nm_alt_tset"] = _sim(a_core, b_alt, fuzz.token_set_ratio) / 100
    f["nm_len1"] = np.array([len(x) for x in a_ns], dtype=np.float32)
    f["nm_len2"] = np.array([len(x) for x in b_ns], dtype=np.float32)
    f["nm_ntok1"] = np.array([x.count(" ") + 1 if x else 0 for x in a_core], dtype=np.float32)
    f["nm_ntok2"] = np.array([x.count(" ") + 1 if x else 0 for x in b_core], dtype=np.float32)

    la, lb = g1("name_legal"), g2("name_legal")
    both = (la != "") & (lb != "")
    f["legal_both"] = both.astype(np.float32)
    f["legal_eq"] = np.where(both, (la == lb).astype(np.float32), np.nan).astype(np.float32)
    disjoint = np.array([not (set(x.split()) & set(y.split())) for x, y in zip(la, lb)])
    f["legal_conflict"] = np.where(both, disjoint.astype(np.float32), np.nan).astype(np.float32)

    for col in ("name_nonlatin", "name_is_domain", "name_has_dba", "addr_empty", "addr_pobox"):
        f[f"x_{col}"] = g2(col).astype(np.float32)
    for col in ("name_dup", "name_tok_mindf", "name_tok_maxdf"):
        f[f"s1_{col}"] = g1(col)
        f[f"x_{col}"] = g2(col)

    a_ad, b_ad = list(g1("addr_full")), list(g2("addr_full"))
    a_aw, b_aw = list(g1("addr_words")), list(g2("addr_words"))
    f["ad_full_ratio"] = _sim(a_ad, b_ad, fuzz.ratio) / 100
    f["ad_full_tset"] = _sim(a_ad, b_ad, fuzz.token_set_ratio) / 100
    f["ad_full_tsort"] = _sim(a_ad, b_ad, fuzz.token_sort_ratio) / 100
    f["ad_words_tset"] = _sim(a_aw, b_aw, fuzz.token_set_ratio) / 100
    f["ad_words_partial"] = _sim(a_aw, b_aw, fuzz.partial_token_set_ratio) / 100
    ctry = g1("country")
    sa = _street_part(g1("addr_comps"), ctry, geo)
    sb = _street_part(g2("addr_comps"), ctry, geo)
    f["ad_street_tset"] = _sim(sa, sb, fuzz.token_set_ratio) / 100
    f["ad_street_ratio"] = _sim(sa, sb, fuzz.ratio) / 100
    f["ad_len1"] = np.array([len(x) for x in a_ad], dtype=np.float32)
    f["ad_len2"] = np.array([len(x) for x in b_ad], dtype=np.float32)

    n1, n2 = g1("addr_nums"), g2("addr_nums")
    c1, c2 = g1("addr_comps"), g2("addr_comps")
    step = 200_000
    spans = [(i, min(i + step, len(cand))) for i in range(0, len(cand), step)]
    res = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_num_comp_chunk)(n1[a:b], n2[a:b], c1[a:b], c2[a:b], ctry[a:b], geo)
        for a, b in spans
    )
    nc = np.vstack(res) if res else np.zeros((0, len(_NUM_COMP_COLS)), np.float32)
    for j, col in enumerate(_NUM_COMP_COLS):
        f[col] = nc[:, j]

    # Joint name+address evidence.
    f["joint_min"] = np.minimum(f["nm_core_tset"], f["ad_full_tset"])
    f["joint_prod"] = f["nm_core_tset"] * f["ad_full_tset"]
    return pd.DataFrame(f)


def group_features(cand: pd.DataFrame) -> pd.DataFrame:
    """Rank / competition features on the blocking score, over ALL candidates."""
    bs = cand["block_score"]
    g_s1s = cand.groupby(["s1_idx", "src"])["block_score"]
    g_x = cand.groupby("x_idx")["block_score"]
    dn = cand["cos_dense"]
    d_s1s = cand.groupby(["s1_idx", "src"])["cos_dense"]
    d_x = cand.groupby("x_idx")["cos_dense"]
    out = pd.DataFrame({
        "blk_rank_s1src": g_s1s.rank(ascending=False, method="first").values,
        "blk_gap_s1src": (bs - g_s1s.transform("max")).values,
        "blk_rank_x": g_x.rank(ascending=False, method="first").values,
        "blk_gap_x": (bs - g_x.transform("max")).values,
        "n_claim_x": cand.groupby("x_idx")["s1_idx"].transform("size").values,
        "n_cand_s1": cand.groupby("s1_idx")["x_idx"].transform("size").values,
        # the same competition view on the embedding cosine (ties -> equal rank, so a
        # constant cos_dense when the dense view is off gives a constant feature)
        "dns_rank_s1src": d_s1s.rank(ascending=False, method="min").values,
        "dns_gap_s1src": (dn - d_s1s.transform("max")).values,
        "dns_rank_x": d_x.rank(ascending=False, method="min").values,
        "dns_gap_x": (dn - d_x.transform("max")).values,
    })
    return out.astype(np.float32)


def stage1_matrix(cand: pd.DataFrame, rows: np.ndarray, S1: pd.DataFrame, X: pd.DataFrame,
                  geo: set, grp: pd.DataFrame, n_jobs: int = -1,
                  cols: list | None = None) -> pd.DataFrame:
    """Stage-1 features for candidate rows ``rows`` (pairwise + group columns)."""
    F = pair_features(cand.iloc[rows], S1, X, geo, n_jobs).reset_index(drop=True)
    for c in grp.columns:
        F[c] = grp[c].values[rows]
    if cols is not None:
        F = F[cols]
    return F


# --------------------------------------------------------------------------
# Stage-2 context features from stage-1 probabilities
# --------------------------------------------------------------------------
def _second_largest(keys: np.ndarray, p: np.ndarray) -> np.ndarray:
    order = np.lexsort((-p, keys))
    ks, ps = keys[order], p[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    rank = np.arange(len(ks)) - np.maximum.accumulate(np.where(first, np.arange(len(ks)), 0))
    sec = pd.Series(ps[rank == 1], index=ks[rank == 1])
    return pd.Series(keys).map(sec).fillna(0.0).values


def context_features(cand: pd.DataFrame, p1: np.ndarray, X: pd.DataFrame,
                     n_anchors: int, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Context features for candidate ``rows`` (default: all rows).

    Aggregates are computed over ALL candidates (every competing S1 / record),
    anchors only for the requested rows.
    """
    df = pd.DataFrame({"s1": cand["s1_idx"].values, "x": cand["x_idx"].values,
                       "src": cand["src"].values, "p": p1.astype(np.float32)})
    if rows is None:
        rows = np.arange(len(df))
    g_s1 = df.groupby("s1")["p"]
    g_s1s = df.groupby(["s1", "src"])["p"]
    g_x = df.groupby("x")["p"]
    hi = (df["p"] > 0.5).astype(np.float32)
    out = {"p1": df["p"].values[rows]}
    out["p1_rank_s1"] = g_s1.rank(ascending=False, method="first").values[rows]
    out["p1_rank_s1src"] = g_s1s.rank(ascending=False, method="first").values[rows]
    out["p1_gap_s1src"] = (df["p"] - g_s1s.transform("max")).values[rows]
    out["p1_sum_s1"] = g_s1.transform("sum").values[rows]
    out["p1_sum_s1src"] = g_s1s.transform("sum").values[rows]
    out["p1_nhi_s1"] = hi.groupby(df["s1"]).transform("sum").values[rows]
    out["p1_rank_x"] = g_x.rank(ascending=False, method="first").values[rows]
    out["p1_gap_x"] = (df["p"] - g_x.transform("max")).values[rows]
    out["p1_x_second"] = _second_largest(df["x"].values, df["p"].values)[rows]
    out["p1_nhi_x"] = hi.groupby(df["x"]).transform("sum").values[rows]

    # Anchor agreement: similarity of this record to the entity's most
    # probable other candidates (siblings of the same real-world business).
    order = df.sort_values(["s1", "p"], ascending=[True, False])
    order["r"] = order.groupby("s1").cumcount()
    top = order[order["r"] < n_anchors]
    anc_x = top.pivot(index="s1", columns="r", values="x")
    anc_p = top.pivot(index="s1", columns="r", values="p")
    names = X["name_full"].values
    addrs = X["addr_full"].values
    s1r = df["s1"].values[rows]
    xr = df["x"].values[rows]
    best_n = np.zeros(len(rows), np.float32)
    best_a = np.zeros(len(rows), np.float32)
    best_j = np.zeros(len(rows), np.float32)
    for r in anc_x.columns:
        ax = anc_x[r].reindex(s1r).values.astype(np.float64)
        ap = anc_p[r].reindex(s1r).values.astype(np.float64)
        valid = ~np.isnan(ax)
        valid[valid] &= ax[valid].astype(np.int64) != xr[valid]
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            continue
        axi = ax[idx].astype(np.int64)
        xi = xr[idx]
        ns = _sim(list(names[xi]), list(names[axi]), fuzz.token_set_ratio) / 100
        as_ = _sim(list(addrs[xi]), list(addrs[axi]), fuzz.token_set_ratio) / 100
        w = ap[idx].astype(np.float32)
        best_n[idx] = np.maximum(best_n[idx], w * ns)
        best_a[idx] = np.maximum(best_a[idx], w * as_)
        best_j[idx] = np.maximum(best_j[idx], w * np.minimum(ns, as_))
    out["anc_name"] = best_n
    out["anc_addr"] = best_a
    out["anc_joint"] = best_j
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def stage3_context(cand: pd.DataFrame, p2: np.ndarray, X: pd.DataFrame,
                   n_anchors: int, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Stage-3 context: the same competition / anchor features recomputed from
    the stage-2 probabilities (prefix ``q_``), plus the record's margin over its
    best competing claimant - the quantity that decides who owns a record."""
    ctx = context_features(cand, p2, X, n_anchors, rows).add_prefix("q_")
    rows = np.arange(len(cand)) if rows is None else rows
    p = p2.astype(np.float32)
    best_x = pd.Series(p).groupby(cand["x_idx"].values).transform("max").values
    second_x = _second_largest(cand["x_idx"].values, p)
    # own p minus the strongest OTHER claimant of the same record
    other = np.where(p >= best_x, second_x, best_x)
    ctx["q_claim_margin"] = (p - other)[rows].astype(np.float32)
    return ctx
