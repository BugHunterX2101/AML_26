"""Multi-view sparse candidate generation (blocking).

Per (country, target source) block, five TF-IDF "views" are built on the
records of that block only (IDF is a property of the corpus being matched, so
the same procedure adapts to an unseen country such as France):

  name_tok   core-name word tokens + the no-space name (typo-free matches,
             word-order transpositions, domain-style names)
  name_char  character 4-grams of the no-space core name (typos, OCR noise,
             partial transliterations)
  name_skel  phonetic-skeleton tokens of the core name (Indic-script names
             transliterated to Latin: 'teknalaji' ~ 'technology')
  addr       address words, house numbers and (number x word) conjunctions
             inside one address component (robust to component reordering)
  name_addr  (core-name token x address token) conjunctions: very specific
             keys that survive even when every single token is common

Tokens whose document frequency exceeds a cap are dropped (they carry no
blocking signal and make the sparse join quadratic). For every view we take
the forward top-k S2/S3 records per S1 entity and the reverse top-k S1
entities per S2/S3 record, and union everything (one block at a time, so a
run can checkpoint and resume block by block).

A sixth, dense view (dense.py) adds the exact embedding-cosine top-k in both
directions when record embeddings are given; it has no frequency cap. Every
union pair carries ``via``, a bitmask of the (view, direction) lists that
retrieved it (VIA_BIT), and ``cos_dense``.

The union is then pruned to ``keep_per_source`` candidates per (S1, source)
by a small learned ranker on the view cosines and the retrieval bitmask,
trained on the union rows of a sample of training entities - exactly the
distribution it prunes. Records whose single best S1 is this entity are always
kept. The pruned set is exactly what the matching model scores and what is
written to candidate_pairs.tsv.
"""
from __future__ import annotations

import logging
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

from . import dense
from .config import BlockingConfig

log = logging.getLogger(__name__)

VIEWS = ("name_tok", "name_char", "name_skel", "addr", "name_addr")     # sparse views
SPARSE_COS = [f"cos_{v}" for v in VIEWS]
COS_COLS = SPARSE_COS + ["cos_dense"]            # cos_dense is 0 when the dense view is off
# ``via`` bits: forward / reverse list of each view that retrieved the pair
VIA_BIT = {**{(v, d): 1 << (2 * i + j) for i, v in enumerate(VIEWS) for j, d in enumerate(("fwd", "rev"))},
           ("dense", "fwd"): 1 << 10, ("dense", "rev"): 1 << 11}
VIA_SPARSE = (1 << 10) - 1
VIA_DENSE = (1 << 10) | (1 << 11)
_POPCOUNT = np.array([bin(i).count("1") for i in range(1 << 12)], dtype=np.uint8)


def n_via(via: np.ndarray, mask: int = VIA_SPARSE | VIA_DENSE) -> np.ndarray:
    """Number of retrieval lists (of ``mask``) that produced each pair."""
    return _POPCOUNT[np.asarray(via, dtype=np.int64) & mask]


# --------------------------------------------------------------------------
# View documents
# --------------------------------------------------------------------------
def _doc_name_tok(df: pd.DataFrame) -> list[str]:
    out = []
    for core, ns, alt in zip(df["name_core"].values, df["name_ns"].values, df["name_alt"].values):
        toks = set(core.split()) | set(alt.split())
        if ns:
            toks.add("ns:" + ns)
        out.append(" ".join(sorted(toks)))   # sorted: identical matrices on every run
    return out


def _doc_name_skel(df: pd.DataFrame) -> list[str]:
    out = []
    for sk in df["name_skel"].values:
        toks = {t for t in sk.split() if len(t) >= 2}
        if sk:
            toks.add("sk:" + sk.replace(" ", ""))
        out.append(" ".join(sorted(toks)))   # sorted: identical matrices on every run
    return out


def _doc_name_char(df: pd.DataFrame) -> list[str]:
    return [f" {ns} " if ns else "" for ns in df["name_ns"].values]


def _doc_addr(df: pd.DataFrame) -> list[str]:
    out = []
    for comps in df["addr_comps"].values:
        toks = set()
        for comp in comps.split("|") if comps else ():
            ct = comp.split()
            nums = [t for t in ct if t.isdigit()]
            words = [t for t in ct if not t.isdigit() and len(t) >= 2]
            toks.update(words)
            toks.update("n:" + n for n in nums)
            for n in nums[:3]:
                for w in words:
                    if len(w) >= 3:
                        toks.add(f"{n}_{w}")
        out.append(" ".join(sorted(toks)))   # sorted: identical matrices on every run
    return out


def _doc_name_addr(df: pd.DataFrame) -> list[str]:
    out = []
    for core, nums, words in zip(df["name_core"].values, df["addr_nums"].values,
                                 df["addr_words"].values):
        ntoks = [t for t in dict.fromkeys(core.split()) if len(t) >= 2][:4]
        akeys = list(dict.fromkeys(nums.split()))[:4] + \
            [w for w in dict.fromkeys(words.split()) if len(w) >= 4][:8]
        out.append(" ".join(f"{n}^{a}" for n in ntoks for a in akeys))
    return out


_DOC_BUILDERS = {
    "name_tok": _doc_name_tok, "name_char": _doc_name_char, "name_skel": _doc_name_skel,
    "addr": _doc_addr, "name_addr": _doc_name_addr,
}


def _split(s: str) -> list[str]:
    return s.split()


def _vectorizer(view: str, max_df: int) -> TfidfVectorizer:
    if view == "name_char":
        return TfidfVectorizer(analyzer="char", ngram_range=(4, 4), lowercase=False,
                               min_df=2, max_df=max_df, sublinear_tf=True,
                               dtype=np.float32)
    return TfidfVectorizer(analyzer=_split, lowercase=False, min_df=2,
                           max_df=max_df, sublinear_tf=True, dtype=np.float32)


# Columns each view's document builder reads (only these are shipped to workers).
_VIEW_COLS = {"name_tok": ["name_core", "name_ns", "name_alt"], "name_char": ["name_ns"],
              "name_skel": ["name_skel"], "addr": ["addr_comps"],
              "name_addr": ["name_core", "addr_nums", "addr_words"]}


def _fit_view(view: str, s1: pd.DataFrame, xt: pd.DataFrame, max_df: int):
    """(A, B) L2-normalised TF-IDF matrices of one view; vocabulary / IDF fitted on S1 + target."""
    da = _DOC_BUILDERS[view](s1)
    db = _DOC_BUILDERS[view](xt)
    # Absolute max_df must stay above min_df for tiny blocks.
    vec = _vectorizer(view, max(max_df, 3))
    try:
        M = vec.fit_transform(da + db).tocsr()   # identical to fit() then transform(), one tokenisation
    except ValueError:  # empty vocabulary in a tiny block
        return (sp.csr_matrix((len(da), 1), dtype=np.float32),
                sp.csr_matrix((len(db), 1), dtype=np.float32))
    return M[:len(da)], M[len(da):]


def build_view_matrices(s1: pd.DataFrame, xt: pd.DataFrame, cfg: BlockingConfig, n_jobs: int = -1):
    """Return {view: (A, B)} for one block; the five views are built in parallel processes."""
    caps = {"name_tok": cfg.max_df_name_tok, "name_char": cfg.max_df_name_char,
            "name_skel": cfg.max_df_name_skel, "addr": cfg.max_df_addr,
            "name_addr": cfg.max_df_name_addr}
    workers = len(VIEWS) if n_jobs == -1 else max(1, min(len(VIEWS), n_jobs))
    res = Parallel(n_jobs=workers)(
        delayed(_fit_view)(v, s1[_VIEW_COLS[v]], xt[_VIEW_COLS[v]], caps[v]) for v in VIEWS)
    return dict(zip(VIEWS, res))


# --------------------------------------------------------------------------
# Sparse top-k join
# --------------------------------------------------------------------------
def _topk_chunk(A_chunk: sp.csr_matrix, BT: sp.csc_matrix, k: int, row_offset: int):
    M = (A_chunk @ BT).tocsr()
    if M.nnz == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    indptr = M.indptr
    rows = np.repeat(np.arange(M.shape[0]), np.diff(indptr))
    order = np.lexsort((-M.data, rows))
    rank = np.arange(len(order)) - indptr[rows[order]]
    keep = order[rank < k]
    return rows[keep].astype(np.int64) + row_offset, M.indices[keep].astype(np.int64)


def topk_join(A: sp.csr_matrix, B: sp.csr_matrix, k: int, chunk_rows: int, n_jobs: int):
    if A.shape[0] == 0 or B.shape[0] == 0 or k <= 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    BT = B.T.tocsc()
    spans = [(i, min(i + chunk_rows, A.shape[0])) for i in range(0, A.shape[0], chunk_rows)]
    res = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_topk_chunk)(A[a:b], BT, k, a) for a, b in spans
    )
    ra = np.concatenate([r[0] for r in res]) if res else np.empty(0, np.int64)
    rb = np.concatenate([r[1] for r in res]) if res else np.empty(0, np.int64)
    return ra, rb


def rowwise_cosine(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                   chunk: int = 1_000_000) -> np.ndarray:
    out = np.zeros(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


# --------------------------------------------------------------------------
# Candidate generation, one (country, target source) block at a time
# --------------------------------------------------------------------------
def block_keys(S1: pd.DataFrame, X: pd.DataFrame) -> list[tuple[str, int, str]]:
    """(country, source, file tag) of every non-empty block, in a fixed order."""
    keys = []
    countries = sorted(set(S1["country"].unique()) | set(X["country"].unique()))
    for i, country in enumerate(countries):
        if not (S1["country"].values == country).any():
            continue
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(country)).strip("_") or "none"
        for src in (2, 3):
            if ((X["country"].values == country) & (X["src"].values == src)).any():
                keys.append((country, src, f"{i:03d}_{safe}_S{src}"))
    return keys


def block_union(S1: pd.DataFrame, X: pd.DataFrame, country: str, src: int,
                cfg: BlockingConfig, n_jobs: int = -1, emb=None) -> pd.DataFrame:
    """Unpruned candidate union of one block, for every S1 entity of the country.

    ``emb``: optional (S1 embeddings, S2/S3 embeddings), row-aligned with S1 / X,
    which adds the dense view.
    Columns: s1_idx, x_idx, src, cos_* (five sparse view cosines + cos_dense), via.
    """
    ks = {"name_tok": cfg.k_name_tok, "name_char": cfg.k_name_char,
          "name_skel": cfg.k_name_skel, "addr": cfg.k_addr, "name_addr": cfg.k_name_addr}
    s1_pos = np.flatnonzero(S1["country"].values == country)
    x_pos = np.flatnonzero((X["country"].values == country) & (X["src"].values == src))
    mats = build_view_matrices(S1.iloc[s1_pos], X.iloc[x_pos], cfg, n_jobs)
    ra_all, rb_all, bits = [], [], []

    def add(ra, rb, bit):
        ra_all.append(ra)
        rb_all.append(rb)
        bits.append(np.full(len(ra), bit, dtype=np.uint16))

    for view in VIEWS:
        A, B = mats[view]
        fa, fb = topk_join(A, B, ks[view], cfg.chunk_rows, n_jobs)
        rb_, ra_ = topk_join(B, A, cfg.k_reverse, cfg.chunk_rows, n_jobs)
        add(fa, fb, VIA_BIT[(view, "fwd")])
        add(ra_, rb_, VIA_BIT[(view, "rev")])
    E1 = EX = None
    if emb is not None:
        E1, EX = emb[0][s1_pos], emb[1][x_pos]
        fa, fb = dense.topk_dense(E1, EX, cfg.k_dense)
        rb_, ra_ = dense.topk_dense(EX, E1, cfg.k_dense_reverse)
        add(fa, fb, VIA_BIT[("dense", "fwd")])
        add(ra_, rb_, VIA_BIT[("dense", "rev")])
    # union of every list, OR-ing the retrieval bits of duplicate pairs
    key_all = np.concatenate(ra_all) * len(x_pos) + np.concatenate(rb_all)
    bits_all = np.concatenate(bits)
    del ra_all, rb_all, bits
    order = np.argsort(key_all, kind="stable")
    key_all, bits_all = key_all[order], bits_all[order]
    del order
    start = np.r_[0, np.flatnonzero(key_all[1:] != key_all[:-1]) + 1] if len(key_all) else np.zeros(0, np.int64)
    key = key_all[start]
    via = np.bitwise_or.reduceat(bits_all, start) if len(start) else np.zeros(0, np.uint16)
    del key_all, bits_all
    la, lb = key // len(x_pos), key % len(x_pos)
    block = pd.DataFrame({
        "s1_idx": s1_pos[la].astype(np.int32),
        "x_idx": x_pos[lb].astype(np.int32),
        "src": np.int8(src),
    })
    for view in VIEWS:
        A, B = mats[view]
        block[f"cos_{view}"] = rowwise_cosine(A, B, la, lb)
    block["cos_dense"] = dense.rowwise_cos(E1, EX, la, lb) if E1 is not None else np.float32(0.0)
    block["via"] = via.astype(np.uint16)
    log.info("blocking %s / S%d: %d S1 x %d records -> %d union pairs (%d from the dense view only)",
             country, src, len(s1_pos), len(x_pos), len(block), int(((via & VIA_SPARSE) == 0).sum()))
    return block


# --------------------------------------------------------------------------
# Learned pruning
# --------------------------------------------------------------------------
def _ranker_features(cand: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    # the first nine columns are the layout of pruners trained before the dense view
    c = cand[SPARSE_COS].values.astype(np.float32)
    src = X["src"].values[cand["x_idx"].values].astype(np.float32)[:, None]
    empty = X["addr_empty"].values[cand["x_idx"].values].astype(np.float32)[:, None]
    via = cand["via"].values
    return np.hstack([c, c.max(1, keepdims=True), c.sum(1, keepdims=True), src, empty,
                      cand[["cos_dense"]].values.astype(np.float32),
                      n_via(via, VIA_SPARSE).astype(np.float32)[:, None],
                      n_via(via, VIA_DENSE).astype(np.float32)[:, None]])


def fit_ranker(cand: pd.DataFrame, X: pd.DataFrame, labels: np.ndarray, seed: int = 0):
    feats = _ranker_features(cand, X)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63,
                  min_data_in_leaf=500, verbose=-1, seed=seed)
    return lgb.train(params, lgb.Dataset(feats, labels.astype(np.float32)), num_boost_round=300)


def prune_candidates(cand: pd.DataFrame, X: pd.DataFrame, ranker, keep: int) -> pd.DataFrame:
    """Keep the top ``keep`` per (S1, source) plus every record's best S1.

    Called per (country, source) block: every record belongs to exactly one
    block, so "best S1 for this record" is global.
    """
    cand = cand.copy()
    feats = _ranker_features(cand, X)
    cand["block_score"] = ranker.predict(feats[:, :ranker.num_feature()]).astype(np.float32)
    del feats
    cand["src"] = X["src"].values[cand["x_idx"].values].astype(np.int8)
    cand = cand.sort_values(["s1_idx", "src", "block_score"], ascending=[True, True, False])
    rank = cand.groupby(["s1_idx", "src"], sort=False).cumcount()
    best_for_x = cand.groupby("x_idx")["block_score"].transform("max") == cand["block_score"]
    return cand[(rank < keep) | best_for_x].reset_index(drop=True)
