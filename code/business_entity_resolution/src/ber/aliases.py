"""Data-driven token aliases and noise tokens, learned from training labels only.

For every labelled (S1, S2/S3) true pair we look at the tokens that differ
between the two records. Token ``u`` that repeatedly appears in the S2/S3
record exactly when token ``v`` is missing from it (and ``u`` is rare in the
clean S1 source) is a spelling/abbreviation/transliteration variant of ``v``:
e.g. texas->tx, tg->telangana, praivet->pvt, madras->chennai. S1 is the clean
reference, so its form is the canonical one.

Name tokens that are frequently *inserted* into S2/S3 names without being in
the S1 name (center, services, shri, ...) are learned as noise tokens and are
dropped from the "core" name view.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import Indel

from .normalize import HONORIFIC_TOKENS, skeleton

log = logging.getLogger(__name__)


def _resolve_chains(amap: dict) -> dict:
    out = {}
    for u, v in amap.items():
        seen = {u}
        while v in amap and v not in seen:
            seen.add(v)
            v = amap[v]
        if v != u:
            out[u] = v
    return out


def _mine(s1_tok: np.ndarray, x_tok: np.ndarray, s1_corpus: pd.Series,
          min_count: int, max_diff: int = 4) -> dict:
    s1_freq = Counter(t for s in s1_corpus.values for t in s.split())
    co: Counter = Counter()
    du: Counter = Counter()
    for a, b in zip(s1_tok, x_tok):
        ta, tb = set(a.split()), set(b.split())
        d1 = [t for t in ta - tb if not t.isdigit()]
        dx = [t for t in tb - ta if not t.isdigit()]
        if not d1 or not dx or len(d1) > max_diff or len(dx) > max_diff:
            continue
        for u in dx:
            du[u] += 1
            for v in d1:
                co[(u, v)] += 1
    cands: dict = {}
    for (u, v), c in co.items():
        if c >= min_count:
            cands.setdefault(u, []).append((c, v))
    amap = {}
    for u, lst in cands.items():
        if u in HONORIFIC_TOKENS:
            continue
        for c, v in sorted(lst, reverse=True):
            if (c / du[u] >= 0.3 and u != v and _plausible(u, v)
                    and s1_freq.get(u, 0) <= 0.05 * s1_freq.get(v, 0)):
                amap[u] = v
                break
    return _resolve_chains(amap)


def _plausible(u: str, v: str) -> bool:
    """An alias must look like its target: same phonetic onset or similar spelling.

    Rejects co-occurrence artefacts such as 'mr'->'ltd' or 'li'->'pvt' while
    keeping abbreviations (tg->telangana), transliterations (phud->food) and
    typos (lnc->inc).
    """
    su, sv = skeleton(u.replace("_", "")), skeleton(v.replace("_", ""))
    return (su[:1] == sv[:1]) or u[:1] == v[:1] or Indel.normalized_similarity(u, v) >= 0.5


def _noise_tokens(s1_tok: np.ndarray, x_tok: np.ndarray, min_count: int = 200,
                  min_rate: float = 0.6) -> set:
    ins: Counter = Counter()
    tot: Counter = Counter()
    for a, b in zip(s1_tok, x_tok):
        ta = set(a.split())
        for t in set(b.split()):
            tot[t] += 1
            if t not in ta:
                ins[t] += 1
    return {t for t, n in tot.items()
            if n >= min_count and ins[t] / n >= min_rate and not t.isdigit()}


def learn_aliases(s1n: pd.DataFrame, xn: pd.DataFrame, pairs: pd.DataFrame,
                  min_count: int, max_pairs: int = 3_000_000, seed: int = 0):
    """pairs: DataFrame with columns s1_id, x_id (true matches, training split)."""
    if len(pairs) > max_pairs:
        pairs = pairs.sample(max_pairs, random_state=seed)
    s1_pos = pd.Series(np.arange(len(s1n)), index=s1n["entity_id"].values)
    x_pos = pd.Series(np.arange(len(xn)), index=xn["entity_id"].values)
    ok = pairs["s1_id"].isin(s1_pos.index) & pairs["x_id"].isin(x_pos.index)
    pairs = pairs[ok]
    ia = s1_pos.loc[pairs["s1_id"].values].values
    ib = x_pos.loc[pairs["x_id"].values].values

    name_s1 = s1n["name_tok"].values[ia]
    name_x = xn["name_tok"].values[ib]
    addr_s1 = s1n["addr_tok"].values[ia]
    addr_x = xn["addr_tok"].values[ib]

    name_alias = _mine(name_s1, name_x, s1n["name_tok"], min_count)
    addr_alias = _mine(addr_s1, addr_x, s1n["addr_tok"], min_count)
    noise = _noise_tokens(np.array([" ".join(name_alias.get(t, t) for t in s.split()) for s in name_s1]),
                          np.array([" ".join(name_alias.get(t, t) for t in s.split()) for s in name_x]))
    log.info("learned %d name aliases, %d address aliases, %d noise tokens",
             len(name_alias), len(addr_alias), len(noise))
    return name_alias, addr_alias, noise
