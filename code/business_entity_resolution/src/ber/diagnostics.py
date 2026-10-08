"""Read-only diagnostics of a trained state, as JSON-ready dicts.

The same measurements the notebook prints in 10.1 / 10.2 (blocking), 14.1 /
14.3 / 14.4 (score and where it is lost), 14.5 (ambiguity ceiling) and 15.2
(corpus-size check), for runs without a notebook (SageMaker training jobs).
Training data only; nothing here changes a model or a checkpoint.

Entities evaluated: the hold-out when the run has one; otherwise (hold-out
fraction 0) the fit entities scored by their out-of-fold predictions.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .blocking import VIA_BIT, VIA_DENSE, VIA_SPARSE, VIEWS
from .decision import decide

log = logging.getLogger(__name__)


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def per_entity_f(codes, label, sel, n_gt, beta: float = 0.5):
    """Per-entity F_beta (1 for a singleton predicted empty) plus tp and #predicted."""
    n = len(n_gt)
    label, sel = np.asarray(label, bool), np.asarray(sel, bool)
    tp = np.bincount(codes, weights=(label & sel).astype(float), minlength=n)
    npred = np.bincount(codes, weights=sel.astype(float), minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(n_gt > 0, tp / n_gt, 0.0)
        f = np.where(tp > 0, (1 + beta ** 2) * P * R / (beta ** 2 * P + R), 0.0)
    return np.where((n_gt == 0) & (npred == 0), 1.0, f), tp, npred


def eval_view(st) -> SimpleNamespace:
    """Rows, final probabilities and selection of the evaluated entities."""
    nf, dc = st.n_fit_rows, st.cfg.decision
    if st.is_hold.any():
        ents, rows, p = np.flatnonzero(st.is_hold), st.model_rows[nf:], st.pfin[nf:]
        sel, name = np.asarray(st.holdout_sel, bool), "hold-out"
    else:
        ents, rows, p = np.flatnonzero(st.is_fit), st.model_rows[:nf], st.pfin[:nf]
        sel, name = None, "out-of-fold (fit entities; no hold-out in this run)"
    codes = np.searchsorted(ents, st.cand["s1_idx"].values[rows])
    if sel is None:
        sel = decide(codes, st.cand["x_idx"].values[rows], p, st.decision, dc.beta, dc.max_items)
    return SimpleNamespace(ents=ents, rows=rows, p=p, sel=np.asarray(sel, bool), codes=codes,
                           y=st.y[rows].astype(bool), n_gt=st.n_gt[ents], name=name)


def blocking_report(st) -> dict:
    """10.1 metrics + 10.2 retrieval breakdown of the true candidates."""
    keys = ("union_pairs", "cand_pairs", "cand_per_s1", "union_recall_eval_sparse_only",
            "union_recall_eval", "cand_recall_eval")
    out = {k: _f(st.metrics.get(k)) for k in keys}
    cand = st.cand
    is_true = st.owner[cand["x_idx"].values] == cand["s1_idx"].values
    ev = (st.is_hold if st.is_hold.any() else st.is_fit)[cand["s1_idx"].values] & is_true
    via = cand["via"].values[ev]
    sp, dn = (via & VIA_SPARSE) > 0, (via & VIA_DENSE) > 0
    if len(via):
        out["true_candidates_found_by"] = {"sparse and dense": _f((sp & dn).mean()),
                                           "sparse views only": _f((sp & ~dn).mean()),
                                           "dense view only": _f((~sp & dn).mean())}
        out["true_candidates_found_by_view"] = {
            v: _f(((via & (VIA_BIT[(v, "fwd")] | VIA_BIT[(v, "rev")])) > 0).mean()) for v in (*VIEWS, "dense")}
        rank = cand.groupby(["s1_idx", "src"], sort=False).cumcount().values + 1
        rt = rank[ev]
        out["true_pruner_rank_1_3"] = _f((rt <= 3).mean())
        out["true_pruner_rank_1_6"] = _f((rt <= 6).mean())
    return out


def score_report(st) -> dict:
    """14.1 metrics (stages, decision rule, hold-out scores)."""
    keys = ("cand_recall_eval", "oof_macro_f05_stage2", "oof_macro_f05_stage3", "final_stage", "oof_macro_f05",
            "holdout_macro_f05_stage2", "holdout_macro_f05_stage3", "holdout_macro_f05",
            "holdout_macro_f05_stage1_t05", "holdout_macro_f05_oracle", "holdout_pair_precision",
            "holdout_pair_recall")
    out = {k: (st.metrics.get(k) if k == "final_stage" else _f(st.metrics.get(k))) for k in keys}
    out["decision"] = st.metrics.get("decision")
    return out


def loss_report(st, ev) -> dict:
    """14.3 by number of true matches + 14.4 where the F0.5 is lost."""
    f_act, tp, npred = per_entity_f(ev.codes, ev.y, ev.sel, ev.n_gt)
    f_orc, reach, _ = per_entity_f(ev.codes, ev.y, ev.y, ev.n_gt)
    n_gt, n = ev.n_gt, len(ev.ents)
    kind = np.select(
        [(n_gt == 0) & (npred > 0), (n_gt > 0) & (npred == 0) & (reach > 0),
         (n_gt > 0) & (npred > tp), (n_gt > 0) & (tp < reach)],
        ["singleton given a match (scores 0)", "matchable entity left empty (scores 0)",
         "entity with >=1 wrong match", "entity missing reachable matches (no wrong one)"], "correct")
    loss = pd.DataFrame({"kind": kind, "loss": f_orc - f_act}).groupby("kind")["loss"]
    by_kind = {k: {"entities": int(g.size), "loss_points": _f(g.sum() / n)} for k, g in loss}
    by_n = pd.DataFrame({"m": np.minimum(n_gt, 8), "f": f_act}).groupby("m")["f"].agg(["mean", "size"])
    xi = st.cand["x_idx"].values[ev.rows]
    fp, fn = ev.sel & ~ev.y, ~ev.sel & ev.y
    q = [0.1, 0.25, 0.5, 0.75, 0.9]

    def share(col, mask):
        return _f(st.X[col].values[xi[mask]].mean()) if mask.any() else None

    ctry = st.S1["country"].values[ev.ents]
    by_c = pd.DataFrame({"country": ctry, "f": f_act, "ceiling": f_orc}).groupby("country")
    return {
        "evaluated_on": ev.name, "entities": n,
        "macro_f05": _f(f_act.mean()), "ceiling_with_these_candidates": _f(f_orc.mean()),
        "lost_at_blocking": _f(1 - f_orc.mean()), "lost_by_model_and_decision": _f(f_orc.mean() - f_act.mean()),
        "loss_by_kind": by_kind,
        "f05_by_true_matches": {(str(int(m)) if m < 8 else "8+"): {"f05": _f(r["mean"]), "entities": int(r["size"])}
                                for m, r in by_n.iterrows()},
        "wrong_matches": int(fp.sum()),
        "wrong_match_record_owned_by_other_s1": _f((st.owner[xi[fp]] >= 0).mean()) if fp.any() else None,
        "p_of_missed_true_matches_q10_25_50_75_90": [_f(v) for v in np.quantile(ev.p[fn], q)] if fn.any() else None,
        "p_of_wrong_matches_q10_25_50_75_90": [_f(v) for v in np.quantile(ev.p[fp], q)] if fp.any() else None,
        "by_country": {c: {"f05": _f(g["f"].mean()), "ceiling": _f(g["ceiling"].mean()), "entities": int(len(g))}
                       for c, g in by_c},
        "record_name_non_latin_share": {"all true pairs": share("name_nonlatin", ev.y),
                                        "missed (FN)": share("name_nonlatin", fn), "wrong (FP)": share("name_nonlatin", fp)},
        "record_address_empty_share": {"all true pairs": share("addr_empty", ev.y),
                                       "missed (FN)": share("addr_empty", fn), "wrong (FP)": share("addr_empty", fp)},
    }


def ambiguity_report(st, ev) -> dict:
    """14.5: estimated ceiling when records with an empty address whose entity shares its
    exact core name with other S1 entities of the country are left out (unresolvable)."""
    key = pd.Series(st.S1["country"].values).astype(str) + "|" + pd.Series(st.S1["name_core"].values)
    n_same = key.map(key.value_counts()).values
    gx = pd.Index(np.asarray(st.X["entity_id"].values, dtype=object)).get_indexer(st.gt_x)
    ok = (st.gt_gi >= 0) & (gx >= 0)
    gi, gxx = st.gt_gi[ok], gx[ok]
    in_ev = np.zeros(len(st.S1), bool)
    in_ev[ev.ents] = True
    gi, gxx = gi[in_ev[gi]], gxx[in_ev[gi]]
    amb = (st.X["addr_empty"].values[gxx] == 1) & (n_same[gi] >= 2)
    tp_floor = np.bincount(np.searchsorted(ev.ents, gi[~amb]), minlength=len(ev.ents)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(ev.n_gt > 0, tp_floor / ev.n_gt, 0.0)
        f_ceiling = np.where(tp_floor > 0, 1.25 * R / (0.25 + R), 0.0)
    f_ceiling = np.where(ev.n_gt == 0, 1.0, f_ceiling)
    has_amb = np.bincount(np.searchsorted(ev.ents, gi[amb]), minlength=len(ev.ents)) > 0
    f_now, _, _ = per_entity_f(ev.codes, ev.y, ev.sel, ev.n_gt)
    lost = 1 - f_now
    n = max(len(lost), 1)
    return {"evaluated_on": ev.name, "true_pairs": int(len(amb)), "ambiguous_true_pairs": int(amb.sum()),
            "ambiguous_share": _f(amb.mean()) if len(amb) else 0.0,
            "entities_with_ambiguous_record": int(has_amb.sum()),
            "AMBIGUITY_CEILING": _f(f_ceiling.mean()), "current_macro_f05": _f(f_now.mean()),
            "loss_points_on_entities_with_ambiguous_records": _f(lost[has_amb].sum() / n),
            "loss_points_on_other_entities": _f(lost[~has_amb].sum() / n),
            "irreducible_loss_points": _f((1 - f_ceiling[has_amb]).sum() / n)}


def _f05(pred: set, gt: set) -> float:
    if not gt:
        return float(not pred)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    P, R = tp / len(pred), tp / len(gt)
    return 1.25 * P * R / (0.25 * P + R)


def _dump(path: Path, frame: pd.DataFrame, mask: np.ndarray) -> None:
    # the raw text exactly as read (no field contains a tab), in the challenge's TSV layout
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for r in zip(*(frame[c].values[mask] for c in ("entity_id", "raw_name", "raw_addr", "country"))):
            fh.write("\t".join(r) + "\n")


def corpus_size_check(st, ev, fracs, sim_root: Path, out_root: Path, predict_fn, **predict_kw) -> list[dict]:
    """15.2: rebuild smaller corpora from the TRAINING files (a fraction of the S1 entities,
    all their records, the same fraction of unowned records), run the trained pipeline on each
    exactly as on the test set, and score the hold-out entities inside. Needs a hold-out."""
    if not fracs or not st.is_hold.any():
        return []
    owned = st.owner >= 0
    f_full = pd.Series(per_entity_f(ev.codes, ev.y, ev.sel, ev.n_gt)[0], index=st.S1["entity_id"].values[ev.ents])
    rng = np.random.default_rng(11)
    res = []
    for frac in fracs:
        name = f"sim{int(round(frac * 1000)):04d}"
        pick = rng.random(len(st.S1)) < frac
        keep_x = np.where(owned, pick[np.clip(st.owner, 0, None)], rng.random(len(st.X)) < frac)
        d = Path(sim_root) / name
        if not (d / f"{name}_source3.tsv").exists():
            d.mkdir(parents=True, exist_ok=True)
            _dump(d / f"{name}_source1.tsv", st.S1, pick)
            for s in (2, 3):
                _dump(d / f"{name}_source{s}.tsv", st.X, keep_x & (st.X["src"].values == s))
        out = Path(out_root) / name
        t0 = time.time()
        predict_fn(Path(sim_root), name, st.work, out, **predict_kw)
        pred = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        pm = dict(zip(pred["source1_entity_id"], pred["matched_entity_ids"]))
        ids = st.S1["entity_id"].values[pick & st.is_hold]
        f_sim = np.array([_f05(set(filter(None, pm.get(i, "").split(","))), set(st.gt.get(i, []))) for i in ids])
        res.append({"corpus_size": frac, "s1_entities": int(pick.sum()), "records": int(keep_x.sum()),
                    "holdout_entities_scored": int(len(ids)),
                    "f05_in_this_corpus": _f(f_sim.mean()) if len(ids) else None,
                    "same_entities_full_corpus": _f(f_full.loc[ids].mean()) if len(ids) else None,
                    "minutes": round((time.time() - t0) / 60, 1)})
        log.info("corpus-size check: %s", res[-1])
    return res


def full_report(st) -> dict:
    ev = eval_view(st)
    return {"blocking": blocking_report(st), "score": score_report(st),
            "loss": loss_report(st, ev), "ambiguity": ambiguity_report(st, ev)}
