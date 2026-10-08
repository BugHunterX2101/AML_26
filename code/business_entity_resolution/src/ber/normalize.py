"""Deterministic, country-agnostic text normalisation.

Two passes:
  * ``normalize_frame``   - static rules (transliteration, punctuation / noise
    cleanup, legal-form and street-type canonicalisation, number parsing).
  * ``finalize_frame``    - applies the aliases learned from training labels
    (see ``aliases.py``) and derives the views used by blocking and features
    (core name, no-space name, phonetic skeleton, address words / numbers).

Nothing here is specific to US / India: the rules are generic, the French legal
forms and street types are included because France is announced for the test
set, and every learned alias comes from the training data.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from anyascii import anyascii
from joblib import Parallel, delayed

# --------------------------------------------------------------------------
# Static dictionaries (generic domain knowledge, no external lookup)
# --------------------------------------------------------------------------
LEGAL_CANON = {
    # English / Indian
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt",
    "limited": "ltd", "ltd": "ltd", "limitd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cos": "co", "compnay": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pc": "pc", "plc": "plc",
    "pllc": "pllc", "lllp": "lllp", "pa": "pa", "opc": "opc",
    # French / continental
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl", "scop": "scop",
    "scm": "scm", "sca": "sca", "eirl": "eirl", "gie": "gie", "societe": "ste",
    "ste": "ste", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl",
    "spa": "spa",
}
LEGAL_TOKENS = set(LEGAL_CANON.values())

# Tokens that decorate a name without identifying it. The data-driven list
# learned in aliases.py is added on top of this one.
HONORIFIC_TOKENS = {
    "the", "m", "s", "ms", "messrs", "mr", "mrs", "dr", "shri", "sri", "smt",
    "sh", "and", "of", "dba", "aka", "le", "la", "les", "l", "du", "de", "des",
    "et", "d",
}

STREET_CANON = {
    "street": "st", "str": "st", "saint": "st", "st": "st", "sainte": "ste",
    "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "circle": "cir",
    "cir": "cir", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "pky": "pkwy", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "suite": "ste",
    "ste": "ste", "apartment": "apt", "apt": "apt", "building": "bldg",
    "bldg": "bldg", "floor": "fl", "flr": "fl", "fl": "fl", "north": "n",
    "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "number": "no", "num": "no",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "expressway": "expy",
    "freeway": "fwy", "way": "way", "plaza": "plz", "plz": "plz",
    # French
    "rue": "rue", "chemin": "ch", "impasse": "imp", "imp": "imp",
    "allee": "all", "route": "rte", "rte": "rte", "faubourg": "fbg",
    "fbg": "fbg", "quai": "quai", "cours": "crs", "residence": "res",
    "batiment": "bat", "bat": "bat", "etage": "fl", "lieu": "ld",
    # India
    "marg": "marg", "nagar": "nagar", "nr": "near", "opp": "opposite",
    "opposite": "opposite",
}

ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3",
}

NULL_WORDS = {"null", "none", "nan", "nil", "undefined"}

_DOMAIN_RE = re.compile(
    r"\b(?:https?\s*:?\s*/*\s*)?(?:www\s*\.\s*)?([a-z0-9][a-z0-9\-]*)\s*\.\s*"
    r"(?:com|c0m|net|org|0rg|in|co|biz|info|us|fr|io|online|shop|store|app)\b"
)
_ACRONYM_RE = re.compile(r"\b(?:[a-z]\s?\.\s?){2,}")
_ID_NOISE_RE = re.compile(r"\(?\b(?:id|ref|no)\s*[:#]\s*\d+\)?")
_MS_RE = re.compile(r"\bm\s*/\s*s\b")
_DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|aka|t/a|trading as)\b")
_POBOX_RE = re.compile(r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s*box|boite\s*postale|bp)\s*#?\s*(\d+)")
_NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bnone\b|\bnan\b")
_ORDINAL_SUFFIX_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme)\b")
_TOKEN_RE = re.compile(r"[a-z]+|\d+")
_NONLATIN_RE = re.compile(r"[^\x00-\u024f\u1e00-\u1eff\u2000-\u206f\u20a0-\u20cf]")
_OCR_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _fold(s: str) -> str:
    """Unicode-normalise, transliterate any script to ASCII, lowercase."""
    s = unicodedata.normalize("NFKC", s)
    return anyascii(s).lower()


def _fix_ocr(tok: str) -> str:
    """Fix digit-for-letter substitutions inside a word ('h0mes', 'zi1via')."""
    if tok.isalpha() or tok.isdigit():
        return tok
    letters = sum(ch.isalpha() for ch in tok)
    if letters >= 2 and letters >= 2 * (len(tok) - letters):
        return tok.translate(_OCR_MAP)
    return tok


def _name_tokens(s: str) -> list[str]:
    s = _MS_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = _ID_NOISE_RE.sub(" ", s)
    s = _DOMAIN_RE.sub(lambda m: " " + m.group(1).replace("-", "") + " ", s)
    s = _ACRONYM_RE.sub(lambda m: " " + re.sub(r"[\s.]", "", m.group(0)) + " ", s)
    s = s.replace("'", "").replace("`", "")
    toks = re.findall(r"[a-z0-9]+", s)
    out = []
    for t in toks:
        t = _fix_ocr(t)
        out.append(LEGAL_CANON.get(t, t))
    return out


def normalize_name(raw: str) -> tuple[str, str, int, int, int]:
    """Return (tokens, dba_alt_tokens, nonlatin, is_domain, has_dba)."""
    nonlatin = int(bool(_NONLATIN_RE.search(raw)))
    s = _fold(raw)
    is_domain = int(bool(_DOMAIN_RE.search(s)) or s.strip().startswith("#"))
    parts = _DBA_RE.split(s)
    has_dba = int(len(parts) > 1)
    toks = _name_tokens(" ".join(parts))
    alt = _name_tokens(parts[-1]) if has_dba else toks
    return " ".join(toks), " ".join(alt), nonlatin, is_domain, has_dba


def normalize_address(raw: str) -> tuple[str, str, int]:
    """Return (tokens, components joined by '|', has_pobox)."""
    s = _fold(raw)
    s = _NULL_RE.sub(" ", s)
    has_pobox = 0
    m = _POBOX_RE.search(s)
    if m:
        has_pobox = 1
        s = _POBOX_RE.sub(" ", s)
    s = _ORDINAL_SUFFIX_RE.sub(r"\1", s)
    comps_out, all_toks = [], []
    for comp in s.split(","):
        toks = []
        for t in _TOKEN_RE.findall(comp):
            if t.isdigit():
                t = t.lstrip("0")
                if not t:
                    continue
            else:
                if t in NULL_WORDS:
                    continue
                t = ORDINAL_WORDS.get(t, t)
                t = STREET_CANON.get(t, t)
            toks.append(t)
        if toks:
            comps_out.append(" ".join(toks))
            all_toks.extend(toks)
    return " ".join(all_toks), "|".join(comps_out), has_pobox


def _normalize_chunk(names: list[str], addrs: list[str]):
    n_out = [normalize_name(x) for x in names]
    a_out = [normalize_address(x) for x in addrs]
    return n_out, a_out


def normalize_frame(df: pd.DataFrame, n_jobs: int = -1, chunk: int = 50_000) -> pd.DataFrame:
    """Static normalisation of a source frame (parallel over chunks)."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    spans = [(i, min(i + chunk, len(df))) for i in range(0, len(df), chunk)]
    res = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_normalize_chunk)(names[a:b], addrs[a:b]) for a, b in spans
    )
    n_rows = [r for part in res for r in part[0]]
    a_rows = [r for part in res for r in part[1]]
    out = pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "country": df["country"].values,
        "raw_name": df["business_name"].values,
        "raw_addr": df["business_address"].values,
    })
    out["name_tok"] = [r[0] for r in n_rows]
    out["name_alt_tok"] = [r[1] for r in n_rows]
    out["name_nonlatin"] = np.array([r[2] for r in n_rows], dtype=np.int8)
    out["name_is_domain"] = np.array([r[3] for r in n_rows], dtype=np.int8)
    out["name_has_dba"] = np.array([r[4] for r in n_rows], dtype=np.int8)
    out["addr_tok"] = [r[0] for r in a_rows]
    out["addr_comps"] = [r[1] for r in a_rows]
    out["addr_pobox"] = np.array([r[2] for r in a_rows], dtype=np.int8)
    return out


# --------------------------------------------------------------------------
# Multi-word locality phrases ("uttar pradesh", "north carolina",
# "nouvelle aquitaine") are joined into one token so that an abbreviation or a
# transliteration can be aliased to the whole phrase (up -> uttar_pradesh).
# The phrase list is read, without labels, from the S1 records of whichever
# corpus is being processed, so it adapts to an unseen country.
# --------------------------------------------------------------------------
def learn_phrases(S1: pd.DataFrame, min_count: int) -> set:
    counts: dict = {}
    for cs in S1["addr_comps"].values:
        for c in cs.split("|") if cs else ():
            toks = c.split()
            if 2 <= len(toks) <= 3 and not any(t.isdigit() for t in toks):
                counts[c] = counts.get(c, 0) + 1
    return {tuple(c.split()) for c, n in counts.items() if n >= min_count}


def _join_chunk(comps_list: list[str], phrases: set) -> tuple[list[str], list[str]]:
    out_c, out_t = [], []
    for cs in comps_list:
        new_comps = []
        for c in cs.split("|") if cs else ():
            toks = c.split()
            res, i = [], 0
            while i < len(toks):
                if i + 3 <= len(toks) and tuple(toks[i:i + 3]) in phrases:
                    res.append("_".join(toks[i:i + 3]))
                    i += 3
                elif i + 2 <= len(toks) and tuple(toks[i:i + 2]) in phrases:
                    res.append("_".join(toks[i:i + 2]))
                    i += 2
                else:
                    res.append(toks[i])
                    i += 1
            new_comps.append(" ".join(res))
        out_c.append("|".join(new_comps))
        out_t.append(" ".join(new_comps))
    return out_c, out_t


def join_phrases(df: pd.DataFrame, phrases: set, n_jobs: int = -1, chunk: int = 200_000) -> None:
    """Rewrite addr_comps / addr_tok in place with joined locality phrases."""
    if not phrases:
        return
    comps = df["addr_comps"].tolist()
    spans = [(i, min(i + chunk, len(comps))) for i in range(0, len(comps), chunk)]
    res = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_join_chunk)(comps[a:b], phrases) for a, b in spans
    )
    df["addr_comps"] = [c for part in res for c in part[0]]
    df["addr_tok"] = [t for part in res for t in part[1]]


# --------------------------------------------------------------------------
# Phonetic skeleton: bridges transliterations ('praivet' ~ 'private',
# 'kelksi knstrksns' ~ 'galaxy constructions', 'sophtveyr' ~ 'software').
# --------------------------------------------------------------------------
_DIGRAPHS = [("ph", "f"), ("bh", "v"), ("sh", "s"), ("ch", "c"), ("th", "t"),
             ("dh", "t"), ("kh", "k"), ("gh", "k"), ("ck", "k"), ("qu", "k"),
             ("x", "ks")]
_SKEL_MAP = str.maketrans({"b": "p", "d": "t", "g": "k", "j": "k", "c": "k",
                           "q": "k", "z": "s", "w": "v"})


def skeleton(tok: str) -> str:
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _DIGRAPHS:
        t = t.replace(a, b)
    t = t.translate(_SKEL_MAP)
    head, tail = t[0], re.sub(r"[aeiouyh]", "", t[1:])
    out = [head]
    for ch in tail:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def _apply_alias(tokstr: str, amap: dict) -> str:
    if not amap or not tokstr:
        return tokstr
    return " ".join(amap.get(t, t) for t in tokstr.split())


def finalize_frame(df: pd.DataFrame, name_alias: dict, addr_alias: dict,
                   noise_tokens: set) -> pd.DataFrame:
    """Apply learned aliases and derive every string view used downstream."""
    drop = LEGAL_TOKENS | HONORIFIC_TOKENS | set(noise_tokens)
    soft_drop = LEGAL_TOKENS | {"the", "m", "s", "and", "of"}

    name_full, name_core, name_ns, name_skel, legal, alt_core = [], [], [], [], [], []
    for tok, alt in zip(df["name_tok"].values, df["name_alt_tok"].values):
        toks = _apply_alias(tok, name_alias).split()
        core = [t for t in toks if t not in drop]
        if not core:
            core = [t for t in toks if t not in soft_drop] or toks
        atoks = _apply_alias(alt, name_alias).split()
        acore = [t for t in atoks if t not in drop] or core
        name_full.append(" ".join(toks))
        name_core.append(" ".join(core))
        name_ns.append("".join(core))
        name_skel.append(" ".join(skeleton(t) for t in core))
        legal.append(" ".join(sorted({t for t in toks if t in LEGAL_TOKENS})))
        alt_core.append(" ".join(acore))

    addr_full, addr_words, addr_nums, addr_comps = [], [], [], []
    for tok, comps in zip(df["addr_tok"].values, df["addr_comps"].values):
        toks = _apply_alias(tok, addr_alias).split()
        addr_full.append(" ".join(toks))
        addr_words.append(" ".join(t for t in toks if not t.isdigit()))
        addr_nums.append(" ".join(t for t in toks if t.isdigit()))
        addr_comps.append("|".join(_apply_alias(c, addr_alias) for c in comps.split("|")) if comps else "")

    out = df[["entity_id", "country", "raw_name", "raw_addr", "name_nonlatin",
              "name_is_domain", "name_has_dba", "addr_pobox"]].copy()
    out["name_full"] = name_full
    out["name_core"] = name_core
    out["name_ns"] = name_ns
    out["name_skel"] = name_skel
    out["name_legal"] = legal
    out["name_alt"] = alt_core
    out["addr_full"] = addr_full
    out["addr_words"] = addr_words
    out["addr_nums"] = addr_nums
    out["addr_comps"] = addr_comps
    out["addr_empty"] = (out["addr_full"] == "").astype(np.int8)
    return out
