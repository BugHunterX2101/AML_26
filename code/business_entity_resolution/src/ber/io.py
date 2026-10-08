"""TSV readers / writers. Every file in the challenge is tab-separated."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path) -> pd.DataFrame:
    """Read one source file as plain strings (empty string for missing)."""
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        quoting=csv.QUOTE_NONE, engine="c", encoding="utf-8",
    )
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[SOURCE_COLS].copy()
    for c in SOURCE_COLS:
        df[c] = df[c].fillna("").astype(str)
    df["entity_id"] = df["entity_id"].str.strip()
    df["country"] = df["country"].str.strip()
    df = df[df["entity_id"] != ""].drop_duplicates("entity_id").reset_index(drop=True)
    log.info("read %s: %d rows", Path(path).name, len(df))
    return df


def read_split(data_dir: Path, split: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir) / split
    return {
        src: read_source(data_dir / f"{split}_source{src[-1]}.tsv")
        for src in ("s1", "s2", "s3")
    }


def read_ground_truth(path: Path) -> dict[str, list[str]]:
    gt: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            s1 = s1.strip()
            if s1:
                # a record listed twice is one match (the metric works on sets); an entity on
                # two lines keeps the records of both
                ids = gt.get(s1, []) + [x.strip() for x in rest.split(",") if x.strip()]
                gt[s1] = list(dict.fromkeys(ids))
    log.info("read ground truth: %d S1 entities", len(gt))
    return gt


def write_id_lists(path: Path, header: tuple[str, str], s1_ids, mapping: dict) -> None:
    """Write one row per S1 id; ids are de-duplicated preserving order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s1 in s1_ids:
            ids = mapping.get(s1, ())
            seen, out = set(), []
            for x in ids:
                if x not in seen and (x.startswith("S2-") or x.startswith("S3-")):
                    seen.add(x)
                    out.append(x)
            f.write(f"{s1}\t{','.join(out)}\n")
    log.info("wrote %s (%d rows)", path, len(s1_ids))
