"""Stage checkpoints on local disk, mirrored to S3 so that any instance can resume.

Every expensive step writes its outputs into a *stage* directory whose name
carries a hash of everything those outputs depend on (data, settings and the
upstream stages):

    <root>/<stage>-<hash12>/<file>          local copy (fast working storage)
    <s3_uri><stage>-<hash12>/<file>         durable copy

Each file is written atomically and uploaded as soon as it is complete (one
blocking block, one feature part, one model fold, ...), and a stage is marked
finished by ``_DONE.json``, written last. A later session - after a kernel
crash, or on a brand-new instance with an empty disk - looks for every file
locally first and then on S3, so it continues from the last completed file
instead of from the start of the step.

Settings that only change speed or memory are left out of the hashes. Changing
any other setting starts fresh stages for that step and everything downstream
of it, while the upstream stages are reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DONE = "_DONE.json"


def fingerprint(obj) -> str:
    """Short stable hash of a JSON-serialisable object."""
    text = json.dumps(obj, sort_keys=True, default=list)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link a file (no extra disk) or copy it when linking is impossible."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Checkpoints:
    """Factory of stages under one local root and (optionally) one S3 prefix."""

    def __init__(self, root: Path, s3_uri: str | None = None, region: str | None = None,
                 endpoint: str | None = None, enabled: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.s3_uri = (s3_uri.rstrip("/") + "/") if (s3_uri and enabled) else None
        self.s3 = None
        if self.s3_uri:
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config as BotoConfig

            assert self.s3_uri.startswith("s3://"), f"not an s3:// URI: {s3_uri}"
            self.bucket, _, self.prefix = self.s3_uri[5:].partition("/")
            self.s3 = boto3.client("s3", region_name=region, endpoint_url=endpoint, config=BotoConfig(
                max_pool_connections=32, retries={"max_attempts": 10, "mode": "adaptive"}))
            self.xfer = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20,
                                       max_concurrency=16)

    def stage(self, name: str, fp: dict) -> "Stage":
        return Stage(self, name, fp)


class Stage:
    """One step's checkpoint directory (local + S3)."""

    def __init__(self, ck: Checkpoints, name: str, fp: dict):
        self.ck = ck
        self.name = name
        self.hash = fingerprint(fp)
        self.key = f"{name}-{self.hash}"
        self.dir = ck.root / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        fp_path = self.dir / "_fingerprint.json"
        if not fp_path.exists():
            fp_path.write_text(json.dumps(fp, indent=1, sort_keys=True, default=list))
        self._remote: dict | None = None

    def __repr__(self) -> str:
        return f"Stage({self.key}, done={self.done})"

    # ---------------------------------------------------------------- lookup
    def _s3_key(self, rel: str) -> str:
        return f"{self.ck.prefix}{self.key}/{rel}"

    def remote(self) -> dict:
        """rel path -> size of every object of this stage on S3 (listed once)."""
        if self._remote is None:
            self._remote = {}
            if self.ck.s3 is not None:
                pre = self._s3_key("")
                pages = self.ck.s3.get_paginator("list_objects_v2").paginate(
                    Bucket=self.ck.bucket, Prefix=pre)
                for page in pages:
                    for obj in page.get("Contents", []):
                        self._remote[obj["Key"][len(pre):]] = obj["Size"]
        return self._remote

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def has(self, rel: str) -> bool:
        """Is this file complete, locally or on S3? (local files are written atomically)"""
        if not self.ck.enabled:
            return False
        return self.path(rel).exists() or rel in self.remote()

    @property
    def done(self) -> bool:
        return self.has(DONE)

    def fetch(self, rel: str) -> Path:
        """Local path of a checkpoint file, downloading it from S3 when needed."""
        p = self.path(rel)
        if p.exists():
            return p
        if rel not in self.remote():
            raise FileNotFoundError(f"{rel} is in neither {self.dir} nor {self.ck.s3_uri}{self.key}/")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        self.ck.s3.download_file(self.ck.bucket, self._s3_key(rel), str(tmp), Config=self.ck.xfer)
        tmp.replace(p)
        return p

    def fetch_many(self, rels: list[str]) -> list[Path]:
        missing = [r for r in rels if not self.path(r).exists()]
        if missing:
            log.info("downloading %d checkpoint file(s) of %s from S3 (%.2f GB)", len(missing), self.key,
                     sum(self.remote().get(r, 0) for r in missing) / 1e9)
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(self.fetch, missing))
        return [self.path(r) for r in rels]

    def fetch_dir(self, rel_dir: str) -> Path:
        pre = rel_dir.rstrip("/") + "/"
        self.fetch_many([r for r in self.remote() if r.startswith(pre)])
        return self.path(rel_dir)

    # ---------------------------------------------------------------- upload
    def commit(self, rel: str) -> Path:
        """Upload a completed local file (or a whole directory) to S3."""
        p = self.path(rel)
        files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
        if self.ck.s3 is not None and files:
            def up(q: Path) -> None:
                r = q.relative_to(self.dir).as_posix()
                self.ck.s3.upload_file(str(q), self.ck.bucket, self._s3_key(r), Config=self.ck.xfer)
                self.remote()[r] = q.stat().st_size

            with ThreadPoolExecutor(8) as ex:
                list(ex.map(up, files))
        return p

    def _atomic(self, rel: str, write) -> Path:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        write(tmp)
        tmp.replace(p)
        return self.commit(rel)

    def save_bytes(self, rel: str, data: bytes) -> Path:
        return self._atomic(rel, lambda t: t.write_bytes(data))

    def save_json(self, rel: str, obj) -> Path:
        return self.save_bytes(rel, json.dumps(obj, indent=1, default=float).encode("utf-8"))

    def save_pickle(self, rel: str, obj) -> Path:
        return self.save_bytes(rel, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def save_npy(self, rel: str, arr: np.ndarray) -> Path:
        def write(t: Path) -> None:
            with open(t, "wb") as f:
                np.save(f, np.asarray(arr))
        return self._atomic(rel, write)

    def save_frame(self, rel: str, df: pd.DataFrame) -> Path:
        return self._atomic(rel, lambda t: df.to_parquet(t, index=False))

    def save_model(self, rel: str, booster, num_iteration: int | None = None) -> Path:
        return self._atomic(rel, lambda t: booster.save_model(str(t), num_iteration=num_iteration))

    # ---------------------------------------------------------------- load
    def load_json(self, rel: str):
        return json.loads(self.fetch(rel).read_text(encoding="utf-8"))

    def load_pickle(self, rel: str):
        return pickle.loads(self.fetch(rel).read_bytes())

    def load_npy(self, rel: str, mmap: bool = False) -> np.ndarray:
        return np.load(self.fetch(rel), mmap_mode="r" if mmap else None)

    def load_frame(self, rel: str) -> pd.DataFrame:
        return pd.read_parquet(self.fetch(rel))

    # ---------------------------------------------------------------- lifecycle
    def finish(self, info: dict | None = None) -> None:
        """Mark the stage complete (after uploading anything a crash left local-only)."""
        if self.ck.s3 is not None:
            remote = self.remote()
            for q in self.dir.rglob("*"):
                r = q.relative_to(self.dir).as_posix()
                if q.is_file() and r != DONE and r not in remote and not q.name.endswith((".tmp", ".part")):
                    self.commit(r)
        self.save_json(DONE, info or {})

    def drop_local(self, rel: str) -> None:
        """Free local disk: delete a local copy that is safely on S3 (kept when there is no S3)."""
        p = self.path(rel)
        if self.ck.s3 is None or not p.exists():
            return
        files = [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
        if all(q.relative_to(self.dir).as_posix() in self.remote() for q in files):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
