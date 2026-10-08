"""Entry point of the SageMaker TRAINING JOBS (runs inside the job's container).

Launched by ``launch_jobs.py``, never by hand. Three modes (hyperparameter ``mode``):

  check    environment preflight (packages, GPU, shared memory, disk, model downloads,
           S3 access) - a few minutes; run it once before the first long job.
  train    channel "train" = s3://.../dataset/train/. Trains every stage with EXACTLY the
           configuration of the notebook (cell 3.2 + 12.1), so every checkpoint key is the
           same and every stage already finished by the notebook - or by an earlier or
           interrupted job - is restored from the shared S3 checkpoints instead of being
           recomputed. Writes the diagnostics of notebook cells 10.1/10.2/14.1/14.3-14.5
           (and 15.2) and uploads artefacts + reports to <output_uri>runs/<run_name>/.
  predict  channels "test" = s3://.../dataset/test/, "artifacts" = <run>/artifacts/ and
           optionally "utils" = s3://.../utils/ (official validator). Writes both
           submission files, checks them, and uploads them to <run>/output/.

The test split is never mounted in a training job: it only exists in the predict job.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
# joblib / loky worker processes must import the library too
os.environ["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(CODE_DIR))
for _k, _v in (("USE_TF", "0"), ("USE_TORCH", "1"), ("USE_FLAX", "0"), ("USE_JAX", "0"),
               ("TRANSFORMERS_NO_TF", "1"), ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_k, _v)

INPUT = Path(os.environ.get("SM_INPUT_DIR", "/opt/ml/input")) / "data"
OUTPUT = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
MODEL = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def _free_gb(p: Path) -> float:
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def choose_work_dir() -> Path:
    """Working storage (local checkpoint copies, embeddings, feature parts: 100+ GB) must be on
    the job's ML storage volume, not the container's root filesystem. Which container folders
    are backed by that volume depends on the instance family, so take the candidate with the
    most free space. A folder under /opt/ml/model is removed at the end of a successful job
    (after everything is on S3), so the model archive stays tiny."""
    if os.environ.get("BER_WORK_DIR"):
        return Path(os.environ["BER_WORK_DIR"])
    cands = [MODEL / "ber_work", Path("/opt/ml/checkpoints/ber_work"), INPUT.parent / "ber_work", Path("/tmp/ber_work")]
    return max(cands, key=_free_gb)


WORK = choose_work_dir()
os.environ.setdefault("HF_HOME", str(WORK / "hf"))                 # model downloads on the big volume
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(WORK / "joblib"))  # large arrays shared with workers: not /dev/shm

log = logging.getLogger("ber.job")


def _cleanup_work() -> None:
    if MODEL in WORK.parents and WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)


def _bool(s) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _floats(s) -> tuple:
    s = str(s).strip().strip("[]()")
    return tuple(float(x) for x in s.replace(" ", "").split(",") if x) if s and s.lower() != "none" else ()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("check", "train", "predict"), required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--output_uri", required=True, help="s3://.../ber_outputs/")
    ap.add_argument("--ckpt_uri", default="", help="defaults to <output_uri>checkpoints/ (shared with the notebook)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-south-1"))
    # the notebook's switches (cell 3.1); the defaults are the notebook's defaults
    ap.add_argument("--holdout_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--profile", default="auto", choices=("auto", "full", "large", "medium"))
    ap.add_argument("--block_k_fwd", type=int, default=15)
    ap.add_argument("--block_k_rev", type=int, default=5)
    ap.add_argument("--block_cap_x", type=float, default=1)
    ap.add_argument("--use_dense", default="true")
    ap.add_argument("--use_cross_encoder", default="true")
    ap.add_argument("--use_stage3", default="true")
    ap.add_argument("--ce_model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--dense_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--size_check_fracs", default="0.1,0.3")
    ap.add_argument("--use_cache", default="true")
    a, unknown = ap.parse_known_args(argv)
    if unknown:
        log.info("ignoring extra arguments: %s", unknown)
    a.use_dense, a.use_cross_encoder = _bool(a.use_dense), _bool(a.use_cross_encoder)
    a.use_stage3, a.use_cache = _bool(a.use_stage3), _bool(a.use_cache)
    a.size_check_fracs = _floats(a.size_check_fracs)
    a.output_uri = a.output_uri.rstrip("/") + "/"
    a.ckpt_uri = (a.ckpt_uri or a.output_uri + "checkpoints/").rstrip("/") + "/"
    a.run_uri = f"{a.output_uri}runs/{a.run_name}/"
    return a


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return max(1, (os.cpu_count() or 2) // 2)


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except ImportError:
        return ""


def shm_gb() -> float:
    try:
        return shutil.disk_usage("/dev/shm").total / 1e9
    except OSError:
        return 0.0


def _model_source(channel: str, hub_name: str) -> str:
    """A pretrained model from an optional input channel (offline jobs), else its Hub name."""
    d = INPUT / channel
    return str(d) if d.is_dir() and any(d.iterdir()) else hub_name


def _loadable(name: str) -> str | None:
    try:
        from ber import cross_encoder
        cross_encoder._quiet_transformers()
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(name)
        return None
    except Exception as e:        # no internet / no weights
        return f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Configuration: identical to notebook cell 3.2 (+ 12.1), so the checkpoint keys match
# --------------------------------------------------------------------------
def build_config(a, gpu: str, ram: float, n_phys: int):
    from ber.config import Config

    profile = a.profile
    if profile == "auto":
        profile = "full" if ram >= 60 else "large" if ram >= 40 else "medium" if ram >= 24 else "dev"
    if profile == "dev":
        raise SystemExit(f"{ram:.0f} GB of RAM is too little for the full corpus - use ml.g6.8xlarge / ml.g5.8xlarge")
    has_gpu = bool(gpu)
    cfg = Config()
    cfg.n_jobs = -1
    cfg.model.seed = a.seed
    cfg.model.holdout_frac = a.holdout_frac
    cfg.model.max_train_s1 = {"full": None, "large": 900_000, "medium": 400_000}[profile]
    cfg.model.ranker_s1 = 150_000
    for v in ("name_tok", "name_char", "name_skel", "addr", "name_addr"):
        setattr(cfg.blocking, f"k_{v}", int(a.block_k_fwd))
        setattr(cfg.blocking, f"max_df_{v}", int(getattr(cfg.blocking, f"max_df_{v}") * a.block_cap_x))
    cfg.blocking.k_reverse = int(a.block_k_rev)
    cfg.model.stage3 = a.use_stage3
    for p in (cfg.model.lgb_stage1, cfg.model.lgb_stage2, cfg.model.lgb_stage3):
        p["num_threads"] = n_phys                      # speed only (not in any checkpoint key)
    cfg.cross_encoder.enabled = a.use_cross_encoder and has_gpu
    g = gpu.upper()
    if "A100" in g or "H100" in g:
        cfg.cross_encoder.max_train_pairs, cfg.cross_encoder.batch_size, cfg.cross_encoder.infer_batch_size = 2_000_000, 128, 1024
    elif "L4" in g or "A10" in g:
        cfg.cross_encoder.max_train_pairs, cfg.cross_encoder.batch_size, cfg.cross_encoder.infer_batch_size = 1_200_000, 96, 512
    else:  # T4 / V100 / other 16 GB cards
        cfg.cross_encoder.max_train_pairs, cfg.cross_encoder.batch_size, cfg.cross_encoder.infer_batch_size = 350_000, 64, 256
    cfg.cross_encoder.model_name = a.ce_model
    if "large" in a.ce_model.lower():
        cfg.cross_encoder.batch_size = max(16, cfg.cross_encoder.batch_size // 3)
        cfg.cross_encoder.infer_batch_size = max(64, cfg.cross_encoder.infer_batch_size // 2)
        cfg.cross_encoder.lr = 1e-5
    cfg.dense.enabled = a.use_dense and has_gpu
    if a.dense_model:
        cfg.dense.model_name = a.dense_model
    return cfg, profile


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------
def _s3(region):
    import boto3
    from botocore.config import Config as BotoConfig
    return boto3.client("s3", region_name=region,
                        config=BotoConfig(max_pool_connections=32, retries={"max_attempts": 10, "mode": "adaptive"}))


def s3_upload(region: str, path: Path, uri: str) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from boto3.s3.transfer import TransferConfig
    s3 = _s3(region)
    bucket, _, key = uri[5:].partition("/")
    path = Path(path)
    if path.is_file():
        jobs = [(path, key + path.name if key.endswith("/") else key)]
    else:
        jobs = [(p, f"{key.rstrip('/')}/{p.relative_to(path).as_posix()}") for p in path.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts]
    xfer = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=16)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: s3.upload_file(str(j[0]), bucket, j[1], Config=xfer), jobs))
    log.info("uploaded %s -> %s (%d file(s))", path, uri, len(jobs))


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def _metric_lines(d: dict, prefix: str = "") -> None:
    # "METRIC name=value" lines feed the job's CloudWatch metric definitions
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(f"METRIC {prefix}{k}={v}", flush=True)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def env_report(a) -> dict:
    gpu, ram = gpu_name(), ram_gb()
    WORK.mkdir(parents=True, exist_ok=True)
    rep = {"python": sys.version.split()[0], "gpu": gpu or "none", "ram_gb": round(ram, 1),
           "cpu_logical": os.cpu_count(), "cpu_physical": physical_cores(), "shm_gb": round(shm_gb(), 2),
           "work_dir": str(WORK), "work_free_gb": round(shutil.disk_usage(WORK).free / 1e9, 1),
           "channels": sorted(p.name for p in INPUT.iterdir()) if INPUT.is_dir() else []}
    import importlib
    for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "rapidfuzz", "anyascii", "pyarrow",
              "joblib", "torch", "transformers", "sentencepiece", "boto3"):
        try:
            rep[m] = getattr(importlib.import_module(m), "__version__", "installed")
        except Exception as e:
            rep[m] = f"MISSING ({type(e).__name__})"
    return rep


def mode_check(a) -> None:
    rep = env_report(a)
    rep["ce_model"] = _loadable(_model_source("ce_model", a.ce_model)) or "loadable"
    rep["dense_model"] = _loadable(_model_source("dense_model", a.dense_model)) or "loadable"
    bucket, _, prefix = a.output_uri[5:].partition("/")
    try:
        _s3(a.region).list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        rep["s3_output"] = "accessible"
    except Exception as e:
        rep["s3_output"] = f"NOT accessible: {type(e).__name__}: {e}"
    problems = [k for k, v in rep.items() if isinstance(v, str) and ("MISSING" in v or "Error" in v or "NOT " in v)]
    if rep["gpu"] == "none":
        problems.append("gpu")
    if rep["ram_gb"] < 60:
        problems.append("ram_gb (< 60 GB: not the full profile)")
    if rep["work_free_gb"] < 150:
        problems.append("work_free_gb (< 150 GB: raise the volume size)")
    rep["problems"] = problems
    print(json.dumps(rep, indent=1), flush=True)
    _write_json(OUTPUT / "environment_check.json", rep)
    s3_upload(a.region, OUTPUT / "environment_check.json", a.run_uri + "reports/")
    if problems:
        raise SystemExit(f"environment check FAILED: {problems}")
    print("environment check PASSED", flush=True)


def mode_train(a) -> None:
    from ber import diagnostics, pipeline

    assert (INPUT / "train" / "train_source1.tsv").exists(), "channel 'train' must hold dataset/train/*.tsv"
    assert not (INPUT / "test").exists(), "the test split must not be mounted in a training job"
    rep = env_report(a)
    log.info("environment: %s", rep)
    cfg, profile = build_config(a, gpu_name(), ram_gb(), physical_cores())
    # pretrained weights: an optional input channel (offline jobs) or the Hugging Face Hub
    cfg.cross_encoder.model_name = _model_source("ce_model", cfg.cross_encoder.model_name)
    cfg.dense.model_name = _model_source("dense_model", cfg.dense.model_name)
    if cfg.dense.enabled and (err := _loadable(cfg.dense.model_name)):
        log.warning("dense view disabled - cannot load %s: %s", cfg.dense.model_name, err)
        cfg.dense.enabled = False
    if cfg.cross_encoder.enabled and (err := _loadable(cfg.cross_encoder.model_name)):
        log.warning("cross-encoder disabled - cannot load %s: %s", cfg.cross_encoder.model_name, err)
        cfg.cross_encoder.enabled = False
    if shm_gb() < 2:          # DataLoader worker processes need shared memory (speed only)
        cfg.cross_encoder.num_workers = 0
    log.info("profile=%s  dense=%s  cross-encoder=%s  stage3=%s  hold-out=%s", profile, cfg.dense.enabled,
             cfg.cross_encoder.enabled, cfg.model.stage3, cfg.model.holdout_frac)

    ck = dict(ckpt_uri=a.ckpt_uri if a.use_cache else None, s3_region=a.region)
    st = pipeline.train_init(INPUT, WORK, cfg, use_cache=a.use_cache, **ck)
    stages = {k: {"key": v.key, "restored": bool(v.done)} for k, v in st.stg.items()}
    log.info("checkpoint stages: %s", stages)
    t0 = time.time()
    for step in pipeline.TRAIN_STEPS:
        ts = time.time()
        step(st)
        log.info("step %s finished in %.1f min", step.__name__, (time.time() - ts) / 60)
    log.info("training finished in %.1f h", (time.time() - t0) / 3600)

    report = {"run_name": a.run_name, "profile": profile, "environment": rep, "stages": stages,
              "config": asdict(cfg), **diagnostics.full_report(st)}
    _write_json(OUTPUT / "report.json", report)
    s3_upload(a.region, st.art, a.run_uri + "artifacts/")          # the predict job's input
    s3_upload(a.region, OUTPUT / "report.json", a.run_uri + "reports/")
    _metric_lines(report["score"])
    _metric_lines({k: v for k, v in report["blocking"].items() if not isinstance(v, dict)})
    print(f"METRIC ambiguity_ceiling={report['ambiguity']['AMBIGUITY_CEILING']}", flush=True)

    if a.size_check_fracs and st.is_hold.any():                   # 15.2 (read-only, training data only)
        report["corpus_size_check"] = diagnostics.corpus_size_check(
            st, diagnostics.eval_view(st), a.size_check_fracs, WORK / "sim_data", WORK / "sim_out",
            pipeline.predict, use_cache=a.use_cache, **ck)
        _write_json(OUTPUT / "report.json", report)
        s3_upload(a.region, OUTPUT / "report.json", a.run_uri + "reports/")
    _write_json(MODEL / "artifacts_location.json", {"artifacts": a.run_uri + "artifacts/", "reports": a.run_uri + "reports/"})
    print(json.dumps({k: report[k] for k in ("score", "ambiguity")}, indent=1, default=str), flush=True)
    _cleanup_work()               # everything that matters is on S3 (checkpoints, artefacts, reports)


def _read_lists(path: Path) -> dict:
    import pandas as pd
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1].map(lambda s: [x for x in s.split(",") if x])))


def mode_predict(a) -> None:
    from ber import pipeline

    art_in = INPUT / "artifacts"
    assert (art_in / "meta.json").exists(), "channel 'artifacts' must hold <run>/artifacts/ (meta.json missing)"
    assert (INPUT / "test" / "test_source1.tsv").exists(), "channel 'test' must hold dataset/test/*.tsv"
    log.info("environment: %s", env_report(a))
    art = WORK / "artifacts"
    if art.exists():
        shutil.rmtree(art)
    shutil.copytree(art_in, art)
    out_dir = OUTPUT / "submission"
    ps = pipeline.predict_init(INPUT, "test", WORK, out_dir, use_cache=a.use_cache,
                               ckpt_uri=a.ckpt_uri if a.use_cache else None, s3_region=a.region)
    for step in pipeline.PREDICT_STEPS:
        ts = time.time()
        step(ps)
        log.info("step %s finished in %.1f min", step.__name__, (time.time() - ts) / 60)

    # internal checks (notebook 18.1) + the official validator (18.2)
    m, c = _read_lists(out_dir / "matching_results.tsv"), _read_lists(out_dir / "candidate_pairs.tsv")
    s1_required, x_valid = set(ps.S1["entity_id"]), set(ps.X["entity_id"])
    checks = {"one row per S1 entity": set(m) == s1_required and len(m) == len(s1_required),
              "candidate file covers every S1": set(c) == s1_required,
              "no duplicate IDs in a list": all(len(v) == len(set(v)) for v in m.values()),
              "only S2-/S3- IDs": all(x[:3] in ("S2-", "S3-") for v in m.values() for x in v),
              "all IDs exist": all(x in x_valid for v in m.values() for x in v),
              "matches are a subset of candidates": all(set(m[k]) <= set(c.get(k, [])) for k in m)}
    validator = INPUT / "utils" / "validate_submission.py"
    if validator.exists():
        r = subprocess.run([sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
                            "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(INPUT / "test")],
                           capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
        checks["official validator"] = r.returncode == 0
    summary = {"run_name": a.run_name, "final_stage": ps.meta.get("final_stage", 2), "checks": checks,
               "s1_entities": len(m), "s1_with_matches": sum(bool(v) for v in m.values()),
               "matched_ids": sum(len(v) for v in m.values())}
    _write_json(out_dir / "predict_summary.json", summary)
    s3_upload(a.region, out_dir, a.run_uri + "output/")
    print(json.dumps(summary, indent=1), flush=True)
    _cleanup_work()
    if not all(checks.values()):
        raise SystemExit(f"submission check FAILED: {[k for k, v in checks.items() if not v]}")
    print("leaderboard file:", a.run_uri + "output/matching_results.tsv", flush=True)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    a = parse_args(argv)
    for d in (WORK, WORK / "joblib", OUTPUT, MODEL):
        d.mkdir(parents=True, exist_ok=True)
    log.info("mode=%s run=%s -> %s", a.mode, a.run_name, a.run_uri)
    {"check": mode_check, "train": mode_train, "predict": mode_predict}[a.mode](a)


if __name__ == "__main__":
    main()
