"""Build the SageMaker notebook (notebooks/Business_Entity_Resolution_Colab.ipynb) from the library sources.

The notebook embeds every module of src/ber verbatim (as %%writefile cells), so
the notebook and the submission package can never drift apart. Re-run this
script after changing any module:

    python tools/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Business_Entity_Resolution_Colab.ipynb"

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _lines(text)})


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def writefile(rel: str, title: str, blurb: str) -> None:
    md(f"### {title}\n\n{blurb}")
    src = (ROOT / rel).read_text(encoding="utf-8")
    code(f"%%writefile {{PKG}}/{rel}\n{src}")


# =============================================================================
# 0. Title
# =============================================================================
md(r"""
# Business Entity Resolution: AWS SageMaker Training Notebook
### Amazon ML Challenge 2026: matching S2/S3 business records to Source 1 entities

This notebook runs end to end inside **AWS SageMaker** (Notebook Instance or Studio JupyterLab). It reads the challenge data from **Amazon S3**, trains the full pipeline on the **training split only**, and writes the trained artefacts and the two submission files back to S3:

| file | content |
|---|---|
| `matching_results.tsv` | final matches per Source 1 entity (the scored file) |
| `candidate_pairs.tsv` | the exact candidate set the matching model scored |

**Data location:** `s3://aws-ml-162599956323-ap-south-1-an/6ab10eb3b23ba_student_resource/student_resource/` (`dataset/train`, `dataset/test`, `utils/`).
**Checkpoints:** `.../student_resource/ber_outputs/checkpoints/`. Every finished piece of work (each blocking block, each feature part, each model fold) is uploaded as soon as it is complete.
**Results location:** `.../student_resource/ber_outputs/runs/<profile>_<timestamp>/` (artefacts, output TSVs, submission zip).

> **Resuming after a crash, a kernel restart or on a new instance:** run the notebook again from the top (*Run All*). Every finished stage, and every finished block, part or fold inside the interrupted stage, is restored from S3, and the work continues from the first missing piece. A resumed run gives bit-identical results to an uninterrupted one.

**Pipeline**

```
raw TSV ─► normalisation (any script → Latin, noise cleanup, learned aliases, locality phrases)
        ─► DENSE VIEW (GPU): MiniLM-L12 bi-encoder (Apache-2.0, 118M) fine-tuned on training pairs
        ─► BLOCKING: 5 sparse TF-IDF views + the dense view per (country, source), forward + reverse top-k
                     ─► learned pruner (16 per entity per source)      ═► candidate_pairs.tsv
        ─► STAGE 1: ~80 pairwise features → LightGBM (5-fold, grouped by entity, out-of-fold)
        ─► CROSS-ENCODER (optional, GPU): mDeBERTa-v3-base (MIT, 278M) on uncertain pairs
        ─► STAGE 2: + context features (competing candidates / claimants / sibling agreement) → LightGBM
        ─► STAGE 3: context recomputed from the stage-2 probabilities → LightGBM (kept only if it wins out-of-fold)
        ─► DECISION: per entity, the subset maximising the exact expected F0.5   ═► matching_results.tsv
```

**Contents**
1. SageMaker environment & S3 data
2. Pipeline library (written to disk, identical to the submission `src/`)
3. Configuration
4. Load the training data
5. Exploratory data analysis
6. Text normalisation
7. Entity split
8. Learned aliases & noise tokens
9. Final views & corpus statistics
10. Candidate generation (blocking)
11. Stage-1 matcher
12. Cross-encoder (optional)
13. Stage-2 and stage-3 matchers
14. Decision rule, hold-out validation, where the score is lost, ambiguity floor
15. Error analysis and corpus-size check
16. Upload the trained artefacts to S3
17. Inference on the test set
18. Validate the submission files
19. Package the submission zip and upload it to S3
20. Final refit & tuning levers
21. Unseen-country check (optional)

> **Fair play.** No external data, APIs or geocoders are used. Every learned artefact comes from `dataset/train`; the test split is downloaded and read only in Section 17, after training is finished, and nothing is ever fitted or tuned on it.
""")

# =============================================================================
# 1. SageMaker + S3
# =============================================================================
md(r"""
## 1. SageMaker environment & S3 data

**Recommended setup**

| item | recommendation |
|---|---|
| instance | **`ml.g6.8xlarge`** (32 vCPU, 128 GiB RAM, 1× L4 24 GB) or `ml.g5.8xlarge` (1× A10G 24 GB): full profile + dense view + cross-encoder. The dense view and the cross-encoder need the GPU; without one both are skipped automatically. |
| storage | EBS volume **≥ 100 GB** (data 2.5 GB + local working copies of the checkpoints, up to ~60 GB at full size). The default 5 GB volume is far too small. |
| kernel | Notebook Instance: **`conda_pytorch_p310`**. Studio JupyterLab: the default **Python 3** kernel of the SageMaker Distribution image. Both ship PyTorch + CUDA. |
| IAM | the SageMaker execution role needs `s3:ListBucket`, `s3:GetObject` and `s3:PutObject` on the bucket. |
| internet | needed once to `pip install` packages and to download `microsoft/mdeberta-v3-base` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` from the Hugging Face Hub. Without internet, upload the models to S3 and set `CE_MODEL_S3_URI` / `DENSE_MODEL_S3_URI` in 3.1 (otherwise the stage is skipped automatically). |

Everything runs on the instance's local EBS volume (fast working storage). S3 is the source of the data, the durable home of every checkpoint, and the destination of the results, so nothing is lost when the kernel dies or the instance is replaced.
""")

code(r"""
# ── 1.1 Inspect the instance: RAM, CPU cores, GPU ────────────────────────────
import os, sys, gc, json, time, shutil, zipfile, platform, subprocess, logging, random
from pathlib import Path

try:
    import psutil
except ImportError:
    %pip install -q psutil
    import psutil

# The cross-encoder runs on PyTorch: keep transformers from loading TensorFlow / JAX (preinstalled on
# SageMaker images), whose CUDA plugins only print "factory already registered" noise.
for _k, _v in (("USE_TF", "0"), ("USE_TORCH", "1"), ("USE_FLAX", "0"), ("USE_JAX", "0"),
               ("TRANSFORMERS_NO_TF", "1"), ("TF_CPP_MIN_LOG_LEVEL", "3")):
    os.environ.setdefault(_k, _v)

ON_SAGEMAKER = Path("/opt/ml/metadata/resource-metadata.json").exists()
RAM_GB = psutil.virtual_memory().total / 1e9
N_CPU = os.cpu_count()
try:
    GPU_INFO = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
except Exception:
    GPU_INFO = ""
print(f"SageMaker     : {ON_SAGEMAKER}")
print(f"Python        : {platform.python_version()}  ({sys.executable})")
print(f"RAM           : {RAM_GB:.1f} GB")
print(f"CPU cores     : {N_CPU}")
print(f"GPU (driver)  : {GPU_INFO or 'none'}")
# What actually matters: can PyTorch use the GPU? (the cross-encoder needs it)
try:
    import torch
    TORCH_CUDA = torch.cuda.is_available()
    if TORCH_CUDA:
        cap = torch.cuda.get_device_capability(0)
        print(f"PyTorch CUDA  : yes - {torch.cuda.get_device_name(0)}, compute {cap[0]}.{cap[1]}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB, "
              f"{'bf16' if cap[0] >= 8 else 'fp16'} mixed precision (torch {torch.__version__})")
    else:
        print(f"PyTorch CUDA  : NO (torch {torch.__version__}) -> the cross-encoder will be skipped")
except ImportError:
    TORCH_CUDA = False
    print("PyTorch CUDA  : torch not installed -> the cross-encoder will be skipped "
          "(use the conda_pytorch_p310 / SageMaker Distribution kernel)")
if GPU_INFO and not TORCH_CUDA:
    print("WARNING: a GPU is attached but PyTorch cannot use it - pick a PyTorch GPU kernel to train the cross-encoder.")
""")

code(r"""
# ── 1.2 Install the pipeline dependencies ────────────────────────────────────
# numpy / pandas / scipy / scikit-learn / boto3 (and torch on the PyTorch kernels) are pre-installed.
import importlib.util
%pip install -q "lightgbm>=4.3" "rapidfuzz>=3.6" "anyascii>=0.3.2" "pyarrow>=14" boto3
if importlib.util.find_spec("torch") is not None:
    # cross-encoder extras; transformers is installed only if missing, so the kernel's torch stays compatible
    if importlib.util.find_spec("transformers") is None:
        %pip install -q "transformers>=4.40"
    %pip install -q sentencepiece protobuf
else:
    print("PyTorch not found in this kernel -> the cross-encoder will be skipped "
          "(use conda_pytorch_p310 / the SageMaker Distribution kernel to enable it)")

import numpy as np, pandas as pd, scipy, sklearn, lightgbm, rapidfuzz, anyascii, boto3
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
for m in (np, pd, scipy, sklearn, lightgbm, rapidfuzz, boto3):
    print(f"{m.__name__:<12} {m.__version__}")
""")

code(r"""
# ── 1.3 S3 locations and local working directories (EDIT HERE) ───────────────
S3_URI        = "s3://aws-ml-162599956323-ap-south-1-an/6ab10eb3b23ba_student_resource/student_resource/"  # <── EDIT if moved
S3_REGION     = "ap-south-1"                                   # region of the bucket
S3_OUTPUT_URI = S3_URI.rstrip("/") + "/ber_outputs/"           # artefacts, submissions and zips go here
S3_CKPT_URI   = S3_OUTPUT_URI + "checkpoints/"                 # every checkpoint is mirrored here (resume point)
S3_ENDPOINT   = os.environ.get("BER_S3_ENDPOINT") or None      # only for a custom / VPC endpoint URL

# Local EBS volume: /home/ec2-user/SageMaker (Notebook Instance) or /home/sagemaker-user (Studio).
LOCAL_ROOT = Path(os.environ.get("BER_LOCAL_ROOT") or next(
    (Path(p) / "ber" for p in ("/home/ec2-user/SageMaker", "/home/sagemaker-user") if Path(p).is_dir()),
    Path.home() / "ber"))
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_SR = LOCAL_ROOT / "student_resource"                     # local mirror of the S3 prefix
WORK_DIR = LOCAL_ROOT / "work"                                 # local copies of the checkpoints + artefacts
LEGACY_WORK_DIR = LOCAL_ROOT / "ber_work"                      # earlier notebook version (its frames are reused if valid)
OUT_DIR = LOCAL_ROOT / "output"                                # submission TSVs
PKG = (LOCAL_ROOT / "business_entity_resolution").resolve().as_posix()   # library + submission code
for d in (WORK_DIR, OUT_DIR, Path(PKG) / "src" / "ber", Path(PKG) / "docs"):
    d.mkdir(parents=True, exist_ok=True)

free_gb = shutil.disk_usage(LOCAL_ROOT).free / 1e9
print("S3 data   :", S3_URI)
print("S3 output :", S3_OUTPUT_URI)
print("S3 ckpts  :", S3_CKPT_URI)
print("LOCAL_ROOT:", LOCAL_ROOT, f"({free_gb:.0f} GB free)")
if free_gb < 70:
    print("WARNING: full-size training needs ~70 GB free on the local volume - enlarge the space's storage "
          f"(Space settings, up to 100 GB), or delete {LEGACY_WORK_DIR} (only an earlier version's cache).")
if free_gb < 30 and not os.environ.get("BER_DEV_SAMPLE"):
    raise RuntimeError(f"only {free_gb:.0f} GB free on {LOCAL_ROOT}: a full-size run fails part-way with "
                       "'No space left on device'. Stop the space, set its storage to 100 GB, and start again "
                       "(finished work is restored from S3).")
""")

code(r"""
# ── 1.4 S3 helpers (boto3: parallel, resumable, size-checked) ────────────────
from concurrent.futures import ThreadPoolExecutor
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError

S3 = boto3.client("s3", region_name=S3_REGION, endpoint_url=S3_ENDPOINT,
                  config=BotoConfig(max_pool_connections=64, retries={"max_attempts": 10, "mode": "adaptive"}))
XFER = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=16)


def split_s3(uri: str) -> tuple:
    assert uri.startswith("s3://"), f"not an s3:// URI: {uri}"
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def s3_list(uri: str):
    # yields (key, size) for every object under the prefix
    bucket, prefix = split_s3(uri)
    for page in S3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def s3_download(uri: str, dest: Path) -> None:
    # mirror every object under the prefix into dest; complete files are skipped, so re-runs are free
    uri = uri if uri.endswith("/") else uri + "/"
    bucket, prefix = split_s3(uri)
    todo = []
    for key, size in s3_list(uri):
        rel = key[len(prefix):]
        if not rel or key.endswith("/") or "__pycache__" in rel:
            continue
        path = Path(dest) / rel
        if not (path.exists() and path.stat().st_size == size):
            todo.append((key, path, size))

    def get(item):
        key, path, _ = item
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        S3.download_file(bucket, key, str(tmp), Config=XFER)
        tmp.replace(path)                     # atomic: a half-downloaded file is never mistaken as complete

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, todo))
    print(f"{uri} -> {dest}: {len(todo)} file(s) downloaded "
          f"({sum(t[2] for t in todo) / 1e9:.2f} GB), others already present")


def s3_upload(path: Path, uri: str) -> None:
    # upload one file (to uri, or into uri if it ends with '/') or a whole directory tree
    path = Path(path)
    bucket, key = split_s3(uri)
    if path.is_file():
        jobs = [(path, key + path.name if key.endswith("/") or not key else key)]
    else:
        jobs = [(p, f"{key.rstrip('/')}/{p.relative_to(path).as_posix()}")
                for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: S3.upload_file(str(j[0]), bucket, j[1], Config=XFER), jobs))
    print(f"{path} -> {uri} ({len(jobs)} file(s))")


def s3_latest_run(profile: str):
    # newest ber_outputs/runs/<profile>_<timestamp>/ that holds complete artefacts (meta.json)
    bucket, prefix = split_s3(S3_OUTPUT_URI + "runs/")
    runs = []
    for page in S3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        runs += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
    for run in sorted(r for r in runs if r[len(prefix):].startswith(profile + "_"))[::-1]:
        if next(s3_list(f"s3://{bucket}/{run}artifacts/meta.json"), None):
            return f"s3://{bucket}/{run}"
    return None


# access check: fail early with a clear message instead of deep inside the pipeline
try:
    _root = split_s3(S3_URI)[1]
    listing = {k[len(_root):].split("/")[0] for k, _ in s3_list(S3_URI)}
except NoCredentialsError as e:
    raise RuntimeError("no AWS credentials - run this notebook on SageMaker with an execution role") from e
except ClientError as e:
    raise RuntimeError(f"cannot list {S3_URI}: {e.response['Error'].get('Code')} - the SageMaker execution "
                       "role needs s3:ListBucket / s3:GetObject / s3:PutObject on this bucket") from e
print("top-level entries under S3_URI:", sorted(listing))
assert "dataset" in listing, "S3_URI must point at the student_resource/ folder (it should contain dataset/)"
""")

code(r"""
# ── 1.5 Download the TRAINING split (+ validator) from S3 to the local volume ─
# The test split is downloaded only in Section 17, after training is finished.
s3_download(S3_URI.rstrip("/") + "/dataset/train/", LOCAL_SR / "dataset" / "train")
s3_download(S3_URI.rstrip("/") + "/utils/", LOCAL_SR / "utils")
DATA_DIR = LOCAL_SR / "dataset"
VALIDATOR = LOCAL_SR / "utils" / "validate_submission.py"
assert (DATA_DIR / "train" / "train_source1.tsv").exists(), f"train_source1.tsv missing under {DATA_DIR / 'train'}"
for f in sorted((DATA_DIR / "train").glob("*.tsv")):
    print(f"  {f.name:<28} {f.stat().st_size / 1e6:9.1f} MB")
print("validator present:", VALIDATOR.exists())
""")

code(r"""
# ── 1.6 Logging (every pipeline step prints start / end / duration) ─────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logging.getLogger("ber").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARNING)
random.seed(2026); np.random.seed(2026)
""")

# =============================================================================
# 2. Library
# =============================================================================
md(r"""
## 2. Pipeline library (`ber` package)

The cells below write the library to `business_entity_resolution/src/ber/`. It is **the same code** as the submission package (`code/business_entity_resolution/src/`), so what you train here is exactly what you submit. Each module is small and single-purpose:

| module | role |
|---|---|
| `config.py` | every hyper-parameter in one place |
| `io.py` | tab-separated readers / writers |
| `normalize.py` | transliteration, noise cleanup, canonicalisation, locality phrases, phonetic skeleton |
| `aliases.py` | token aliases and noise tokens learned from training labels |
| `blocking.py` | five-view sparse blocking (+ the dense view) and the learned pruner |
| `dense.py` | learned dense retrieval view (bi-encoder training, embeddings, exact GPU top-k) |
| `features.py` | pairwise, group and context features |
| `models.py` | grouped K-fold LightGBM with out-of-fold predictions |
| `cross_encoder.py` | optional multilingual cross-encoder |
| `decision.py` | exact expected-F0.5 selection and the macro F0.5 metric |
| `checkpoint.py` | stage checkpoints on local disk, mirrored to S3 |
| `pipeline.py` | the resumable train / predict steps used below |
""")

writefile("src/ber/__init__.py", "2.1 Package marker", "Makes `ber` an importable package.")
writefile("src/ber/config.py", "2.2 Configuration (`config.py`)",
          "Blocking sizes, LightGBM parameters, cross-encoder and decision settings. "
          "Section 3 adapts a few of them to the runtime.")
writefile("src/ber/io.py", "2.3 Input / output (`io.py`)",
          "All files are TAB-separated; ID lists are comma-separated. Outputs contain one row per "
          "Source 1 entity, de-duplicated, S2-/S3- IDs only.")
writefile("src/ber/normalize.py", "2.4 Text normalisation (`normalize.py`)",
          "Static rules (any script → Latin with `anyascii`, OCR-digit repair, domain / DBA / acronym "
          "handling, legal-form and street-type canonicalisation incl. French forms, number parsing), "
          "locality-phrase joining, and a transliteration-aware **phonetic skeleton** "
          "(`sophtveyr` and `software` share a skeleton).")
writefile("src/ber/aliases.py", "2.5 Learned aliases (`aliases.py`)",
          "Token substitutions mined from training true pairs (e.g. `texas→tx`, `up→uttar_pradesh`, "
          "`praivet→pvt`), with a plausibility check, plus noise tokens inserted by the sources "
          "(`center`, `services`, ...).")
writefile("src/ber/blocking.py", "2.6 Blocking (`blocking.py`)",
          "Five TF-IDF views per (country, source) block, forward top-k per entity and reverse top-k "
          "per record, then a learned LightGBM pruner applied block by block (memory-safe).")
writefile("src/ber/features.py", "2.7 Features (`features.py`)",
          "Pairwise similarities (rapidfuzz, vectorised), house-number and locality agreement, "
          "genericness, blocking-rank features, and stage-2 context features.")
writefile("src/ber/models.py", "2.8 Gradient-boosted models (`models.py`)",
          "LightGBM trained with 5 folds **grouped by Source 1 entity**; out-of-fold predictions for "
          "fitting rows, fold-average predictions for everything else.")
writefile("src/ber/cross_encoder.py", "2.9 Cross-encoder (`cross_encoder.py`)",
          "`microsoft/mdeberta-v3-base` (MIT, 278M parameters, multilingual tokenizer) fine-tuned on "
          "uncertain pairs; mixed precision on GPU.")
writefile("src/ber/dense.py", "2.9b Dense retrieval view (`dense.py`)",
          "`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters) "
          "fine-tuned as a siamese bi-encoder with the symmetric in-batch-negative contrastive loss on "
          "training true pairs (country-homogeneous batches, sibling records masked); exact cosine top-k "
          "on the GPU. Unlike the sparse views it has no document-frequency cap.")
writefile("src/ber/decision.py", "2.10 Decision rule & metric (`decision.py`)",
          "Macro F0.5 (singletons included) and the exact expected-F0.5 subset selection via "
          "Poisson-binomial dynamic programming.")
writefile("src/ber/diagnostics.py", "2.10b Diagnostics (`diagnostics.py`)",
          "The measurements of cells 10.1/10.2/14.1/14.3-14.5/15.2 as JSON-ready functions, used by the "
          "SageMaker training jobs (`sagemaker/`). The notebook shows the same numbers in its own cells.")
writefile("src/ber/checkpoint.py", "2.11 Checkpoints (`checkpoint.py`)",
          "Every expensive step writes its outputs into a stage folder whose name carries a hash of the "
          "data, the settings and the upstream stages. Each file is uploaded to S3 the moment it is "
          "complete, so a new session (or a new instance) restores finished work and resumes at the first "
          "missing file.")
writefile("src/ber/pipeline.py", "2.12 Orchestration (`pipeline.py`)",
          "Resumable steps with S3-mirrored checkpoints. The notebook calls them one by one below.")
writefile("src/run.py", "2.13 Command-line entry point (`run.py`)",
          "Same pipeline from a terminal: `python src/run.py train|predict|package ...`.")
writefile("README.md", "2.14 Package README", "Reproduction instructions shipped in the zip.")
writefile("requirements.txt", "2.15 Pinned requirements", "Library versions used for the submission.")
writefile("docs/Documentation_template.md", "2.16 Methodology document",
          "The filled challenge template, packaged at the root of the submission zip "
          "(fill in the team fields and the final metrics before submitting).")

code(r"""
# ── 2.17 Import the library (and make it visible to worker processes) ──────
import importlib
src_dir = str(Path(PKG) / "src")
ber_dir = Path(src_dir) / "ber"
missing = [m for m in ("__init__", "config", "io", "normalize", "aliases", "blocking", "dense", "features",
                       "models", "cross_encoder", "decision", "diagnostics", "checkpoint", "pipeline")
           if not (ber_dir / f"{m}.py").exists()]
assert not missing, f"library files missing in {ber_dir}: {missing} - run the %%writefile cells of Section 2 first"

# The library folder must come first on the path, in this kernel and in worker processes. Any other
# folder named "ber" - e.g. LOCAL_ROOT itself when the notebook runs from its parent folder - would
# otherwise be imported as an empty namespace package ("cannot import name 'config' from 'ber'").
sys.path[:] = [src_dir] + [p for p in sys.path if p != src_dir]
os.environ["PYTHONPATH"] = os.pathsep.join(
    [src_dir] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != src_dir])
# Forget any earlier import of ber (stale code, or the wrong folder) so that the modules below are
# exactly the files written above, and stop idle joblib workers that still hold the old modules.
for _m in [m for m in sys.modules if m == "ber" or m.startswith("ber.")]:
    del sys.modules[_m]
importlib.invalidate_caches()
try:
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
except Exception:
    pass

import ber
from ber import config as ber_config, normalize, aliases, blocking, features, models, \
    cross_encoder, dense, decision, checkpoint, pipeline, io as ber_io
_loaded = Path(getattr(ber, "__file__", None) or "?").resolve().parent
assert _loaded == ber_dir.resolve(), f"'ber' was imported from {_loaded}, expected {ber_dir}"
print("ber library loaded from", _loaded)
""")

# =============================================================================
# 3. Configuration
# =============================================================================
md(r"""
## 3. Configuration

Edit the switches in 3.1. `PROFILE="auto"` picks the largest setting that fits the instance's memory:

| profile | RAM | example SageMaker instance | entities used to fit the GBDTs | notes |
|---|---|---|---|---|
| `full` | ≥ 60 GB | `ml.g5.4xlarge`, `ml.g5.8xlarge`, `ml.r6i.2xlarge`, `ml.m6i.4xlarge` | all non-hold-out (~1.9 M) | best accuracy |
| `large` | ≥ 40 GB | 48 GiB instances | 900 000 | near-full accuracy |
| `medium` | ≥ 24 GB | `ml.g5.2xlarge`, `ml.m6i.2xlarge` | 400 000 | |
| `dev` | any | any | train-derived subset (`DEV_SAMPLE_S1`) | workflow check only |

Whatever the profile, **candidates are generated and scored for every Source 1 entity**, so the "competition" features look exactly as they will on the test set; the profile only limits how many entities the models are *fitted* on. Feature matrices are kept on disk and streamed into LightGBM, and the text tables are released while the models train, which is why 64 GiB is enough for the full profile.

**Checkpoints.** Each step's checkpoint is keyed by a hash of the data and of the settings it depends on. Changing a setting re-runs only the steps that depend on it (for example, changing `PROFILE` reuses the normalisation and blocking checkpoints). `USE_CACHE = False` ignores all checkpoints.
""")

code(r"""
# ── 3.1 Main switches (EDIT HERE) ────────────────────────────────────────────
PROFILE           = os.environ.get("BER_PROFILE", "auto")   # "auto" | "full" | "large" | "medium" | "dev"
USE_CROSS_ENCODER = True     # used only when a CUDA GPU is present
HOLDOUT_FRAC      = 0.15     # S1 entities kept out for validation (0.0 = final refit on everything)
DEV_SAMPLE_S1     = int(os.environ.get("BER_DEV_SAMPLE", 0)) or None   # e.g. 20000 forces DEV mode
USE_CACHE         = True     # resume from the checkpoints (local copy first, then S3_CKPT_URI)
# Stop at once if the finished work up to stage 1 is not found on S3 (4.2), instead of silently recomputing
# it (10-20 hours), or if the cross-encoder is disabled (12.1). Set False only for a deliberately fresh run.
REQUIRE_RESTORE   = True
SEED              = 2026
CE_MODEL_S3_URI   = None     # e.g. "s3://<bucket>/models/mdeberta-v3-base/" if the instance has no internet
USE_DENSE         = True     # learned dense retrieval view (GPU); see Section 9.2
DENSE_MODEL_S3_URI = None    # e.g. "s3://<bucket>/models/paraphrase-multilingual-MiniLM-L12-v2/" (no internet)
# Sparse blocking breadth. The previous run used 10 / 3 (hold-out candidate recall 98.0%); cell 10.3
# measures the hold-out recall of other values. Changing any of them re-runs blocking and everything after it.
BLOCK_K_FWD       = 15       # forward top-k per sparse view
BLOCK_K_REV       = 5        # reverse top-k per sparse view
BLOCK_CAP_X       = 1        # multiplier on the document-frequency caps (max_df_*)
USE_STAGE3        = True     # collective refinement (13.2); kept only if it wins on out-of-fold data
# Cross-encoder backbone: "microsoft/mdeberta-v3-base" (MIT, 278M) or the stronger
# "FacebookAI/xlm-roberta-large" (MIT, 560M; about 3x slower to train and score).
CE_MODEL          = "microsoft/mdeberta-v3-base"
# Cross-encoder training pairs per half. None = sized by GPU class in 3.2 (1.2M on L4 / A10G: ~2-2.5 h per
# half). 600_000 about halves the cross-encoder's GPU time at a small accuracy cost. Changing it retrains
# the cross-encoder (only when it is not already finished on S3 with the current value).
CE_TRAIN_PAIRS    = None
# Read-only robustness checks (training data only; they never change the trained models):
SIZE_CHECK_FRACS  = ()           # 15.2: e.g. (0.1, 0.3) re-runs inference on train-derived corpora of these
                                 # sizes (~1-2 extra GPU hours); off by default to save credits
RUN_COUNTRY_CHECK = False        # 21.1: train on one country, score the other (a separate, long run)
RUN_BLOCKING_WHATIF = False      # 10.3: sparse-view recall at other k / caps (rebuilds every block's TF-IDF twice: slow)
""")

code(r"""
# ── 3.2 Resolve the profile and build the configuration ─────────────────────
from dataclasses import asdict
from ber.config import Config

if PROFILE == "auto":
    PROFILE = "full" if RAM_GB >= 60 else "large" if RAM_GB >= 40 else "medium" if RAM_GB >= 24 else "dev"
if DEV_SAMPLE_S1:
    PROFILE = "dev"
if PROFILE == "dev" and not DEV_SAMPLE_S1:
    DEV_SAMPLE_S1 = 40_000
    print("WARNING: not enough RAM for the full training corpus -> DEV mode on a train-derived subset.")

cfg = Config()
cfg.n_jobs = -1
cfg.model.seed = SEED
cfg.model.holdout_frac = HOLDOUT_FRAC
cfg.model.max_train_s1 = {"full": None, "large": 900_000, "medium": 400_000, "dev": None}[PROFILE]
cfg.model.ranker_s1 = 150_000
for _v in ("name_tok", "name_char", "name_skel", "addr", "name_addr"):
    setattr(cfg.blocking, f"k_{_v}", int(BLOCK_K_FWD))
    setattr(cfg.blocking, f"max_df_{_v}", int(getattr(cfg.blocking, f"max_df_{_v}") * BLOCK_CAP_X))
cfg.blocking.k_reverse = int(BLOCK_K_REV)
cfg.model.stage3 = USE_STAGE3
# LightGBM is fastest with one thread per physical core (speed only: not part of any checkpoint key)
N_PHYS = psutil.cpu_count(logical=False) or N_CPU
for _p in (cfg.model.lgb_stage1, cfg.model.lgb_stage2, cfg.model.lgb_stage3):
    _p["num_threads"] = N_PHYS

# Cross-encoder sizing by GPU class (training pairs per half, batch sizes).
HAS_GPU = TORCH_CUDA                       # PyTorch must see the GPU, not just the driver
cfg.cross_encoder.enabled = USE_CROSS_ENCODER and HAS_GPU
gpu = (torch.cuda.get_device_name(0) if TORCH_CUDA else GPU_INFO).upper()
if "A100" in gpu or "H100" in gpu:
    cfg.cross_encoder.max_train_pairs, cfg.cross_encoder.batch_size, cfg.cross_encoder.infer_batch_size = 2_000_000, 128, 1024
elif "L4" in gpu or "A10" in gpu:
    cfg.cross_encoder.max_train_pairs, cfg.cross_encoder.batch_size, cfg.cross_encoder.infer_batch_size = 1_200_000, 96, 512
else:  # T4 / V100 / other 16 GB cards
    cfg.cross_encoder.max_train_pairs, cfg.cross_encoder.batch_size, cfg.cross_encoder.infer_batch_size = 350_000, 64, 256
cfg.cross_encoder.model_name = CE_MODEL
if "large" in CE_MODEL.lower():            # 560M backbone: smaller batches and learning rate
    cfg.cross_encoder.batch_size = max(16, cfg.cross_encoder.batch_size // 3)
    cfg.cross_encoder.infer_batch_size = max(64, cfg.cross_encoder.infer_batch_size // 2)
    cfg.cross_encoder.lr = 1e-5
if CE_TRAIN_PAIRS:
    cfg.cross_encoder.max_train_pairs = int(CE_TRAIN_PAIRS)
if PROFILE == "dev":
    cfg.cross_encoder.max_train_pairs = min(cfg.cross_encoder.max_train_pairs, 60_000)

# Dense retrieval view: needs the GPU and the pretrained encoder. Decided here, before the
# checkpoint keys are computed in 4.2, so that the keys describe what actually runs.
cfg.dense.enabled = USE_DENSE and HAS_GPU
if cfg.dense.enabled:
    try:
        if DENSE_MODEL_S3_URI:                     # offline instances: pretrained weights from S3
            _local = LOCAL_ROOT / "hf_models" / "paraphrase-multilingual-MiniLM-L12-v2"
            s3_download(DENSE_MODEL_S3_URI, _local)
            cfg.dense.model_name = str(_local)
        cross_encoder._quiet_transformers()
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(cfg.dense.model_name)   # fail fast (no internet / no weights)
    except Exception as e:
        print(f"dense view disabled - cannot load {cfg.dense.model_name}: {type(e).__name__}: {e}")
        cfg.dense.enabled = False
elif USE_DENSE:
    print("dense view disabled - PyTorch cannot use a GPU in this kernel")

# every run uploads its artefacts and submission files to its own S3 folder
RUN_NAME = f"{PROFILE}_{time.strftime('%Y%m%d-%H%M%S')}"
RUN_URI = f"{S3_OUTPUT_URI}runs/{RUN_NAME}/"

print(f"PROFILE={PROFILE}  dense view={'on' if cfg.dense.enabled else 'off'}  "
      f"cross-encoder={'on' if cfg.cross_encoder.enabled else 'off'}  "
      f"hold-out={HOLDOUT_FRAC}  max_train_s1={cfg.model.max_train_s1}")
print("results will be uploaded to", RUN_URI)
print("checkpoints (resume point)  :", S3_CKPT_URI if USE_CACHE else "disabled (USE_CACHE=False)")
display(pd.json_normalize(asdict(cfg), sep=".").T.rename(columns={0: "value"}).astype(str))
""")

# =============================================================================
# 4. Load
# =============================================================================
md(r"""
## 4. Load the training data

In DEV mode a train-derived subset is built first: `DEV_SAMPLE_S1` Source 1 entities, all their true S2/S3 records, and a proportional share of other records as distractors. A second disjoint subset, `deveval`, stands in for the test set in Section 17 so that the whole workflow can be scored. Both come from `dataset/train`; the real test data is never used for training or tuning.
""")

code(r"""
# ── 4.1 (DEV mode only) build train-derived subsets ─────────────────────────
def build_dev_dataset(src: Path, dst: Path, n_train: int, n_eval: int, seed: int = 7) -> Path:
    if (dst / "deveval" / "deveval_ground_truth.tsv").exists():
        print("dev dataset already present:", dst); return dst
    rng = random.Random(seed)
    gt = {}
    with open(src / "train" / "train_ground_truth.tsv", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t"); gt[a] = b
    ids = list(gt); rng.shuffle(ids)
    splits = {"train": set(ids[:n_train]), "deveval": set(ids[n_train:n_train + n_eval])}
    owner = {x: sp for sp, S in splits.items() for a in S for x in gt[a].split(",") if x}
    rate = {sp: 0.5 * len(S) / len(ids) for sp, S in splits.items()}
    outs = {}
    for sp in splits:
        (dst / sp).mkdir(parents=True, exist_ok=True)
        for k in (1, 2, 3):
            outs[sp, k] = open(dst / sp / f"{sp}_source{k}.tsv", "w", encoding="utf-8", newline="\n")
    for k in (1, 2, 3):
        with open(src / "train" / f"train_source{k}.tsv", encoding="utf-8") as f:
            h = next(f)
            for sp in splits: outs[sp, k].write(h)
            for line in f:
                i = line[:line.find("\t")]
                if k == 1:
                    for sp, S in splits.items():
                        if i in S: outs[sp, k].write(line)
                else:
                    o = owner.get(i)
                    if o: outs[o, k].write(line)
                    else:
                        r = rng.random()
                        if r < rate["train"]: outs["train", k].write(line)
                        elif r < rate["train"] + rate["deveval"]: outs["deveval", k].write(line)
    for v in outs.values(): v.close()
    for sp, S in splits.items():
        with open(dst / sp / f"{sp}_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as g:
            g.write(header)
            for a in S: g.write(f"{a}\t{gt[a]}\n")
    return dst

if PROFILE == "dev":
    TRAIN_DATA_DIR = build_dev_dataset(DATA_DIR, LOCAL_ROOT / "dev_dataset", DEV_SAMPLE_S1,
                                       max(DEV_SAMPLE_S1 // 4, 2000))
    EVAL_SPLIT, EVAL_DATA_DIR = "deveval", TRAIN_DATA_DIR
else:
    TRAIN_DATA_DIR = DATA_DIR
    EVAL_SPLIT, EVAL_DATA_DIR = "test", DATA_DIR
print("training data :", Path(TRAIN_DATA_DIR) / "train")
print("inference on  :", Path(EVAL_DATA_DIR) / EVAL_SPLIT)
""")

code(r"""
# ── 4.2 Initialise the training state (checkpoints on S3) and read the data ─
st = pipeline.train_init(TRAIN_DATA_DIR, WORK_DIR, cfg, use_cache=USE_CACHE, ckpt_uri=S3_CKPT_URI,
                         s3_region=S3_REGION, s3_endpoint=S3_ENDPOINT, legacy_work_dir=LEGACY_WORK_DIR)
(Path(WORK_DIR) / "run_uri.txt").unlink(missing_ok=True)   # rewritten by 16.1 for these artefacts
display(pd.DataFrame([{"stage": k, "checkpoint": v.key, "status": "complete" if v.done else "to do"}
                      for k, v in st.stg.items()]))
_missing = pipeline.missing_restores(st)
if _missing and REQUIRE_RESTORE:
    raise RuntimeError(
        f"not restorable from {S3_CKPT_URI}: {_missing} - these steps would be recomputed from scratch. "
        "Check S3_URI / the settings of 3.1 (HOLDOUT_FRAC, SEED, BLOCK_K_*, BLOCK_CAP_X, USE_DENSE) and that "
        "the earlier run finished them; set REQUIRE_RESTORE = False in 3.1 only for a deliberately fresh run.")
print("restorable up to stage 1:", "yes" if not _missing else f"NO - will recompute {_missing}")
pipeline.train_load(st)
print(f"ground-truth rows: {len(st.gt):,}")
if getattr(st, "raw", None) is not None:
    for k, v in st.raw.items():
        print(f"{k}: {len(v):,} rows")
""")

# =============================================================================
# 5. EDA
# =============================================================================
md(r"""
## 5. Exploratory data analysis

Findings that drive the design:
- **Singletons** (no true match) score 1.0 only with an empty prediction.
- Every S2/S3 record belongs to **at most one** Source 1 entity, which motivates the exclusivity option.
- True matches always share the **country** label, so blocking runs per country.
- Names and addresses are heavily noised (see examples), and many names are written in **Indic scripts**.
""")

code(r"""
# ── 5.1 Ground-truth structure ───────────────────────────────────────────────
sizes = np.array([len(v) for v in st.gt.values()])
n_s2 = np.array([sum(x.startswith("S2-") for x in v) for v in st.gt.values()])
n_s3 = sizes - n_s2
all_ids = [x for v in st.gt.values() for x in v]
print(f"S1 entities            : {len(sizes):,}")
print(f"singleton rate         : {np.mean(sizes == 0):.2%}")
print(f"mean matches / entity  : {sizes.mean():.2f}   (max {sizes.max()})")
print(f"max S2 / S3 per entity : {n_s2.max()} / {n_s3.max()}")
print(f"records matched twice  : {len(all_ids) - len(set(all_ids))}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
vals, cnts = np.unique(sizes, return_counts=True)
ax[0].bar(vals, cnts, color="#4C72B0"); ax[0].set_title("True matches per Source 1 entity")
ax[0].set_xlabel("number of matches"); ax[0].set_ylabel("entities")
w = 0.4
for off, arr, lab in ((-w/2, n_s2, "from S2"), (w/2, n_s3, "from S3")):
    v, c = np.unique(arr, return_counts=True); ax[1].bar(v + off, c, width=w, label=lab)
ax[1].set_title("Matches per source"); ax[1].legend(); ax[1].set_xlabel("matches")
plt.tight_layout(); plt.show()
del all_ids
""")

code(r"""
# ── 5.2 Field quality per source ─────────────────────────────────────────────
import re
NONLATIN = r"[^\u0000-\u024f\u1e00-\u1eff\u2000-\u206f]"   # anything outside Latin + punctuation

def eda_frames():
    # raw frames if still in memory, otherwise the restored normalised frames (they keep the raw text)
    if getattr(st, "raw", None) is not None:
        return {k: v.rename(columns={"business_name": "raw_name", "business_address": "raw_addr"})
                for k, v in st.raw.items()}
    S1 = st.S1 if st.S1 is not None else st.S1s
    X = st.X if st.X is not None else st.Xs
    return {"s1": S1, "s2": X[X["src"] == 2], "s3": X[X["src"] == 3]}

rows = []
for k, f in eda_frames().items():
    samp = f.sample(min(len(f), 200_000), random_state=0)
    rows.append({"source": k.upper(), "rows": len(f),
                 **{f"country={c}": f"{p:.1%}" for c, p in f["country"].value_counts(normalize=True).items()},
                 "empty address": f"{(samp['raw_addr'].str.strip() == '').mean():.2%}",
                 "non-Latin script": f"{samp['raw_name'].str.contains(NONLATIN, regex=True).mean():.2%}"})
display(pd.DataFrame(rows).fillna("-"))
""")

code(r"""
# ── 5.3 Examples of true-match groups (the noise the model must see through) ─
frames = eda_frames()
lookup = pd.concat([f[["entity_id", "raw_name", "raw_addr", "country"]] for f in frames.values()]).set_index("entity_id")
rng = np.random.default_rng(3)
multi = [k for k, v in st.gt.items() if len(v) >= 2]
for s1 in rng.choice(multi, min(4, len(multi)), replace=False):
    ids = [s1] + st.gt[s1]
    display(Markdown(f"**{s1}**"))
    display(lookup.loc[[i for i in ids if i in lookup.index]])
del frames, lookup, multi; gc.collect()
""")

# =============================================================================
# 6. Normalisation
# =============================================================================
md(r"""
## 6. Text normalisation

Deterministic and country-agnostic:
- **Transliteration.** Every script becomes Latin (`anyascii`): ఇండో పవర్ ఎల్‌ఎల్‌పీ → `imdo pavar elelpi`.
- **Cleanup.** Accent folding, case folding, OCR-digit repair (`H0mes` → `homes`), `m/s`, domains (`redcafe.com` → `redcafe`), dotted acronyms (`L.L.C.` → `llc`) and DBA splitting.
- **Canonical forms.** Legal forms (`Private`→`pvt`, `SARL`, `SAS`, ...), street types (`Road`→`rd`, `Boulevard`/`Bd`→`blvd`, `rue`, ...), ordinals and house numbers.
- **Phonetic skeleton.** A consonant skeleton that bridges transliterations to their English source.
""")

code(r"""
# ── 6.1 Normalisation on real-looking examples ──────────────────────────────
from ber.normalize import normalize_name, normalize_address, skeleton
names = ["Pvt. Sacha Téchnologies Ltd.", "LLC FRANKLIN, DANIEL + ARRINGTNO", "Lappin Gbn H0mes LLC",
         "Marshallbuilders.C0m", "Brixumbra dba Crystal Emera LLC", "ఇండో పవర్ ఎల్‌ఎల్‌పీ",
         "इनोवेटिव इंफ्रास्ट्रक्चर प्राइवेट लिमिटेड", "SCI Ptit Àmicale"]
display(pd.DataFrame([{"raw name": n, "normalised tokens": normalize_name(n)[0],
                       "phonetic skeleton": " ".join(skeleton(t) for t in normalize_name(n)[0].split())}
                      for n in names]))
addrs = ["##308, VIJAYA TOWERS NAGARJUNANAGAR COLONY, HYDERABAD, Telangana",
         "IL, 1606 11TH STREET, SPRINGFIELD CPD", "11003 89nd Avenue, Vancouver, Washington",
         "<NULL>, 5Th Main Road, Plot D/427 #A3&A4 Brundavana, KA",
         "175 Bd du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine"]
display(pd.DataFrame([{"raw address": a, "tokens": normalize_address(a)[0],
                       "components": normalize_address(a)[1]} for a in addrs]))
""")

code(r"""
# ── 6.2 Static normalisation of the training corpus ─────────────────────────
pipeline.train_normalize(st)
if st.S1s is not None:
    display(st.S1s[["raw_name", "name_tok", "raw_addr", "addr_comps"]].head(5))
else:
    print("restored from checkpoint - static pass skipped")
""")

# =============================================================================
# 7. Split
# =============================================================================
md(r"""
## 7. Entity split

Source 1 entities are split **before anything is learned**:
- **hold-out** (`HOLDOUT_FRAC`): never used for fitting anything. It gives the honest validation score.
- **dense** (`dense_frac`, only with the dense view): their true pairs train the bi-encoder of 9.2 and nothing else. Their candidates are still generated and scored as competitors, exactly like the test set's.
- **fit**: the entities whose candidates train the GBDTs, in 5 folds grouped by entity.
- **ranker**: a sample (disjoint from the fit entities whenever enough remain) whose candidates train the blocking pruner. It does not depend on the profile, so the blocking checkpoint survives a profile change.
""")

code(r"""
# ── 7.1 Split entities and build the label arrays ───────────────────────────
pipeline.train_split(st)
display(pd.Series({k: st.metrics[k] for k in ("n_s1", "n_x", "singleton_rate", "n_fit_s1",
                                              "n_holdout_s1", "n_ranker_s1", "n_dense_s1")}, name="value").to_frame())
""")

# =============================================================================
# 8. Aliases
# =============================================================================
md(r"""
## 8. Learned aliases & noise tokens

These are learned from **training-split true pairs only**. For a record matched to a Source 1 entity, tokens the record has and the entity lacks are paired with tokens the entity has and the record lacks. Pairs that recur consistently become aliases, with the clean Source 1 form as the canonical side: state codes (`tg→telangana`), transliterations (`praivet→pvt`, `sophtveyr→software`) and typos. Tokens that sources routinely *insert* into names (`center`, `services`) become noise tokens and are dropped from the core-name view.
""")

code(r"""
# ── 8.1 Learn (or restore) aliases and show samples ─────────────────────────
pipeline.train_aliases(st)
al = json.loads((st.art / "aliases.json").read_text())
print(f"name aliases: {len(al['name_alias'])}   address aliases: {len(al['addr_alias'])}   "
      f"noise tokens: {al['noise_tokens']}")
show = lambda d, n: pd.DataFrame(list(d.items())[:n], columns=["variant", "canonical"])
display(pd.concat([show(al["name_alias"], 25).add_prefix("name "),
                   show(al["addr_alias"], 25).add_prefix("address ")], axis=1))
""")

# =============================================================================
# 9. Finalise
# =============================================================================
md(r"""
## 9. Final views & corpus statistics

The learned aliases are applied, then the views used downstream are derived: core name, space-free name, phonetic skeleton, legal forms, address words and numbers, and components. Corpus statistics are computed on the corpus being matched. The test set gets its own statistics in Section 17, which is what lets the model adapt to the unseen French data:
- name genericness: exact-name duplicate counts and token document frequencies;
- the **locality vocabulary**: address components that recur in Source 1.
""")

code(r"""
# ── 9.1 Finalise the normalised frames (checkpointed) ───────────────────────
pipeline.train_finalize(st)
display(st.S1[["raw_name", "name_core", "name_skel", "name_legal", "addr_full", "addr_nums"]].head(6))
print(f"locality vocabulary: {len(st.geo):,} components")
print(f"memory: S1 {st.S1.memory_usage(deep=True).sum()/1e9:.2f} GB,  "
      f"S2+S3 {st.X.memory_usage(deep=True).sum()/1e9:.2f} GB")
""")

md(r"""
**9.2 Dense retrieval view (GPU).** The previous run lost most of its score at blocking: hold-out candidate recall was 98.0%, while the same code reached 99.9% on a 20k-entity slice. The cause is scale. The sparse views drop every token whose document frequency exceeds an absolute cap (3,000). In a block of about a million entities that removes most of a generic name, and a fixed top-k gets crowded out by look-alikes. The dense view has no cap:

- **Model.** `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters) embeds "name | address" (normalised text) into 384 dimensions.
- **Training.** Symmetric in-batch-negative contrastive loss (InfoNCE, as in DPR / Sentence-BERT) on the true pairs of a separate **dense slice** of training entities (`dense_frac`, 15% of the non-hold-out entities). Batches stay inside one country, so the negatives are hard. Records of the same entity are masked, because they are not negatives.
- **No leakage.** No other model is fitted on the dense slice: not the pruner, the GBDTs or the cross-encoder. The embedding cosine therefore stays an honest feature, the same reason the cross-encoder is trained 2-fold. The hold-out never touches the encoder.
- **Retrieval.** Exact cosine top-30 per entity and top-10 per record, computed on the GPU per (country, source) block. The cosine and the list of views that retrieved each pair become pruner and matcher features.

The encoder and every 2M-record embedding part are checkpointed to S3. When the blocking checkpoint already exists, the embeddings are not recomputed.
""")

code(r"""
# ── 9.2 Dense retrieval view: fine-tune the bi-encoder, embed every record (GPU, checkpointed) ──
pipeline.train_dense(st)
if cfg.dense.enabled:
    pairs = st.stg["dense"].load_json("dense_encoder.ok")["pairs"]
    print(f"dense encoder: trained on {pairs:,} true pairs of {st.is_dense.sum():,} dense-slice entities "
          f"(kept out of every other fit)")
    if st.emb is not None:
        print(f"embeddings: S1 {st.emb[0].shape}, S2+S3 {st.emb[1].shape} ({st.emb[0].dtype})")
    else:
        print("blocking results already saved - embeddings not needed")
else:
    print("dense view off (USE_DENSE=False, no GPU, or the model could not be loaded) - sparse views only")
""")

# =============================================================================
# 10. Blocking
# =============================================================================
md(r"""
## 10. Candidate generation (blocking)

The five TF-IDF views are built per (country, target source) block, and IDF is always fitted on the whole block:

| view | catches |
|---|---|
| name tokens (+ space-free name) | exact tokens, word-order swaps, domain names |
| name character 4-grams | typos, OCR noise |
| name phonetic skeleton | Indic-script names transliterated to Latin |
| address words / numbers / number×word | reordered components, missing parts |
| name×address conjunctions | very specific keys when every single token is common |

Tokens above a document-frequency cap are dropped. Each sparse view contributes the forward top-15 records per entity and the reverse top-5 entities per record. The **dense view** (9.2) adds its exact embedding-cosine top-30 per entity and top-10 per record, with no frequency cap. Every pair records which lists retrieved it (`via`). The steps are:

1. **Union, one block at a time, for every Source 1 entity.** The five sparse views of a block are built in parallel, the dense top-k runs on the GPU, and each block's union is checkpointed to S3 as soon as it is done.
2. **Pruner.** A LightGBM is trained on the union rows of the ranker entities, which is exactly the candidate distribution it will prune. Its features are the view cosines, including the dense one, and the number of sparse and dense lists that retrieved the pair.
3. **Pruning.** Each block keeps **16 candidates per entity per source**, plus every record's single best entity, again checkpointed block by block.

**Recall here is the ceiling for everything downstream.**
""")

code(r"""
# ── 10.1 Candidate union per block, pruner, pruning (checkpointed per block) ─
pipeline.train_blocking(st)
bm = {k: st.metrics.get(k) for k in ("union_pairs", "cand_pairs", "cand_per_s1", "union_recall_eval_sparse_only",
                                     "union_recall_eval", "cand_recall_eval")}
display(pd.Series(bm, name="value").to_frame())
print("previous run (sparse top-10 / top-3, 12 kept): union_recall_eval 0.98111, cand_recall_eval 0.97956")
""")

code(r"""
# ── 10.2 Candidate-set diagnostics ───────────────────────────────────────────
cand = st.cand
per_s1 = cand.groupby("s1_idx").size()
is_true = st.owner[cand["x_idx"].values] == cand["s1_idx"].values
# candidates are stored sorted by (entity, source, pruner score desc) inside each block
rank = cand.groupby(["s1_idx", "src"], sort=False).cumcount().values + 1
rt = rank[is_true]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].hist(per_s1.values, bins=range(0, per_s1.max() + 2), color="#55A868")
ax[0].set_title("Candidates per S1 entity"); ax[0].set_xlabel("candidates")
ax[1].hist(np.minimum(rt, 15), bins=range(1, 17), color="#C44E52")
ax[1].set_title("Pruner rank of the TRUE matches (per source)"); ax[1].set_xlabel("rank (15 = 15+)")
plt.tight_layout(); plt.show()
print(f"true matches ranked 1-3: {(rt <= 3).mean():.2%}   ranked 1-6: {(rt <= 6).mean():.2%}")
# which retrieval lists found the true candidates (evaluation entities only)
ev = (st.is_hold if st.is_hold.any() else st.is_fit)[cand["s1_idx"].values] & is_true
via = cand["via"].values[ev]
sp, dn = (via & blocking.VIA_SPARSE) > 0, (via & blocking.VIA_DENSE) > 0
display(pd.Series({"sparse and dense": (sp & dn).mean(), "sparse views only": (sp & ~dn).mean(),
                   "dense view only": (~sp & dn).mean()}, name="share of true candidates").to_frame().round(4))
display(pd.Series({v: ((via & (blocking.VIA_BIT[(v, "fwd")] | blocking.VIA_BIT[(v, "rev")])) > 0).mean()
                   for v in (*blocking.VIEWS, "dense")}, name="found by view").to_frame().round(4))
del rank, rt, is_true, per_s1, ev, via, sp, dn
""")

md(r"""
**10.3 (optional) Blocking what-if, sparse views only.** Recall is the ceiling of everything downstream. This read-only cell measures, on a sample of hold-out entities, the rank of every true record in every **sparse** view. That gives the exact sparse-union recall for any forward / reverse top-k and for higher frequency caps, together with the union size each setting would produce. Set `BLOCK_K_FWD`, `BLOCK_K_REV` and `BLOCK_CAP_X` in 3.1 from its table. It is off by default (`RUN_BLOCKING_WHATIF` in 3.1) because it rebuilds every block's TF-IDF matrices twice.

- **The dense view comes on top.** Its extra recall is added to these numbers: compare `union_recall_eval` with `union_recall_eval_sparse_only` in 10.1.
- **Without a hold-out** (`HOLDOUT_FRAC = 0`) the sample is drawn from the fit entities instead. The sparse views learn nothing from labels except the token aliases, so this is only slightly optimistic.
- **Union sizes** are calibrated on each block's sparse-only union. Blocks built before that count was recorded show a rough, uncalibrated estimate.
""")

code(r"""
# ── 10.3 (optional, read-only) Blocking what-if: sparse-view recall at larger k / higher caps ──
# Training data only. For a sample of evaluation entities (hold-out; fit entities
# when HOLDOUT_FRAC = 0) the rank of every true record is measured in every sparse
# view - forward (among the block's records, for the entity) and reverse (among the
# country's entities, for the record) - so the union recall of any (k, k_reverse)
# is known exactly without rebuilding the union. Each cap multiplier rebuilds a
# block's TF-IDF matrices once. Nothing is written and no checkpoint changes.
import time
from dataclasses import replace
from ber import blocking

S1, X, bc0 = st.S1, st.X, st.cfg.blocking
SAMPLE_PER_COUNTRY = 30_000          # evaluation entities analysed per country
K_FWD = tuple(sorted({10, 15, 20, 30, 50, bc0.k_name_tok}))    # forward top-k per sparse view (current included)
K_REV = tuple(sorted({3, 5, 10, 20, bc0.k_reverse}))           # reverse top-k per sparse view (current included)
CAP_MULTS = (1, 3)                   # max_df caps x this factor (1 = current caps)

t_all = time.time()
KMAX = max(max(K_FWD), max(K_REV))
rng = np.random.default_rng(0)
s1_country, x_country, x_src = S1["country"].values, X["country"].values, X["src"].values
if st.is_hold.any():
    pool, pool_name = st.is_hold, "hold-out"
else:                                # HOLDOUT_FRAC = 0: sparse views use no labels except the aliases
    pool, pool_name = st.is_fit, "fit (no hold-out in this run; slightly optimistic)"
sample = np.zeros(len(S1), bool)
for c in np.unique(s1_country[pool]):
    pos = np.flatnonzero(pool & (s1_country == c))
    sample[rng.choice(pos, min(SAMPLE_PER_COUNTRY, len(pos)), replace=False)] = True
gx = pd.Index(np.asarray(X["entity_id"].values, dtype=object)).get_indexer(st.gt_x)
in_s = (st.gt_gi >= 0) & sample[np.clip(st.gt_gi, 0, None)]
n_true = int(in_s.sum())                                   # every true pair of the sample (recall denominator)
tp_ = pd.DataFrame({"s1": st.gt_gi[in_s], "x": gx[in_s]})
tp_ = tp_[tp_["x"] >= 0].reset_index(drop=True)
same_ctry = x_country[tp_["x"].values] == s1_country[tp_["s1"].values]
print(f"sample: {int(sample.sum()):,} {pool_name} entities, {n_true:,} true pairs "
      f"({n_true - len(tp_):,} with an id absent from the files, {int((~same_ctry).sum()):,} cross-country)")
RUN_WHATIF = RUN_BLOCKING_WHATIF and n_true > 0
if not RUN_BLOCKING_WHATIF:
    print("blocking what-if skipped (RUN_BLOCKING_WHATIF=False in 3.1)")
elif not RUN_WHATIF:
    print("no true pairs in the sample - nothing to measure, cell skipped")


def _ranks(A, B, r, c, chunk):
    # rank (1-based, ties counted ahead) of column c[i] in row r[i] of A @ B.T - inf when
    # the pair shares no token - and every row's top-KMAX as (row, col, rank0)
    BT = B.T.tocsc()
    out = np.full(len(r), np.inf)
    ur, inv = np.unique(r, return_inverse=True)
    top = ([], [], [])
    for s in range(0, len(ur), chunk):
        rows = ur[s:s + chunk]
        M = (A[rows] @ BT).tocsr()
        M.sum_duplicates()
        rr = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
        order = np.lexsort((-M.data, rr))
        rs, ds = rr[order], M.data[order].astype(np.float64)
        key = rs * 4.0 + (2.0 - ds)                          # ascending: row, then cosine descending
        start = M.indptr[:-1]
        rank0 = np.arange(len(order)) - start[rs]
        keep = rank0 < KMAX
        top[0].append(rows[rs[keep]]); top[1].append(M.indices[order][keep]); top[2].append(rank0[keep])
        sel = np.flatnonzero((inv >= s) & (inv < s + len(rows)))
        if len(sel):
            lr = inv[sel] - s
            tv = np.asarray(M[lr, c[sel]]).ravel().astype(np.float64)
            ge = np.searchsorted(key, lr * 4.0 + (2.0 - tv), side="right") - start[lr]
            out[sel] = np.where(tv > 0, ge, np.inf)
    return out, tuple(np.concatenate(t) if t else np.zeros(0, np.int64) for t in top)


res = {m: {v: [np.full(len(tp_), np.inf), np.full(len(tp_), np.inf)] for v in blocking.VIEWS} for m in CAP_MULTS}
size_rows, calibrated = [], True
for country, src, tag in (blocking.block_keys(S1, X) if RUN_WHATIF else []):
    s1_pos = np.flatnonzero(s1_country == country)
    x_pos = np.flatnonzero((x_country == country) & (x_src == src))
    x_in = np.zeros(len(X), bool)
    x_in[x_pos] = True
    pm = np.flatnonzero(same_ctry & (s1_country[tp_["s1"].values] == country) & x_in[tp_["x"].values])
    la = np.searchsorted(s1_pos, tp_["s1"].values[pm])
    lb = np.searchsorted(x_pos, tp_["x"].values[pm])
    s_rows = np.searchsorted(s1_pos, np.flatnonzero(sample & (s1_country == country)))
    n_xs = max(len(np.unique(lb)), 1)
    info = st.stg["block"].load_json(f"cand/{tag}.json")
    # this cell estimates SPARSE unions: calibrate on the block's sparse-only union when it was recorded
    real_union = info.get("union_pairs_sparse")
    calibrated &= real_union is not None
    for m in CAP_MULTS:
        t0 = time.time()
        bc = replace(bc0, **{f"max_df_{v}": getattr(bc0, f"max_df_{v}") * m for v in blocking.VIEWS})
        mats = blocking.build_view_matrices(S1.iloc[s1_pos], X.iloc[x_pos], bc, st.cfg.n_jobs)
        t_build, t1 = time.time() - t0, time.time()
        chunk = max(200, 1000 // m)
        fwd_top, rev_top = [], []
        for v in blocking.VIEWS:
            A, B = mats[v]
            # one product for the true pairs and the union-size rows (same sampled entities)
            f, ft = _ranks(A, B, np.r_[la, s_rows], np.r_[lb, np.zeros(len(s_rows), np.int64)], chunk)
            r, rt = _ranks(B, A, lb, la, chunk)
            res[m][v][0][pm], res[m][v][1][pm] = f[:len(la)], r
            fwd_top.append(ft)
            rev_top.append(rt)
        t_rank = time.time() - t1
        for kf in K_FWD:
            fk = len(np.unique(np.concatenate([t[0][t[2] < kf] * len(x_pos) + t[1][t[2] < kf] for t in fwd_top])))
            for kr in K_REV:
                n_rev = sum(int((t[2] < kr).sum()) for t in rev_top) / n_xs
                size_rows.append({"cap_x": m, "k_fwd": kf, "k_rev": kr, "block": tag, "real_union": real_union,
                                  "raw": len(s1_pos) * fk / max(len(s_rows), 1) + len(x_pos) * n_rev})
        print(f"{country}/S{src} caps x{m}: {len(s1_pos):,} S1 x {len(x_pos):,} records, {len(pm):,} true pairs "
              f"| TF-IDF build {t_build:.0f}s, ranks {t_rank:.0f}s  (elapsed {time.time() - t_all:.0f}s)")
        del mats
        gc.collect()

if RUN_WHATIF:
    # Union size per setting: the sample-based estimate, scaled per block so that the CURRENT
    # setting reproduces the block's real sparse-only union (rough, unscaled estimate otherwise)
    sz = pd.DataFrame(size_rows)
    sz["real_union"] = pd.to_numeric(sz["real_union"], errors="coerce")   # None (not recorded) -> NaN
    cur = sz[(sz["cap_x"] == 1) & (sz["k_fwd"] == bc0.k_name_tok) & (sz["k_rev"] == bc0.k_reverse)]
    sz = sz.merge(cur[["block", "raw"]].rename(columns={"raw": "raw_cur"}), on="block", how="left")
    ok = sz["real_union"].notna() & (sz["raw_cur"] > 0)
    sz["est"] = np.where(ok, sz["raw"] / sz["raw_cur"].where(ok, 1.0) * sz["real_union"].where(ok, 0.0), sz["raw"])
    size = sz.groupby(["cap_x", "k_fwd", "k_rev"])["est"].sum().div(1e6).rename("union_M").reset_index()

    tab = []
    for m in CAP_MULTS:
        fmin = np.vstack([res[m][v][0] for v in blocking.VIEWS]).min(0)
        rmin = np.vstack([res[m][v][1] for v in blocking.VIEWS]).min(0)
        for kf in K_FWD:
            for kr in K_REV:
                tab.append({"cap_x": m, "k_fwd": kf, "k_rev": kr,
                            "recall": float(((fmin <= kf) | (rmin <= kr)).sum() / n_true)})
        never = np.isinf(fmin) & np.isinf(rmin)
        print(f"caps x{m}: true pairs sharing no token in any SPARSE view (unreachable by them at any k): "
              f"{int(never.sum()):,} = {never.sum() / n_true:.3%}")
    tab = pd.DataFrame(tab).merge(size, on=["cap_x", "k_fwd", "k_rev"])
    print("\nreal union_recall_eval (all views):", round(st.metrics["union_recall_eval"], 5),
          "  sparse views only:", round(st.metrics.get("union_recall_eval_sparse_only", float("nan")), 5),
          "  real union pairs (M):", round(st.metrics["union_pairs"] / 1e6, 1))
    print(f"{pool_name} sample, sparse-union recall (ties counted against the true record; "
          f"current = caps x1, k_fwd {bc0.k_name_tok}, k_rev {bc0.k_reverse}):")
    display(tab.pivot_table(index=["cap_x", "k_fwd"], columns="k_rev", values="recall").round(5))
    print("estimated SPARSE union pairs over all blocks (millions)"
          + ("" if calibrated else " - rough: some blocks predate the sparse-only count, re-run blocking to calibrate") + ":")
    display(tab.pivot_table(index=["cap_x", "k_fwd"], columns="k_rev", values="union_M").round(0))
    best = tab["recall"].max()
    near = tab[tab["recall"] >= best - 0.0005].sort_values("union_M")
    if len(near):
        pick = near.iloc[0]
        print(f"best recall {best:.5f}; cheapest setting within 0.05 pt of it: caps x{int(pick.cap_x)}, "
              f"k_fwd {int(pick.k_fwd)}, k_rev {int(pick.k_rev)} -> recall {pick.recall:.5f}, ~{pick.union_M:.0f}M union pairs")
print(f"done in {time.time() - t_all:.0f}s")
# global references to the text frames would keep them in memory while LightGBM trains
# (the pipeline releases st.S1 / st.X there): drop them and the sample arrays
del S1, X, s1_country, x_country, x_src, sample, gx, in_s, tp_, same_ctry, res
gc.collect()
""")

# =============================================================================
# 11. Stage 1
# =============================================================================
md(r"""
## 11. Stage-1 matcher

About 80 features per candidate pair:
- **Names.** Several string similarities on the full, core, space-free and phonetic-skeleton views; legal-form agreement or conflict; name genericness.
- **Addresses.** Full, word and street-part similarity; house-number agreement (exact, fuzzy, numeric distance); locality-component agreement or conflict.
- **Blocking.** View cosines, pruner score, and ranks and gaps within the entity and within the record.

LightGBM is trained with **5 folds grouped by Source 1 entity**, so no entity appears in both the training and validation folds. The out-of-fold probability `p1` of every fitting row feeds stage 2. Every other candidate (hold-out entities and competing entities) is scored by the fold average, exactly as the test set will be.

**Scale and resume.** The features are written to disk in parts of 2 million rows, and each part is checkpointed to S3. LightGBM reads the parts as a stream into its compact binned format, so the float matrix never has to fit in RAM. Each finished fold (model and predictions) is checkpointed too, so an interrupted training continues at the next fold.
""")

code(r"""
# ── 11.1 Stage-1 features + grouped 5-fold LightGBM (checkpointed per part / fold)
pipeline.train_stage1(st)
print(f"stage-1 features: {len(st.f1_cols)}   model rows: {len(st.model_rows):,}   "
      f"(fit rows {st.n_fit_rows:,})")
""")

code(r"""
# ── 11.2 Stage-1 diagnostics: out-of-fold quality + feature importance ──────
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
nf = st.n_fit_rows
y_fit = st.y[st.model_rows][:nf]; p_fit = st.p1[st.model_rows][:nf]
print(f"OOF  AUC {roc_auc_score(y_fit, p_fit):.5f}   PR-AUC {average_precision_score(y_fit, p_fit):.5f}   "
      f"log-loss {log_loss(y_fit, np.clip(p_fit, 1e-7, 1-1e-7)):.5f}")
imp = pd.read_csv(st.art / "stage1_importance.csv", index_col=0)["gain"].head(25)[::-1]
imp.plot.barh(figsize=(8, 7), color="#4C72B0", title="Stage-1 feature importance (gain, top 25)")
plt.tight_layout(); plt.show()
""")

# =============================================================================
# 12. Cross-encoder
# =============================================================================
md(r"""
## 12. Cross-encoder (optional, GPU)

`microsoft/mdeberta-v3-base` (MIT licence, 278M parameters, well under the 8B limit) reads both records jointly as "name | address" pairs. Its multilingual vocabulary natively covers Devanagari, Telugu, Tamil, Bengali and French.
- **Training.** Uncertain stage-1 pairs (0.01 ≤ p1 ≤ 0.99), plus 20% confident ones for calibration, with sequences up to 128 pieces. It is trained 2-fold over the fit entities, so its scores are also out-of-fold.
- **Use.** Its score is a stage-2 feature (NaN outside the band), so stage 2 learns how far to trust it.

The step is skipped automatically without a GPU. The model is downloaded from the Hugging Face Hub, or from `CE_MODEL_S3_URI` when the instance has no internet access. If it cannot be loaded, the step is skipped with a message instead of failing. Each trained half (weights and scores) is checkpointed to S3.
""")

code(r"""
# ── 12.1 Train / score the cross-encoder (skipped without GPU) ──────────────
if cfg.cross_encoder.enabled:
    try:
        if CE_MODEL_S3_URI:                        # offline instances: pretrained weights from S3
            local_model = LOCAL_ROOT / "hf_models" / "mdeberta-v3-base"
            s3_download(CE_MODEL_S3_URI, local_model)
            cfg.cross_encoder.model_name = str(local_model)
        cross_encoder._quiet_transformers()       # expected messages only (new head, tokenizer note)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(cfg.cross_encoder.model_name)   # fail fast (no internet / no weights)
        print("cross-encoder will train on:", torch.cuda.get_device_name(0))
    except Exception as e:
        print(f"cross-encoder disabled - cannot load {cfg.cross_encoder.model_name}: {type(e).__name__}: {e}")
        cfg.cross_encoder.enabled = False
    cfg.save(st.art / "config.json")
if REQUIRE_RESTORE and not cfg.cross_encoder.enabled:
    raise RuntimeError(
        "the cross-encoder is disabled (no GPU, or the model could not be loaded - see the message above): "
        "stage 2 would train without it. Fix the cause, or set REQUIRE_RESTORE = False in 3.1 to go on without it.")
_ce_restored = cfg.cross_encoder.enabled and pipeline.ce_stage(st).done
print("cross-encoder:", "restored from checkpoint" if _ce_restored else
      "not on S3 yet - training it now (both halves: roughly 3-5 GPU hours; each finished half is saved to S3). "
      "Never run a second notebook on the same checkpoints meanwhile.")
pipeline.train_cross_encoder(st)
band = np.isfinite(st.ce_m)
print(f"cross-encoder enabled: {st.ce_enabled}   scored pairs: {band.sum():,}")
ym = st.y[st.model_rows]
if band.any() and len(np.unique(ym[band])) == 2:
    print(f"cross-encoder AUC on its band: {roc_auc_score(ym[band], st.ce_m[band]):.4f}  "
          f"(stage-1 AUC on the same pairs: {roc_auc_score(ym[band], st.p1[st.model_rows][band]):.4f})")
""")

# =============================================================================
# 13. Stage 2
# =============================================================================
md(r"""
## 13. Stage-2 matcher

Stage 2 adds **context** computed from the out-of-fold `p1` of *all* candidates:
- **Within the entity.** The pair's rank and gap versus the entity's best candidate per source; the sum of p1, which estimates the expected number of matches; the count of confident candidates.
- **Within the record.** The pair's rank among every entity that claims the same record, the second-best claimant, and the number of confident claimants. This targets generic names.
- **Sibling agreement.** Name and address similarity of the record to the entity's most probable *other* candidates, weighted by their p1. This catches records renamed to an unrelated brand at the same address, and rejects look-alikes.
- **Ownership (collective evidence).** Most remaining errors are records, usually with an empty address, that several look-alike entities compete for. The record's strongest rival claimant and its probability; how well that rival is already served in the same source; the entity's own best other candidate in this source and in the other source; how many Source 1 entities of the country share exactly this name; and how many of the record's claimants carry exactly this name, with their probability mass. An exact-name tie is a coin flip, which a precision-weighted metric prefers to leave out.
- **Cross-encoder score.**

After training, the rows of the competitor-only entities (the dense slice) are scored by the fold-average stage-2 models, cross-encoder included, exactly as every test row is. Stage 3 and the decision step then see a stage-2 probability for every candidate. The context features, every finished fold and these scores are checkpointed to S3.
""")

code(r"""
# ── 13.1 Context features + grouped 5-fold LightGBM ─────────────────────────
pipeline.train_stage2(st)
nf = st.n_fit_rows
y_fit = st.y[st.model_rows][:nf]
print(f"stage-2 OOF AUC {roc_auc_score(y_fit, st.p2m[:nf]):.5f}   "
      f"PR-AUC {average_precision_score(y_fit, st.p2m[:nf]):.5f}")
imp = pd.read_csv(st.art / "stage2_importance.csv", index_col=0)["gain"].head(25)[::-1]
imp.plot.barh(figsize=(8, 7), color="#8172B2", title="Stage-2 feature importance (gain, top 25)")
plt.tight_layout(); plt.show()
""")

md(r"""
**13.2 Stage 3: collective refinement.** Most remaining errors are ownership questions: several look-alike Source 1 entities (often the same name, and a record with no address) compete for one record. Stage 2 judged those competitions with **stage-1** probabilities of the competitors. Stage 3 recomputes the same competition and sibling-anchor features from the **stage-2** probabilities, which already include the cross-encoder. It adds the record's margin over its strongest competing claimant (`q_claim_margin`) and trains another grouped 5-fold LightGBM. This is stacked collective classification: each round sees better estimates of its neighbours.

- **Consistency.** The stage-2 inputs are out-of-fold for the fit entities, and the fold assignment is the same, so stage 3 stays honest. Competitors outside the fit and hold-out entities (the dense slice) get the fold-average stage-2 probability, cross-encoder included, exactly as every test row does, so stage 3 sees the same kind of input in training and at inference. After training, those competitor rows are scored by the fold-average stage-3 models too.
- **Safety.** Section 14 tunes the decision rule on stage 2, on stage 3 and on their mean, and keeps the one that wins on **out-of-fold** data. The hold-out scores of all three are reported and are never used for the choice.
""")

code(r"""
# ── 13.2 Stage 3: context from p2 + grouped 5-fold LightGBM (checkpointed per fold) ──
pipeline.train_stage3(st)
if st.p3m is not None:
    nf = st.n_fit_rows
    y_fit = st.y[st.model_rows][:nf]
    print(f"stage-3 OOF AUC {roc_auc_score(y_fit, st.p3m[:nf]):.5f}   PR-AUC {average_precision_score(y_fit, st.p3m[:nf]):.5f}"
          f"   (stage 2: {roc_auc_score(y_fit, st.p2m[:nf]):.5f} / {average_precision_score(y_fit, st.p2m[:nf]):.5f})")
    imp = pd.read_csv(st.art / "stage3_importance.csv", index_col=0)["gain"].head(20)[::-1]
    imp.plot.barh(figsize=(8, 6), color="#937860", title="Stage-3 feature importance (gain, top 20)")
    plt.tight_layout(); plt.show()
else:
    print("stage 3 off (USE_STAGE3=False)")
""")

# =============================================================================
# 14. Decision
# =============================================================================
md(r"""
## 14. Decision rule & hold-out validation

The metric is F0.5 per Source 1 entity, averaged, and a singleton scores 1 **only** for an empty list. For each entity the pipeline selects the top-k candidates that maximise the **exact expected F0.5**:

$$E[F_{0.5}\mid k]=\sum_{a,b}P(TP=a\mid\text{top-}k)\,P(FN=b\mid\text{rest})\,\frac{1.25\,a}{0.25\,(a+b)+k},\qquad E[F_{0.5}\mid 0]=P(\text{no true match})$$

Both distributions are Poisson-binomial, computed exactly by dynamic programming. The grid search also tries:
- **Recalibration.** A temperature and a calibration offset on the logit scale (Platt scaling). The previous run's best rule was a plain threshold, a sign that the probabilities were shifted.
- **Probability sharing.** A record belongs to at most one entity, but each pair was scored on its own. The rule either uses the raw probabilities, divides each record's claims by their total when it exceeds 1, or lets only the record's most probable claimant keep its probability.
- **A plain global threshold.**
- **One-owner exclusivity.** A record goes only to its most probable entity.

The rule is applied to the **whole candidate graph at once**, exactly as on the test set, so sharing and exclusivity see every entity claiming a record, not only the evaluated ones. The winner, and the final probability (stage 2, stage 3 or their mean), are chosen on **out-of-fold** data of the fit entities and then scored once on the untouched **hold-out** entities.
""")

code(r"""
# ── 14.1 Tune on OOF, evaluate on the hold-out ──────────────────────────────
metrics = pipeline.train_decision(st)
keys = ["cand_recall_eval", "oof_macro_f05_stage2", "oof_macro_f05_stage3", "oof_macro_f05_stage23", "final_stage",
        "oof_macro_f05", "holdout_macro_f05_stage2", "holdout_macro_f05_stage3", "holdout_macro_f05_stage23",
        "holdout_macro_f05", "holdout_macro_f05_stage1_t05", "holdout_macro_f05_oracle",
        "holdout_pair_precision", "holdout_pair_recall"]
display(pd.Series({k: metrics.get(k) for k in keys}, name="value").to_frame())
print("chosen decision rule:", metrics["decision"], "  on stage", metrics.get("final_stage", 2),
      "(23 = mean of stages 2 and 3)")
print("previous run: holdout_macro_f05 0.98843, oracle 0.99350, leaderboard 0.985")
""")

code(r"""
# ── 14.2 Decision grid: macro F0.5 of every rule tried (out-of-fold) ────────
g = pd.DataFrame([{**p, "score": s} for s, p in st.decision_grid])
g["prob"] = g["prob"].fillna("raw") if "prob" in g else "raw"
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for (prob, excl), sub in g[g["rule"] == "threshold"].groupby(["prob", "exclusive"]):
    axes[0].plot(sub["threshold"], sub["score"], marker=".", label=f"{prob}, exclusive={excl}")
axes[0].set_xlabel("threshold"); axes[0].set_ylabel("OOF macro F0.5"); axes[0].set_title("Global threshold")
axes[0].legend(fontsize=8)
ef = g[(g["rule"] == "expf") & (g.get("bias", 0.0) == 0.0)]
for (prob, excl), sub in ef.groupby(["prob", "exclusive"]):
    axes[1].plot(sub["temperature"], sub["score"], marker=".", label=f"{prob}, exclusive={excl}")
axes[1].set_xlabel("temperature"); axes[1].set_title("Expected-F (calibration offset 0)")
axes[1].legend(fontsize=8); plt.tight_layout(); plt.show()
print("best 10 rules (out-of-fold):")
display(g.sort_values("score", ascending=False).head(10))
""")

code(r"""
# ── 14.3 F0.5 broken down by the number of true matches ─────────────────────
from types import SimpleNamespace


def per_entity_f(codes, label, sel, n_gt, beta=0.5):
    n = len(n_gt)
    tp = np.bincount(codes, weights=(label * sel).astype(float), minlength=n)
    npred = np.bincount(codes, weights=sel.astype(float), minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0); R = np.where(n_gt > 0, tp / n_gt, 0)
        f = np.where(tp > 0, (1 + beta**2) * P * R / (beta**2 * P + R), 0.0)
    return np.where((n_gt == 0) & (npred == 0), 1.0, f)


def eval_view():
    # The entities 14.3-15.1 evaluate: the hold-out when this run has one; otherwise
    # (HOLDOUT_FRAC = 0) the fit entities scored by their OUT-OF-FOLD predictions with
    # the chosen decision rule - honest per entity, the rule itself was tuned on them.
    nf = st.n_fit_rows
    if st.is_hold.any():
        ents, rows, p = np.flatnonzero(st.is_hold), st.model_rows[nf:], st.pfin[nf:]
        sel, name = np.asarray(st.holdout_sel, bool), "hold-out"
    else:
        ents, rows, p = np.flatnonzero(st.is_fit), st.model_rows[:nf], st.pfin[:nf]
        sel, name = None, "out-of-fold (fit entities; no hold-out in this run)"
    codes = np.searchsorted(ents, st.cand["s1_idx"].values[rows])
    if sel is None:                   # the rule applied to the whole candidate graph, as on the test set
        sel = st.sel_full[rows]
    return SimpleNamespace(ents=ents, rows=rows, p=p, sel=np.asarray(sel, bool), codes=codes,
                           y=st.y[rows].astype(bool), n_gt=st.n_gt[ents], name=name)


EV = eval_view()
f = per_entity_f(EV.codes, EV.y, EV.sel, EV.n_gt)
print(f"evaluated on: {EV.name} - {len(EV.ents):,} entities, macro F0.5 {f.mean():.5f}")
br = pd.DataFrame({"true matches": np.minimum(EV.n_gt, 8), "F0.5": f}) \
       .groupby("true matches")["F0.5"].agg(["mean", "size"])
br.index = [str(i) if i < 8 else "8+" for i in br.index]
display(br.rename(columns={"mean": "F0.5", "size": "entities"}))
""")

code(r"""
# ── 14.4 (optional, read-only) Where the F0.5 is lost ────────────────────────
# Splits (1 - macro F0.5) of the evaluated entities (14.3: hold-out, or out-of-fold
# fit entities without a hold-out) into the part lost at blocking (true records
# that never became candidates) and the part lost by the model / decision, the
# latter by kind of error and per country. Training data only; nothing is written.
def _ent_f(codes, label, sel, n_gt, beta=0.5):
    n = len(n_gt)
    tp = np.bincount(codes, weights=(label & sel).astype(float), minlength=n)
    npred = np.bincount(codes, weights=sel.astype(float), minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(n_gt > 0, tp / n_gt, 0.0)
        f = np.where(tp > 0, (1 + beta**2) * P * R / (beta**2 * P + R), 0.0)
    return np.where((n_gt == 0) & (npred == 0), 1.0, f), tp, npred


codes, yh, selh, n_gt = EV.codes, EV.y, EV.sel, EV.n_gt
f_act, tp, npred = _ent_f(codes, yh, selh, n_gt)
f_orc, reach, _ = _ent_f(codes, yh, yh, n_gt)
n = len(EV.ents)
kind = np.select(
    [(n_gt == 0) & (npred > 0), (n_gt > 0) & (npred == 0) & (reach > 0),
     (n_gt > 0) & (npred > tp), (n_gt > 0) & (tp < reach)],
    ["singleton given a match (scores 0)", "matchable entity left empty (scores 0)",
     "entity with >=1 wrong match", "entity missing reachable matches (no wrong one)"], "correct")
print(f"[{EV.name}] macro F0.5 {f_act.mean():.5f} | ceiling with these candidates {f_orc.mean():.5f} | "
      f"lost at blocking {1 - f_orc.mean():.5f} | lost by model + decision {f_orc.mean() - f_act.mean():.5f}")
d = pd.DataFrame({"kind": kind, "loss": f_orc - f_act})
display(d.groupby("kind")["loss"].agg(entities="size", loss_points=lambda s: s.sum() / n)
        .sort_values("loss_points", ascending=False).round(5))

xi = st.cand["x_idx"].values[EV.rows]
fp, fn = selh & ~yh, ~selh & yh
own = st.owner[xi[fp]]
print(f"wrong matches: {int(fp.sum()):,}   record owned by another S1 entity: "
      f"{(own >= 0).mean() if fp.any() else 0:.1%}")
if fn.any():
    print("final probability of the missed reachable true matches (10/25/50/75/90 %):",
          np.round(np.quantile(EV.p[fn], [0.1, 0.25, 0.5, 0.75, 0.9]), 3))
if fp.any():
    print("final probability of the wrong matches (10/25/50/75/90 %):",
          np.round(np.quantile(EV.p[fp], [0.1, 0.25, 0.5, 0.75, 0.9]), 3))

ctry = st.S1["country"].values[EV.ents]
by = pd.DataFrame({"country": ctry, "F0.5": f_act, "ceiling": f_orc}).groupby("country").mean()
by["entities"] = pd.Series(ctry).value_counts()
display(by.round(5))


def _share(col, mask):
    return float(st.X[col].values[xi[mask]].mean()) if mask.any() else np.nan


display(pd.DataFrame({name: [_share("name_nonlatin", m), _share("addr_empty", m)]
                      for name, m in (("all true pairs", yh), ("missed (FN)", fn), ("wrong (FP)", fp))},
                     index=["record name non-Latin", "record address empty"]).round(3))
""")

md(r"""
**14.5 Ambiguity floor: the best score any model could reach.** Some true records carry no information that picks their owner. A record with an **empty address**, whose Source 1 entity shares its exact normalised name with **other Source 1 entities of the same country**, reads identically against all of them. Under a precision-weighted F0.5 the best action for such a record is to leave it out. Scoring the hold-out as if every other true record were found, and every such record were left out, estimates the ceiling for *any* pipeline on this data. If this number is below your target, no model change can reach the target.

It is an estimate, not a strict bound:
- a model may still resolve some of these records from small name differences (legal form, word order);
- other kinds of ambiguity are not counted.
""")

code(r"""
# ── 14.5 (read-only) Ambiguity floor ─────────────────────────────────────────
# Uses the entities of 14.3 (hold-out, or out-of-fold fit entities without a hold-out);
# the ceiling itself needs labels only, no model.
key = pd.Series(st.S1["country"].values).astype(str) + "|" + pd.Series(st.S1["name_core"].values)
n_same = key.map(key.value_counts()).values                 # S1 entities with this (country, core name)
gx = pd.Index(np.asarray(st.X["entity_id"].values, dtype=object)).get_indexer(st.gt_x)
ok = (st.gt_gi >= 0) & (gx >= 0)
gi, gxx = st.gt_gi[ok], gx[ok]
in_ev = np.zeros(len(st.S1), bool); in_ev[EV.ents] = True
gi, gxx = gi[in_ev[gi]], gxx[in_ev[gi]]
amb = (st.X["addr_empty"].values[gxx] == 1) & (n_same[gi] >= 2)
n_gt_e = EV.n_gt
tp_floor = np.bincount(np.searchsorted(EV.ents, gi[~amb]), minlength=len(EV.ents)).astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    R = np.where(n_gt_e > 0, tp_floor / n_gt_e, 0.0)
    f_ceiling = np.where(tp_floor > 0, 1.25 * R / (0.25 + R), 0.0)   # precision 1 on what is predicted
f_ceiling = np.where(n_gt_e == 0, 1.0, f_ceiling)
has_amb = np.bincount(np.searchsorted(EV.ents, gi[amb]), minlength=len(EV.ents)) > 0
f_now = per_entity_f(EV.codes, EV.y, EV.sel, n_gt_e)
print(f"[{EV.name}] true pairs: {len(amb):,}   ambiguous (empty address, name shared by >=2 S1 of the country): "
      f"{int(amb.sum()):,} = {amb.mean() if len(amb) else 0:.3%}")
print(f"entities with >=1 ambiguous true record: {has_amb.sum():,} of {len(EV.ents):,} ({has_amb.mean():.2%})")
print(f"AMBIGUITY CEILING (every resolvable record found, every ambiguous one left out): {f_ceiling.mean():.5f}")
print(f"current macro F0.5 on the same entities: {f_now.mean():.5f}")
lost = (1 - f_now)
display(pd.DataFrame({"entities": [int(has_amb.sum()), int((~has_amb).sum())],
                      "loss points (of 1 - F0.5)": [lost[has_amb].sum() / len(lost), lost[~has_amb].sum() / len(lost)],
                      "irreducible part": [(1 - f_ceiling[has_amb]).sum() / len(lost), 0.0]},
                     index=["with ambiguous records", "without"]).round(5))
print("leaderboard ~ hold-out - 0.003 so far: a leaderboard above 0.996 needs a hold-out near 0.999, "
      "so this ceiling must be at least that high")
""")

# =============================================================================
# 15. Error analysis
# =============================================================================
md(r"""
## 15. Error analysis

A random sample of wrong merges (false positives) and missed links (false negatives), shown side by side. They are drawn from the hold-out, or from the out-of-fold fit entities when the run has no hold-out.
""")

code(r"""
# ── 15.1 False positives and false negatives on the hold-out ────────────────
def show_pairs(mask, rows, ph, title, n=10):
    idx = np.flatnonzero(mask)
    idx = np.random.default_rng(0).choice(idx, min(n, len(idx)), replace=False) if len(idx) else idx
    r = rows[idx]
    s1i, xi = st.cand["s1_idx"].values[r], st.cand["x_idx"].values[r]
    df = pd.DataFrame({"S1 name": st.S1["raw_name"].values[s1i], "S1 address": st.S1["raw_addr"].values[s1i],
                       "record name": st.X["raw_name"].values[xi], "record address": st.X["raw_addr"].values[xi],
                       "p": np.round(ph[idx], 3)})
    display(Markdown(f"**{title}** ({mask.sum():,} in total)")); display(df)

print("evaluated on:", EV.name)
show_pairs(EV.sel & ~EV.y, EV.rows, EV.p, "False positives (predicted, not a true match)")
show_pairs(~EV.sel & EV.y, EV.rows, EV.p, "False negatives among candidates (true match, not predicted)")
""")

md(r"""
**15.2 Corpus-size check (read-only, training data only).** The test corpus has its own size, and several signals depend on how many look-alikes a corpus holds: name-duplicate counts, claimants per record, IDF, top-k crowding. The hold-out is always scored inside the full training corpus. This cell measures whether the score moves with corpus size, which is one possible cause of the hold-out → leaderboard gap:

1. For each fraction in `SIZE_CHECK_FRACS`, a corpus is rebuilt **from the training files**: that fraction of the Source 1 entities, every record they own, and the same fraction of the records that belong to no entity. The composition matches the training data.
2. The trained pipeline runs on it **exactly as on the test set**: its own normalisation statistics, dense embeddings, blocking, all stages and the decision rule.
3. Only the **hold-out** entities inside it are scored, and they are compared with the same entities' scores in the full corpus (14.1).

Nothing is trained or tuned here. Each fraction costs roughly that fraction of a test inference.
""")

code(r"""
# ── 15.2 (optional, read-only) Corpus-size check on train-derived corpora ────
def _dump(path, frame, mask):
    # the raw text exactly as read (no field contains a tab), in the challenge's TSV layout
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for r in zip(*(frame[c].values[mask] for c in ("entity_id", "raw_name", "raw_addr", "country"))):
            fh.write("\t".join(r) + "\n")


def _f05(pred: set, gt: set) -> float:
    if not gt:
        return float(not pred)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    P, R = tp / len(pred), tp / len(gt)
    return 1.25 * P * R / (0.25 * P + R)


if SIZE_CHECK_FRACS and st.is_hold.any():
    sim_root = LOCAL_ROOT / "sim_data"
    owned = st.owner >= 0
    hold_ents = np.flatnonzero(st.is_hold); nf = st.n_fit_rows
    codes = np.searchsorted(hold_ents, st.cand["s1_idx"].values[st.model_rows][nf:])
    f_full = pd.Series(per_entity_f(codes, st.y[st.model_rows][nf:], st.holdout_sel, st.n_gt[hold_ents]),
                       index=st.S1["entity_id"].values[hold_ents])
    rng = np.random.default_rng(11)
    res = []
    for frac in SIZE_CHECK_FRACS:
        name = f"sim{int(round(frac * 1000)):04d}"
        pick = rng.random(len(st.S1)) < frac                               # Source 1 entities kept
        keep_x = np.where(owned, pick[np.clip(st.owner, 0, None)], rng.random(len(st.X)) < frac)
        d = sim_root / name
        if not (d / f"{name}_source3.tsv").exists():
            d.mkdir(parents=True, exist_ok=True)
            _dump(d / f"{name}_source1.tsv", st.S1, pick)
            for s in (2, 3):
                _dump(d / f"{name}_source{s}.tsv", st.X, keep_x & (st.X["src"].values == s))
        out = LOCAL_ROOT / "sim_out" / name
        t0 = time.time()
        pipeline.predict(sim_root, name, st.work, out, use_cache=USE_CACHE, ckpt_uri=S3_CKPT_URI,
                         s3_region=S3_REGION, s3_endpoint=S3_ENDPOINT)
        pred = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        pm = dict(zip(pred["source1_entity_id"], pred["matched_entity_ids"]))
        ids = st.S1["entity_id"].values[pick & st.is_hold]
        f_sim = np.array([_f05(set(filter(None, pm.get(i, "").split(","))), set(st.gt.get(i, []))) for i in ids])
        res.append({"corpus size": f"{frac:.0%}", "S1 entities": int(pick.sum()), "records": int(keep_x.sum()),
                    "hold-out entities scored": len(ids), "F0.5 in this corpus": f_sim.mean(),
                    "same entities, full corpus (14.1)": f_full.loc[ids].mean(),
                    "minutes": (time.time() - t0) / 60})
        print(res[-1])
    display(pd.DataFrame(res).round(5))
    print("If the score changes clearly with size, corpus-size effects explain part of the leaderboard gap.")
else:
    print("corpus-size check skipped (SIZE_CHECK_FRACS empty or no hold-out)")
""")

# =============================================================================
# 16. Save
# =============================================================================
md(r"""
## 16. Upload the trained artefacts to S3

`WORK_DIR/artifacts` holds everything inference needs: config, aliases, pruner, fold models, cross-encoder weights, feature lists, the decision rule and the metrics. The same files already live in the S3 checkpoints; this step also uploads them as one self-contained folder, `RUN_URI/artifacts/`, which Section 17 can restore on any instance.
""")

code(r"""
# ── 16.1 Upload artefacts + metrics to S3 ────────────────────────────────────
s3_upload(st.art, RUN_URI + "artifacts/")
(st.work / "run_uri.txt").write_text(RUN_URI)       # 17.1 / 19.1 upload next to these artefacts after a restart
print("artefacts saved to", RUN_URI + "artifacts/")
print(json.dumps(json.loads((st.art / "metrics.json").read_text()), indent=1)[:1500])
""")

# =============================================================================
# 17. Inference
# =============================================================================
md(r"""
## 17. Inference on the test set

The frozen artefacts are applied to the test split with **the same transforms**. Corpus statistics (IDF, genericness, locality vocabulary and phrases) are computed on the test corpus itself; this needs no labels and adapts the features to France. Candidate rows are processed in chunks, and stage-1 features are staged on disk. The training state is released first to free memory.

In DEV mode the train-derived `deveval` subset is used instead, and it is scored in 17.8.

Inference is checkpointed too: the normalised test frames, each block's candidates and each feature part are uploaded to `S3_CKPT_URI/predict-test-<hash>/`, so an interrupted inference also resumes where it stopped. These files are derived from the test split for inference only; nothing is trained or tuned on them.

> **After a kernel restart** you do not need to retrain. Either run the notebook from the top (every training step restores from its checkpoint in minutes), or run Sections 1–3 and cell 4.1 and continue from here: the trained artefacts are read from `WORK_DIR/artifacts`, or restored from the latest S3 run uploaded in 16.1.
""")

code(r"""
# ── 17.1 Free the training state and locate the trained artefacts ──────────
TRAINED_WORK_DIR = st.work if "st" in globals() else WORK_DIR
for _name in ("st", "cand", "y_fit", "p_fit", "ym", "band", "rows", "sel", "yh", "ph", "codes", "f", "g", "imp",
              "EV", "selh", "fp", "fn", "xi", "own", "d", "kind", "f_act", "f_orc", "tp", "npred", "reach",
              "amb", "gi", "gxx", "gx", "in_ev", "n_same", "key", "f_now", "f_ceiling", "tp_floor", "has_amb", "lost",
              "tp_", "res", "sample", "in_s", "same_ctry", "S1", "X"):
    globals().pop(_name, None)
gc.collect()

if not (TRAINED_WORK_DIR / "artifacts" / "meta.json").exists():
    latest = s3_latest_run(PROFILE)
    assert latest, f"no trained artefacts locally or under {S3_OUTPUT_URI}runs/ - run Sections 4-16 first"
    print("restoring artefacts from", latest)
    s3_download(latest + "artifacts/", TRAINED_WORK_DIR / "artifacts")
    RUN_URI = latest                               # outputs go next to the artefacts that produced them
    (TRAINED_WORK_DIR / "run_uri.txt").write_text(RUN_URI)
elif (TRAINED_WORK_DIR / "run_uri.txt").exists():  # local artefacts: their run folder, not this session's timestamp
    RUN_URI = (TRAINED_WORK_DIR / "run_uri.txt").read_text().strip()
print("results go to", RUN_URI)
print("artefacts :", TRAINED_WORK_DIR / "artifacts")
print(f"RAM in use: {psutil.virtual_memory().used/1e9:.1f} GB of {RAM_GB:.1f} GB")
""")

code(r"""
# ── 17.2 Download the test split from S3, then load + normalise it ──────────
if EVAL_SPLIT == "test":
    s3_download(S3_URI.rstrip("/") + "/dataset/test/", DATA_DIR / "test")
ps = pipeline.predict_init(EVAL_DATA_DIR, EVAL_SPLIT, TRAINED_WORK_DIR, OUT_DIR, use_cache=USE_CACHE,
                          ckpt_uri=S3_CKPT_URI, s3_region=S3_REGION, s3_endpoint=S3_ENDPOINT)
pipeline.predict_prepare(ps)
print(f"{EVAL_SPLIT}: {len(ps.S1):,} S1 entities, {len(ps.X):,} S2+S3 records")
display(ps.S1["country"].value_counts().rename("S1 entities per country").to_frame())
""")

code(r"""
# ── 17.3 Dense embeddings + candidate generation with the trained pruner ────
pipeline.predict_dense(ps)                         # skipped when the dense view was not trained
pipeline.predict_blocking(ps)
print(f"candidate pairs: {len(ps.cand):,}  ({len(ps.cand)/len(ps.S1):.1f} per S1 entity)")
display(pd.DataFrame({"country": ps.S1["country"].values[ps.cand["s1_idx"].values],
                      "source": ps.cand["src"].values})
          .value_counts().rename("candidate pairs").to_frame())
""")

code(r"""
# ── 17.4 Stage-1 features + scoring (parts on disk, checkpointed) ───────────
pipeline.predict_stage1(ps)
print(f"p1 > 0.5 for {(ps.p1 > 0.5).mean():.2%} of candidate pairs")
""")

code(r"""
# ── 17.5 Cross-encoder scoring (only if it was trained) ─────────────────────
pipeline.predict_cross_encoder(ps)
print(f"cross-encoder scored pairs: {np.isfinite(ps.ce).sum():,}")
""")

code(r"""
# ── 17.6 Stage-2 (+ stage-3 when it was chosen) scoring ─────────────────────
pipeline.predict_stage2(ps)
pipeline.predict_stage3(ps)                        # keeps p2 when stage 2 was the final stage
print("final stage:", ps.meta.get("final_stage", 2))
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(ps.pfin, bins=50, log=True, color="#8172B2"); ax.set_title("Final probabilities (log scale)")
plt.tight_layout(); plt.show()
""")

code(r"""
# ── 17.7 Decision rule + write both submission files ─────────────────────────
summary = pipeline.predict_write(ps)
display(pd.Series(summary, name="value").to_frame())
print(f"entities predicted as singletons: {1 - summary['s1_with_matches']/summary['s1_entities']:.2%}")
display(pd.read_csv(OUT_DIR / "matching_results.tsv", sep="\t", nrows=5, keep_default_na=False))
""")

code(r"""
# ── 17.8 (DEV mode) score the train-derived evaluation subset ───────────────
if EVAL_SPLIT == "deveval":
    gt_eval = ber_io.read_ground_truth(Path(EVAL_DATA_DIR) / "deveval" / "deveval_ground_truth.tsv")
    pred = pd.read_csv(OUT_DIR / "matching_results.tsv", sep="\t", keep_default_na=False, dtype=str)
    F = []
    for s1, m in zip(pred["source1_entity_id"], pred["matched_entity_ids"]):
        p, g = set(filter(None, m.split(","))), set(gt_eval.get(s1, []))   # absent = singleton
        if not g:
            F.append(float(not p)); continue
        tp = len(p & g)
        F.append(0.0 if tp == 0 else 1.25 * (tp/len(p)) * (tp/len(g)) / (0.25 * tp/len(p) + tp/len(g)))
    print(f"DEV evaluation macro F0.5 = {np.mean(F):.5f} on {len(F):,} entities")
""")

# =============================================================================
# 18. Validate
# =============================================================================
md(r"""
## 18. Validate the submission files

This runs the challenge's official validator, `utils/validate_submission.py`, and an internal consistency check:
- one row per Source 1 entity, with no duplicates;
- S2-/S3- IDs only;
- matches ⊆ candidates.
""")

code(r"""
# ── 18.1 Internal checks ─────────────────────────────────────────────────────
def read_lists(path):
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1].map(lambda s: [x for x in s.split(",") if x])))
m = read_lists(OUT_DIR / "matching_results.tsv"); c = read_lists(OUT_DIR / "candidate_pairs.tsv")
s1_required = set(ps.S1["entity_id"])
x_valid = set(ps.X["entity_id"])
checks = {
    "one row per S1 entity": set(m) == s1_required and len(m) == len(s1_required),
    "candidate file covers every S1": set(c) == s1_required,
    "no duplicate IDs in a list": all(len(v) == len(set(v)) for v in m.values()),
    "only S2-/S3- IDs": all(x[:3] in ("S2-", "S3-") for v in m.values() for x in v),
    "all IDs exist": all(x in x_valid for v in m.values() for x in v),
    "matches are a subset of candidates": all(set(m[k]) <= set(c.get(k, ())) for k in m),
}
display(pd.Series(checks, name="passed").to_frame())
assert all(checks.values()), "submission check failed"
del m, c, x_valid
""")

code(r"""
# ── 18.2 Official validator (test split only) ────────────────────────────────
if EVAL_SPLIT == "test" and VALIDATOR.exists():
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUT_DIR / "matching_results.tsv"),
                        "--candidate", str(OUT_DIR / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA_DIR / "test")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, f"official validator failed (exit code {r.returncode})"
elif EVAL_SPLIT == "test":
    print("utils/validate_submission.py not found - only the internal checks in 18.1 were run")
else:
    print("official validator runs on the real test split only (skipped in DEV mode)")
""")

# =============================================================================
# 19. Package
# =============================================================================
md(r"""
## 19. Package the submission zip and upload it to S3

This builds `<team>_submission.zip` in the required layout and uploads it, together with the two TSVs, to `RUN_URI`:

```
<team>_submission.zip
├── output/  matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/  src/, README.md, requirements.txt, docs/
└── Documentation_template.md
```

Before submitting, fill in the team fields and the final metrics in `Documentation_template.md`.
""")

code(r"""
# ── 19.1 Build the zip ────────────────────────────────────────────────────────
TEAM = "Expecto Patronum"                              # <── EDIT
# <team_name>_submission.zip; spaces become underscores so the file name is safe in any upload / shell
zpath = LOCAL_ROOT / f"{'_'.join(TEAM.split())}_submission.zip"
pkg = Path(PKG)
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(OUT_DIR / name, f"output/{name}")
    for p in pkg.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, f"code/business_entity_resolution/{p.relative_to(pkg).as_posix()}")
    z.write(pkg / "docs" / "Documentation_template.md", "Documentation_template.md")
print("wrote", zpath, f"({zpath.stat().st_size/1e6:.1f} MB)")
s3_upload(zpath, RUN_URI)
for name in ("matching_results.tsv", "candidate_pairs.tsv"):
    s3_upload(OUT_DIR / name, RUN_URI + "output/")
print("leaderboard file:", RUN_URI + "output/matching_results.tsv")
print("submission zip  :", RUN_URI + zpath.name)
""")

# =============================================================================
# 20. Final notes
# =============================================================================
md(r"""
## 20. Final refit & tuning levers

**Final refit.** After you are happy with the hold-out score, set `HOLDOUT_FRAC = 0.0` in 3.1 and run the notebook again (each run gets its own `ber_outputs/runs/...` folder on S3). Every entity is then used for fitting; the out-of-fold score in `metrics.json` stays an honest estimate. The static normalisation checkpoint is reused; every later step gets new checkpoints, because its hash includes the hold-out setting, so old and new work are never mixed.

**Reading the metrics**

| metric | meaning |
|---|---|
| `cand_recall_eval` | share of true matches that survive blocking: the recall ceiling |
| `holdout_macro_f05_oracle` | best achievable macro F0.5 with these candidates |
| `holdout_macro_f05` | the honest validation score of the full pipeline |
| `holdout_pair_precision` / `_recall` | where the remaining errors are |

**Levers** (in `cfg`, Section 3)
- **Recall-bound** (candidate recall well below the pair recall you need): raise `cfg.blocking.k_*`, `k_dense*` or `keep_per_source`, raise the `max_df_*` caps (`BLOCK_CAP_X`), or give the dense encoder more pairs / epochs (`cfg.model.dense_frac`, `cfg.dense.epochs`).
- **Precision-bound:** enable the cross-encoder (GPU), switch `CE_MODEL` to `FacebookAI/xlm-roberta-large`, increase `cross_encoder.max_train_pairs`, or widen `band_low` / `band_high`.
- **Ceiling:** 14.5 estimates the best score any model could reach on this data. If it is below a target, no model change will reach that target.
- **More training signal:** use a larger profile or `max_train_s1`.
- **Checkpoints:** `S3_CKPT_URI/<step>-<hash>/` (local copy in `WORK_DIR/ckpt/`). Changing a setting creates new checkpoints only for the steps that depend on it. After editing the library code itself (e.g. `normalize.py`), set `USE_CACHE = False` once, or bump `CKPT_VERSION` in `pipeline.py`. Old checkpoint folders can be deleted from S3 at any time; they are only read, never required.
""")

# =============================================================================
# 21. Unseen-country check
# =============================================================================
md(r"""
## 21. Unseen-country check (optional, `RUN_COUNTRY_CHECK`)

The test set contains **France**, which never appears in training, and the leaderboard has been about 0.003 below the hold-out for both pipelines. This section measures how much a *country the models never saw* costs:

1. Train the whole pipeline on the entities of one training country only. The aliases, noise tokens, dense encoder, pruner, GBDTs, cross-encoder and decision rule are all learned from that country.
2. Score a sample of the other country, exactly as the test set would be scored.
3. Compare with the trained country's own hold-out score.

Everything comes from `dataset/train`. It is a separate, long run: one full training per direction, at `LOCO_TRAIN_S1` entities, in its own work folder and S3 checkpoint prefix. It never touches the main run's models. India and the US differ more from each other than France differs from the US, so treat the drop as an upper estimate of France's cost.
""")

code(r"""
# ── 21.1 (optional) Unseen-country check: train on one country, score the other ──
LOCO_PAIRS    = (("US", "India"), ("India", "US"))   # (training country, scored country)
LOCO_TRAIN_S1 = 300_000                                # Source 1 entities of the training country
LOCO_EVAL_S1  = 100_000                                # Source 1 entities of the scored country
import re


def _f05(pred: set, gt: set) -> float:
    # per-entity F0.5 of the challenge (same as in 15.2; repeated so this cell runs on its own)
    if not gt:
        return float(not pred)
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    P, R = tp / len(pred), tp / len(gt)
    return 1.25 * P * R / (0.25 * P + R)


def _loco_split(raw, gt_owner, country, n_s1, rng):
    # n_s1 entities of one country, every record they own, and the same share of that country's unowned records
    s1 = raw["s1"][raw["s1"]["country"].values == country]
    pick = set(rng.choice(s1["entity_id"].values, min(n_s1, len(s1)), replace=False))
    rate = len(pick) / max(len(s1), 1)
    out = {"s1": s1[s1["entity_id"].isin(pick).values]}
    for k in ("s2", "s3"):
        f = raw[k][raw[k]["country"].values == country]
        own = f["entity_id"].map(gt_owner)
        out[k] = f[(own.isin(pick) | (own.isna() & (rng.random(len(f)) < rate))).values]
    return out, pick


def _loco_write(root, split, frames, gt_all, pick):
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    for k, f in frames.items():
        with open(d / f"{split}_source{k[-1]}.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in zip(f["entity_id"].values, f["business_name"].values, f["business_address"].values,
                         f["country"].values):
                fh.write("\t".join(r) + "\n")
    with open(d / f"{split}_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s in frames["s1"]["entity_id"].values:
            fh.write(f"{s}\t{','.join(gt_all.get(s, []))}\n")


if RUN_COUNTRY_CHECK:
    import copy
    raw = ber_io.read_split(TRAIN_DATA_DIR, "train")
    gt_all = ber_io.read_ground_truth(Path(TRAIN_DATA_DIR) / "train" / "train_ground_truth.tsv")
    gt_owner = pd.Series({x: s for s, lst in gt_all.items() for x in lst})
    loco_rows = []
    for train_c, eval_c in LOCO_PAIRS:
        root = LOCAL_ROOT / "loco" / re.sub(r"[^A-Za-z0-9]+", "_", f"{train_c}_to_{eval_c}")
        rng = np.random.default_rng(5)
        tr, tr_ids = _loco_split(raw, gt_owner, train_c, LOCO_TRAIN_S1, rng)
        ev, ev_ids = _loco_split(raw, gt_owner, eval_c, LOCO_EVAL_S1, rng)
        if not (root / "train" / "train_ground_truth.tsv").exists():
            _loco_write(root, "train", tr, gt_all, tr_ids)
        if not (root / "loeval" / "loeval_ground_truth.tsv").exists():
            _loco_write(root, "loeval", ev, gt_all, ev_ids)          # its labels are used only to score
        del tr, ev
        cfg_l = copy.deepcopy(cfg)
        cfg_l.model.holdout_frac = 0.15          # the trained country's own hold-out is the comparison point
        cfg_l.model.max_train_s1 = None
        cfg_l.model.ranker_s1 = min(cfg.model.ranker_s1, LOCO_TRAIN_S1 // 2)
        ck = dict(ckpt_uri=f"{S3_CKPT_URI}loco/{root.name}/", s3_region=S3_REGION, s3_endpoint=S3_ENDPOINT)
        m = pipeline.train(root, root / "work", cfg_l, use_cache=USE_CACHE, **ck)
        pipeline.predict(root, "loeval", root / "work", root / "out", use_cache=USE_CACHE, **ck)
        gt_e = ber_io.read_ground_truth(root / "loeval" / "loeval_ground_truth.tsv")
        pred = pd.read_csv(root / "out" / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        pm = dict(zip(pred["source1_entity_id"], pred["matched_entity_ids"]))
        f_ev = np.mean([_f05(set(filter(None, pm.get(s, "").split(","))), set(g)) for s, g in gt_e.items()])
        loco_rows.append({"trained on": train_c, "scored on (never seen)": eval_c,
                          "hold-out F0.5, trained country": m.get("holdout_macro_f05"),
                          "F0.5, unseen country": f_ev})
        print(loco_rows[-1])
        gc.collect()
    display(pd.DataFrame(loco_rows).round(5))
    del raw, gt_all, gt_owner
else:
    print("unseen-country check skipped (RUN_COUNTRY_CHECK=False)")
""")

# =============================================================================
def _md_to_comments(src: str) -> list[str]:
    out = []
    for ln in src.split("\n"):
        s = ln.rstrip()
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            out += ["#", f"# ▌{title.upper() if s.startswith('## ') or s.startswith('# ') else title}"]
        else:
            out.append(f"# {s}" if s else "#")
    return out


def _runnable(cell: dict) -> dict:
    """Every explanation cell becomes a code cell of '#' comments: running it does nothing,
    so neither 'Run All' nor cells pasted one by one into code cells can stop on prose."""
    if cell["cell_type"] != "markdown":
        return cell
    text = "\n".join(_md_to_comments("".join(cell["source"]).strip("\n"))).strip("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(text)}


nb = {
    "cells": [_runnable(c) for c in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(CELLS)} cells)")


# =============================================================================
# Plain copy-paste version: notebooks/sagemaker_cells.py
# =============================================================================
# One block per notebook cell, separated by a banner. Markdown is folded into
# comments at the top of the following code cell, and %%writefile cells become
# write_module(...) calls, so every block (banner included) is valid Python.
CELLS_OUT = ROOT / "notebooks" / "sagemaker_cells.py"
BAR = "═" * 76
WRITE_HELPER = '''
def write_module(rel_path: str, text: str) -> None:
    """Write one library file under PKG (same effect as %%writefile)."""
    path = Path(PKG) / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\\n"), encoding="utf-8")
    print("wrote", path)
'''.strip("\n")


def _as_python(src: str, first_module: bool) -> str:
    if not src.startswith("%%writefile {PKG}/"):
        return src
    head, _, body = src.partition("\n")
    rel = head[len("%%writefile {PKG}/"):].strip()
    assert "'''" not in body and not body.endswith("\\"), rel
    call = f"write_module({rel!r}, r'''\n{body}''')"
    return (WRITE_HELPER + "\n\n" + call) if first_module else call


blocks, pending_md, n_code, seen_module = [], [], sum(c["cell_type"] == "code" for c in CELLS), False
for c in CELLS:
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        pending_md.append(src)
        continue
    is_module = src.startswith("%%writefile")
    body = _as_python(src, is_module and not seen_module)
    seen_module |= is_module
    first = body.split("\n", 1)[0]
    title = first.strip("# ─").strip() if first.startswith("# ──") else \
        f"write {src.split(chr(10), 1)[0].split('}/', 1)[1]}" if is_module else "code"
    lines = [f"# {BAR}", f"# CELL {len(blocks) + 1:02d} / {n_code:02d}   {title}", f"# {BAR}"]
    for m in pending_md:
        lines += _md_to_comments(m) + ["#"]
    pending_md = []
    if first.startswith("# ──"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
    blocks.append("\n".join(lines + [body.rstrip("\n")]))
if pending_md:  # trailing notes (Section 20): comments only
    notes = [f"# {BAR}", "# NOTES (comments only - nothing to run)", f"# {BAR}"]
    for m in pending_md:
        notes += _md_to_comments(m)
    blocks.append("\n".join(notes))

header = f'''# {BAR}
# Business Entity Resolution - AWS SageMaker notebook as plain copy-paste cells
# {BAR}
# HOW TO USE
#   1. SageMaker Notebook Instance (kernel conda_pytorch_p310) or Studio JupyterLab
#      (Python 3 kernel); ml.g6.8xlarge (or ml.g5.8xlarge), EBS volume >= 100 GB.
#   2. For every "CELL NN" block below: create a new CODE cell, paste the whole
#      block (from its banner down to the next banner) and run it, in order.
#   3. Data is read from S3_URI (CELL 03, already set to the challenge bucket) and
#      results are uploaded to S3_URI/ber_outputs/runs/<profile>_<timestamp>/.
#      Only TEAM in the packaging cell needs editing (search for "<── EDIT").
#   4. Every checkpoint is mirrored to S3_URI/ber_outputs/checkpoints/. After a
#      crash, a kernel restart or on a new instance, run all cells again from
#      CELL 01: finished work is restored and training continues where it stopped.
# Generated by tools/build_notebook.py - identical code to the .ipynb.
'''
CELLS_OUT.write_text(header + "\n\n" + "\n\n\n".join(blocks) + "\n", encoding="utf-8")
print(f"wrote {CELLS_OUT} ({len(blocks)} code cells)")
