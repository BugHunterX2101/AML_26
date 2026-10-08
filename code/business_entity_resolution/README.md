# Business Entity Resolution — reproducible pipeline

Links every Source 1 business to its Source 2 / Source 3 records with a
**blocking (sparse views + learned dense view) → two-stage GBDT (+ multilingual
cross-encoder) → expected-F0.5 decision** pipeline. All models are trained on `dataset/train` only; the test
split is touched only by `predict`.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
```

`torch` / `transformers` are needed for the dense retrieval view and the
cross-encoder. Both run on a CUDA GPU and are skipped automatically without
one (or with `--no-dense` / `--no-cross-encoder`); everything else is CPU
(multi-core) code.

## 2. Data layout

Point `--data-dir` at the challenge `dataset/` folder:

```
dataset/
  train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 3. Run end-to-end

```bash
# (a) Train + validate. Uses ONLY dataset/train. 15% of S1 entities are held out
#     and never trained on; their macro F0.5 is written to work/artifacts/metrics.json
python src/run.py train --data-dir ../../dataset --work-dir work

# (b) Optional final refit on 100% of the training entities (the OOF estimate
#     in metrics.json remains honest)
python src/run.py train --data-dir ../../dataset --work-dir work_full --holdout-frac 0

# (c) Inference on the test split -> output/matching_results.tsv + output/candidate_pairs.tsv
python src/run.py predict --data-dir ../../dataset --split test --work-dir work_full --out-dir ../../output

# (d) Validate the submission format (from the student_resource/ directory)
python utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test

# (e) Build <team>_submission.zip (output/ + code/ + filled Documentation_template.md)
python src/run.py package --out-dir ../../output --team <team_name> --dest ../..
```

### AWS SageMaker (S3)

`notebooks/Business_Entity_Resolution_Colab.ipynb` (despite the file name, it targets **AWS SageMaker**) runs the same pipeline cell by cell; `notebooks/sagemaker_cells.py` holds the identical cells as plain copy-paste blocks. The notebook downloads `dataset/train` and `utils/` from `s3://aws-ml-162599956323-ap-south-1-an/6ab10eb3b23ba_student_resource/student_resource/` to the instance's EBS volume. It then trains every stage with diagnostics and plots, downloads `dataset/test` only for inference, validates the submission and uploads the artefacts, both TSVs and the submission zip to `.../student_resource/ber_outputs/runs/<profile>_<timestamp>/`. Every checkpoint is mirrored to `.../student_resource/ber_outputs/checkpoints/`: after a crash, a kernel restart or on a new instance, run the notebook again from the top and it continues from the last completed block, feature part or fold. Recommended: `ml.g5.8xlarge` (128 GiB RAM, A10G GPU for the cross-encoder; `ml.g5.4xlarge` with 64 GiB also runs the full profile), EBS volume >= 100 GB, kernel `conda_pytorch_p310` (Notebook Instance) or Python 3 (Studio). The execution role needs `s3:ListBucket`, `s3:GetObject` and `s3:PutObject` on the bucket. The notebook is generated from the sources by `python tools/build_notebook.py`; re-run it after editing any module.

Every expensive step checkpoints into `<work-dir>/ckpt/<step>-<hash>/`, and with `--ckpt-uri s3://bucket/prefix/` (as the notebook does) each file is also uploaded to S3 the moment it is complete. The granularity is one blocking block, one 2-million-row feature part, one model fold or one cross-encoder half. Re-running `train` or `predict`, on the same machine or a new one, restores finished work and continues from the first missing file; a resumed run is bit-identical to an uninterrupted one. Each hash covers the training files, the settings the step depends on and its upstream steps, so changing a setting re-runs only the affected steps. After editing the library code itself, run once with caching off or bump `CKPT_VERSION` in `pipeline.py`.

Useful switches for `train`:

| flag | effect |
| --- | --- |
| `--no-cross-encoder` | skip the mDeBERTa stage |
| `--no-dense` | block without the learned dense view (with both flags: pure CPU pipeline) |
| `--holdout-frac F` | S1 fraction held out for validation (default 0.15, `0` = train on all) |
| `--max-train-s1 N` | fit the GBDTs on at most N S1 entities (candidates are still generated and scored for every entity, so competition features match the test set) |
| `--n-jobs N` | CPU workers for normalisation / blocking / features (default all cores) |

Every other hyper-parameter lives in `src/ber/config.py` and is saved to
`work/artifacts/config.json` for each run.

### SageMaker training jobs (alternative to the notebook)

`sagemaker/launch_jobs.py` runs the same pipeline as managed SageMaker training
jobs. It uses boto3 only, and should be run from a Studio or notebook-instance
terminal inside `code/business_entity_resolution`:

```bash
python sagemaker/launch_jobs.py check  --wait                     # ~5 min: packages, GPU, disk, model downloads, S3
python sagemaker/launch_jobs.py train  --wait --tail              # hold-out 0.15: honest score + diagnostics
python sagemaker/launch_jobs.py predict --run-name <run> --wait   # test inference -> submission files
python sagemaker/launch_jobs.py all --holdout-frac 0              # final refit, then predict
```

**What each job does**
- **The training job** mounts `dataset/train/` only. The test split exists only in the predict job.
- **Same configuration as the notebook.** It builds exactly the configuration of notebook cell 3.2, so it shares the notebook's S3 checkpoints. Stages the notebook already finished are restored, not recomputed. A failed, stopped or spot-interrupted job simply resumes when you launch it again.

**What each run folder contains** (`ber_outputs/runs/<run>/`)

| folder | content |
| --- | --- |
| `artifacts/` | the trained models (the predict job's input) |
| `reports/report.json` | the diagnostics of notebook cells 10.1/10.2/14.1/14.3–14.5/15.2 |
| `output/` | `matching_results.tsv`, `candidate_pairs.tsv` and `predict_summary.json` (internal checks and the official validator) |

**Launcher defaults**
- **Instance:** `ml.g6.8xlarge`, with a 300 GB volume and the AWS PyTorch 2.5.1 GPU image (Python 3.11).
- **Job dependencies:** installed from `sagemaker/requirements.txt`, which leaves the image's torch, numpy and CUDA untouched.
- **Spot pricing:** add `--spot`.
- **No internet access:** pass `--ce-model-s3` / `--dense-model-s3`.

## 4. Code map

| file | role |
| --- | --- |
| `src/run.py` | CLI: `train`, `predict`, `package` |
| `src/ber/io.py` | TSV readers/writers (tab-separated, no quoting) |
| `src/ber/normalize.py` | transliteration (any script → Latin), noise cleanup, legal-form / street-type canonicalisation, number parsing, locality-phrase joining, phonetic skeleton |
| `src/ber/aliases.py` | token aliases (tx↔texas, up↔uttar_pradesh, praivet↔pvt …) and noise tokens learned from training labels |
| `src/ber/blocking.py` | 5-view sparse TF-IDF top-k blocking (forward + reverse), the dense view's top-k, and learned candidate pruning |
| `src/ber/dense.py` | learned dense retrieval view: MiniLM-L12 bi-encoder (Apache-2.0) fine-tuned on training pairs, exact GPU top-k |
| `src/ber/features.py` | ~75 pairwise features + stage-2 context features |
| `src/ber/models.py` | entity-grouped K-fold LightGBM with out-of-fold predictions |
| `src/ber/cross_encoder.py` | optional mDeBERTa-v3-base cross-encoder (MIT) |
| `src/ber/decision.py` | exact expected-F0.5 selection, exclusivity, macro-F0.5 metric |
| `src/ber/checkpoint.py` | stage checkpoints on local disk, mirrored to S3 |
| `src/ber/diagnostics.py` | blocking / score / loss-breakdown / ambiguity-ceiling / corpus-size reports (JSON) |
| `sagemaker/launch_jobs.py` | starts and follows the SageMaker training jobs (boto3) |
| `sagemaker/sagemaker_entry.py` | what runs inside a job: `check`, `train`, `predict` |
| `sagemaker/requirements.txt` | the jobs' extra packages (installed by the container) |
| `src/ber/pipeline.py` | resumable train / predict steps (checkpointed, chunked, memory-safe) |
| `notebooks/Business_Entity_Resolution_Colab.ipynb` | step-by-step AWS SageMaker / S3 notebook (generated) |
| `notebooks/sagemaker_cells.py` | the same notebook as plain copy-paste cells (generated) |
| `tools/build_notebook.py` | regenerates the notebook from the sources |

## 5. Artefacts written by `train`

`work/artifacts/`: `config.json`, `aliases.json`, `block_ranker.txt`,
`dense_encoder/` (bi-encoder), `stage1_fold*.txt`, `stage2_fold*.txt`, `stage3_fold*.txt`,
`*_importance.csv`, `ce_0/`, `ce_1/` (cross-encoder), `meta.json` (feature lists + chosen decision rule) and
`metrics.json` (recall ceilings, OOF and hold-out macro F0.5). `work/run.log`
holds the full log.

## 6. Fair play / licences

No external data, APIs, gazetteers or geocoders are used. The only built-in
knowledge is generic abbreviation lists in `normalize.py` (Rd→road,
Pvt→private, SARL/SAS…). Every learned artefact comes from `dataset/train`.
Libraries are BSD / MIT / Apache / ISC licensed. The two neural models are
`microsoft/mdeberta-v3-base` (cross-encoder, MIT, 278M parameters) and
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dense retrieval
bi-encoder, Apache-2.0, 118M parameters), both far under the 8B limit and both
fine-tuned on `dataset/train` only.
