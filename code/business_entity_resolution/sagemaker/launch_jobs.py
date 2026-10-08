"""Run the Business Entity Resolution pipeline as SageMaker TRAINING JOBS.

Run this from a SageMaker Studio / Notebook-instance terminal (it uses that
session's AWS credentials and execution role), from the folder
code/business_entity_resolution:

    python sagemaker/launch_jobs.py check  --wait               # ~5 min preflight: packages, GPU, disk, models, S3
    python sagemaker/launch_jobs.py train  --wait               # training job (hold-out 0.15 -> honest score)
    python sagemaker/launch_jobs.py predict --run-name <run> --wait   # test inference with that run's artefacts
    python sagemaker/launch_jobs.py all                         # train, then predict, waiting for both
    python sagemaker/launch_jobs.py status <job-name>           # status (add --wait --tail to follow the log)

    final refit on every training entity:   python sagemaker/launch_jobs.py all --holdout-frac 0

What the jobs do (sagemaker_entry.py):
  * train   - the whole training pipeline with EXACTLY the notebook's configuration, so it
              shares the notebook's S3 checkpoints (ber_outputs/checkpoints/): stages the
              notebook already finished are restored, not recomputed. The test split is not
              mounted. Diagnostics of notebook cells 10.1/10.2/14.1/14.3-14.5/15.2 are
              written to <run>/reports/report.json; artefacts to <run>/artifacts/.
  * predict - test inference with a run's artefacts; both submission files plus checks
              (and the official validator) in <run>/output/.
Every stage checkpoints to S3, so a failed / stopped / spot-interrupted job is simply
launched again (same command) and continues where it stopped.

Uses boto3 only (no SageMaker Python SDK), so it works with whichever SDK version is installed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../business_entity_resolution/sagemaker
CODE_ROOT = HERE.parent                            # .../business_entity_resolution
S3_ROOT = "s3://aws-ml-162599956323-ap-south-1-an/6ab10eb3b23ba_student_resource/student_resource/"
REGION = "ap-south-1"
# AWS Deep Learning Container: PyTorch 2.5.1 training, GPU, Python 3.11, CUDA 12.4 (L4 / A10G / A100)
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
MAX_RUNTIME_LIMIT = 432_000                        # 5 days: the SageMaker training-job maximum
METRICS = ("holdout_macro_f05", "oof_macro_f05", "holdout_macro_f05_oracle", "cand_recall_eval",
           "union_recall_eval", "union_recall_eval_sparse_only", "holdout_pair_precision",
           "holdout_pair_recall", "ambiguity_ceiling", "final_stage")


def _bool(s: str) -> bool:
    if str(s).lower() in ("1", "true", "yes", "on"):
        return True
    if str(s).lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _boto(service: str, region: str):
    import boto3
    return boto3.client(service, region_name=region)


def split_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), f"not an s3:// URI: {uri}"
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def source_bundle() -> bytes:
    """sourcedir.tar.gz: the entry script, the job requirements (at the root, where the
    container installs them) and the ber library - nothing else."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(HERE / "sagemaker_entry.py", arcname="sagemaker_entry.py")
        tar.add(HERE / "requirements.txt", arcname="requirements.txt")
        for p in sorted((CODE_ROOT / "src" / "ber").glob("*.py")):
            tar.add(p, arcname=f"ber/{p.name}")
    return buf.getvalue()


def resolve_role(region: str, role_arn: str | None) -> str:
    """The execution role of this SageMaker session (or --role-arn)."""
    if role_arn:
        return role_arn
    arn = _boto("sts", region).get_caller_identity()["Arn"]
    acct = arn.split(":")[4]
    if ":assumed-role/" in arn:
        name = arn.split(":assumed-role/")[1].split("/")[0]
    elif ":role/" in arn:
        return arn
    else:
        raise SystemExit(f"cannot derive an execution role from {arn}: pass --role-arn arn:aws:iam::...:role/...")
    try:
        return _boto("iam", region).get_role(RoleName=name)["Role"]["Arn"]
    except Exception:
        # console-created SageMaker execution roles live under the service-role/ path
        guess = f"arn:aws:iam::{acct}:role/service-role/{name}" if "ExecutionRole" in name else \
            f"arn:aws:iam::{acct}:role/{name}"
        print(f"(iam:GetRole not allowed - assuming {guess}; pass --role-arn if the job is refused)")
        return guess


def channel(name: str, uri: str) -> dict:
    return {"ChannelName": name, "InputMode": "File", "CompressionType": "None",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri.rstrip("/") + "/",
                                            "S3DataDistributionType": "FullyReplicated"}}}


def job_request(a, mode: str, job_name: str, run_name: str, role: str, source_uri: str) -> dict:
    root = a.s3_root.rstrip("/") + "/"
    output_uri = root + "ber_outputs/"
    run_uri = f"{output_uri}runs/{run_name}/"
    chans = []
    if mode == "train":
        chans.append(channel("train", root + "dataset/train/"))          # training split ONLY
    elif mode == "predict":
        chans += [channel("test", root + "dataset/test/"),
                  channel("artifacts", a.artifacts_uri or run_uri + "artifacts/"),
                  channel("utils", root + "utils/")]
    if mode in ("train", "check"):
        if a.ce_model_s3:
            chans.append(channel("ce_model", a.ce_model_s3))
        if a.dense_model_s3:
            chans.append(channel("dense_model", a.dense_model_s3))
    hp = {"mode": mode, "run_name": run_name, "output_uri": output_uri, "region": a.region,
          "holdout_frac": a.holdout_frac, "seed": a.seed, "block_k_fwd": a.block_k_fwd,
          "block_k_rev": a.block_k_rev, "block_cap_x": a.block_cap_x, "use_dense": a.use_dense,
          "use_cross_encoder": a.use_cross_encoder, "use_stage3": a.use_stage3, "ce_model": a.ce_model,
          "size_check_fracs": a.size_check_fracs, "use_cache": not a.no_cache,
          # script mode of the SageMaker PyTorch container
          "sagemaker_program": "sagemaker_entry.py", "sagemaker_submit_directory": source_uri,
          "sagemaker_region": a.region, "sagemaker_container_log_level": 20}
    max_run = min(int(a.max_hours * 3600), MAX_RUNTIME_LIMIT)
    stop = {"MaxRuntimeInSeconds": max_run}
    if a.spot:
        stop["MaxWaitTimeInSeconds"] = max_run + 86_400        # up to one extra day waiting for capacity
    req = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": a.image_uri or IMAGE.format(region=a.region), "TrainingInputMode": "File",
            "MetricDefinitions": [{"Name": m, "Regex": rf"METRIC {m}=([-0-9.eE]+)"} for m in METRICS]},
        "RoleArn": role,
        "OutputDataConfig": {"S3OutputPath": f"{output_uri}jobs/"},
        "ResourceConfig": {"InstanceType": a.instance_type, "InstanceCount": 1, "VolumeSizeInGB": a.volume_gb},
        "StoppingCondition": stop,
        # the container's script mode reads every hyperparameter as a JSON value
        "HyperParameters": {k: json.dumps(v) for k, v in hp.items()},
        "Environment": {"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
        "EnableManagedSpotTraining": bool(a.spot),
        "Tags": [{"Key": "project", "Value": "business-entity-resolution"}, {"Key": "run", "Value": run_name}],
    }
    if chans:
        req["InputDataConfig"] = chans
    return req


def submit(a, mode: str, run_name: str) -> str:
    job_name = f"ber-{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    root = a.s3_root.rstrip("/") + "/"
    source_uri = f"{root}ber_outputs/jobs/{job_name}/source/sourcedir.tar.gz"
    bundle = source_bundle()
    role = "arn:aws:iam::000000000000:role/DRY-RUN" if a.dry_run else resolve_role(a.region, a.role_arn)
    req = job_request(a, mode, job_name, run_name, role, source_uri)
    if a.dry_run:
        print(json.dumps(req, indent=1))
        print(f"(dry run: source bundle of {len(bundle) / 1e3:.0f} kB not uploaded, job not created)")
        return job_name
    bucket, key = split_s3(source_uri)
    _boto("s3", a.region).put_object(Bucket=bucket, Key=key, Body=bundle)
    try:
        _boto("sagemaker", a.region).create_training_job(**req)
    except Exception as e:
        msg, hint = str(e), ""
        if "ResourceLimitExceeded" in msg:
            hint = (f"\n-> the account's training-job quota for {a.instance_type} is too low: request an increase in "
                    f"Service Quotas > Amazon SageMaker > '{a.instance_type} for "
                    f"{'spot ' if a.spot else ''}training job usage'.")
        elif "AccessDenied" in msg or "not authorized" in msg:
            hint = "\n-> pass --role-arn with the SageMaker execution role (it needs S3 read/write on the bucket)."
        raise SystemExit(f"create_training_job failed: {e}{hint}") from None
    print(f"started {job_name}  ({a.instance_type}{', spot' if a.spot else ''}, run '{run_name}')")
    print(f"  run folder : {root}ber_outputs/runs/{run_name}/")
    print(f"  console    : https://{a.region}.console.aws.amazon.com/sagemaker/home?region={a.region}#/jobs/{job_name}")
    print(f"  logs       : CloudWatch /aws/sagemaker/TrainingJobs, stream {job_name}/algo-1-...")
    return job_name


def wait(region: str, job_name: str, tail: bool = True, poll: int = 60) -> str:
    sm = _boto("sagemaker", region)
    logs = _boto("logs", region) if tail else None
    token, stream, last = None, None, None
    while True:
        d = sm.describe_training_job(TrainingJobName=job_name)
        status, secondary = d["TrainingJobStatus"], d.get("SecondaryStatus")
        if (status, secondary) != last:
            print(f"[{time.strftime('%H:%M:%S')}] {job_name}: {status} / {secondary}", flush=True)
            last = (status, secondary)
        if logs is not None:
            try:
                if stream is None:
                    s = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs",
                                                  logStreamNamePrefix=job_name + "/").get("logStreams", [])
                    stream = s[0]["logStreamName"] if s else None
                if stream:
                    kw = {"nextToken": token} if token else {"startFromHead": True}
                    ev = logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", logStreamName=stream, **kw)
                    for e in ev["events"]:
                        print(e["message"], flush=True)
                    token = ev["nextForwardToken"]
            except Exception:
                pass                                   # the log is a convenience; the status is what counts
        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                print(f"{job_name} {status}: {d.get('FailureReason', '')}")
                print("-> fix the cause and launch the same command again: finished stages are restored from S3.")
            for m in d.get("FinalMetricDataList", []):
                print(f"  {m['MetricName']:<34} {m['Value']:.5f}")
            return status
        time.sleep(poll)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SageMaker training jobs for the BER pipeline")
    ap.add_argument("command", choices=("check", "train", "predict", "all", "status"))
    ap.add_argument("job", nargs="?", help="job name (status only)")
    ap.add_argument("--run-name", default=None, help="run folder name (default job_<timestamp>); "
                                                     "reuse it to predict a run")
    ap.add_argument("--artifacts-uri", default=None,
                    help="predict with other artefacts, e.g. a notebook run: s3://.../ber_outputs/runs/<run>/artifacts/")
    ap.add_argument("--s3-root", default=S3_ROOT)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--role-arn", default=None)
    ap.add_argument("--instance-type", default="ml.g6.8xlarge", help="32 vCPU, 128 GiB, 1x L4 (or ml.g5.8xlarge)")
    ap.add_argument("--volume-gb", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=120)
    ap.add_argument("--spot", action="store_true", help="managed spot capacity (cheaper; resumes from S3 checkpoints)")
    ap.add_argument("--image-uri", default=None)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--tail", action="store_true", help="with --wait: stream the job log")
    ap.add_argument("--dry-run", action="store_true", help="print the job request only (no AWS calls)")
    # pipeline switches - defaults = the notebook's (cell 3.1), so the checkpoints are shared
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--block-k-fwd", type=int, default=15)
    ap.add_argument("--block-k-rev", type=int, default=5)
    ap.add_argument("--block-cap-x", type=float, default=1)
    ap.add_argument("--use-dense", type=_bool, default=True)
    ap.add_argument("--use-cross-encoder", type=_bool, default=True)
    ap.add_argument("--use-stage3", type=_bool, default=True)
    ap.add_argument("--ce-model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--ce-model-s3", default=None, help="pretrained cross-encoder on S3 (no-internet jobs)")
    ap.add_argument("--dense-model-s3", default=None, help="pretrained dense encoder on S3 (no-internet jobs)")
    ap.add_argument("--size-check-fracs", default="0.1,0.3", help="15.2 corpus-size check; '' skips it")
    ap.add_argument("--no-cache", action="store_true", help="ignore every checkpoint (recompute all)")
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    if a.command == "status":
        if not a.job:
            raise SystemExit("status needs a job name")
        if a.wait:
            sys.exit(0 if wait(a.region, a.job, tail=a.tail, poll=30) == "Completed" else 1)
        d = _boto("sagemaker", a.region).describe_training_job(TrainingJobName=a.job)
        keep = ("TrainingJobStatus", "SecondaryStatus", "FailureReason", "TrainingStartTime",
                "TrainingEndTime", "FinalMetricDataList")
        print(json.dumps({k: d[k] for k in keep if k in d}, indent=1, default=str))
        return
    if a.command == "predict" and not (a.run_name or a.artifacts_uri):
        raise SystemExit("predict needs --run-name <run> (or --artifacts-uri s3://.../artifacts/)")
    run_name = a.run_name or f"job_{time.strftime('%Y%m%d-%H%M%S')}"
    if a.command in ("check", "train", "predict"):
        job = submit(a, a.command, run_name)
        if a.wait and not a.dry_run:
            sys.exit(0 if wait(a.region, job, tail=a.tail) == "Completed" else 1)
        return
    # all: train, then predict with this run's artefacts
    job = submit(a, "train", run_name)
    if a.dry_run:
        submit(a, "predict", run_name)
        return
    if wait(a.region, job, tail=a.tail) != "Completed":
        sys.exit(1)
    job = submit(a, "predict", run_name)
    sys.exit(0 if wait(a.region, job, tail=a.tail) == "Completed" else 1)


if __name__ == "__main__":
    main()
