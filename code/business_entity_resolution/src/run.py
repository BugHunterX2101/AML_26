"""Command-line entry point.

    python src/run.py train   --data-dir <student_resource>/dataset --work-dir work
    python src/run.py predict --data-dir <student_resource>/dataset --work-dir work --out-dir output
    python src/run.py package --out-dir output --team <team_name>

Add ``--ckpt-uri s3://bucket/prefix/`` to train / predict to mirror every
checkpoint to S3: an interrupted run then resumes from the last completed
block / feature part / fold, on the same machine or a new one.
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber.config import Config  # noqa: E402
from ber.pipeline import predict, train  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[1]


def _setup_logging(work_dir: Path | None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(work_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _package(out_dir: Path, team: str, dest: Path) -> Path:
    zpath = dest / f"{team}_submission.zip"
    doc = CODE_ROOT / "docs" / "Documentation_template.md"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(out_dir / name, f"output/{name}")
        for p in CODE_ROOT.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                z.write(p, f"code/business_entity_resolution/{p.relative_to(CODE_ROOT).as_posix()}")
        z.write(doc, "Documentation_template.md")
    return zpath


def main() -> None:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit every model on dataset/train only")
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--work-dir", type=Path, required=True)
    t.add_argument("--no-cross-encoder", action="store_true")
    t.add_argument("--no-dense", action="store_true", help="block without the learned dense view")
    t.add_argument("--max-train-s1", type=int, default=None)
    t.add_argument("--holdout-frac", type=float, default=None,
                   help="S1 fraction kept out for validation (default 0.15; 0 = train on all)")
    t.add_argument("--n-jobs", type=int, default=-1)
    t.add_argument("--ckpt-uri", default=None, help="s3://... prefix for checkpoints (optional)")
    t.add_argument("--s3-region", default=None)

    p = sub.add_parser("predict", help="apply trained artefacts to a split")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ckpt-uri", default=None, help="s3://... prefix for checkpoints (optional)")
    p.add_argument("--s3-region", default=None)

    k = sub.add_parser("package", help="build <team>_submission.zip")
    k.add_argument("--out-dir", type=Path, required=True)
    k.add_argument("--team", required=True)
    k.add_argument("--dest", type=Path, default=Path("."))

    args = ap.parse_args()
    if args.cmd == "train":
        _setup_logging(args.work_dir)
        cfg = Config()
        cfg.n_jobs = args.n_jobs
        if args.no_cross_encoder:
            cfg.cross_encoder.enabled = False
        if args.no_dense:
            cfg.dense.enabled = False
        if args.holdout_frac is not None:
            cfg.model.holdout_frac = args.holdout_frac
        if args.max_train_s1:
            cfg.model.max_train_s1 = args.max_train_s1
        train(args.data_dir, args.work_dir, cfg, ckpt_uri=args.ckpt_uri, s3_region=args.s3_region)
    elif args.cmd == "predict":
        _setup_logging(args.work_dir)
        predict(args.data_dir, args.split, args.work_dir, args.out_dir,
                ckpt_uri=args.ckpt_uri, s3_region=args.s3_region)
    else:
        _setup_logging(None)
        print(_package(args.out_dir, args.team, args.dest))


if __name__ == "__main__":
    main()
