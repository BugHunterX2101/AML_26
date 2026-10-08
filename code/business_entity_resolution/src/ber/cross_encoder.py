"""Optional multilingual cross-encoder re-scorer (mDeBERTa-v3-base, MIT licence).

The GBDT is excellent on engineered similarities but cannot "read" a Telugu
business name against its English form, or reason about token-level
correspondences across two noisy strings. A cross-encoder attends jointly over
both records ("name | address" [SEP] "name | address") with a multilingual
SentencePiece vocabulary that natively covers Indic scripts and French.

It is trained only on training pairs and applied only to the uncertain band of
stage-1 probabilities; its score becomes a stage-2 feature (NaN outside the
band), so the GBDT learns how much to trust it.
"""
from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# The model runs on PyTorch. Keep transformers from importing TensorFlow / JAX
# (preinstalled on SageMaker images), whose CUDA plugins only print
# "factory already registered" noise. Must be set before transformers loads.
# The DataLoader's worker processes are forked after the tokenizer was used in the
# main process: Rust-side parallelism must be off to avoid the fork warning / deadlock.
for _k, _v in (("USE_TF", "0"), ("USE_TORCH", "1"), ("USE_FLAX", "0"), ("USE_JAX", "0"),
               ("TRANSFORMERS_NO_TF", "1"), ("TF_CPP_MIN_LOG_LEVEL", "3"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_k, _v)


def _quiet_transformers() -> None:
    """Silence expected messages: the freshly initialised classification head
    (it is trained here) and the fast-tokenizer byte-fallback note."""
    import transformers
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=".*byte fallback option.*")


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The cross-encoder stage needs `torch` and `transformers` "
            "(pip install -r requirements.txt), or run with --no-cross-encoder."
        ) from e


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _grad_scaler(torch, enabled: bool):
    try:  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def pair_text(names: np.ndarray, addrs: np.ndarray) -> list[str]:
    return [f"{' '.join(n.split())} | {' '.join(a.split())}" for n, a in zip(names, addrs)]


def _amp_dtype(torch, fp16: bool):
    if not fp16 or not torch.cuda.is_available():
        return None
    # Native bf16 needs Ampere+ (A100 / L4 / H100); newer torch reports emulated
    # bf16 as "supported" on a T4, which is far slower than fp16 there.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def train_cross_encoder(text_a: list[str], text_b: list[str], labels: np.ndarray,
                        cfg, out_dir: Path, seed: int) -> Path:
    _require_torch()
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    _quiet_transformers()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("cross-encoder training on %s", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=1).to(device)
    idx = np.random.default_rng(seed).permutation(len(labels))

    def collate(batch_ids):
        enc = tok([text_a[i] for i in batch_ids], [text_b[i] for i in batch_ids],
                  truncation=True, max_length=cfg.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels[batch_ids], dtype=torch.float32)
        return enc

    loader = DataLoader(idx, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=getattr(cfg, "num_workers", 2), drop_last=False)
    steps = cfg.epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * steps), steps)
    amp = _amp_dtype(torch, cfg.fp16)
    scaler = _grad_scaler(torch, amp == torch.float16)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    for _ in range(cfg.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits = model(**batch).logits.squeeze(-1)
            loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                log.info("cross-encoder step %d/%d loss %.4f", step, steps, loss.item())
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def score_pairs(model_dir: Path, text_a: list[str], text_b: list[str], cfg) -> np.ndarray:
    _require_torch()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not text_a:
        return np.zeros(0, dtype=np.float32)
    _quiet_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    amp = _amp_dtype(torch, cfg.fp16)
    lengths = np.array([len(a) + len(b) for a, b in zip(text_a, text_b)])
    order = np.argsort(lengths)
    out = np.zeros(len(text_a), dtype=np.float32)
    bs = cfg.infer_batch_size
    with torch.no_grad():
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            enc = tok([text_a[i] for i in ids], [text_b[i] for i in ids], truncation=True,
                      max_length=cfg.max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits = model(**enc).logits.squeeze(-1).float()
            out[ids] = torch.sigmoid(logits).cpu().numpy()
            if (s // bs) % 2000 == 0:
                log.info("cross-encoder scoring %d / %d", s, len(order))
    return out


def sample_training_pairs(band_idx: np.ndarray, other_idx: np.ndarray, labels: np.ndarray,
                          max_pairs: int, seed: int) -> np.ndarray:
    """Mostly uncertain pairs, plus 20% confident ones for calibration."""
    rng = np.random.default_rng(seed)
    n_band = min(len(band_idx), int(math.ceil(0.8 * max_pairs)))
    n_other = min(len(other_idx), max_pairs - n_band)
    pick = np.concatenate([rng.choice(band_idx, n_band, replace=False) if n_band else band_idx[:0],
                           rng.choice(other_idx, n_other, replace=False) if n_other else other_idx[:0]])
    log.info("cross-encoder training pairs: %d (positives %.3f)", len(pick),
             labels[pick].mean() if len(pick) else 0.0)
    return pick
