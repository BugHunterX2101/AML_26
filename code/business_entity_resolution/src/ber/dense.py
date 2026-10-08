"""Learned dense retrieval view (bi-encoder), a sixth blocking view.

The sparse TF-IDF views drop every token above a document-frequency cap, and
at full scale (about a million entities per country block) that removes most
of a generic name ("Prime Money", "Shree Ganesh Traders"): such records keep
an almost empty vector and their true entity is crowded out of every top-k.
This is why candidate recall fell from 99.9% on a 20k-entity slice to 98.1%
on the full training set.

A dense view has no cap. Every record is embedded as "name | address"
(normalised text) by a siamese sentence encoder and retrieved by exact cosine
top-k on the GPU, so a common name is still disambiguated by its address and
a transliterated one by its sub-word pieces.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0, 118M parameters, 384-dim). It is fine-tuned on training true pairs
with the symmetric in-batch-negative contrastive loss (InfoNCE / "multiple
negatives ranking", as in DPR and Sentence-BERT):

  * batches are drawn inside one country, so the negatives are hard;
  * records of the same entity inside a batch are masked out as negatives;
  * the pairs come from a separate slice of training entities that no other
    model is fitted on, so the embedding cosine stays an honest feature for
    the pruner and the GBDTs (see ModelConfig.dense_frac).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_encoder import _amp_dtype, _grad_scaler, _quiet_transformers, _require_torch

log = logging.getLogger(__name__)


def record_text(frame: pd.DataFrame, idx: np.ndarray | None = None) -> list[str]:
    """Encoder input of every record (or of rows ``idx``): normalised name | address."""
    names, addrs = frame["name_full"].values, frame["addr_full"].values
    if idx is not None:
        names, addrs = names[idx], addrs[idx]
    return [f"{n} | {a}" for n, a in zip(names, addrs)]


def _embed(torch, model, tok, texts: list[str], max_len: int, device, amp):
    """Mean-pooled, L2-normalised sentence embeddings (float32 tensor)."""
    enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    v = (h * m).sum(1).float() / m.sum(1).float().clamp(min=1.0)
    return torch.nn.functional.normalize(v, dim=-1)


def _batches(groups: np.ndarray, batch_size: int, rng) -> list[np.ndarray]:
    """Shuffled batches, each drawn inside one group (country)."""
    out = []
    for g in np.unique(groups):
        idx = rng.permutation(np.flatnonzero(groups == g))
        out += [idx[s:s + batch_size] for s in range(0, len(idx), batch_size) if len(idx) - s >= 2]
    return [out[i] for i in rng.permutation(len(out))]


def train_encoder(text_a: list[str], text_b: list[str], entity: np.ndarray, groups: np.ndarray,
                  cfg, out_dir: Path, seed: int) -> Path:
    """Fine-tune the siamese encoder: text_a[i] (S1) and text_b[i] (S2/S3) are a
    true pair of S1 entity ``entity[i]``; ``groups[i]`` (country) keeps batches
    homogeneous."""
    _require_torch()
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    _quiet_transformers()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("dense encoder training on %s: %d pairs, %d epoch(s)",
             torch.cuda.get_device_name(0) if device == "cuda" else "CPU", len(text_a), cfg.epochs)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name).to(device)
    rng = np.random.default_rng(seed)
    entity = np.asarray(entity)
    n_batches = len(_batches(groups, cfg.batch_size, np.random.default_rng(seed)))
    steps = max(1, cfg.epochs * n_batches)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * steps), steps)
    amp = _amp_dtype(torch, cfg.fp16)
    scaler = _grad_scaler(torch, amp == torch.float16)
    ce = torch.nn.functional.cross_entropy
    model.train()
    step = 0
    for _ in range(cfg.epochs):
        for b in _batches(groups, cfg.batch_size, rng):
            ea = _embed(torch, model, tok, [text_a[i] for i in b], cfg.max_len, device, amp)
            eb = _embed(torch, model, tok, [text_b[i] for i in b], cfg.max_len, device, amp)
            logits = ea @ eb.T / cfg.temperature
            e = torch.as_tensor(entity[b], device=device)
            # another record of the same entity is a true match, not a negative
            sibling = (e[:, None] == e[None, :]) & ~torch.eye(len(b), dtype=torch.bool, device=device)
            logits = logits.masked_fill(sibling, -1e4)
            target = torch.arange(len(b), device=device)
            loss = (ce(logits, target) + ce(logits.T, target)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 250 == 0:
                log.info("dense encoder step %d/%d loss %.4f", step, steps, loss.item())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def encode(model_dir: Path, texts: list[str], cfg) -> np.ndarray:
    """L2-normalised float16 embeddings of ``texts`` (batched by length)."""
    _require_torch()
    import torch
    from transformers import AutoModel, AutoTokenizer

    _quiet_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    amp = _amp_dtype(torch, cfg.fp16)
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    order = np.argsort(np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts)), kind="stable")
    bs = cfg.infer_batch_size
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            e = _embed(torch, model, tok, [texts[i] for i in ids], cfg.max_len, device, amp)
            out[ids] = e.half().cpu().numpy()
            if (s // bs) % 500 == 0:
                log.info("dense encoding %d / %d", s, len(order))
    return out


def topk_dense(A: np.ndarray, B: np.ndarray, k: int):
    """For every row of A, the k rows of B with the highest cosine (rows are
    L2-normalised). Exact search in row chunks on the GPU (CPU fallback).
    Returns (row of A, row of B) int64 arrays."""
    empty = np.empty(0, dtype=np.int64)
    if len(A) == 0 or len(B) == 0 or k <= 0:
        return empty, empty
    _require_torch()
    import torch

    k = min(k, len(B))
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    dt = torch.float16 if cuda else torch.float32
    Bt = torch.from_numpy(np.ascontiguousarray(B)).to(dev, dt)
    # score-matrix rows per chunk: a quarter of the free GPU memory (speed only; results are identical)
    budget = torch.cuda.mem_get_info()[0] // 4 if cuda else 2 * 2**30
    chunk = int(max(16, min(len(A), budget // (len(B) * (2 if cuda else 4)))))
    ia, ib = [], []
    with torch.inference_mode():
        for s in range(0, len(A), chunk):
            a = torch.from_numpy(np.ascontiguousarray(A[s:s + chunk])).to(dev, dt)
            idx = torch.topk(a @ Bt.T, k, dim=1, sorted=False).indices
            ib.append(idx.cpu().numpy().astype(np.int64).ravel())
            ia.append(np.repeat(np.arange(s, s + len(a), dtype=np.int64), k))
    del Bt
    if cuda:
        torch.cuda.empty_cache()
    return np.concatenate(ia), np.concatenate(ib)


def rowwise_cos(A: np.ndarray, B: np.ndarray, ia: np.ndarray, ib: np.ndarray,
                chunk: int = 2_000_000) -> np.ndarray:
    out = np.zeros(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.einsum("ij,ij->i", A[ia[s:e]].astype(np.float32), B[ib[s:e]].astype(np.float32))
    return out
