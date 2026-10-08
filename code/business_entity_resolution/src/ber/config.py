"""Central configuration for the Business Entity Resolution pipeline.

Every tunable knob lives here so that a run is fully described by one object
(it is serialised next to the trained artefacts as ``config.json``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class BlockingConfig:
    # Per-view forward top-k (S1 -> target source) retrieved by sparse cosine.
    # At full scale (~1M entities per country block) a top-10 is crowded out by
    # look-alike names, which is why recall fell from 99.9% on a 20k slice to 98.1%.
    k_name_tok: int = 15
    k_name_char: int = 15
    k_name_skel: int = 15
    k_addr: int = 15
    k_name_addr: int = 15
    # Reverse top-k (target record -> S1) per view; protects records whose true
    # S1 is crowded out of the forward list by generic look-alikes.
    k_reverse: int = 5
    # Dense view (learned bi-encoder, see dense.py): exact top-k by embedding
    # cosine, forward per S1 entity and reverse per S2/S3 record. It has no
    # document-frequency cap, so it reaches records whose every name token is
    # too common for the sparse views. Used when DenseConfig.enabled.
    k_dense: int = 30
    k_dense_reverse: int = 10
    # Document-frequency caps (absolute counts inside one country block). Tokens
    # more frequent than this carry no blocking signal and blow up the join.
    max_df_name_tok: int = 3000
    max_df_name_char: int = 4000
    max_df_name_skel: int = 3000
    max_df_addr: int = 3000
    max_df_name_addr: int = 400
    # Final pruning: candidates kept per (S1 entity, target source) after the
    # learned blocking ranker. GT has at most 6 true matches per source; the
    # margin absorbs pruner mistakes (union -> candidate recall lost 0.16 pt at 12).
    keep_per_source: int = 16
    # Rows of S1 processed per sparse matmul chunk.
    chunk_rows: int = 1000


@dataclass
class ModelConfig:
    n_folds: int = 5
    seed: int = 2026
    # Fraction of training S1 entities held out (never trained on) to give an
    # honest validation F0.5, as the challenge statement recommends.
    holdout_frac: float = 0.15
    # Optional cap on S1 entities used to fit the GBDTs (None = all of the
    # non-holdout ones). Rows of the remaining entities are still scored.
    max_train_s1: int | None = None
    # S1 entities (disjoint from fit / hold-out when possible) used to train the
    # blocking pruner.
    ranker_s1: int = 150_000
    # Fraction of the non-hold-out S1 entities whose true pairs train the dense
    # bi-encoder (only when the dense view is enabled). These entities are kept
    # out of every other fit (pruner, GBDTs, cross-encoder): the encoder has
    # seen their pairs, so their embedding cosines would be optimistic - the
    # same stacking leak the cross-encoder avoids with its 2-fold scheme.
    dense_frac: float = 0.15
    lgb_stage1: dict = field(default_factory=lambda: dict(
        objective="binary", learning_rate=0.05, num_leaves=255,
        min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=10.0, max_bin=255, verbose=-1,
        num_threads=0,
    ))
    lgb_stage2: dict = field(default_factory=lambda: dict(
        objective="binary", learning_rate=0.03, num_leaves=127,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=10.0, max_bin=255, verbose=-1,
        num_threads=0,
    ))
    # Stage 3 (collective refinement): the competition / sibling-anchor context is
    # recomputed from the stage-2 probabilities - which already include the
    # cross-encoder - and a third grouped-OOF LightGBM is trained on it. Most
    # remaining errors are "which of several look-alike entities owns this
    # record", exactly what better competitor probabilities inform. The decision
    # step keeps stage 3 only if it beats stage 2 on out-of-fold data.
    stage3: bool = True
    lgb_stage3: dict = field(default_factory=lambda: dict(
        objective="binary", learning_rate=0.03, num_leaves=127,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=10.0, max_bin=255, verbose=-1,
        num_threads=0,
    ))
    num_boost_round: int = 5000
    early_stopping: int = 100
    # Number of top candidates (by stage-1 prob) used as sibling "anchors".
    n_anchors: int = 3


@dataclass
class CrossEncoderConfig:
    enabled: bool = True
    # mDeBERTa-v3-base: MIT licence, 278M params, multilingual tokenizer that
    # natively covers Devanagari/Telugu/Tamil/Bengali/... and French accents.
    model_name: str = "microsoft/mdeberta-v3-base"
    # Two "name | address" records in Indic scripts easily exceed 96 pieces.
    max_len: int = 128
    # Only pairs whose stage-1 probability falls in this band are re-scored;
    # outside it the GBDT is already (near-)certain. Remaining hold-out errors
    # include confident ones, so the band is wider than the original 0.02-0.98.
    band_low: float = 0.01
    band_high: float = 0.99
    max_train_pairs: int = 3_000_000
    epochs: int = 1
    lr: float = 2e-5
    batch_size: int = 128
    infer_batch_size: int = 512
    warmup_frac: float = 0.06
    fp16: bool = True
    # DataLoader worker processes for training (speed only, not part of any checkpoint key);
    # 0 where /dev/shm is too small for worker processes (some container setups).
    num_workers: int = 2


@dataclass
class DenseConfig:
    """Learned dense retrieval view (dense.py): a siamese sentence encoder
    fine-tuned with in-batch-negative contrastive loss on training true pairs."""
    enabled: bool = True
    # Apache-2.0, 118M parameters, 384-dim embeddings, multilingual vocabulary.
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_len: int = 64
    epochs: int = 2
    lr: float = 5e-5
    batch_size: int = 512          # pairs per batch -> 511 in-batch negatives per pair
    temperature: float = 0.05
    warmup_frac: float = 0.05
    max_train_pairs: int = 2_000_000
    fp16: bool = True
    infer_batch_size: int = 2048   # speed only (not part of any checkpoint key)


@dataclass
class DecisionConfig:
    beta: float = 0.5
    # Candidates per S1 entering the exact expected-F optimisation.
    max_items: int = 20
    # Grid searched on out-of-fold predictions.
    temperatures: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5)
    thresholds: tuple = tuple(round(0.30 + 0.02 * i, 2) for i in range(34))


def _build(cls, values: dict):
    """Dataclass from a dict, ignoring keys an older / newer version wrote."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in known})


@dataclass
class Config:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    n_jobs: int = -1
    # Minimum co-occurrence count for a learned token alias.
    alias_min_count: int = 25
    # A component seen at least this many times in a country's S1 records is
    # treated as a locality (city / district / state / region) component.
    geo_min_count: int = 30

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        cfg = cls()
        cfg.blocking = _build(BlockingConfig, raw["blocking"])
        cfg.model = _build(ModelConfig, raw["model"])
        cfg.cross_encoder = _build(CrossEncoderConfig, raw["cross_encoder"])
        # artefacts trained before the dense view existed have no "dense" section
        cfg.dense = _build(DenseConfig, raw["dense"]) if "dense" in raw else DenseConfig(enabled=False)
        dec = dict(raw["decision"])
        dec["temperatures"] = tuple(dec["temperatures"])
        dec["thresholds"] = tuple(dec["thresholds"])
        cfg.decision = _build(DecisionConfig, dec)
        for k in ("n_jobs", "alias_min_count", "geo_min_count"):
            setattr(cfg, k, raw[k])
        return cfg
