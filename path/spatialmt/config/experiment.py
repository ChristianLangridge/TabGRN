"""
spatialmt.config.experiment — ExperimentConfig and all sub-configs.

Single source of truth for every hyperparameter. Every training run serialises
its config to experiments/{run_id}/config.json, making runs fully reproducible
from that file alone.

Sub-configs
-----------
DataConfig      Data shape, feature selection, target definitions
ContextConfig   ICL context window layout
ModelConfig     Learning rates, warmup, head initialisation

Named presets
-------------
ExperimentConfig.debug_preset()              256 genes, debug tier
ExperimentConfig.rotation_finetune_dirichlet()  512 genes, standard tier, Dirichlet NLL
ExperimentConfig.full_finetune()             1024 genes, full tier, KL
ExperimentConfig.full_finetune_dirichlet()   1024 genes, full tier, Dirichlet NLL
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Sentinel exception
# ---------------------------------------------------------------------------

class ConfigurationError(ValueError):
    """Raised for invalid ExperimentConfig values or memory budget violations."""


# ---------------------------------------------------------------------------
# TabICLv2 pretrained architecture — single source of truth
#
# These values are read directly from the published pretrained checkpoint:
#   col_embedder  : embed_dim=128, nhead=8, 3 ISAB blocks, 128 inducing points
#   row_interactor: num_cls=4  →  d_model = 4 × 128 = 512
#   tf_icl        : d_model=512, nhead=8, 12 Encoder blocks
#
# DO NOT change these to match a different checkpoint without also retraining
# from scratch — the model's pretrained weights are tied to these exact shapes.
# ---------------------------------------------------------------------------

TABICL_V2_ARCH: dict = {
    "embed_dim":   128,
    "n_heads":     8,
    "num_cls":     4,    # d_model = num_cls × embed_dim = 512
    "col_num_inds": 128, # inducing points in ColEmbedding's Set Transformer
    "n_layers_col": 3,   # ISAB blocks in ColEmbedding (pretrained)
    "n_layers_row": 3,   # blocks in RowInteraction (not in checkpoint; kept symmetric)
    "n_layers_icl": 12,  # blocks in tf_icl Encoder (pretrained)
}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    max_genes: int
    test_timepoint: int = 11
    hardware_tier: str = "standard"
    n_cell_states: int = 8
    label_softening_temperature: float = 1.0
    log1p_transform: bool = True

    def __post_init__(self) -> None:
        if not self.log1p_transform:
            raise ValueError(
                "log1p_transform must be True — raw counts are not supported. "
                "Apply sc.pp.normalize_total() + sc.pp.log1p() before constructing DataConfig."
            )


@dataclass
class ContextConfig:
    n_bins: int = 6
    cells_per_bin: int = 5
    max_context_cells: int = 50
    allow_replacement: bool = True
    n_composition_anchors: int = 0
    n_pseudotime_anchors: int = 0
    n_pseudotime_bins: int = 5

    def __post_init__(self) -> None:
        total = (
            self.n_bins * self.cells_per_bin
            + self.n_pseudotime_anchors
            + self.n_composition_anchors
        )
        if total > self.max_context_cells:
            raise ValueError(
                f"n_bins ({self.n_bins}) × cells_per_bin ({self.cells_per_bin}) "
                f"+ n_pseudotime_anchors ({self.n_pseudotime_anchors}) "
                f"+ n_composition_anchors ({self.n_composition_anchors}) = {total} "
                f"exceeds max_context_cells ({self.max_context_cells}). "
                "Increase max_context_cells or reduce context parameters."
            )


@dataclass
class ModelConfig:
    # ------------------------------------------------------------------
    # TabICLv2 backbone architecture
    # These must match the pretrained checkpoint exactly.
    # Defaults are read from TABICL_V2_ARCH — do not override per-preset.
    # ------------------------------------------------------------------
    embed_dim: int   = TABICL_V2_ARCH["embed_dim"]
    n_heads: int     = TABICL_V2_ARCH["n_heads"]
    num_cls: int     = TABICL_V2_ARCH["num_cls"]
    col_num_inds: int = TABICL_V2_ARCH["col_num_inds"]
    n_layers_col: int = TABICL_V2_ARCH["n_layers_col"]
    n_layers_row: int = TABICL_V2_ARCH["n_layers_row"]
    n_layers_icl: int = TABICL_V2_ARCH["n_layers_icl"]

    # Path to a pretrained TabICLv2 .ckpt file.
    # None = random initialisation (scratch ablation only).
    backbone_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    lr_col: float = 1e-6       # column attention — gene × gene
    lr_row: float = 1e-4       # row attention — feature → cell repr
    lr_icl: float = 5e-5       # ICL attention — cell × cell
    lr_emb: float = 1e-3       # column embeddings (always re-initialised)
    lr_head: float = 1e-3      # pseudotime head + composition head
    warmup_col_steps: int = 500   # steps before column attention is unfrozen
    warmup_icl_steps: int = 100   # steps before ICL attention is unfrozen
    output_head_init_bias: float = 0.5   # PseudotimeHead bias → sigmoid(0.5)≈0.62
    output_head_init_std: float = 0.01   # near-zero weight init for both heads
    bio_plausibility_passed: Optional[bool] = None  # populated post-training
    supervised_batch_size: int = 64      # mini-batch size for SupervisedTrainer
    # "kl" = KL-divergence CompositionHead (rotation_001)
    # "dirichlet" = Dirichlet NLL DirichletCompositionHead (rotation_002)
    composition_loss_type: str = "kl"


# ---------------------------------------------------------------------------
# Hardware tier defaults (used by named presets)
# ---------------------------------------------------------------------------

HARDWARE_TIERS: dict[str, dict] = {
    # max_context_cells = 5 active days × cells_per_bin (D11 excluded at runtime).
    "debug":    {"max_genes": 256,  "max_context_cells": 50},
    "standard": {"max_genes": 512,  "max_context_cells": 50},
    "full":     {"max_genes": 2000, "max_context_cells": 50},
}


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    run_id: str
    data: DataConfig
    context: ContextConfig
    model: ModelConfig

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write config to experiments/{run_id}/config.json under PROJECT_ROOT."""
        from spatialmt.config.paths import PROJECT_ROOT
        run_dir = PROJECT_ROOT / "experiments" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        with open(config_path, "w") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    # ------------------------------------------------------------------
    # Hash — SHA-256 of serialised hyperparameters (run_id excluded)
    # ------------------------------------------------------------------

    @property
    def config_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("run_id", None)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def _pretrained_model_config(cls) -> "ModelConfig":
        """ModelConfig with backbone_path set to the canonical TabICLv2 checkpoint."""
        from spatialmt.config.paths import Paths
        return ModelConfig(backbone_path=str(Paths.tabicl_checkpoint))

    @classmethod
    def debug_preset(cls, run_id: str = "debug") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["debug"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="debug",
            ),
            context=ContextConfig(
                n_bins=6,
                cells_per_bin=5,
                max_context_cells=tier["max_context_cells"],
            ),
            model=cls._pretrained_model_config(),
        )

    @classmethod
    def rotation_finetune_dirichlet(cls, run_id: str = "rotation_002") -> "ExperimentConfig":
        """Dirichlet NLL composition head, standard hardware tier (512 genes)."""
        tier = HARDWARE_TIERS["standard"]
        model_cfg = cls._pretrained_model_config()
        model_cfg.composition_loss_type = "dirichlet"
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="standard",
            ),
            context=ContextConfig(
                n_bins=6,
                cells_per_bin=5,
                max_context_cells=tier["max_context_cells"],
            ),
            model=model_cfg,
        )

    @classmethod
    def full_finetune(cls, run_id: str = "rotation_001") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["full"]
        model_cfg = cls._pretrained_model_config()
        model_cfg.supervised_batch_size = 8  # 1024 genes × 64 batch = 65k col ops → OOM on A100
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="full",
            ),
            context=ContextConfig(
                cells_per_bin=10,
                max_context_cells=tier["max_context_cells"],
            ),
            model=model_cfg,
        )

    @classmethod
    def full_finetune_dirichlet(cls, run_id: str = "rotation_002") -> "ExperimentConfig":
        """Dirichlet NLL composition head, full hardware tier (1024 genes)."""
        tier = HARDWARE_TIERS["full"]
        model_cfg = cls._pretrained_model_config()
        model_cfg.composition_loss_type = "dirichlet"
        model_cfg.supervised_batch_size = 4  # 2000 genes × 4 batch = 8k col ops, safe on A100
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="full",
            ),
            context=ContextConfig(
                cells_per_bin=10,
                max_context_cells=tier["max_context_cells"],
            ),
            model=model_cfg,
        )
