"""
spatialmt.config.experiment — ExperimentConfig and all sub-configs.

Single source of truth for every hyperparameter. Every training run serialises
its config to experiments/{run_id}/config.json, making runs fully reproducible
from that file alone.

Sub-configs
-----------
DataConfig          Data shape, feature selection, target definitions
ContextConfig       ICL context window layout
ModelConfig         Learning rates, warmup, head initialisation
ExplainabilityConfig  SHAP background, biological plausibility gate
PerturbationConfig  In-silico knockout definitions and thresholds
BenchmarkConfig     Baseline ladder for dual-axis justification

Named presets
-------------
ExperimentConfig.debug_preset()         128 genes, CPU, 2 cells/bin
ExperimentConfig.rotation_finetune()    512 genes, V100, dual-head
ExperimentConfig.rotation_baselines()   Baseline ladder only
ExperimentConfig.full_finetune()        1024 genes, A100, extended training
ExperimentConfig.scratch_preset()       No pretrained weights [Phase 6]
ExperimentConfig.no_icl_preset()        Single cell, no context [Phase 6]
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
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

    def __post_init__(self) -> None:
        total = self.n_bins * self.cells_per_bin
        if total > self.max_context_cells:
            raise ValueError(
                f"n_bins ({self.n_bins}) × cells_per_bin ({self.cells_per_bin}) = {total} "
                f"exceeds max_context_cells ({self.max_context_cells}). "
                "Increase max_context_cells or reduce n_bins/cells_per_bin."
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
    # Set by every preset that uses pretrained weights (all except scratch_preset).
    # None = random initialisation (scratch ablation only).
    backbone_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    lr_col: float = 1e-5       # column attention — gene × gene
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


@dataclass
class ExplainabilityConfig:
    # --- SHAP ---
    # GradientSHAP: model-aware, uses gradients — fast cross-validation of
    # AttentionScorer findings at the node (gene) level.
    shap_background_size: int = 100
    shap_background_seed: int = 42

    # --- Biological plausibility gate ---
    # Column attention scores for these genes must rank in the top quartile
    # of their row after training, or bio_plausibility_passed is set False.
    bio_plausibility_required: tuple[str, ...] = ("SOX2",)

    # --- Integrated Hessians ---
    # Pairwise gene interaction matrix (n_genes, n_genes), task-coupled and
    # developmental-state-specific. Run separately per prediction head and
    # per cell state to produce multiple IH matrices. Captures higher-order
    # feature interactions beyond what attention weights expose directly.
    ih_n_steps: int = 50
    ih_baseline_strategy: str = "day5_mean"   # "zero" | "dataset_mean" | "day5_mean"
    ih_top_k_edges: int = 500                 # threshold before biological validation


@dataclass
class AblationTarget:
    """Specification for a single in-silico gene ablation experiment.

    Ablation variants (composable)
    ──────────────────────────────
    zero_in_query=True only
        Cell-autonomous contribution — query cell KO in wild-type neighbourhood.

    zero_in_context_states=None (all states)
        Population-wide KO approximation — comparable to real KO dataset.
        Concordance with wet-lab KO → gene effect is cell-autonomous.
        Discordance → effect is non-cell-autonomous / trajectory-level.

    zero_in_context_states=["neural_progenitor"]
        State-specific non-cell-autonomous contribution — does this gene in
        progenitor anchors inform downstream cell predictions specifically.

    zero_in_context_pseudotime_below / _above
        Temporal window ablation — is the gene's contextual effect gated to
        early or late trajectory positions. Uniquely enabled by ICL design.

    Combining all fields
        Full ablation within a lineage — approximates lineage-restricted KO.
    """
    gene: str
    zero_in_query: bool = True
    zero_in_context_states: list[str] | None = None   # None = all context states
    zero_in_context_pseudotime_below: float | None = None
    zero_in_context_pseudotime_above: float | None = None


@dataclass
class PerturbationConfig:
    # List of ablation experiments to run post-training.
    # Default: single cell-autonomous WLS ablation.
    ablations: list[AblationTarget] = field(default_factory=lambda: [
        AblationTarget(gene="WLS", zero_in_query=True)
    ])
    # Thresholds for calling a significant perturbation effect
    pseudotime_delta_threshold: float = -0.05
    attention_drop_fraction: float = 0.1
    composition_shift_threshold: float = 0.05


@dataclass
class BenchmarkConfig:
    baselines: tuple[str, ...] = (
        "mean",
        "ridge_pca",
        "xgboost_regressor",
        "tabicl_finetune",
    )


# ---------------------------------------------------------------------------
# Hardware tier defaults (used by named presets)
# ---------------------------------------------------------------------------

HARDWARE_TIERS: dict[str, dict] = {
    # max_context_cells = n_bins × cells_per_bin (theoretical max; day-11 bin is excluded
    # at runtime, so actual cells fed = (n_bins-1) × cells_per_bin).
    "debug":    {"max_genes": 256,  "max_context_cells": 30},
    "standard": {"max_genes": 512,  "max_context_cells": 50},
    "full":     {"max_genes": 1024, "max_context_cells": 100},
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
    explainability: ExplainabilityConfig
    perturbation: PerturbationConfig
    benchmark: BenchmarkConfig

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
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def rotation_finetune(cls, run_id: str = "rotation_001") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["standard"]
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
            model=cls._pretrained_model_config(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def rotation_finetune_dirichlet(cls, run_id: str = "rotation_002") -> "ExperimentConfig":
        """Dirichlet NLL variant of rotation_finetune.

        Identical hardware tier and context to rotation_001, but uses
        DirichletCompositionHead + DirichletDualHeadLoss instead of the
        KL-divergence composition head. Produces per-class uncertainty estimates
        (α₀, Var(p̂_k)) alongside point predictions.
        """
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
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def rotation_baselines(cls, run_id: str = "rotation_baselines") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["standard"]
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
            model=cls._pretrained_model_config(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(
                baselines=("mean", "ridge_pca", "xgboost_regressor"),
            ),
        )

    @classmethod
    def full_finetune(cls, run_id: str = "rotation_001") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["full"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="full",
            ),
            context=ContextConfig(
                n_bins=10,
                cells_per_bin=10,
                max_context_cells=tier["max_context_cells"],
            ),
            model=cls._pretrained_model_config(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def full_finetune_dirichlet(cls, run_id: str = "rotation_002") -> "ExperimentConfig":
        """Dirichlet NLL variant of full_finetune."""
        tier = HARDWARE_TIERS["full"]
        model_cfg = cls._pretrained_model_config()
        model_cfg.composition_loss_type = "dirichlet"
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="full",
            ),
            context=ContextConfig(
                n_bins=10,
                cells_per_bin=10,
                max_context_cells=tier["max_context_cells"],
            ),
            model=model_cfg,
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def scratch_preset(cls, run_id: str = "scratch") -> "ExperimentConfig":
        """No pretrained weights — random init ablation [Phase 6]."""
        tier = HARDWARE_TIERS["standard"]
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
            model=ModelConfig(backbone_path=None),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def no_icl_preset(cls, run_id: str = "no_icl") -> "ExperimentConfig":
        """Single cell input — no in-context learning [Phase 6]."""
        tier = HARDWARE_TIERS["standard"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="standard",
            ),
            context=ContextConfig(
                n_bins=1,
                cells_per_bin=1,
                max_context_cells=tier["max_context_cells"],
            ),
            model=cls._pretrained_model_config(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )
