"""
spatialmt.config.experiment
============================
ExperimentConfig — single source of truth for all hyperparameters.

ROTATION SCOPE fields are active and used in training now.
FULL PROJECT fields are present but gated — they are validated
at construction time and will raise if used before their phase gate.

Phase gates
-----------
ROTATION (now → July 3rd):
    DataConfig.soft_label fields      → present, not used in training
    ModelConfig.n_cell_states         → present, CompositionHead not wired
    ModelConfig.loss_balancing        → present, NormalisedDualLoss not active
    BenchmarkConfig.composition_*     → present, not computed

FULL PROJECT (Phase 5A, July 3rd onwards):
    Call model.enable_composition_head() to wire in CompositionHead
    Switch Trainer to NormalisedDualLoss
    All composition metrics become active
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from spatialmt.config.paths import PROJECT_ROOT, Paths


# ---------------------------------------------------------------------------
# Hardware tiers
# ---------------------------------------------------------------------------

HARDWARE_TIERS: dict[str, dict] = {
    "debug":    {"max_genes": 128,  "batch_size": 2,  "max_context_cells": 10},
    "standard": {"max_genes": 512,  "batch_size": 16, "max_context_cells": 50},
    "full":     {"max_genes": 1024, "batch_size": 32, "max_context_cells": 100},
}


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Feature selection, hold-out split, and label softening."""

    # Feature space
    max_genes: int = 512
    hvg_flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = "seurat_v3"
    log1p_transform: bool = True  # Must be True — see dataset.py

    # Hold-out split
    test_timepoint: int = 11
    val_fraction: float = 0.2
    random_seed: int = 42

    # Hardware tier
    hardware_tier: Literal["debug", "standard", "full", "custom"] = "custom"

    # Data paths
    tpm_dir: str = str(Paths.processed_tpm.parent)
    external_validation_path: str = ""

    # ------------------------------------------------------------------
    # Soft label configuration (FULL PROJECT — Phase 5A)
    # Present now so ProcessedDataset schema is stable.
    # Not used in training until model.enable_composition_head() is called.
    # ------------------------------------------------------------------
    n_cell_states: int = 5
    """K=5: neuroectodermal, neural tube, prosencephalic, telencephalic, early neuron"""

    label_softening_method: Literal["distance_centroid", "fuzzy_cmeans", "gmm"] = (
        "distance_centroid"
    )
    """Method A: distance-to-centroid softmax with temperature. Recommended."""

    label_softening_temperature: float = 1.0
    """Controls softness. τ→0 recovers hard labels. τ=1.0 moderate softness.
    Tunable hyperparameter — stored in config for reproducibility."""

    cell_state_names: list[str] = field(default_factory=lambda: [
        "neuroectodermal_progenitor",
        "neural_tube_neuroepithelial",
        "prosencephalic_progenitor",
        "telencephalic_progenitor",
        "early_neuron",
    ])

    def __post_init__(self) -> None:
        if self.hardware_tier != "custom":
            tier = HARDWARE_TIERS[self.hardware_tier]
            self.max_genes = tier["max_genes"]
        if not self.log1p_transform:
            raise ValueError(
                "log1p_transform must be True. Raw TPM dominates attention."
            )
        if len(self.cell_state_names) != self.n_cell_states:
            raise ValueError(
                f"cell_state_names length {len(self.cell_state_names)} "
                f"!= n_cell_states {self.n_cell_states}"
            )


# ---------------------------------------------------------------------------
# ContextConfig
# ---------------------------------------------------------------------------

@dataclass
class ContextConfig:
    """ICL context window: 5 pseudotime bins anchored to collection days."""

    n_bins: int = 5
    cells_per_bin: int = 5
    max_context_cells: int = 50
    allow_replacement: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        total = self.n_bins * self.cells_per_bin
        if total > self.max_context_cells:
            raise ValueError(
                f"n_bins ({self.n_bins}) × cells_per_bin ({self.cells_per_bin}) "
                f"= {total} exceeds max_context_cells ({self.max_context_cells})."
            )


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """TabICLv2 architecture and fine-tuning strategy."""

    # Architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # Cell states (used by CompositionHead — Phase 5A)
    n_cell_states: int = 5

    # Fine-tuning strategy
    finetune_strategy: Literal[
        "full_differential",
        "freeze_column",
        "scratch",
    ] = "full_differential"

    # Differential learning rates — 5 component groups
    lr_col: float = 1e-5   # Column attention (TabICLv2 stage 1)
    lr_row: float = 1e-4   # Row attention    (TabICLv2 stage 2)
    lr_icl: float = 5e-5   # ICL attention    (TabICLv2 stage 3)
    lr_emb: float = 1e-3   # Column embeddings (always from scratch)
    lr_head: float = 1e-3  # Output head(s)   (always from scratch)

    # Warmup — column and ICL frozen for these many steps
    warmup_col_steps: int = 500
    warmup_icl_steps: int = 100

    # Regression head initialisation
    output_head_init_bias: float = 0.5   # Trajectory midpoint prior
    output_head_init_std: float = 0.01   # Near-zero → stable initial gradients

    # ------------------------------------------------------------------
    # Loss balancing (FULL PROJECT — Phase 5A)
    # Present now so config serialisation is stable.
    # NormalisedDualLoss is not active until enable_composition_head().
    # ------------------------------------------------------------------
    loss_balancing: Literal["normalised", "fixed_lambda", "manual"] = "normalised"
    """Strategy A: divide each loss by its initial value. Automatic."""

    lambda_pseudotime: float = 1.0
    lambda_composition: float = 1.0
    """Used only when loss_balancing='manual'. Ignored for 'normalised'."""

    # Fallback state — set programmatically after plausibility gate
    bio_plausibility_passed: bool | None = None


# ---------------------------------------------------------------------------
# ExplainabilityConfig
# ---------------------------------------------------------------------------

@dataclass
class ExplainabilityConfig:
    """GeneScorer and ExplainabilityReport configuration."""

    # SHAP background
    shap_background_size: int = 100
    shap_background_seed: int = 42

    # Disagreement taxonomy thresholds
    attention_high_threshold: float = 0.01
    shap_high_threshold: float = 0.0

    # Biological plausibility gate
    bio_plausibility_genes: list[str] = field(default_factory=lambda: [
        "SOX2", "POU5F1", "WLS", "YAP1", "SIX3", "LHX2",
    ])
    bio_plausibility_top_k: int = 20
    bio_plausibility_required: list[str] = field(
        default_factory=lambda: ["SOX2"]
    )
    """SOX2 MUST be in top-k attention. Failure triggers fallback strategy."""


# ---------------------------------------------------------------------------
# PerturbationConfig
# ---------------------------------------------------------------------------

@dataclass
class PerturbationConfig:
    """In-silico perturbation experiments."""

    name: str = "wls_knockout"
    perturbation_mask: dict[str, float] = field(
        default_factory=lambda: {"WLS": 0.0}
    )

    # Signal 1 — pseudotime shift (ROTATION SCOPE — active)
    pseudotime_delta_threshold: float = -0.05

    # Signal 2 — attention drop (ROTATION SCOPE — active)
    attention_drop_fraction: float = 0.1

    # Signal 3 — composition shift (FULL PROJECT — Phase 5B)
    # Telencephalic index in the K=5 state vector
    telencephalic_state_idx: int = 3
    non_telencephalic_state_idx: int = 2
    composition_shift_threshold: float = 0.05
    """predict_comp(ko)[telencephalic] - predict_comp(baseline)[telencephalic]
    must exceed this. Active only after enable_composition_head()."""


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Baseline suite and metric logging."""

    # ROTATION SCOPE: tabicl_finetune + xgboost only
    # FULL PROJECT:   all five presets
    baselines: list[str] = field(default_factory=lambda: [
        "tabicl_finetune",
        "xgboost",
        # Phase 6 additions — uncomment July onwards:
        # "tabicl_scratch",
        # "tabicl_no_icl",
        # "tabpfn_v2",
    ])

    xgboost_n_estimators: int = 500
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.05
    xgboost_random_seed: int = 42

    # ROTATION SCOPE metrics (active now)
    metrics: list[str] = field(default_factory=lambda: [
        "mae_day11",
        "attention_entropy",
        # Phase 5A additions — uncomment July onwards:
        # "composition_nll",
        # "top20_bio_overlap",
        # "telencephalic_affinity_accuracy",
    ])

    mae_regression_threshold: float = 1.10


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Complete specification for one TabGRN-ICL experiment run.

    Serialised to experiments/{run_id}/config.json at startup.

    ROTATION SCOPE PRESETS
    ----------------------
    ExperimentConfig.rotation_finetune()  — primary rotation model
    ExperimentConfig.rotation_xgboost()  — XGBoost baseline
    ExperimentConfig.debug_preset()      — laptop CPU, 128 genes

    FULL PROJECT PRESETS (July onwards)
    ------------------------------------
    ExperimentConfig.full_finetune()     — dual-head, 1024 genes, A100
    ExperimentConfig.scratch_preset()    — ablation: no pre-training
    ExperimentConfig.no_icl_preset()     — ablation: no ICL mechanism
    """

    run_id: str = "run_001"
    description: str = ""

    data: DataConfig = field(default_factory=DataConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    perturbations: list[PerturbationConfig] = field(
        default_factory=lambda: [PerturbationConfig()]
    )
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    config_hash: str = ""
    output_dir: str = ""

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(PROJECT_ROOT / "experiments" / self.run_id)

    def save(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["config_hash"] = self._hash(data)
        path = out / "config.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        data.pop("config_hash", None)
        data["data"] = DataConfig(**data["data"])
        data["context"] = ContextConfig(**data["context"])
        data["model"] = ModelConfig(**data["model"])
        data["explainability"] = ExplainabilityConfig(**data["explainability"])
        data["perturbations"] = [PerturbationConfig(**p) for p in data["perturbations"]]
        data["benchmark"] = BenchmarkConfig(**data["benchmark"])
        return cls(**data)

    @staticmethod
    def _hash(data: dict) -> str:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:12]

    # ------------------------------------------------------------------
    # ROTATION SCOPE presets
    # ------------------------------------------------------------------

    @classmethod
    def debug_preset(cls, run_id: str = "debug") -> "ExperimentConfig":
        """128 genes, 2 cells/bin. Runs on any laptop CPU."""
        return cls(
            run_id=run_id,
            description="Debug: 128 genes, smoke test on CPU",
            data=DataConfig(hardware_tier="debug"),
            context=ContextConfig(cells_per_bin=2, max_context_cells=10),
            model=ModelConfig(n_layers=2, d_model=32, n_heads=2, warmup_col_steps=0),
        )

    @classmethod
    def rotation_finetune(cls, run_id: str = "rotation_finetune") -> "ExperimentConfig":
        """
        Primary rotation model.
        TabICLv2 fine-tuned, pseudotime head only, 512 genes, V100.
        """
        return cls(
            run_id=run_id,
            description=(
                "Rotation primary: TabICLv2 fine-tune, pseudotime-only, "
                "standard tier (512 genes, V100)"
            ),
            data=DataConfig(hardware_tier="standard"),
            context=ContextConfig(cells_per_bin=5, max_context_cells=50),
            model=ModelConfig(finetune_strategy="full_differential"),
            benchmark=BenchmarkConfig(baselines=["tabicl_finetune", "xgboost"]),
        )

    @classmethod
    def rotation_xgboost(cls, run_id: str = "rotation_xgboost") -> "ExperimentConfig":
        """XGBoost baseline. Same HVG feature set as rotation_finetune."""
        return cls(
            run_id=run_id,
            description="Rotation baseline: XGBoost on HVG expression",
            data=DataConfig(hardware_tier="standard"),
            benchmark=BenchmarkConfig(baselines=["xgboost"]),
        )

    # ------------------------------------------------------------------
    # FULL PROJECT presets (July onwards)
    # ------------------------------------------------------------------

    @classmethod
    def full_finetune(cls, run_id: str = "full_finetune") -> "ExperimentConfig":
        """
        Full dual-head model. Phase 5A onwards.
        Requires: model.enable_composition_head() before training.
        """
        return cls(
            run_id=run_id,
            description=(
                "Full project: dual-head (pseudotime + composition), "
                "1024 genes, A100"
            ),
            data=DataConfig(hardware_tier="full"),
            context=ContextConfig(cells_per_bin=10, max_context_cells=100),
            model=ModelConfig(
                finetune_strategy="full_differential",
                loss_balancing="normalised",
            ),
            benchmark=BenchmarkConfig(baselines=[
                "tabicl_finetune", "xgboost", "tabicl_scratch",
                "tabicl_no_icl", "tabpfn_v2",
            ]),
        )

    @classmethod
    def scratch_preset(cls, run_id: str = "scratch") -> "ExperimentConfig":
        """Ablation: TabICLv2 architecture, no pre-trained weights."""
        return cls(
            run_id=run_id,
            description="Ablation: scratch training — quantifies pre-training benefit",
            model=ModelConfig(
                finetune_strategy="scratch",
                warmup_col_steps=0,
                warmup_icl_steps=0,
            ),
        )

    @classmethod
    def no_icl_preset(cls, run_id: str = "no_icl") -> "ExperimentConfig":
        """Ablation: single cell input, no context window."""
        return cls(
            run_id=run_id,
            description="Ablation: no ICL — quantifies context window benefit",
            context=ContextConfig(cells_per_bin=0, max_context_cells=1),
        )