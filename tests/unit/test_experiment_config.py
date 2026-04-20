"""
Unit tests for spatialmt.config.experiment.ExperimentConfig and sub-configs.

Tests are organised into:
  - Sub-config validation (DataConfig, ContextConfig raise on invalid inputs)
  - Preset values (debug, rotation_finetune, full_finetune field assertions)
  - Serialisation (JSON round-trip, save/load with tmp_path)
  - Hash (determinism, sensitivity to hyperparameter changes)
  - ModelConfig parameter group layout (6 LR groups present)
"""
import dataclasses
import json

import pytest

from spatialmt.config.experiment import (
    BenchmarkConfig,
    ConfigurationError,
    ContextConfig,
    DataConfig,
    ExperimentConfig,
    ExplainabilityConfig,
    ModelConfig,
    PerturbationConfig,
)


# ---------------------------------------------------------------------------
# DataConfig validation
# ---------------------------------------------------------------------------

def test_data_config_log1p_transform_false_raises():
    with pytest.raises(ValueError, match="log1p_transform"):
        DataConfig(max_genes=512, log1p_transform=False)


def test_data_config_valid_constructs():
    cfg = DataConfig(max_genes=512)
    assert cfg.log1p_transform is True
    assert cfg.n_cell_states == 8
    assert cfg.test_timepoint == 11


def test_data_config_label_softening_temperature_default():
    cfg = DataConfig(max_genes=128)
    assert cfg.label_softening_temperature == 1.0


# ---------------------------------------------------------------------------
# ContextConfig validation
# ---------------------------------------------------------------------------

def test_context_config_bins_exceed_max_context_raises():
    # 6 bins × 10 cells/bin = 60 > max_context_cells=30
    with pytest.raises(ValueError):
        ContextConfig(n_bins=6, cells_per_bin=10, max_context_cells=30)


def test_context_config_exact_capacity_is_valid():
    # 5 × 5 = 25 == max_context_cells=25 — should not raise
    cfg = ContextConfig(n_bins=5, cells_per_bin=5, max_context_cells=25)
    assert cfg.n_bins == 5


def test_context_config_defaults():
    cfg = ContextConfig()
    assert cfg.n_bins == 6
    assert cfg.cells_per_bin == 5
    assert cfg.max_context_cells == 50
    assert cfg.allow_replacement is True


# ---------------------------------------------------------------------------
# Preset values
# ---------------------------------------------------------------------------

def test_debug_preset_max_genes():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.data.max_genes == 128


def test_debug_preset_hardware_tier():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.data.hardware_tier == "debug"


def test_debug_preset_cells_per_bin():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.context.cells_per_bin == 2


def test_rotation_finetune_max_genes():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.data.max_genes == 512


def test_rotation_finetune_hardware_tier():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.data.hardware_tier == "standard"


def test_rotation_finetune_n_bins():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.context.n_bins == 6


def test_rotation_finetune_n_cell_states():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.data.n_cell_states == 8


def test_rotation_finetune_test_timepoint():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.data.test_timepoint == 11


def test_full_finetune_max_genes():
    cfg = ExperimentConfig.full_finetune()
    assert cfg.data.max_genes == 1024


def test_full_finetune_hardware_tier():
    cfg = ExperimentConfig.full_finetune()
    assert cfg.data.hardware_tier == "full"


# ---------------------------------------------------------------------------
# ModelConfig — six parameter groups
# ---------------------------------------------------------------------------

def test_model_config_has_six_lr_fields():
    cfg = ExperimentConfig.rotation_finetune()
    # Six groups from TDD §3.5: column_attention, row_attention, icl_attention,
    # column_embeddings, pseudotime_head, composition_head
    # Represented by: lr_col, lr_row, lr_icl, lr_emb, lr_head (heads share lr_head)
    assert cfg.model.lr_col == pytest.approx(1e-5)
    assert cfg.model.lr_row == pytest.approx(1e-4)
    assert cfg.model.lr_icl == pytest.approx(5e-5)
    assert cfg.model.lr_emb == pytest.approx(1e-3)
    assert cfg.model.lr_head == pytest.approx(1e-3)


def test_model_config_warmup_steps():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.model.warmup_col_steps == 500
    assert cfg.model.warmup_icl_steps == 100


def test_model_config_head_init():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.model.output_head_init_bias == pytest.approx(0.5)
    assert cfg.model.output_head_init_std == pytest.approx(0.01)


def test_model_config_bio_plausibility_none_by_default():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.model.bio_plausibility_passed is None


# ---------------------------------------------------------------------------
# ExplainabilityConfig
# ---------------------------------------------------------------------------

def test_explainability_config_defaults():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.explainability.shap_background_size == 100
    assert cfg.explainability.shap_background_seed == 42
    assert "SOX2" in cfg.explainability.bio_plausibility_required


# ---------------------------------------------------------------------------
# PerturbationConfig
# ---------------------------------------------------------------------------

def test_perturbation_config_default_ablation():
    """Default ablation target is a cell-autonomous WLS KO in the query cell."""
    cfg = ExperimentConfig.rotation_finetune()
    assert len(cfg.perturbation.ablations) == 1
    ablation = cfg.perturbation.ablations[0]
    assert ablation.gene == "WLS"
    assert ablation.zero_in_query is True
    assert ablation.zero_in_context_states is None


def test_perturbation_config_thresholds():
    cfg = ExperimentConfig.rotation_finetune()
    assert cfg.perturbation.pseudotime_delta_threshold == pytest.approx(-0.05)
    assert cfg.perturbation.composition_shift_threshold == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_config_is_json_serialisable():
    cfg = ExperimentConfig.debug_preset(run_id="ser_test")
    d = dataclasses.asdict(cfg)
    s = json.dumps(d)
    assert "run_id" in s
    assert "max_genes" in s


def test_config_round_trips_through_json():
    cfg = ExperimentConfig.debug_preset(run_id="rt_test")
    d = dataclasses.asdict(cfg)
    s = json.dumps(d, sort_keys=True)
    d2 = json.loads(s)
    assert d2["data"]["max_genes"] == 128
    assert d2["context"]["n_bins"] == 6


def test_save_writes_config_json(tmp_path, monkeypatch):
    import spatialmt.config.paths as paths_module
    monkeypatch.setattr(paths_module, "PROJECT_ROOT", tmp_path)
    cfg = ExperimentConfig.debug_preset(run_id="save_test")
    cfg.save()
    config_file = tmp_path / "experiments" / "save_test" / "config.json"
    assert config_file.exists()
    d = json.loads(config_file.read_text())
    assert d["run_id"] == "save_test"
    assert d["data"]["max_genes"] == 128


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------

def test_config_hash_is_deterministic():
    cfg1 = ExperimentConfig.debug_preset(run_id="hash_a")
    cfg2 = ExperimentConfig.debug_preset(run_id="hash_a")
    assert cfg1.config_hash == cfg2.config_hash


def test_config_hash_run_id_independent():
    """Same hyperparameters, different run_id → same hash (run_id is not a hyperparameter)."""
    cfg1 = ExperimentConfig.debug_preset(run_id="run_1")
    cfg2 = ExperimentConfig.debug_preset(run_id="run_2")
    assert cfg1.config_hash == cfg2.config_hash


def test_config_hash_changes_with_different_lr():
    cfg1 = ExperimentConfig.rotation_finetune(run_id="lr_a")
    model_modified = ModelConfig(lr_col=9e-5)  # non-default
    cfg2 = ExperimentConfig(
        run_id="lr_a",
        data=cfg1.data,
        context=cfg1.context,
        model=model_modified,
        explainability=cfg1.explainability,
        perturbation=cfg1.perturbation,
        benchmark=cfg1.benchmark,
    )
    assert cfg1.config_hash != cfg2.config_hash


def test_config_hash_changes_with_different_max_genes():
    cfg1 = ExperimentConfig.debug_preset()
    data_modified = DataConfig(max_genes=256, hardware_tier="debug")
    cfg2 = ExperimentConfig(
        run_id=cfg1.run_id,
        data=data_modified,
        context=cfg1.context,
        model=cfg1.model,
        explainability=cfg1.explainability,
        perturbation=cfg1.perturbation,
        benchmark=cfg1.benchmark,
    )
    assert cfg1.config_hash != cfg2.config_hash


def test_config_hash_is_64_char_hex():
    cfg = ExperimentConfig.debug_preset()
    h = cfg.config_hash
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------

def test_rotation_baselines_preset_baseline_list():
    cfg = ExperimentConfig.rotation_baselines()
    assert "mean" in cfg.benchmark.baselines
    assert "ridge_pca" in cfg.benchmark.baselines
    assert "xgboost_regressor" in cfg.benchmark.baselines
