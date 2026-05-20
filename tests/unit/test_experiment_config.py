"""
Unit tests for spatialmt.config.experiment.ExperimentConfig and sub-configs.

Tests are organised into:
  - Sub-config validation (DataConfig, ContextConfig raise on invalid inputs)
  - Preset values (debug, full_finetune field assertions)
  - Serialisation (JSON round-trip, save/load with tmp_path)
  - Hash (determinism, sensitivity to hyperparameter changes)
  - ModelConfig parameter group layout (5 LR groups present)
"""
import dataclasses
import json

import pytest

from spatialmt.config.experiment import (
    ConfigurationError,
    ContextConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
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
    # 5 active days × 11 cells/bin = 55 > max_context_cells=50
    with pytest.raises(ValueError):
        ContextConfig(cells_per_bin=11, max_context_cells=50)


def test_context_config_exact_capacity_is_valid():
    # 5 × 10 = 50 == max_context_cells=50 — should not raise
    cfg = ContextConfig(cells_per_bin=10, max_context_cells=50)
    assert cfg.cells_per_bin == 10


def test_context_config_defaults():
    cfg = ContextConfig()
    assert cfg.cells_per_bin == 5
    assert cfg.max_context_cells == 50
    assert cfg.allow_replacement is True


# ---------------------------------------------------------------------------
# Preset values
# ---------------------------------------------------------------------------

def test_debug_preset_max_genes():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.data.max_genes == 256


def test_debug_preset_hardware_tier():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.data.hardware_tier == "debug"


def test_debug_preset_cells_per_bin():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.context.cells_per_bin == 5


def test_full_finetune_max_genes():
    cfg = ExperimentConfig.full_finetune()
    assert cfg.data.max_genes == 2000


def test_full_finetune_hardware_tier():
    cfg = ExperimentConfig.full_finetune()
    assert cfg.data.hardware_tier == "full"



# ---------------------------------------------------------------------------
# ModelConfig — parameter groups and warmup schedule
# ---------------------------------------------------------------------------

def test_model_config_has_five_lr_fields():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.model.lr_col  == pytest.approx(1e-6)
    assert cfg.model.lr_row  == pytest.approx(1e-4)
    assert cfg.model.lr_icl  == pytest.approx(5e-5)
    assert cfg.model.lr_emb  == pytest.approx(1e-3)
    assert cfg.model.lr_head == pytest.approx(1e-3)


def test_model_config_warmup_steps():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.model.warmup_col_steps == 500
    assert cfg.model.warmup_icl_steps == 100


def test_model_config_head_init():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.model.output_head_init_bias == pytest.approx(0.5)
    assert cfg.model.output_head_init_std  == pytest.approx(0.01)


def test_model_config_bio_plausibility_none_by_default():
    cfg = ExperimentConfig.debug_preset()
    assert cfg.model.bio_plausibility_passed is None


def test_model_config_composition_loss_type_default_is_kl():
    cfg = ModelConfig()
    assert cfg.composition_loss_type == "kl"


def test_model_config_accepts_dirichlet_loss_type():
    cfg = ModelConfig(composition_loss_type="dirichlet")
    assert cfg.composition_loss_type == "dirichlet"


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
    assert d2["data"]["max_genes"] == 256
    assert d2["context"]["cells_per_bin"] == 5


def test_save_writes_config_json(tmp_path, monkeypatch):
    import spatialmt.config.paths as paths_module
    monkeypatch.setattr(paths_module, "PROJECT_ROOT", tmp_path)
    cfg = ExperimentConfig.debug_preset(run_id="save_test")
    cfg.save()
    config_file = tmp_path / "experiments" / "save_test" / "config.json"
    assert config_file.exists()
    d = json.loads(config_file.read_text())
    assert d["run_id"] == "save_test"
    assert d["data"]["max_genes"] == 256


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
    cfg1 = ExperimentConfig.debug_preset(run_id="lr_a")
    cfg2 = ExperimentConfig(
        run_id="lr_a",
        data=cfg1.data,
        context=cfg1.context,
        model=ModelConfig(lr_col=9e-5),
    )
    assert cfg1.config_hash != cfg2.config_hash


def test_config_hash_changes_with_different_max_genes():
    cfg1 = ExperimentConfig.debug_preset()
    cfg2 = ExperimentConfig(
        run_id=cfg1.run_id,
        data=DataConfig(max_genes=512, hardware_tier="debug"),
        context=cfg1.context,
        model=cfg1.model,
    )
    assert cfg1.config_hash != cfg2.config_hash


def test_config_hash_is_64_char_hex():
    cfg = ExperimentConfig.debug_preset()
    h = cfg.config_hash
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# rotation_finetune_dirichlet preset
# ---------------------------------------------------------------------------

def test_rotation_finetune_dirichlet_returns_experiment_config():
    cfg = ExperimentConfig.rotation_finetune_dirichlet()
    assert isinstance(cfg, ExperimentConfig)


def test_rotation_finetune_dirichlet_default_run_id():
    cfg = ExperimentConfig.rotation_finetune_dirichlet()
    assert cfg.run_id == "rotation_002"


def test_rotation_finetune_dirichlet_run_id_overridable():
    cfg = ExperimentConfig.rotation_finetune_dirichlet(run_id="my_run")
    assert cfg.run_id == "my_run"


def test_rotation_finetune_dirichlet_composition_loss_type():
    cfg = ExperimentConfig.rotation_finetune_dirichlet()
    assert cfg.model.composition_loss_type == "dirichlet"


# ---------------------------------------------------------------------------
# full_finetune — KL vs Dirichlet variants
# ---------------------------------------------------------------------------

def test_full_finetune_dirichlet_exists():
    assert hasattr(ExperimentConfig, "full_finetune_dirichlet")


def test_full_finetune_composition_loss_type_is_kl():
    cfg = ExperimentConfig.full_finetune()
    assert cfg.model.composition_loss_type == "kl"


def test_full_finetune_dirichlet_composition_loss_type():
    cfg = ExperimentConfig.full_finetune_dirichlet()
    assert cfg.model.composition_loss_type == "dirichlet"


def test_full_finetune_dirichlet_same_context_as_kl():
    kl  = ExperimentConfig.full_finetune()
    dir = ExperimentConfig.full_finetune_dirichlet()
    assert kl.context.cells_per_bin      == dir.context.cells_per_bin
    assert kl.context.max_context_cells  == dir.context.max_context_cells


def test_full_finetune_dirichlet_different_hash_from_kl():
    """composition_loss_type is a hyperparameter — must produce a distinct config hash."""
    kl  = ExperimentConfig.full_finetune(run_id="x")
    dir = ExperimentConfig.full_finetune_dirichlet(run_id="x")
    assert kl.config_hash != dir.config_hash
