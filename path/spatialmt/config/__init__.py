"""spatialmt.config — pipeline-wide configuration."""

from spatialmt.config.paths import Dirs, Paths, PROJECT_ROOT, setup_output_dirs, validate_raw_inputs
from spatialmt.config.experiment import (
    ExperimentConfig,
    DataConfig,
    ContextConfig,
    ModelConfig,
    ConfigurationError,
    HARDWARE_TIERS,
)

__all__ = [
    "Dirs", "Paths", "PROJECT_ROOT", "setup_output_dirs", "validate_raw_inputs",
    "ExperimentConfig", "DataConfig", "ContextConfig", "ModelConfig",
    "ConfigurationError", "HARDWARE_TIERS",
]
