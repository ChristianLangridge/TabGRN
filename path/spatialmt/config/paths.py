"""
spatialmt.config.paths
======================
Single source of truth for every filesystem path in the pipeline.

Root detection priority (highest → lowest):
  1. Env var  PROJECT_ROOT  — set this on HPC/Docker/Colab for zero ambiguity.
  2. Sentinel walk          — walks up from this file looking for pyproject.toml.
                              Works on any local machine after `pip install -e .`

Usage
-----
    from spatialmt.config.paths import Paths

    df = pd.read_csv(Paths.raw_tpm)          # always correct, on every machine
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Return the absolute project root directory.

    Strategy
    --------
    1. Honour the PROJECT_ROOT environment variable if set.
       Recommended for HPC (set in your .bashrc / SLURM job script),
       Docker (pass via -e PROJECT_ROOT=...), and Colab (os.environ before imports).

    2. Walk upward from *this file* until we find pyproject.toml.
       Reliable after `pip install -e .` on any local machine.

    Raises
    ------
    RuntimeError
        If neither strategy succeeds, with an actionable error message.
    """
    # Strategy 1 — explicit env var (HPC / Docker / Colab)
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"PROJECT_ROOT env var points to a non-existent directory: {root}\n"
                "Fix: export PROJECT_ROOT=/absolute/path/to/your/project"
            )
        return root

    # Strategy 2 — sentinel walk (local dev after pip install -e .)
    sentinel = "pyproject.toml"
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / sentinel).exists():
            return parent

    raise RuntimeError(
        "Could not locate project root.\n\n"
        "Options:\n"
        "  A) Local:  run `pip install -e .` from the project root once.\n"
        "  B) HPC:    add `export PROJECT_ROOT=/path/to/project` to your .bashrc\n"
        "             or SLURM job script before calling Python.\n"
        "  C) Docker: pass `-e PROJECT_ROOT=/workspace` to `docker run`.\n"
        "  D) Colab:  add `os.environ['PROJECT_ROOT'] = '/content/drive/...'`\n"
        "             before importing spatialmt.\n"
    )


PROJECT_ROOT: Path = _find_project_root()


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# Edit these if you reorganise folders — every other file imports from here,
# so one change propagates everywhere.

class Dirs:
    """Top-level directories."""
    root     = PROJECT_ROOT
    data     = PROJECT_ROOT / "data"
    raw      = PROJECT_ROOT / "data" / "raw"
    interim  = PROJECT_ROOT / "data" / "interim"
    processed = PROJECT_ROOT / "data" / "processed"
    results  = PROJECT_ROOT / "results"
    models   = PROJECT_ROOT / "models"
    src      = PROJECT_ROOT / "src"


class Paths:
    """
    Fully-resolved paths to specific files.

    Add a new entry here whenever you introduce a new canonical data file.
    Never hardcode a path anywhere else in the codebase.
    """
    # --- raw inputs --------------------------------------------------------
    raw_tpm        = Dirs.raw      / "Original_TPM_data.csv"
    raw_tpm_txt    = Dirs.raw      / "Original_TPM_data.txt"

    # --- interim (post-cleaning, pre-feature-engineering) ------------------
    clean_tpm      = Dirs.interim  / "clean_tpm.csv"

    # --- processed (model-ready) -------------------------------------------
    processed_tpm  = Dirs.processed / "processed_tpm.csv"


# ---------------------------------------------------------------------------
# Filesystem setup
# ---------------------------------------------------------------------------
# Auto-mkdir has been deliberately removed from import time.
#
# Reason: in distributed / multi-GPU training, this module is imported once
# per process (32+ times on a multi-node SLURM job). Concurrent mkdir/exists
# calls hammer the shared filesystem (Lustre/GPFS) with metadata ops and
# cause metadata storms.
#
# Instead, call setup_output_dirs() explicitly on rank 0 only, before the
# distributed barrier. See the training script snippet in the README.

def setup_output_dirs() -> None:
    """
    Create all derived output directories.

    Call this ONCE before training starts — on rank 0 only in distributed runs.

    Single-GPU / notebook usage:
        from spatialmt.config import setup_output_dirs
        setup_output_dirs()

    Multi-GPU / SLURM usage:
        if dist.get_rank() == 0:
            setup_output_dirs()
            validate_raw_inputs()
        dist.barrier()
    """
    for _dir in (Dirs.interim, Dirs.processed, Dirs.results, Dirs.models):
        _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sanity-check: warn if expected raw inputs are absent
# ---------------------------------------------------------------------------
# Same rule applies: call on rank 0 only in distributed runs.

def validate_raw_inputs() -> None:
    """
    Verify all raw input files exist before a pipeline run begins.

    Raises FileNotFoundError with a clear, actionable message if anything
    is missing — catching the problem at the start of a run rather than
    mid-way through preprocessing.

    Call this ONCE on rank 0 only in distributed runs (see setup_output_dirs).
    """
    required = {
        "Original TPM CSV":  Paths.raw_tpm,
        "Original TPM text": Paths.raw_tpm_txt,
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        lines = "\n".join(f"  {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(
            f"Missing raw input file(s):\n{lines}\n\n"
            f"Expected inside: {Dirs.raw}\n"
            "Copy or symlink your source data there before running the pipeline."
        )