#!/bin/bash -l
#SBATCH --job-name=tabgrn_rotation
#SBATCH --account=<YOUR_PROJECT_CODE>         # ← replace with your Myriad project code
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tabgrn_%j.out
#SBATCH --error=logs/tabgrn_%j.err

# ---------------------------------------------------------------------------
# Paths — set these to match your Myriad home/scratch layout
# ---------------------------------------------------------------------------

PROJECT_ROOT="$HOME/TabGRN"                   # ← adjust if cloned elsewhere
H5AD_PATH="$HOME/TabGRN/data/training_data/AnnData/neurectoderm_with_pseudotime.h5ad"
BACKBONE="$HOME/TabGRN/data/TabICLv2_checkpoint/tabicl-regressor-v2-20260212.ckpt"

export PROJECT_ROOT H5AD_PATH BACKBONE

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module purge
module load python/miniconda3/4.10.3          # ← check available: module avail python
source activate tabgrn                         # ← your conda env name on Myriad

# Install the package in editable mode if not already done
# (safe to run every time — no-op if already installed)
pip install -e "$PROJECT_ROOT" --quiet

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "============================================================"
echo "TabGRN Myriad job"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  H5AD_PATH   : $H5AD_PATH"
echo "  BACKBONE    : $BACKBONE"
echo "============================================================"

# Abort early if required files are missing
for f in "$H5AD_PATH" "$BACKBONE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

mkdir -p "$PROJECT_ROOT/logs"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"

python src/train/tabgrn_myriad_run.py

echo "============================================================"
echo "Job complete: $SLURM_JOB_ID"
echo "============================================================"
