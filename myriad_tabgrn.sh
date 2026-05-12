#!/bin/bash -l
# Batch script to run a GPU job for model fine-tuning.
#$ -N tabgrn_rotation           # job name

#$ -l h_rt=08:00:00             # max wall-clock time
#$ -l mem=6G                    # RAM per core (Myriad: per-core, not total)
#$ -l gpu=1                     # one GPU
#$ -pe smp 8                    # 8 CPU cores (shared-memory parallel env)
#$ -l tmpfs=20G                 # local scratch on the compute node
#$ -wd /home/$USER/Scratch/TabGRN
#$ -o /home/zcbtcl9/Scratch/TabGRN/logs/tabgrn_$JOB_ID.out
#$ -e /home/zcbtcl9/Scratch/TabGRN/logs/tabgrn_$JOB_ID.err
#$ -m bea                       # email on Begin, End, Abort (add your address below)
#$ -M zcbtcl9@ucl.ac.uk

# ---------------------------------------------------------------------------
# Paths — adjust if cloned elsewhere on Myriad scratch
# ---------------------------------------------------------------------------

PROJECT_ROOT="$HOME/Scratch/TabGRN"
H5AD_PATH="$HOME/Scratch/TabGRN/data/training_data/AnnData/neurectoderm_with_pseudotime.h5ad"
BACKBONE="$HOME/Scratch/TabGRN/data/TabICLv2_checkpoint/tabicl-regressor-v2-20260212.ckpt"

export PROJECT_ROOT H5AD_PATH BACKBONE

# Run preset — "dirichlet" uses Dirichlet NLL + fixed lambda_comp
export RUN_PRESET=dirichlet
export N_EPOCHS=3
export N_ICL_WARMUP_STEPS=1000
export SEED=42
export CUDA_LAUNCH_BLOCKING=1

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module purge
module load python/miniconda3/24.3.0-0
module load cuda/11.8.0/gnu-10.2.0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate tabgrn

# ---------------------------------------------------------------------------
# Stage h5ad to local scratch (faster random access than Lustre home)
# ---------------------------------------------------------------------------

echo "Copying h5ad to local scratch..."
cp "$H5AD_PATH" "$TMPDIR/neurectoderm_with_pseudotime.h5ad" \
    || { echo "ERROR: failed to copy h5ad to scratch"; exit 1; }
export H5AD_PATH="$TMPDIR/neurectoderm_with_pseudotime.h5ad"
echo "  staged: $H5AD_PATH  ($(du -sh "$H5AD_PATH" | cut -f1))"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

mkdir -p "$PROJECT_ROOT/logs"

echo "============================================================"
echo "TabGRN Myriad job"
echo "  Job ID      : $JOB_ID"
echo "  Node        : $HOSTNAME"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  RUN_PRESET  : $RUN_PRESET"
echo "  N_EPOCHS    : $N_EPOCHS"
echo "  WARMUP STEPS: $N_ICL_WARMUP_STEPS"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  H5AD_PATH   : $H5AD_PATH"
echo "  BACKBONE    : $BACKBONE"
echo "============================================================"

for f in "$H5AD_PATH" "$BACKBONE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"

python src/train/tabgrn_myriad_run.py

echo "============================================================"
echo "Job complete: $JOB_ID"
echo "============================================================"
