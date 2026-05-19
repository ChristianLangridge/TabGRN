#!/bin/bash -l
# Batch script to run checkpoint inference on Myriad HPC.
# Evaluates a saved .pt checkpoint over day-11 held-out cells.
# EMD computation is parallelised across all allocated CPU cores.
#
# Usage — submit with optional overrides:
#   qsub myriad_inference.sh
#   CKPT_NAME=gene_coexpression.pt qsub myriad_inference.sh
#   NULL_CONTEXT=1 SHUFFLE_GENES=1 qsub myriad_inference.sh

#$ -N trace_inference             # job name
#$ -l h_rt=1:00:00                 # max wall-clock time
#$ -l mem=4G                       # RAM per core
#$ -pe smp 36                      # 36 CPU cores — parallelises EMD computation
#$ -l tmpfs=20G                    # local scratch for h5ad staging
#$ -wd /home/$USER/Scratch/TRACE
#$ -o /home/zcbtcl9/Scratch/TRACE/logs/trace_inference_$JOB_ID.out
#$ -e /home/zcbtcl9/Scratch/TRACE/logs/trace_inference_$JOB_ID.err
#$ -m bea
#$ -M zcbtcl9@ucl.ac.uk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT="$HOME/Scratch/TRACE"
H5AD_SRC="$HOME/Scratch/TRACE/data/training_data/AnnData/neurectoderm_with_pseudotime.h5ad"

# Checkpoint to evaluate — override with CKPT_NAME env var at submission time
# e.g. CKPT_NAME=final.pt qsub myriad_inference.sh
CKPT_NAME="${CKPT_NAME:-gene_coexpression.pt}"
CKPT_PATH="$HOME/Scratch/TRACE/experiments/rotation_002/checkpoints/$CKPT_NAME"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module purge
module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate trace

# ---------------------------------------------------------------------------
# Stage h5ad to local scratch
# ---------------------------------------------------------------------------

echo "Copying h5ad to local scratch..."
cp "$H5AD_SRC" "$TMPDIR/neurectoderm_with_pseudotime.h5ad" \
    || { echo "ERROR: failed to copy h5ad to scratch"; exit 1; }
export H5AD_PATH="$TMPDIR/neurectoderm_with_pseudotime.h5ad"
echo "  staged: $H5AD_PATH  ($(du -sh "$H5AD_PATH" | cut -f1))"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

mkdir -p "$PROJECT_ROOT/logs"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "============================================================"
echo "TRACE inference job"
echo "  Job ID      : $JOB_ID"
echo "  Node        : $HOSTNAME"
echo "  Cores       : $NSLOTS"
echo "  Checkpoint  : $CKPT_PATH"
echo "  H5AD        : $H5AD_PATH"
echo "  NULL_CONTEXT : ${NULL_CONTEXT:-0}"
echo "  SHUFFLE_GENES: ${SHUFFLE_GENES:-0}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"

export CKPT_PATH
export N_WORKERS=$NSLOTS
export NULL_CONTEXT="${NULL_CONTEXT:-0}"
export SHUFFLE_GENES="${SHUFFLE_GENES:-0}"
export DEVICE=cpu   # no GPU needed for inference — forward pass is fast enough

python -u src/inference/checkpoint_inference.py

echo "============================================================"
echo "Inference job complete: $JOB_ID"
echo "============================================================"
