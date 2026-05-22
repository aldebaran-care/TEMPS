#!/bin/bash
# -----------------------------------------------------------------------------
# Jean Zay (IDRIS) — QA post-training of the temporal embeddings model.
#
# Picks up where train.sh left off: starts from the most recent checkpoint
# produced by train.py and fine-tunes on the synthetic QA dataset.
#
# Override the source checkpoint with:
#   POSTTRAIN_CHECKPOINT=/abs/path/to/checkpoint.pth sbatch posttrain.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=temps-posttrain
#SBATCH --output=logs/slurm/temps-posttrain-%j.out
#SBATCH --error=logs/slurm/temps-posttrain-%j.err
#SBATCH --account=zrp@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

set -euo pipefail

source ~/.bashrc

mkdir -p logs/slurm

cd "$WORK"/projects/temporal/temporal-embeddings

uv sync

# Jean Zay compute nodes have no internet — see train.sh for the rationale.
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Pick the most recent pre-training checkpoint unless one was passed in.
# `ls -t` orders by mtime, newest first; the glob matches the per-step
# checkpoints saved by train.py.
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
CHECKPOINT_GLOB="output/trained_models/model_${MODEL_NAME//\//_}_*.pth"
POSTTRAIN_CHECKPOINT="${POSTTRAIN_CHECKPOINT:-$(ls -t ${CHECKPOINT_GLOB} 2>/dev/null | head -n1 || true)}"

if [ -z "${POSTTRAIN_CHECKPOINT:-}" ] || [ ! -f "${POSTTRAIN_CHECKPOINT}" ]; then
    echo "ERROR: no pre-training checkpoint found." >&2
    echo "  Set POSTTRAIN_CHECKPOINT=/abs/path/to/checkpoint.pth or run train.sh first." >&2
    exit 1
fi

echo "Post-training from checkpoint: ${POSTTRAIN_CHECKPOINT}"

uv run torchrun \
    --nproc_per_node=1 \
    posttrain.py \
    --model_path "${POSTTRAIN_CHECKPOINT}" \
    --data_fraction=1.0 \
    --epochs=1 \
    --batch_size=512 \
    --model_name="${MODEL_NAME}" \
    --input_file_path ./data/synthetic_dataset_with_score.csv

# Bundle run artifacts for off-cluster inspection (same convention as train.sh).
RUN_DIR=$(ls -dt logs/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${RUN_DIR:-}" ]; then
    tar -czf "logs/${SLURM_JOB_ID}_posttrain_$(basename "$RUN_DIR").tar.gz" \
        "$RUN_DIR" \
        output/metrics \
        output/log.csv \
        2>/dev/null || true
    echo "Archived run artifacts: logs/${SLURM_JOB_ID}_posttrain_$(basename "$RUN_DIR").tar.gz"
fi
