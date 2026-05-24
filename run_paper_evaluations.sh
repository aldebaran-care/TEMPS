#!/bin/bash
# -----------------------------------------------------------------------------
# Jean Zay (IDRIS) — paper evaluation runner for temporal retrieval benchmarks.
#
# Default config:
#   temporal_embeddings/config/paper_evaluation_config.json
#
# Override config path when submitting:
#   EVAL_CONFIG=/abs/path/to/config.json sbatch run_paper_evaluations.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=temps-paper-eval
#SBATCH --output=logs/slurm/temps-paper-eval-%j.out
#SBATCH --error=logs/slurm/temps-paper-eval-%j.err
#SBATCH --account=zrp@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Stage-2 similarity computation is CPU-parallel. On gpu_p5 the IDRIS share
# is 8 cores per requested GPU, so to get the full 64-core node we need
# --exclusive (which also reserves all 8 GPUs — we only use 1, but with
# --exclusive the node is billed in full regardless).
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=10:00:00

set -euo pipefail

source ~/.bashrc

mkdir -p logs/slurm

cd "$WORK"/projects/temporal/temporal-embeddings

uv sync

# Jean Zay compute nodes are offline. Keep all HF calls local.
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_CONFIG="${EVAL_CONFIG:-temporal_embeddings/config/paper_evaluation_config.json}"

echo "Using evaluation config: ${EVAL_CONFIG}"
uv run python run_paper_evaluations.py --config "${EVAL_CONFIG}"
