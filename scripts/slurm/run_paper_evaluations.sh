#!/bin/bash
# -----------------------------------------------------------------------------
# Paper evaluation runner for temporal retrieval benchmarks.
#
# Default config:
#   temporal_embeddings/config/paper_evaluation_config.json
#
# Override config path when submitting:
#   EVAL_CONFIG=/abs/path/to/config.json sbatch scripts/slurm/run_paper_evaluations.sh
#
# Example SLURM job. Adjust the commented-out ##SBATCH lines (account,
# partition, constraint) and module loads to your cluster, then submit
# from the repository root.
# -----------------------------------------------------------------------------
#SBATCH --job-name=temps-paper-eval
#SBATCH --output=logs/slurm/temps-paper-eval-%j.out
#SBATCH --error=logs/slurm/temps-paper-eval-%j.err
##SBATCH --account=<your_account>
##SBATCH --partition=<your_gpu_partition>
##SBATCH --constraint=<your_gpu_constraint>
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Stage-2 similarity computation is CPU-parallel. If your scheduler allots a
# fixed number of cores per GPU, --exclusive gets the full node (it also
# reserves every GPU, so the node is billed in full).
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=10:00:00

set -euo pipefail

source ~/.bashrc

mkdir -p logs/slurm

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

uv sync

# If compute nodes have no internet, pre-fetch weights on a login node with
# scripts/prefetch_model.py and keep all HF calls local.
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_CONFIG="${EVAL_CONFIG:-temporal_embeddings/config/paper_evaluation_config.json}"

echo "Using evaluation config: ${EVAL_CONFIG}"
uv run python run_paper_evaluations.py --config "${EVAL_CONFIG}"
