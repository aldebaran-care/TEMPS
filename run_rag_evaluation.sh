#!/bin/bash
# -----------------------------------------------------------------------------
# Jean Zay (IDRIS) — RAG evaluation runner for Time-Sensitive QA.
#
# Pairs the trained temporal embedding model with semantic retrievers and a
# generator LLM, then scores retrieval (Recall@k, MRR, NDCG@k) and QA
# (SQuAD EM/F1) on data/evaluation/time_sensitive_qa/human_annotated_test.json.
#
# All outputs land under output/rag/ (config snapshot + per-run metrics +
# per-run predictions + consolidated report).
#
# Default config:
#   temporal_embeddings/config/rag_evaluation_config.json
#
# Override config when submitting:
#   EVAL_CONFIG=/abs/path/to/config.json sbatch run_rag_evaluation.sh
#
# IMPORTANT — model weights:
#   GPU nodes have no internet. Download every HF repo referenced by the
#   config from a login node BEFORE submitting (see the README block at the
#   bottom of this script).
# -----------------------------------------------------------------------------
#SBATCH --job-name=temps-rag-eval
#SBATCH --output=logs/slurm/temps-rag-eval-%j.out
#SBATCH --error=logs/slurm/temps-rag-eval-%j.err
#SBATCH --account=zrp@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Retrieval stage is CPU-parallel; generation stage is GPU-bound on one A100.
# Reserve the full node so we get all 64 cores for the parallel similarity
# computation (gpu_p5 quota is otherwise 8 cores per requested GPU).
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00

set -euo pipefail

source ~/.bashrc

mkdir -p logs/slurm output/rag

cd "$WORK"/projects/temporal/temporal-embeddings

uv sync

# Jean Zay compute nodes are offline. Keep all HF calls local.
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Avoid the noisy fork warning when retrieval spins up its multiprocessing pool.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EVAL_CONFIG="${EVAL_CONFIG:-temporal_embeddings/config/rag_evaluation_config.json}"

echo "Using RAG config: ${EVAL_CONFIG}"
echo "HF cache:        ${HF_HOME}"
nvidia-smi || true

uv run python run_rag_evaluation.py --config "${EVAL_CONFIG}"

# -----------------------------------------------------------------------------
# One-time setup on the LOGIN / INTERFACE node (has internet):
#
#   cd "$WORK"/projects/temporal/temporal-embeddings
#   export HF_HOME="$PWD/.hf_cache"
#   uv sync
#
#   # Default generator (Qwen2.5-7B-Instruct, no gating):
#   uv run python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download(
#       repo_id='Qwen/Qwen2.5-7B-Instruct',
#       allow_patterns=['*.json','*.txt','*.safetensors','tokenizer*','*.model'],
#   )
#   "
#
#   # Alternative generator (Llama-3.1-8B-Instruct — requires HF token + license):
#   #   huggingface-cli login
#   #   uv run python -c "from huggingface_hub import snapshot_download; \
#   #     snapshot_download(repo_id='meta-llama/Llama-3.1-8B-Instruct')"
#
#   # Semantic baselines listed in the default config:
#   uv run python -c "
#   from huggingface_hub import snapshot_download
#   for repo in [
#       'intfloat/e5-base-v2',
#       'BAAI/bge-large-en-v1.5',
#       'sentence-transformers/all-mpnet-base-v2',
#       'Salesforce/SFR-Embedding-Mistral',
#   ]:
#       snapshot_download(repo_id=repo)
#   "
#
# After download, submit the job:
#   sbatch run_rag_evaluation.sh
# -----------------------------------------------------------------------------
