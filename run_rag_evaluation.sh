#!/bin/bash
# Local OpenRouter RAG evaluation for the held-out Time-Sensitive QA split.
# OPENROUTER_API_KEY is loaded from the project .env file by Python.

set -euo pipefail

cd "$(dirname "$0")"

mkdir -p output/rag

uv sync

# Avoid the noisy fork warning when retrieval spins up its multiprocessing pool.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EVAL_CONFIG="${EVAL_CONFIG:-temporal_embeddings/config/rag_evaluation_config.json}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

echo "Using RAG config: ${EVAL_CONFIG}"
echo "Using evaluation split: ${EVAL_SPLIT}"

uv run python run_rag_evaluation.py --config "${EVAL_CONFIG}" --split "${EVAL_SPLIT}"
