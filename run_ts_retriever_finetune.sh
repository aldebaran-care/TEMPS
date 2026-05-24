#!/bin/bash
# -----------------------------------------------------------------------------
# Jean Zay (IDRIS) — TS-Retriever fine-tuning of a trained TEMPS model.
#
# Defaults:
#   train: data/new_training_dataset/ts_retriever_finetuning/train.csv
#   val:   data/new_training_dataset/ts_retriever_finetuning/val.csv
#
# Override when submitting:
#   EPOCHS=2 \
#   BATCH_SIZE=512 \
#   sbatch run_ts_retriever_finetune.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=ts-ret-ft
#SBATCH --output=logs/slurm/ts-ret-ft-%j.out
#SBATCH --error=logs/slurm/ts-ret-ft-%j.err
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
module load arch/a100

mkdir -p logs/slurm

cd "$WORK"/projects/temporal/temporal-embeddings

uv sync

# Jean Zay compute nodes are offline. Keep all HF calls local.
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="${MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
MODEL_NAME_SAFE="${MODEL_NAME//\//_}"

TRAIN_FILE="${TRAIN_FILE:-data/new_training_dataset/ts_retriever_finetuning/train.csv}"
VAL_FILE="${VAL_FILE:-data/new_training_dataset/ts_retriever_finetuning/val.csv}"
FINETUNE_FILE="${FINETUNE_FILE:-data/new_training_dataset/ts_retriever_finetuning/train_val_for_gaussdata.csv}"

DATA_FRACTION="${DATA_FRACTION:-1.0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-2e-5}"
NUM_WARMUP_RATIO="${NUM_WARMUP_RATIO:-0.02}"
NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-100}"

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: train file not found: ${TRAIN_FILE}" >&2
    exit 1
fi

if [ ! -f "${VAL_FILE}" ]; then
    echo "ERROR: validation file not found: ${VAL_FILE}" >&2
    exit 1
fi

# The current GaussData loader accepts one CSV and then slices it as:
# 0-90% train, 98-99% val, 99-100% test. Build a staged CSV where the
# requested train rows fill the training slice and the held-out tail comes
# from the requested validation file.
TRAIN_ROWS=$(($(wc -l < "${TRAIN_FILE}") - 1))
VAL_ROWS=$(($(wc -l < "${VAL_FILE}") - 1))
TOTAL_ROWS=$(((TRAIN_ROWS * 10 + 8) / 9))
APPEND_ROWS=$((TOTAL_ROWS - TRAIN_ROWS))

if [ "${TRAIN_ROWS}" -le 0 ]; then
    echo "ERROR: train file has no data rows: ${TRAIN_FILE}" >&2
    exit 1
fi

if [ "${VAL_ROWS}" -le 0 ]; then
    echo "ERROR: validation file has no data rows: ${VAL_FILE}" >&2
    exit 1
fi

mkdir -p "$(dirname "${FINETUNE_FILE}")"
head -n 1 "${TRAIN_FILE}" > "${FINETUNE_FILE}"
tail -n +2 "${TRAIN_FILE}" >> "${FINETUNE_FILE}"
awk -v need="${APPEND_ROWS}" '
    FNR == 1 { next }
    { rows[++n] = $0 }
    END {
        for (i = 0; i < need; i++) {
            print rows[(i % n) + 1]
        }
    }
' "${VAL_FILE}" >> "${FINETUNE_FILE}"

echo "Prepared fine-tuning CSV: ${FINETUNE_FILE}"
echo "  train rows: ${TRAIN_ROWS}"
echo "  validation rows staged in held-out tail: ${APPEND_ROWS}"

POSTTRAIN_CHECKPOINT="output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2026-05-24_00-00-51.pth"

if [ -z "${POSTTRAIN_CHECKPOINT:-}" ] || [ ! -f "${POSTTRAIN_CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${POSTTRAIN_CHECKPOINT}" >&2
    exit 1
fi

echo "Fine-tuning from checkpoint: ${POSTTRAIN_CHECKPOINT}"
echo "Base model: ${MODEL_NAME}"

uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    posttrain.py \
    --model_path "${POSTTRAIN_CHECKPOINT}" \
    --data_fraction="${DATA_FRACTION}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --lr="${LR}" \
    --num_warmup_ratio="${NUM_WARMUP_RATIO}" \
    --num_eval_steps="${NUM_EVAL_STEPS}" \
    --model_name="${MODEL_NAME}" \
    --input_file_path "${FINETUNE_FILE}"

RUN_DIR=$(ls -dt logs/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${RUN_DIR:-}" ]; then
    tar -czf "logs/${SLURM_JOB_ID:-manual}_ts_retriever_finetune_$(basename "$RUN_DIR").tar.gz" \
        "$RUN_DIR" \
        output/metrics \
        output/log.csv \
        2>/dev/null || true
    echo "Archived run artifacts: logs/${SLURM_JOB_ID:-manual}_ts_retriever_finetune_$(basename "$RUN_DIR").tar.gz"
fi
