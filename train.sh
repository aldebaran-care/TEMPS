#!/bin/bash
#SBATCH --job-name=temps-train-fast
#SBATCH --output=logs/slurm/temps-train-%j.out
#SBATCH --error=logs/slurm/temps-train-%j.err
#SBATCH --account=zrp@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=10:00:00

set -euo pipefail

source ~/.bashrc
module load arch/a100

mkdir -p logs/slurm

cd "$WORK"/projects/temporal/temporal-embeddings

uv sync

export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    --data_fraction=1.0 \
    --epochs=1 \
    --batch_size=2048 \
    --num_eval_steps=500 \
    --model_name=sentence-transformers/all-MiniLM-L6-v2 \
    --input_file_path ./merged_training_data.csv

RUN_DIR=$(ls -dt logs/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${RUN_DIR:-}" ]; then
    tar -czf "logs/${SLURM_JOB_ID}_$(basename "$RUN_DIR").tar.gz" \
        "$RUN_DIR" \
        output/metrics \
        output/log.csv \
        2>/dev/null || true
    echo "Archived run artifacts: logs/${SLURM_JOB_ID}_$(basename "$RUN_DIR").tar.gz"
fi