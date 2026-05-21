#!/bin/bash
# -----------------------------------------------------------------------------
# Jean Zay (IDRIS) — exclusive 8-GPU A100 node training.
#
# Notes:
#   * gpu_p5 nodes are octo-A100 (8 x A100-80GB SXM4) with AMD EPYC 7543 Milan
#     (64 physical cores / node). There is no 4-GPU A100 partition on Jean Zay.
#   * `--ntasks-per-node=1` because torchrun spawns one worker per GPU itself
#     (`--nproc_per_node=8`). The full 64 physical cores are reserved for the
#     8 workers (8 cores / GPU, the IDRIS-recommended ratio).
#   * `--exclusive` makes the node reservation explicit. On gpu_p5 a full-node
#     reservation is billed for the full node anyway.
#   * qos_gpu-t3 caps walltime at 20h; use qos_gpu-t4 (max 100h, fewer slots)
#     if you need longer.
#   * Replace CHANGE_ME with your IDRIS project id (the @a100 suffix is
#     mandatory for A100 jobs).
# -----------------------------------------------------------------------------
#SBATCH --job-name=temps-train
#SBATCH --output=logs/slurm/temps-train-%j.out
#SBATCH --error=logs/slurm/temps-train-%j.err
#SBATCH --account=zrp@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=20:00:00

set -euo pipefail

source ~/.bashrc

mkdir -p logs/slurm

# Project root on Jean Zay — adjust if your checkout lives elsewhere.
cd "$WORK"/projects/temporal/temporal-embeddings

uv sync

# Jean Zay compute nodes have no internet. Read tokenizer/model weights from
# the project-local cache populated by `scripts/prefetch_model.py` on a login
# node. TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE make HF fail fast (no DNS hang)
# if anything is missing from the cache.
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1   # `datasets` reads the local CSV; never hit the Hub

# NCCL transport hints for Jean Zay gpu_p5 (single node, 8 A100 SXM4):
#   - keep NVLink P2P enabled (SXM4 link is the fast intra-node path)
#   - disable IB probing — single-node training doesn't cross IB, and the
#     auto-probe has been observed to segfault when libfabric / UCX isn't
#     on LD_LIBRARY_PATH
#   - DEBUG=INFO prints the chosen transport once per rank; remove once stable
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# torchrun rendezvous on a single node: nproc_per_node = # of A100 on the node.
uv run torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    train.py \
    --data_fraction=1 \
    --epochs=1 \
    --batch_size=512 \
    --model_name=sentence-transformers/all-MiniLM-L6-v2 \
    --input_file_path ./merged_training_data.csv \

# Bundle the run artifacts so they can be scp'd off the cluster in one file
# and opened locally with `tensorboard --logdir logs/runs`.
RUN_DIR=$(ls -dt logs/runs/* 2>/dev/null | head -n1 || true)
if [ -n "${RUN_DIR:-}" ]; then
    tar -czf "logs/${SLURM_JOB_ID}_$(basename "$RUN_DIR").tar.gz" \
        "$RUN_DIR" \
        output/metrics \
        output/log.csv \
        2>/dev/null || true
    echo "Archived run artifacts: logs/${SLURM_JOB_ID}_$(basename "$RUN_DIR").tar.gz"
fi
