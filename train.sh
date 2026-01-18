#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=n53
#SBATCH --cpus-per-task=28
#SBATCH --mem=180G

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env
torchrun --nproc_per_node=3 --nnodes=1 train.py --data_fraction=1 --epochs=1 --batch_size=128 --model_name=sentence-transformers/all-MiniLM-L6-v2 --input_file_path data/new_training_dataset/training_datasets/training_data_chunk_31_50.csv --continue_training --model_path output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2026-01-17_23-12-36.pth
conda deactivate