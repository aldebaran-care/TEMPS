#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n54

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env
python3 train.py --data_fraction=1 --epochs=1 --batch_size=128 --model_name=intfloat/e5-base-v2 --input_file_path=data/new_training_dataset/training_datasets/training_data_chunk_1.csv
conda deactivate