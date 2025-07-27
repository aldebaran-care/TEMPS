#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:3
#SBATCH --nodelist=n54

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env
python3 train.py --data_fraction=0.5 --epochs=1 --batch_size=128 --model_name=sentence-transformers/all-MiniLM-L6-v2 --input_file_path=data/dataset/dataset.csv
conda deactivate