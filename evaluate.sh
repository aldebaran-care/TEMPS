#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n53

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env

BENCHMARKS=("ts_retriever")
EXTERNAL_MODELS=("intfloat/e5-base-v2" "all-mpnet-base-v2" "BAAI/bge-large-en-v1.5" "salesforce")
NUM_NEGATIVE_SAMPLES=(-1)
ALPHA_VALUES=($(seq 0.05 0.05 0.95))
EVAL_ID="New architecture: 50M New ts_retriever (split)"
MODEL_PATH="output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2026-01-19_02-33-51.pth"
BATCH_SIZE=128
MAX_SEQ_LEN=512
TOP_K=3

echo "##########################################"
echo "### EXTERNAL MODELS ONLY (no alpha) ###"
echo "##########################################"

for benchmark in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=== BENCHMARK: $benchmark ==="
    
    for num_neg in "${NUM_NEGATIVE_SAMPLES[@]}"; do
        echo "Num Negative Samples: $num_neg"

        python3 evaluate.py --model_name=all-minilm-l6-v2 --model_path=$MODEL_PATH --batch_size=$BATCH_SIZE --max_seq_len=$MAX_SEQ_LEN --benchmark=$benchmark --eval_id="$EVAL_ID" --top_k=$TOP_K --metric=all --num_negative_samples=$num_neg
        
        for external_model in "${EXTERNAL_MODELS[@]}"; do
            echo "External Model: $external_model"
            python3 evaluate.py --model_name=$external_model --benchmark=$benchmark --eval_id="$EVAL_ID" --top_k=$TOP_K --metric=all --num_negative_samples=$num_neg
        done
    done
done

for alpha in "${ALPHA_VALUES[@]}"; do
    echo "##########################################"
    echo "### ALPHA = $alpha ###"
    echo "##########################################"
    
    for benchmark in "${BENCHMARKS[@]}"; do
        echo ""
        echo "=== BENCHMARK: $benchmark ==="
        
        for num_neg in "${NUM_NEGATIVE_SAMPLES[@]}"; do
            echo "Num Negative Samples: $num_neg"
            
            for external_model in "${EXTERNAL_MODELS[@]}"; do
                echo "External Model: $external_model (with alpha)"
                python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=$external_model --model_path=$MODEL_PATH --batch_size=$BATCH_SIZE --max_seq_len=$MAX_SEQ_LEN --benchmark=$benchmark --eval_id="$EVAL_ID" --top_k=$TOP_K --metric=all --alpha=$alpha --num_negative_samples=$num_neg
            done
        done
    done
done

conda deactivate