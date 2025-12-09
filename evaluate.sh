#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n53

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env

TOP_K_VALUES=(1 3 5 10 50 100)
ALPHA_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
BENCHMARKS=("time_sensitive_qa" "ts_retriever" "temp_reason_l1")
EXTERNAL_MODELS=("intfloat/e5-base-v2")

for alpha in "${ALPHA_VALUES[@]}"; do
    echo "##########################################"
    echo "### ALPHA = $alpha ###"
    echo "##########################################"
    
    for top_k in "${TOP_K_VALUES[@]}"; do
        echo "=========================================="
        echo "=== TOP_K = $top_k ==="
        echo "=========================================="
        
        for benchmark in "${BENCHMARKS[@]}"; do
            echo ""
            echo "=== BENCHMARK: $benchmark ==="
            
            for external_model in "${EXTERNAL_MODELS[@]}"; do
                echo "External Model: $external_model"
                python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=$external_model --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_23-47-04.pth --batch_size=128 --max_seq_len=512 --benchmark=$benchmark --eval_id=2 --top_k=$top_k --metric=all --alpha=$alpha --use_all_paragraphs | grep "^{'top':"
            done
        done
    done
done

conda deactivate