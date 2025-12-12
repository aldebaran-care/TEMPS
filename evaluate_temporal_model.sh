#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n53

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env

BENCHMARKS=("time_sensitive_qa" "ts_retriever" "temp_reason_l1")
NUM_NEGATIVE_SAMPLES=(-1 0 10 100)

echo "##########################################"
echo "### EXTERNAL MODELS ONLY (no alpha) ###"
echo "##########################################"
    
for benchmark in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=== BENCHMARK: $benchmark ==="
    
    for num_neg in "${NUM_NEGATIVE_SAMPLES[@]}"; do
        echo "Num Negative Samples: $num_neg"            
        python3 evaluate.py --model_name=all-minilm-l6-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_23-47-04.pth --batch_size=128 --max_seq_len=512 --benchmark=$benchmark --eval_id="all baselines" --top_k=10 --metric=all --num_negative_samples=$num_neg | grep "^{'top':"
    done
done

conda deactivate