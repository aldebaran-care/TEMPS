#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n102

source ~/.bashrc
cd /mnt/beegfs/home/hassani/training_an_em/project/temporal-embeddings
conda activate train-env

# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=time_sensitive_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=BAAI/bge-large-en-v1.5 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=time_sensitive_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=time_sensitive_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=salesforce --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=time_sensitive_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=mixedbread-ai/mxbai-embed-large-v1 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=time_sensitive_qa --eval_id=1 --top_k=5 --metric=all


# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=temp_reason_l1 --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=BAAI/bge-large-en-v1.5 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=temp_reason_l1 --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=temp_reason_l1 --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=salesforce --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=temp_reason_l1 --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=mixedbread-ai/mxbai-embed-large-v1 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=temp_reason_l1 --eval_id=1 --top_k=5 --metric=all

# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=menat_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=BAAI/bge-large-en-v1.5 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=menat_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=menat_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=salesforce --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=menat_qa --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=mixedbread-ai/mxbai-embed-large-v1 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=menat_qa --eval_id=1 --top_k=5 --metric=all

# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=ts_retriever --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2-full --external_model_name=BAAI/bge-large-en-v1.5 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=ts_retriever --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=all-minilm-l6-v2 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=ts_retriever --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=salesforce --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=ts_retriever --eval_id=1 --top_k=5 --metric=all
# python3 evaluate.py --model_name=mixedbread-ai/mxbai-embed-large-v1 --external_model_name=all-mpnet-base-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_13-23-15.pth --batch_size=128 --max_seq_len=512 --benchmark=ts_retriever --eval_id=1 --top_k=5 --metric=all

python3 evaluate.py --model_name=all-minilm-l6-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_23-47-04.pth --batch_size=128 --max_seq_len=512 --benchmark=time_sensitive_qa --eval_id=1 --top_k=5 --metric=all
python3 evaluate.py --model_name=all-minilm-l6-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_23-47-04.pth --batch_size=128 --max_seq_len=512 --benchmark=menat_qa --eval_id=1 --top_k=5 --metric=all
python3 evaluate.py --model_name=all-minilm-l6-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_23-47-04.pth --batch_size=128 --max_seq_len=512 --benchmark=ts_retriever --eval_id=1 --top_k=5 --metric=all
python3 evaluate.py --model_name=all-minilm-l6-v2 --model_path=output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2025-07-25_23-47-04.pth --batch_size=128 --max_seq_len=512 --benchmark=temp_reason_l1 --eval_id=1 --top_k=5 --metric=all

conda deactivate