TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval
Code, trained temporal module, training-data generation pipeline, and evaluation scripts for "TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval" (Mourad Hassani, Julien Romero, Amel Bouzeghoub, Christian Jacquelinet), EMNLP 2026.

Overview
Dense retrievers and RAG pipelines match queries to documents well on topic but poorly on time, so they surface content that is on-topic yet temporally wrong. This work introduces:

Temporal Textual Similarity (TTS) — a task that measures how well two anchored texts (text + anchor date) align in time, independent of their topical similarity.
TEMPS (Temporal Embedding Model for Precise Search) — a modular temporal branch that attaches to a frozen semantic retriever. It resolves anchored temporal expressions to intervals, moment-matches each interval to a Gaussian, and uses the resulting ordering to supervise an anchor-date-conditioned encoder. At inference the temporal score is fused with the semantic score.
Supervision comes from grounding, so training uses no hand-labeled temporal data (60M automatically generated tuples). The temporal score is the Gaussian-KL inclusion measure from distributional embeddings; what TEMPS adds is the grounding and the moment-matched supervision.

Method in brief
Each grounded interval I = [a, b] (inclusive day range) is moment-matched to a Gaussian:

mu    = (a + b + 1) / 2
sigma^2 = (b - a + 1)^2 / 12

The label is the directional divergence TSF(I1 -> I2) = KL(p_I1 || p_I2), bounded as ts = 1 / (1 + TSF). The model emits a diagonal Gaussian per (text, anchor date) and is trained with a CoSENT ranking loss to reproduce that ordering. At inference:

s(1,2) = (1 - alpha) * cos(E_1, E_2) + alpha * 1 / (1 + KL(p_1 || p_2))

with the two branches min-max normalized per query before fusion. alpha is a benchmark-level calibration parameter selected on a held-out validation split (0.6 TimeQA, 0.8 TempReason, 0.3 TS-Retriever).

Architecture: a frozen all-MiniLM-L6-v2 backbone (512 tokens) + a sinusoidal anchor-date encoder, attention pooling, and mean/variance heads. The temporal module adds ~27M parameters.

Results
Highlights on the held-out test splits (full tables in the paper):

Benchmark	Best system	MRR	Gain over backbone
TimeQA (alpha=0.6)	Mistral + TEMPS	0.5360	+31.9%
TempReason (alpha=0.8)	Mistral + SUTime control	0.6901	+11.5%
TS-Retriever (alpha=0.3)	Mistral + TEMPS	0.8240	+9.3%
Against the prior temporal state of the art on TS-Retriever, E5base-v2 + TEMPS lifts R@1 from 19.92 to 25.39 and P@1 from 58.63 to 69.14 over TSContriever. On downstream TimeQA RAG (top-5 passages, Qwen2.5-7B-Instruct) every +TEMPS retriever improves EM, F1 and containment over its unaugmented counterpart.

TempReason is reported as-is rather than tuned around: at the validation-selected alpha=0.8 the fused score is temporal-dominated, which sharpens the top of the ranking but costs Recall@5 for all four backbones, and the rule-based SUTime control outperforms TEMPS there.

Installation
Training environment (SLURM / GPU servers)
Training dependencies are managed with uv from pyproject.toml. The lockfile pulls the CUDA 12.8 PyTorch wheels (cu128).

uv sync

This creates a .venv/ with everything needed to run train.py. Invoke scripts with uv run, e.g. uv run python train.py ... or uv run torchrun ... (see train.sh).

Jean Zay (offline A100 compute nodes)
Jean Zay's gpu_p5 A100 nodes have no internet, so HuggingFace weights must be pre-cached on a login node first:

On a login node (has internet):
uv sync
uv run python scripts/prefetch_model.py sentence-transformers/all-MiniLM-L6-v2

This populates .hf_cache/ under the project.
Submit the job — train.sh exports HF_HOME=$PWD/.hf_cache, TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1, so the compute node reads from the cache and never touches the network:
sbatch train.sh

After the job, train.sh packages the TensorBoard event dir + metrics JSON + log.csv into logs/<jobid>_<run>.tar.gz. Pull it back and view locally:
scp jean-zay:.../temporal-embeddings/logs/<jobid>_*.tar.gz .
tar xzf <jobid>_*.tar.gz
tensorboard --logdir logs/runs

Data preparation / annotation environment (local)
The data prep and annotation scripts (create_*.py, add_score_v2_to_training_data.py, etc.) have additional dependencies (spaCy, stanza, NLTK, sentence-transformers, datatrove, ...). For that workflow use the legacy requirements.txt:

pip install -r requirements.txt
python setup.py  # installs the Stanza CoreNLP server (needed for the SUTime control)

Usage
1. Creating the Training Dataset
The training pipeline uses a weakly supervised approach combining rule-based temporal annotation with synthetic data generation.

Step 1: Generate Synthetic Data
Synthetic TimeML-style expressions spanning the years 1000–2030, rendered into text with templates and paired with random anchor dates (55M tuples in the paper):

python create_synthetic_dataset.py \
    --output_file_path data/synthetic_data/synthetic_dataset.csv \
    --size 400000

Step 2: Create Real-World Dataset
Extract anchored temporal expressions from the English portion of the Multilingual MLM Temporal Tagging Resources (Wikipedia text, HeidelTime-tagged; 5M tuples in the paper):

python create_real_dataset.py --num_rows 1000000 --skip 0

Step 3: Merge and Prepare Training Data
Combine synthetic and real-world datasets:

python create_training_dataset.py

This script merges data from:

data/new_training_dataset/synthetic_dataset/synthetic_dataset.csv
data/new_training_dataset/synthetic_dataset/temporal_relationships.csv
data/new_training_dataset/real_world_dataset/dataset.csv
Output is saved to data/new_training_dataset/training_datasets/ as tuples (s1, d1, s2, d2, ts).

2. Training the Model
# Single GPU training
python train.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --input_file_path data/new_training_dataset/training_datasets/merged_training_data.csv \
    --epochs 1 \
    --batch_size 128 \
    --lr 3e-4

# Multi-GPU distributed training
torchrun --nproc_per_node=3 --nnodes=1 train.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --input_file_path data/new_training_dataset/training_datasets/merged_training_data.csv \
    --epochs 1 \
    --batch_size 128 \
    --data_fraction 1.0

Paper configuration: 3 epochs over the full 60M-tuple corpus, global batch size 2,048, lr 3e-4, weight decay 0.01, 5% linear warmup, max 512 tokens, fp16, seed 42 — about 4 wall-clock hours on 8×A100-80GB.

Training Arguments
Argument	Default	Description
--model_name	prajjwal1/bert-tiny	Base model name
--batch_size	64	Training batch size
--lr	3e-4	Learning rate
--epochs	1	Number of training epochs
--weight_decay	0.01	Weight decay
--num_warmup_ratio	0.05	Warmup ratio for scheduler
--num_eval_steps	1000	Evaluation frequency
--data_fraction	1.0	Fraction of data to use
--continue_training	False	Resume from checkpoint
--model_path	None	Path to checkpoint for resuming
SLURM Cluster Training
sbatch train.sh

3. Creating Evaluation Benchmarks
# TimeQA (Time-Sensitive QA)
python create_evaluation_dataset.py time_sensitive_qa

# TempReason (levels 1 and 2 combined; temp_reason_l1 / temp_reason_l2 build them separately)
python create_evaluation_dataset.py temp_reason

# TS-Retriever
python create_evaluation_dataset.py ts_retriever

4. Evaluating Models
Evaluate TEMPS (Temporal Model)
python evaluate.py \
    --model_name all-minilm-l6-v2 \
    --model_path output/trained_models/model.pth \
    --benchmark temp_reason \
    --batch_size 128 \
    --max_seq_len 512 \
    --top_k 10 \
    --metric all

Evaluate Semantic Baseline Models
python evaluate.py \
    --model_name all-mpnet-base-v2 \
    --benchmark temp_reason \
    --top_k 10 \
    --metric all

python evaluate.py \
    --model_name intfloat/e5-base-v2 \
    --benchmark temp_reason \
    --top_k 10 \
    --metric all

Evaluate Combined Temporal + Semantic Model
python evaluate.py \
    --model_name all-minilm-l6-v2-full \
    --external_model_name intfloat/e5-base-v2 \
    --model_path output/trained_models/model.pth \
    --benchmark temp_reason \
    --alpha 0.5 \
    --top_k 10 \
    --metric all

Evaluation Arguments
Argument	Default	Description
--model_name	""	Model to evaluate
--model_path	""	Path to trained model weights
--benchmark	""	Benchmark dataset name
--batch_size	32	Evaluation batch size
--max_seq_len	128	Maximum sequence length
--top_k	1	Top-k for retrieval metrics
--metric	top	Metric type (all, top, mrr, ndcg, precision, recall, f1)
--alpha	0.5	Weight for combining temporal and semantic scores
--external_model_name	""	External semantic model for hybrid evaluation
--num_negative_samples	0	Number of negative samples
Batch Evaluation
sbatch evaluate.sh

5. Reproducing the Paper
All paper numbers come from config-driven runs over the same cached similarities. The held-out protocol is: tag each benchmark once into 20% validation / 80% test, select one alpha per benchmark on validation MRR alone, then score the test split once.

# 1. Tag a 20/80 val/test split in place (deterministic given --seed)
python split_and_tune_alpha.py tag

# 2. Sweep alpha on the validation split only, one value per benchmark
python split_and_tune_alpha.py tune-alpha

# 3. Main results table (Table 1) on the test split
python run_paper_evaluations.py --config temporal_embeddings/config/paper_evaluation_config.json

# 4. Paired significance tests (Wilcoxon, t-test, bootstrap CI) — Tables 8-10
python statistical_significance.py --split test

# 5. Downstream RAG on TimeQA (Table 3), Qwen2.5-7B-Instruct generator
python run_rag_evaluation.py --config temporal_embeddings/config/rag_evaluation_config.json --split test

The rule-based + SUTime control (same fusion, normalization and alpha, temporal Gaussian from SUTime grounding instead of the learned encoder) is the sutime_hybrid block of paper_evaluation_config.json — set "enabled": true to include those rows. It requires the CoreNLP server from python setup.py.

Analysis scripts, all reusing the cached similarity pickles (no model inference):

# Alpha sensitivity curves (Figure 4)
python plot_paper_alpha_mrr.py --report output/metrics/paper_evaluation_report.md

# Explicit-timestamp oracle study: is the gain just year matching? (Table 7)
python regex_vs_temps.py --benchmark temp_reason --baseline intfloat/e5-base-v2 --alpha 0.8

# Timestamp-matching reducibility per benchmark
python benchmark_reducibility.py --semantic intfloat/e5-base-v2

# Where the gold passage moves on TimeQA (Tables 11-12)
python qualitative_analysis_timeqa.py --baseline intfloat/e5-base-v2 --alpha 0.6

Reports are written to output/metrics/, similarity caches to output/similarities/.

6. Supported Models
Model Type	Model Names
TEMPS (Temporal)	all-minilm-l6-v2, prajjwal1/bert-tiny
TEMPS + Semantic	all-minilm-l6-v2-full, prajjwal1/bert-tiny-full
Semantic Baselines	all-mpnet-base-v2, intfloat/e5-base-v2, BAAI/bge-large-en-v1.5, salesforce (SFR-Embedding-Mistral)
Other Baselines	bm25, sutime
Scope and Limitations
TEMPS covers temporal expressions that can be anchored and resolved to an interval: absolute dates, deictic expressions and offsets resolved against the anchor date, PRESENT_REF-style references, seasons, durations, and intervals. It does not do discourse-level reasoning — references needing an unobserved antecedent, unanchored vague ordering, and event–event relations (e.g. TempReason Level 3) are out of scope, and the pipeline inherits the tagger's misses. All experiments are in English. The retrieval unit is a passage of up to 512 tokens collapsed into a single Gaussian by attention pooling, which is lossy for long or temporally heterogeneous passages; passages with no temporal expression are held out of temporal training, so their Gaussians are uncalibrated and the semantic branch is what ranks them. Results come from a single training run (seed 42), with significance measured across queries rather than across seeds.

Citation
@inproceedings{hassani-etal-2026-temps,
  title     = {{TEMPS}: Temporal Sentence Embeddings for Temporal Information Retrieval},
  author    = {Hassani, Mourad and Romero, Julien and Bouzeghoub, Amel and Jacquelinet, Christian},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}

Acknowledgments
This project was provided with computing and storage resources by GENCI at IDRIS thanks to the grant 2025-105442 on the supercomputer Jean Zay's A100 partition.

License
The code in this repository is released under the MIT License, intended for research on temporal retrieval.

The trained TEMPS temporal module is a derivative of sentence-transformers/all-MiniLM-L6-v2 and is released under that model's Apache-2.0 license.

Third-party artifacts keep their own licenses and terms, and are used here strictly within their intended research scope: the TimeQA, TempReason and TS-Retriever benchmarks, the Multilingual MLM Temporal Tagging Resources, the pretrained backbones (MPNet-base-v2, E5-base-v2, BGE-large-en-v1.5, SFR-Embedding-Mistral, Qwen2.5-7B-Instruct), and Stanford CoreNLP / SUTime (GPL-3.0, used as a separate server by the rule-based control). The paper itself is published by the ACL under CC BY 4.0 and is not covered by the MIT license.
