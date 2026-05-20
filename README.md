# TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval

This repository contains the code and resources for the research paper **"TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval"**.

## Abstract

Temporal information is a critical yet underrepresented dimension in modern information retrieval (IR) systems, particularly in domains where the timing of events significantly influences interpretation and decision-making. While contemporary Retrieval-Augmented Generation (RAG) frameworks and dense retrieval models excel in semantic matching, they struggle with temporal reasoning due to the limited representation of temporal information during training. This results in the retrieval of topically relevant but temporally misaligned content, undermining the effectiveness of downstream applications.

To address this gap, we introduce **Temporal Textual Similarity (TTS)** as a novel task focused on modeling and evaluating temporal relationships in natural language to enhance retrieval performance. We present **TEMPS (Temporal Embedding Model for Precise Search)**, a pipeline to add temporal awareness to an existing language model to obtain better temporal information retrieval.

TEMPS is composed of:
- A **weakly supervised training pipeline** that integrates rule-based temporal annotation with synthetic data generation, enabling scalable learning of symbolic temporal representations without the need for extensive manual labeling.
- A **specialized embedding architecture** that encodes temporal information and effectively grounds document content to an anchor date, allowing for fine-grained temporal reasoning and improved alignment between temporally sensitive queries and documents.

Our approach addresses key limitations in existing retrieval systems and demonstrates substantial improvements in time-aware retrieval tasks.

## Installation

### Training environment (SLURM / GPU servers)

Training dependencies are managed with [uv](https://docs.astral.sh/uv/) from `pyproject.toml`. The lockfile pulls the CUDA 12.8 PyTorch wheels (`cu128`).

```bash
uv sync
```

This creates a `.venv/` with everything needed to run `train.py`. Invoke scripts with `uv run`, e.g. `uv run python train.py ...` or `uv run torchrun ...` (see `train.sh`).

### Data preparation / annotation environment (local)

The data prep and annotation scripts (`create_*.py`, `add_score_v2_to_training_data.py`, etc.) have additional dependencies (spaCy, stanza, NLTK, sentence-transformers, datatrove, ...). For that workflow use the legacy `requirements.txt`:

```bash
pip install -r requirements.txt
python setup.py  # installs the Stanza CoreNLP server
```

## Usage

### 1. Creating the Training Dataset

The training pipeline uses a weakly supervised approach combining rule-based temporal annotation with synthetic data generation.

#### Step 1: Generate Synthetic Data

```bash
python create_synthetic_dataset.py \
    --output_file_path data/synthetic_data/synthetic_dataset.csv \
    --size 400000
```

#### Step 2: Create Real-World Dataset

Extract temporal relationships from real-world data:

```bash
python create_real_dataset.py --num_rows 1000000 --skip 0
```

#### Step 3: Merge and Prepare Training Data

Combine synthetic and real-world datasets:

```bash
python create_training_dataset.py
```

This script merges data from:
- `data/new_training_dataset/synthetic_dataset/synthetic_dataset.csv`
- `data/new_training_dataset/synthetic_dataset/temporal_relationships.csv`
- `data/new_training_dataset/real_world_dataset/dataset.csv`

Output is saved to `data/new_training_dataset/training_datasets/`.

### 2. Training the Model

Train the TEMPS model using distributed training with PyTorch:

```bash
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
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `prajjwal1/bert-tiny` | Base model name |
| `--batch_size` | `64` | Training batch size |
| `--lr` | `3e-4` | Learning rate |
| `--epochs` | `1` | Number of training epochs |
| `--weight_decay` | `0.01` | Weight decay |
| `--num_warmup_ratio` | `0.05` | Warmup ratio for scheduler |
| `--num_eval_steps` | `1000` | Evaluation frequency |
| `--data_fraction` | `1.0` | Fraction of data to use |
| `--continue_training` | `False` | Resume from checkpoint |
| `--model_path` | `None` | Path to checkpoint for resuming |

#### SLURM Cluster Training

```bash
sbatch train.sh
```

### 3. Creating Evaluation Benchmarks

Generate evaluation datasets for different benchmarks:

```bash
# Time-Sensitive QA benchmark
python create_evaluation_dataset.py time_sensitive_qa

# TempReason benchmark
python create_evaluation_dataset.py temp_reason

# TS-Retriever benchmark
python create_evaluation_dataset.py ts_retriever
```

### 4. Evaluating Models

#### Evaluate TEMPS (Temporal Model)

```bash
python evaluate.py \
    --model_name all-minilm-l6-v2 \
    --model_path output/trained_models/model.pth \
    --benchmark temp_reason \
    --batch_size 128 \
    --max_seq_len 512 \
    --top_k 10 \
    --metric all
```

#### Evaluate Semantic Baseline Models

```bash
# Evaluate sentence-transformers models
python evaluate.py \
    --model_name all-mpnet-base-v2 \
    --benchmark temp_reason \
    --top_k 10 \
    --metric all

# Evaluate other models
python evaluate.py \
    --model_name intfloat/e5-base-v2 \
    --benchmark temp_reason \
    --top_k 10 \
    --metric all
```

#### Evaluate Combined Temporal + Semantic Model

```bash
python evaluate.py \
    --model_name all-minilm-l6-v2-full \
    --external_model_name intfloat/e5-base-v2 \
    --model_path output/trained_models/model.pth \
    --benchmark temp_reason \
    --alpha 0.5 \
    --top_k 10 \
    --metric all
```

#### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `""` | Model to evaluate |
| `--model_path` | `""` | Path to trained model weights |
| `--benchmark` | `""` | Benchmark dataset name |
| `--batch_size` | `32` | Evaluation batch size |
| `--max_seq_len` | `128` | Maximum sequence length |
| `--top_k` | `1` | Top-k for retrieval metrics |
| `--metric` | `top` | Metric type (`all`, `top`, `mrr`, `ndcg`, `precision`, `recall`, `f1`) |
| `--alpha` | `0.5` | Weight for combining temporal and semantic scores |
| `--external_model_name` | `""` | External semantic model for hybrid evaluation |
| `--num_negative_samples` | `0` | Number of negative samples |

#### Batch Evaluation

Run comprehensive evaluation across multiple models and benchmarks:

```bash
sbatch evaluate.sh
```

### 5. Supported Models

| Model Type | Model Names |
|------------|-------------|
| **TEMPS (Temporal)** | `all-minilm-l6-v2`, `prajjwal1/bert-tiny` |
| **TEMPS + Semantic** | `all-minilm-l6-v2-full`, `prajjwal1/bert-tiny-full` |
| **Semantic Baselines** | `all-mpnet-base-v2`, `intfloat/e5-base-v2`, `BAAI/bge-large-en-v1.5` |
| **Other Baselines** | `bm25`, `sutime`, `salesforce` |

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{temps2026,
  title={TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval},
  author={},
  journal={},
  year={2026}
}
```

## License

*TBD*