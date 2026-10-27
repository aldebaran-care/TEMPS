# TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval

Code, trained temporal module, training-data generation pipeline, and evaluation scripts for
**"TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval"**
(Mourad Hassani, Julien Romero, Amel Bouzeghoub, Christian Jacquelinet), **EMNLP 2026**.

## Overview

Dense retrievers and RAG pipelines match queries to documents well on topic but poorly on
time, so they surface content that is on-topic yet temporally wrong. This work introduces:

- **Temporal Textual Similarity (TTS)** — a task that measures how well two *anchored* texts
  (text + anchor date) align in time, independent of their topical similarity.
- **TEMPS (Temporal Embedding Model for Precise Search)** — a modular temporal branch that
  attaches to a **frozen** semantic retriever. It resolves anchored temporal expressions to
  intervals, moment-matches each interval to a Gaussian, and uses the resulting ordering to
  supervise an anchor-date-conditioned encoder. At inference the temporal score is fused with
  the semantic score.

Supervision comes from grounding, so training uses **no hand-labeled temporal data**
(60M automatically generated tuples). The temporal score is the Gaussian-KL inclusion measure
from distributional embeddings; what TEMPS adds is the grounding and the moment-matched
supervision.

### Method in brief

Each grounded interval `I = [a, b]` (inclusive day range) is moment-matched to a Gaussian:

```
mu    = (a + b + 1) / 2
sigma^2 = (b - a + 1)^2 / 12
```

The label is the directional divergence `TSF(I1 -> I2) = KL(p_I1 || p_I2)`, bounded as
`ts = 1 / (1 + TSF)`. The model emits a diagonal Gaussian per (text, anchor date) and is
trained with a CoSENT ranking loss to reproduce that ordering. At inference:

```
s(1,2) = (1 - alpha) * cos(E_1, E_2) + alpha * 1 / (1 + KL(p_1 || p_2))
```

with the two branches min-max normalized per query before fusion. `alpha` is a
benchmark-level calibration parameter selected on a held-out validation split
(0.6 TimeQA, 0.8 TempReason, 0.3 TS-Retriever).

Architecture: a frozen `all-MiniLM-L6-v2` backbone (512 tokens) + a sinusoidal anchor-date
encoder, attention pooling, and mean/variance heads. The temporal module adds ~27M parameters.

## Results

Highlights on the held-out test splits (full tables in the paper):

| Benchmark | Best system | MRR | Gain over backbone |
|---|---|---|---|
| TimeQA (alpha=0.6) | Mistral + TEMPS | 0.5360 | +31.9% |
| TempReason (alpha=0.8) | Mistral + SUTime control | 0.6901 | +11.5% |
| TS-Retriever (alpha=0.3) | Mistral + TEMPS | 0.8240 | +9.3% |

Against the prior temporal state of the art on TS-Retriever, `E5base-v2 + TEMPS` lifts
**R@1 from 19.92 to 25.39** and P@1 from 58.63 to 69.14 over TSContriever. On downstream
TimeQA RAG (top-5 passages, Qwen2.5-7B-Instruct) every `+TEMPS` retriever improves EM, F1
and containment over its unaugmented counterpart.

TempReason is reported as-is rather than tuned around: at the validation-selected alpha=0.8
the fused score is temporal-dominated, which sharpens the top of the ranking but costs
Recall@5 for all four backbones, and the rule-based SUTime control outperforms TEMPS there.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from `pyproject.toml`; on
Linux the lockfile pulls the CUDA 12.8 PyTorch wheels (`cu128`).

```bash
uv sync                    # training, evaluation, analysis
uv sync --extra sutime     # + the rule-based SUTime control (Stanza / CoreNLP)
```

Run every script with `uv run`, e.g. `uv run python evaluate.py ...`. The SUTime control also
needs the CoreNLP server: `uv run python scripts/install_corenlp.py`.

## Pretrained Model

The trained TEMPS temporal module (frozen `all-MiniLM-L6-v2` backbone + temporal head) is on
the Hugging Face Hub at [aldebaran-research/temps](https://huggingface.co/aldebaran-research/temps). This saves it under `output/trained_models/` with the file name the
paper configs expect:

```bash
uv run python scripts/download_model.py
```

## Usage

### 1. Creating the Training Dataset

Training labels come from grounding, not hand annotation: each generator resolves both
expressions of a pair to day intervals and labels the pair with the moment-matched Gaussian
KL score (`temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py`). The
pipeline runs in this order:

```bash
# 1. Synthetic TimeML-style expressions (years 1000-2030), rendered with templates and paired
#    with random anchor dates (55M tuples in the paper)
uv run python create_synthetic_dataset.py \
    --output_file_path data/new_training_dataset/synthetic_dataset/synthetic_dataset.csv --size 400000

# 2. Anchored expressions from the English Multilingual MLM Temporal Tagging Resources
#    (Wikipedia text, JSON; 5M tuples in the paper)
uv run python create_real_world_dataset.py \
    --input_path <temporal-tagging-resources.json> \
    --output_path data/new_training_dataset/real_world_dataset/dataset.csv

# 3. Pairs expressing temporal relations, the kind of reasoning TempReason tests
uv run python create_temporal_relationships_dataset.py \
    --output_file_path data/new_training_dataset/synthetic_dataset/temporal_relationships.csv

# 4. Merge and shuffle with bounded memory into ./merged_training_data.csv
uv run python merge_training_dataset.py \
    data/new_training_dataset/synthetic_dataset/synthetic_dataset.csv \
    data/new_training_dataset/real_world_dataset/dataset.csv \
    data/new_training_dataset/synthetic_dataset/temporal_relationships.csv
```

The result is a CSV of tuples `(sent0, sent0_date, sent1, sent1_date, score)`.

### 2. Training the Model

```bash
# Single GPU
uv run python train.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --input_file_path merged_training_data.csv \
    --epochs 1 --batch_size 128 --lr 3e-4

# Multi-GPU (one node)
uv run torchrun --standalone --nproc_per_node=8 train.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --input_file_path merged_training_data.csv \
    --epochs 1 --batch_size 2048
```

Paper configuration: 3 epochs over the full 60M-tuple corpus, global batch size 2,048,
lr 3e-4, weight decay 0.01, 5% linear warmup, max 512 tokens, fp16, seed 42 — about
4 wall-clock hours on 8×A100-80GB. Run `uv run python train.py --help` for all arguments.

#### SLURM clusters

Example jobs live in [`scripts/slurm/`](scripts/slurm/). Fill in the commented-out
`##SBATCH` account / partition / constraint lines for your cluster and submit from the
repository root (`sbatch scripts/slurm/train.sh`). If compute nodes have no internet, cache
the backbone on a login node first; the jobs then run with `HF_HUB_OFFLINE=1`:

```bash
uv run python scripts/prefetch_model.py sentence-transformers/all-MiniLM-L6-v2
```

### 3. Building the Evaluation Benchmarks

The benchmarks are not redistributed. Download the originals into `data/evaluation/`:

| Benchmark | Source | Expected files |
|---|---|---|
| TimeQA (Chen et al., 2021) | [wenhuchen/Time-Sensitive-QA](https://github.com/wenhuchen/Time-Sensitive-QA) | `time_sensitive_qa/human_annotated_test.json` |
| TempReason (Tan et al., 2023) | [DAMO-NLP-SG/TempReason](https://github.com/DAMO-NLP-SG/TempReason) | `temp_reason_l1/data.json`, `temp_reason_l2/data.json` |
| TS-Retriever (Wu et al., 2024) | [suzhou-22/TS-Retriever](https://github.com/suzhou-22/TS-Retriever/tree/main/evaluation/data/nobel_prize) (`evaluation/data/nobel_prize/`) | `ts_retriever/query.json`, `ts_retriever/doc.json` |

Then build them and tag the 20% validation / 80% test split:

```bash
uv run python create_evaluation_dataset.py time_sensitive_qa
uv run python create_evaluation_dataset.py temp_reason_l1
uv run python create_evaluation_dataset.py temp_reason_l2
uv run python create_evaluation_dataset.py temp_reason      # merges levels 1 and 2
uv run python create_evaluation_dataset.py ts_retriever

uv run python split_and_tune_alpha.py tag                   # deterministic: --seed 42
```

[`splits/`](splits/) lists the paper's split for each benchmark as `(index, sha1, split)`, where
`sha1` hashes each item's `question`, `paragraphs` and `answer`. Use it to check that your
rebuilt benchmarks, and therefore the tagged splits, are identical to ours.

### 4. Evaluating Models

`evaluate.py` scores one model on one benchmark:

```bash
# TEMPS alone
uv run python evaluate.py --model_name all-minilm-l6-v2 \
    --model_path output/trained_models/<model>.pth --benchmark temp_reason --top_k 10 --metric all

# Semantic baseline
uv run python evaluate.py --model_name intfloat/e5-base-v2 --benchmark temp_reason --top_k 10 --metric all

# Semantic + TEMPS fusion
uv run python evaluate.py --model_name all-minilm-l6-v2-full --external_model_name intfloat/e5-base-v2 \
    --model_path output/trained_models/<model>.pth --benchmark temp_reason --alpha 0.8 --top_k 10 --metric all
```

Run `uv run python evaluate.py --help` for all arguments.

### 5. Reproducing the Paper

The protocol: tag each benchmark once into 20% validation / 80% test, select one `alpha` per
benchmark on validation MRR alone, then score the test split once. The first run embeds every
benchmark with every model (GPU hours, dominated by TS-Retriever and SFR-Embedding-Mistral)
and caches the similarities under `output/similarities/`; every later step reuses that cache.

```bash
# 1. Sweep alpha on the validation split only, one value per benchmark
uv run python split_and_tune_alpha.py tune-alpha

# 2. Main results table (Table 1) on the test split
uv run python run_paper_evaluations.py --config temporal_embeddings/config/paper_evaluation_config.json

# 3. Paired significance tests (Wilcoxon, t-test, bootstrap CI) — Tables 8-10
uv run python statistical_significance.py --split test

# 4. Downstream RAG on TimeQA (Table 3), Qwen2.5-7B-Instruct via OpenRouter
#    (set OPENROUTER_API_KEY in .env)
uv run python run_rag_evaluation.py --config temporal_embeddings/config/rag_evaluation_config.json --split test
```

The rule-based **+ SUTime control** (same fusion, normalization and alpha, temporal Gaussian
from SUTime grounding instead of the learned encoder) is the `sutime_hybrid` block of
`paper_evaluation_config.json`; set `"enabled": true` to include those rows. It needs the
`sutime` extra and the CoreNLP server (see Installation).

Analysis scripts, all reusing the cached similarities (no model inference):

```bash
# Alpha sensitivity curves (Figure 4)
uv run python plot_paper_alpha_mrr.py --report output/metrics/paper_evaluation_report.md

# Explicit-timestamp oracle study: is the gain just year matching? (Table 7)
uv run python regex_vs_temps.py --benchmark temp_reason --baseline intfloat/e5-base-v2 --alpha 0.8

# Timestamp-matching reducibility per benchmark
uv run python benchmark_reducibility.py --semantic intfloat/e5-base-v2

# Where the gold passage moves on TimeQA (Tables 11-12)
uv run python qualitative_analysis_timeqa.py --baseline intfloat/e5-base-v2 --alpha 0.6
```

Reports are written to `output/metrics/`.

### 6. Supported Models

| Model Type | Model Names |
|------------|-------------|
| **TEMPS (Temporal)** | `all-minilm-l6-v2`, `prajjwal1/bert-tiny` |
| **TEMPS + Semantic** | `all-minilm-l6-v2-full`, `prajjwal1/bert-tiny-full` |
| **Semantic Baselines** | `all-mpnet-base-v2`, `intfloat/e5-base-v2`, `BAAI/bge-large-en-v1.5`, `salesforce` (SFR-Embedding-Mistral) |
| **Other Baselines** | `bm25`, `sutime` |

## Scope and Limitations

TEMPS covers temporal expressions that can be **anchored and resolved to an interval**:
absolute dates, deictic expressions and offsets resolved against the anchor date,
`PRESENT_REF`-style references, seasons, durations, and intervals. It does not do
discourse-level reasoning — references needing an unobserved antecedent, unanchored vague
ordering, and event–event relations (e.g. TempReason Level 3) are out of scope, and the
pipeline inherits the tagger's misses. All experiments are in English. The retrieval unit is
a passage of up to 512 tokens collapsed into a single Gaussian by attention pooling, which is
lossy for long or temporally heterogeneous passages; passages with no temporal expression are
held out of temporal training, so their Gaussians are uncalibrated and the semantic branch is
what ranks them. Results come from a single training run (seed 42), with significance measured
across queries rather than across seeds.

## Citation

```bibtex
@inproceedings{hassani-etal-2026-temps,
  title     = {{TEMPS}: Temporal Sentence Embeddings for Temporal Information Retrieval},
  author    = {Hassani, Mourad and Romero, Julien and Bouzeghoub, Amel and Jacquelinet, Christian},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgments

This project was provided with computing and storage resources by GENCI at IDRIS thanks to the
grant 2025-105442 on the supercomputer Jean Zay's A100 partition.

## License

The code in this repository is released under the [MIT License](LICENSE), intended for research
on temporal retrieval.

The trained TEMPS temporal module is a derivative of
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and is released under that model's **Apache-2.0** license.

Third-party artifacts keep their own licenses and terms, and are used here strictly within
their intended research scope: the TimeQA, TempReason and TS-Retriever benchmarks, the
Multilingual MLM Temporal Tagging Resources, the pretrained backbones (MPNet-base-v2,
E5-base-v2, BGE-large-en-v1.5, SFR-Embedding-Mistral, Qwen2.5-7B-Instruct), and
Stanford CoreNLP / SUTime (GPL-3.0, used as a separate server by the rule-based control).
The paper itself is published by the ACL under CC BY 4.0 and is not covered by the MIT license.
