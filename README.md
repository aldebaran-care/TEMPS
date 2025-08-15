# TEMPS: Temporal Embedding Model for Precise Search

## Overview

TEMPS addresses a critical gap in modern information retrieval systems: the lack of temporal awareness. While contemporary Retrieval-Augmented Generation (RAG) frameworks and dense retrieval models excel at semantic matching, they struggle with temporal reasoning, often retrieving topically relevant but temporally misaligned content.

This repository implements **Temporal Textual Similarity (TTS)**, a novel task focused on modeling and evaluating temporal relationships in natural language to enhance retrieval performance.

## Key Features

- **Temporal-Aware Embeddings**: Specialized architecture that encodes temporal information and grounds document content to anchor dates
- **Weakly Supervised Training**: Rule-based temporal annotation with synthetic data generation for scalable learning
- **Symbolic Temporal Representations**: Learning of temporal patterns without extensive manual labeling
- **Fine-Grained Temporal Reasoning**: Improved alignment between temporally sensitive queries and documents

## Architecture

TEMPS consists of three main components:

1. **Temporal Annotation Pipeline**: Rule-based system for temporal expression recognition and normalization
2. **Synthetic Data Generation**: Automated creation of temporally-aware training data
3. **Temporal Embedding Model**: Specialized architecture for encoding temporal relationships

## Data Pipeline

### Temporal Expression Processing

The system processes various temporal expressions:

- **Offsets**: Tomorrow, last week, next month
- **References**: This month, next year, yesterday
- **Intervals**: Date ranges and temporal spans
- **Explicit Dates**: ISO format dates with temporal grounding

### Text Generation

Temporal expressions are automatically converted to natural language variations for training data diversity.

## Project Structure

```
temporal-embeddings/
├── temporal_embeddings/
│   ├── data_utils/              # Data processing utilities
│   ├── synthetic_data/          # Synthetic data generation
│   ├── evaluation/              # Evaluation frameworks
│   ├── inference/               # Model inference tools
│   └── utils/                   # General utilities
├── data/
│   ├── evaluation/              # Benchmark datasets
│   └── synthetic/               # Generated datasets
├── models/                      # Trained models
├── output/                      # Evaluation results
└── configs/                     # Configuration files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request