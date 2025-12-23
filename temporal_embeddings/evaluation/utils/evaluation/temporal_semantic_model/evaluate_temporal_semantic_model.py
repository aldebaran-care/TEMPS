from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import compute_temporal_similarities
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import compute_semantic_similarities
from temporal_embeddings.utils.math.normalize import normalize_list
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def reciprocal_rank_fusion(temporal_sims: pd.DataFrame, semantic_sims: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    temporal_ranks = temporal_sims.rank(axis=1, method='min', ascending=False)
    semantic_ranks = semantic_sims.rank(axis=1, method='min', ascending=False)
    
    rrf_scores = 1 / (k + temporal_ranks) + 1 / (k + semantic_ranks)
    
    return rrf_scores

def evaluate_temporal_semantic_model(temporal_model_name: str, semantic_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark: str, benchmark_file_path: Path, eval_id: str, top_k: int, metric: str, alpha: float = 0.5, num_negative_samples: int = 0) -> None:
    print(f"Starting temporal semantic model evaluation")
    print(f"Temporal model: {temporal_model_name} at {temporal_model_path}")
    print(f"Semantic model: {semantic_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    temporal_cache_path, temporal_similarities_path, semantic_cache_path, semantic_similarities_path = set_output_files(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name=semantic_model_name,
        benchmark=benchmark,
    ).values()

    if not temporal_similarities_path.exists():
        print("Running temporal model (not skipping)...")

        temporal_similarities: pd.DataFrame = compute_temporal_similarities(
            temporal_model_name=temporal_model_name,
            temporal_model_path=temporal_model_path,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            temporal_cache_file_path=temporal_cache_path,
            temporal_similarities_file_path=temporal_similarities_path,
            num_negative_samples=num_negative_samples
        )

    else:
        print("Skipping model runs - using existing temporal similarities")

        print("Loading temporal similarities for fusion...")
        temporal_similarities: pd.DataFrame = pd.read_pickle(temporal_similarities_path)

    print(f"Loaded {len(temporal_similarities)} temporal similarity lists")

    if not semantic_similarities_path.exists():
        print("Running semantic model (not skipping)...")

        semantic_similarities: pd.DataFrame = compute_semantic_similarities(
            semantic_model_name=semantic_model_name,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            semantic_cache_file_path=semantic_cache_path,
            semantic_similarities_file_path=semantic_similarities_path,
            num_negative_samples=num_negative_samples
        )

    else:
        print("Skipping model runs - using existing semantic similarities")

        print("Loading semantic similarities for fusion...")
        semantic_similarities: pd.DataFrame = pd.read_pickle(semantic_similarities_path)
    
    print(f"Loaded {len(semantic_similarities)} semantic similarity lists")
    
    print("Loading ground truth for score evaluation...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = add_negative_samples(json.load(f), num_negatives=num_negative_samples)

        for e in benchmark_data:
            ground_truth.append(e["answer"])

    print(f"Loaded ground truth for {len(ground_truth)} items")

    if alpha == -1:
        print("Merging similarities using Reciprocal Rank Fusion...")
        merged_similarities: pd.DataFrame = reciprocal_rank_fusion(temporal_similarities, semantic_similarities)
    else:
        print(f"Merging similarities using linear combination with alpha={alpha}...")
        merged_similarities: pd.DataFrame = (alpha * temporal_similarities) + ((1 - alpha) * semantic_similarities)
    
    print("Filtering merged similarities to only include candidate paragraphs...")
    filtered_similarities: List[List[float]] = []
    
    for _, benchmark_item in enumerate(benchmark_data):
        question_similarities = merged_similarities.loc[benchmark_item["question"]][benchmark_item["paragraphs"]].tolist()
        filtered_similarities.append(question_similarities)
    
    print(f"Filtered to {len(filtered_similarities)} similarity lists with candidate paragraphs only")

    print(f"Computing score metrics with top_k={top_k}, metric={metric}")

    results: Dict[str, float]= compute_metrics(ground_truth, filtered_similarities, top_k, metric)
    log_metrics_to_notion(id=eval_id, model=temporal_model_name, external_model=semantic_model_name, benchmark=benchmark, metrics=results, k=top_k, alpha=alpha, num_negative_samples=num_negative_samples)

    print(results)
    print("Temporal semantic model evaluation completed successfully")
