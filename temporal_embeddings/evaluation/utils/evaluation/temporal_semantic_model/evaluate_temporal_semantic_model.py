from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics, compute_metrics_ranks
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import compute_temporal_similarities
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import compute_semantic_similarities
from temporal_embeddings.utils.math.normalize import normalize_list
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion

def evaluate_temporal_semantic_model(temporal_model_name: str, semantic_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark: str, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, alpha: float = 0.5, num_negative_samples: int = 0) -> None:
    print(f"Starting temporal semantic model evaluation")
    print(f"Temporal model: {temporal_model_name} at {temporal_model_path}")
    print(f"Semantic model: {semantic_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    temporal_cache_path, temporal_similarities_path, semantic_cache_path, semantic_similarities_path = set_output_files(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name=semantic_model_name,
        benchmark=benchmark,
        eval_id=eval_id
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
            use_all_paragraphs=use_all_paragraphs
        )

    else:
        print("Skipping model runs - using existing temporal similarities")

        print("Loading temporal similarities for fusion...")
        temporal_similarities: pd.DataFrame = pd.read_pickle(temporal_similarities_path).to_numpy().tolist()

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
        semantic_similarities: pd.DataFrame = pd.read_pickle(semantic_similarities_path).to_numpy().tolist()
    
    print(f"Loaded {len(semantic_similarities)} semantic similarity lists")

    print("Using score normalization and weighted fusion...")

    print("Normalizing similarity scores...")
    temporal_similarities = normalize_list(temporal_similarities)
    semantic_similarities = normalize_list(semantic_similarities)

    print("Merging similarities with weights (temporal: 1x, external: 2x)...")
    merged_list = [[((alpha*x) + ((1-alpha)*y)) for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(temporal_similarities, semantic_similarities)]

    merged_similarities: List[List[float]] = merged_list
    print("Score fusion completed")
    
    print("Loading ground truth for score evaluation...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = json.load(f)

        for e in benchmark_data:
            ground_truth.append(e["answer"])

    print(f"Computing score metrics with top_k={top_k}, metric={metric}")

    results: Dict[str, float]= compute_metrics(ground_truth, merged_similarities, top_k, metric)
    log_metrics_to_notion(id=str(eval_id), model=temporal_model_name, external_model=semantic_model_name, benchmark=benchmark, metrics=results, k=top_k, alpha=alpha)

    print(results)
    print("Temporal semantic model evaluation completed successfully")
