from pathlib import Path
import json
from typing import List

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import compute_temporal_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def evaluate_temporal_model(temporal_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False, use_all_paragraphs: bool = False, reference_date: str = "09 august 2024") -> None:
    print(f"Starting temporal model evaluation for model: {temporal_model_name}")
    print(f"Model path: {temporal_model_path}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    temporal_cache_path, temporal_similarities_path, _, _ = set_output_files(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name="",
        benchmark_file_path=benchmark_file_path,
        eval_id=eval_id
    ).values()

    if not skip:
        print("Starting similarity computation phase...")
        
        temporal_similarities = compute_temporal_similarities(
            temporal_model_name=temporal_model_name,
            temporal_model_path=temporal_model_path,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            temporal_cache_file_path=temporal_cache_path,
            temporal_similarities_file_path=temporal_similarities_path,
            use_all_paragraphs=use_all_paragraphs,
            reference_date=reference_date
        )
    else:
        print("Skipping similarity computation - using existing results")

        print(f"Loading similarities from: {temporal_similarities_path}")
        temporal_similarities: pd.DataFrame = pd.read_pickle(temporal_similarities_path)

    similarities_list: List[List[float]] = temporal_similarities.to_numpy().tolist()
    
    print(f"Loaded {len(temporal_similarities)} similarity lists")
    
    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = json.load(f)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    print(compute_metrics(ground_truth, similarities_list, top_k, metric))
    print("Evaluation completed successfully")