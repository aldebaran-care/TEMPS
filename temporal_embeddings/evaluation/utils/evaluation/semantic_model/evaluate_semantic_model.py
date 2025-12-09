from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import compute_semantic_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion

def evaluate_semantic_model(semantic_model_name: str, max_seq_len: int, benchmark: str, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, use_all_paragraphs: bool = False) -> None:
    print(f"Starting semantic model evaluation for model: {semantic_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    _, _, semantic_cache_path, semantic_similarities_path = set_output_files(
        temporal_model_name="",
        temporal_model_path=Path(""),
        semantic_model_name=semantic_model_name,
        benchmark=benchmark,
        eval_id=eval_id
    ).values()

    if not semantic_similarities_path.exists():
        print("Starting similarity computation phase...")
        
        semantic_similarities = compute_semantic_similarities(
            semantic_model_name=semantic_model_name,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            semantic_cache_file_path=semantic_cache_path,
            semantic_similarities_file_path=semantic_similarities_path,
            use_all_paragraphs=use_all_paragraphs
        )
    else:
        print("Skipping similarity computation - using existing results")

        print(f"Loading similarities from: {semantic_similarities_path}")
        semantic_similarities: pd.DataFrame = pd.read_pickle(semantic_similarities_path)

    similarities_list: List[List[float]] = semantic_similarities.to_numpy().tolist()
    
    print(f"Loaded {len(semantic_similarities)} similarity lists")
    
    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = json.load(f)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    
    results: Dict[str, float]= compute_metrics(ground_truth, similarities_list, top_k, metric)

    log_metrics_to_notion(id=eval_id, model=semantic_model_name, benchmark=benchmark, metrics=results, k=top_k)
    
    print()
    print("Evaluation completed successfully")
