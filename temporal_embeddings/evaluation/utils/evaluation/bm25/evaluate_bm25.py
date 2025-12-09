from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.bm25.compute_bm25_similarities import compute_bm25_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion

def evaluate_bm25(bm25_model_name: str, benchmark: str, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str) -> None:
    print(f"Starting BM25 evaluation for model: {bm25_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    _, _, bm25_cache_path, bm25_similarities_path = set_output_files(
        temporal_model_name="",
        temporal_model_path=Path(""),
        semantic_model_name=bm25_model_name,
        benchmark=benchmark,
        eval_id=eval_id
    ).values()

    if not bm25_similarities_path.exists():
        print("Starting BM25 similarity computation phase...")
        
        bm25_similarities = compute_bm25_similarities(
            bm25_model_name=bm25_model_name,
            benchmark_file_path=benchmark_file_path,
            bm25_cache_file_path=bm25_cache_path,
            bm25_similarities_file_path=bm25_similarities_path
        )
    else:
        print("Skipping BM25 similarity computation - using existing results")

        print(f"Loading similarities from: {bm25_similarities_path}")
        bm25_similarities: pd.DataFrame = pd.read_pickle(bm25_similarities_path)

    similarities_list: List[List[float]] = bm25_similarities.to_numpy().tolist()
    
    print(f"Loaded {len(bm25_similarities)} similarity lists")
    
    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = json.load(f)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    
    results: Dict[str, float] = compute_metrics(ground_truth, similarities_list, top_k, metric)
    log_metrics_to_notion(id=str(eval_id), model=bm25_model_name, benchmark=benchmark, metrics=results, k=top_k)
    
    print(results)
    print("Evaluation completed successfully")