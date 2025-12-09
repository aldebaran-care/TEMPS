from pathlib import Path
import json
from typing import Dict, List

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.evaluation.sutime.compute_sutime_similarities import compute_sutime_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion

def evaluate_sutime(model_name: str, benchmark: str, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool) -> None:
    print(f"Starting SUTime evaluation with model: {model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    temporal_cache_path, temporal_similarities_path, _, _ = set_output_files(
        temporal_model_name="sutime",
        temporal_model_path=Path(""),
        semantic_model_name="",
        benchmark=benchmark,
        eval_id=eval_id
    ).values()

    if not temporal_similarities_path.exists():
        print("Starting two-stage SUTime evaluation...")
        
        sutime_similarities = compute_sutime_similarities(
            benchmark_file_path=benchmark_file_path,
            cache_file_path=temporal_cache_path,
            similarities_file_path=temporal_similarities_path
        )
    else:
        print("Skipping similarity computation - using existing results")

        print(f"Loading similarities from: {temporal_similarities_path}")
        sutime_similarities: pd.DataFrame = pd.read_pickle(temporal_similarities_path)
        print(f"Loaded {len(sutime_similarities)} similarity entries")

    similarities_list: List[List[float]] = sutime_similarities.to_numpy().tolist()

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data: List[Dict] = json.load(f)

        for element in benchmark_data:
            ground_truth.append(element["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")

    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    
    results: Dict[str, float]= compute_metrics(ground_truth, similarities_list, top_k, metric)
    log_metrics_to_notion(id=str(eval_id), model=model_name, benchmark=benchmark, metrics=results, k=top_k)

    print(results)
    print("SUTime evaluation completed successfully")
