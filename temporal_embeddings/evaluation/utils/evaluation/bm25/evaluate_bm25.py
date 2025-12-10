from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.bm25.compute_bm25_similarities import compute_bm25_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def evaluate_bm25(bm25_model_name: str, benchmark: str, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, num_negative_samples: int = 0) -> None:
    print(f"Starting BM25 evaluation for model: {bm25_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    _, _, bm25_cache_path, bm25_similarities_path = set_output_files(
        temporal_model_name="",
        temporal_model_path=Path(""),
        semantic_model_name=bm25_model_name,
        benchmark=benchmark,
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

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = add_negative_samples(json.load(f), num_negative_samples=num_negative_samples)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    
    print("Filtering similarities to only include candidate paragraphs...")
    similarities_list: List[List[float]] = []
    
    for idx, benchmark_item in enumerate(benchmark_data):
        candidate_paragraphs = benchmark_item["paragraphs"]
        question_similarities = bm25_similarities.iloc[idx][candidate_paragraphs].tolist()
        similarities_list.append(question_similarities)
    
    print(f"Filtered to {len(similarities_list)} similarity lists with candidate paragraphs only")
    
    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    
    results: Dict[str, float] = compute_metrics(ground_truth, similarities_list, top_k, metric)
    log_metrics_to_notion(id=str(eval_id), model=bm25_model_name, benchmark=benchmark, metrics=results, k=top_k, num_negative_samples=num_negative_samples)
    
    print(results)
    print("Evaluation completed successfully")