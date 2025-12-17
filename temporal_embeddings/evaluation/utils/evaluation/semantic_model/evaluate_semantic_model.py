from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import compute_semantic_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def evaluate_semantic_model(semantic_model_name: str, max_seq_len: int, benchmark: str, benchmark_file_path: Path, eval_id: str, top_k: int, metric: str, num_negative_samples: int = 0) -> None:
    print(f"Starting semantic model evaluation for model: {semantic_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    _, _, semantic_cache_path, semantic_similarities_path = set_output_files(
        temporal_model_name="",
        temporal_model_path=Path(""),
        semantic_model_name=semantic_model_name,
        benchmark=benchmark,
    ).values()

    if not semantic_similarities_path.exists():
        print("Starting similarity computation phase...")
        
        semantic_similarities = compute_semantic_similarities(
            semantic_model_name=semantic_model_name,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            semantic_cache_file_path=semantic_cache_path,
            semantic_similarities_file_path=semantic_similarities_path,
            num_negative_samples=num_negative_samples
        )
    else:
        print("Skipping similarity computation - using existing results")

        print(f"Loading similarities from: {semantic_similarities_path}")
        semantic_similarities: pd.DataFrame = pd.read_pickle(semantic_similarities_path)

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = add_negative_samples(json.load(f), num_negatives=num_negative_samples)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    
    print("Filtering similarities to only include candidate paragraphs...")
    similarities_list: List[List[float]] = []
    
    for _, benchmark_item in enumerate(benchmark_data):
        question_similarities = semantic_similarities.loc[benchmark_item["question"]][benchmark_item["paragraphs"]].tolist()
        similarities_list.append(question_similarities)
    
    print(f"Filtered to {len(similarities_list)} similarity lists with candidate paragraphs only")
    
    print(f"Computing metrics with top_k={top_k}, metric={metric}, num_negative_samples={num_negative_samples}")
    
    results: Dict[str, float]= compute_metrics(ground_truth, similarities_list, top_k, metric)
    log_metrics_to_notion(id=eval_id, model=semantic_model_name, benchmark=benchmark, metrics=results, k=top_k, num_negative_samples=num_negative_samples)
    
    print(results)
    print("Evaluation completed successfully")
