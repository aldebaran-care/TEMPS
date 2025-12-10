from pathlib import Path
import json
from typing import List, Dict

import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import compute_temporal_similarities
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.notion.notion import log_metrics_to_notion
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def evaluate_temporal_model(temporal_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark: str, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, num_negative_samples: int = 0, reference_date: str = "09 august 2024") -> None:
    print(f"Starting temporal model evaluation for model: {temporal_model_name}")
    print(f"Model path: {temporal_model_path}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    temporal_cache_path, temporal_similarities_path, _, _ = set_output_files(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name="",
        benchmark=benchmark,
    ).values()

    if not temporal_similarities_path.exists():
        print("Starting similarity computation phase...")
        
        temporal_similarities = compute_temporal_similarities(
            temporal_model_name=temporal_model_name,
            temporal_model_path=temporal_model_path,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            temporal_cache_file_path=temporal_cache_path,
            temporal_similarities_file_path=temporal_similarities_path,
            num_negative_samples=num_negative_samples,
            reference_date=reference_date
        )
    else:
        print("Skipping similarity computation - using existing results")

        print(f"Loading similarities from: {temporal_similarities_path}")
        temporal_similarities: pd.DataFrame = pd.read_pickle(temporal_similarities_path)

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = add_negative_samples(json.load(f), num_negative_samples=num_negative_samples)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    
    print("Filtering similarities to only include candidate paragraphs...")
    similarities_list: List[List[float]] = []
    
    for _, benchmark_item in enumerate(benchmark_data):
        candidate_paragraphs = benchmark_item["paragraphs"]
        question_similarities_series = temporal_similarities.loc[benchmark_item["question"]]

        question_similarities = []
        for paragraph in candidate_paragraphs:
            if paragraph in question_similarities_series.index:
                question_similarities.append(question_similarities_series[paragraph])
                print(f"Paragraph found in similarities: {paragraph[:50]}...")
            else:
                question_similarities.append(float('-inf'))
        similarities_list.append(question_similarities)
    
    print(f"Filtered to {len(similarities_list)} similarity lists with candidate paragraphs only")
    
    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    
    results: Dict[str, float]= compute_metrics(ground_truth, similarities_list, top_k, metric)
    log_metrics_to_notion(id=str(eval_id), model=temporal_model_name, benchmark=benchmark, metrics=results, k=top_k, num_negative_samples=num_negative_samples)
    
    print(results)
    print("Evaluation completed successfully")