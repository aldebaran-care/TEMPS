from pathlib import Path
import json
from typing import List

import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics, compute_metrics_ranks
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import compute_temporal_similarities
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import compute_semantic_similarities
from temporal_embeddings.utils.math.normalize import normalize_list

def evaluate_temporal_semantic_model(temporal_model_name: str, semantic_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False, use_ranking: bool = False, alpha: float = 0.5, use_all_paragraphs: bool = False) -> None:
    print(f"Starting temporal semantic model evaluation")
    print(f"Temporal model: {temporal_model_name} at {temporal_model_path}")
    print(f"Semantic model: {semantic_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    print(f"Use ranking fusion: {use_ranking}")
    print(f"Use all paragraphs: {use_all_paragraphs}")
    
    temporal_cache_path, temporal_similarities_path, semantic_cache_path, semantic_similarities_path = set_output_files(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name=semantic_model_name,
        benchmark_file_path=benchmark_file_path,
        eval_id=eval_id
    ).values()

    if not skip:
        print("Running both models (not skipping)...")

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

        semantic_similarities: pd.DataFrame = compute_semantic_similarities(
            semantic_model_name=semantic_model_name,
            max_seq_len=max_seq_len,
            benchmark_file_path=benchmark_file_path,
            semantic_cache_file_path=semantic_cache_path,
            semantic_similarities_file_path=semantic_similarities_path,
            use_all_paragraphs=use_all_paragraphs
        )

    else:
        print("Skipping model runs - using existing similarities")

        print("Loading similarities for fusion...")
        temporal_similarities: pd.DataFrame = pd.read_pickle(temporal_similarities_path).to_numpy().tolist()
        semantic_similarities: pd.DataFrame = pd.read_pickle(semantic_similarities_path).to_numpy().tolist()
    
    print(f"Loaded {len(temporal_similarities)} temporal similarity lists")
    print(f"Loaded {len(semantic_similarities)} semantic similarity lists")

    if use_ranking:
        print("Using Borda count ranking fusion...")
        def borda_count_fusion(temporal_similarities: List[List[float]], external_similarities: List[List[float]]) -> List[List[int]]:
            merged_ranks = []

            for temp_sim, ext_sim in zip(temporal_similarities, external_similarities):
                scores = {}
                
                temp_ranks: List[float] = sorted(range(len(temp_sim)), key=lambda i: temp_sim[i], reverse=True)
                ext_ranks: List[float] = sorted(range(len(ext_sim)), key=lambda i: ext_sim[i], reverse=True)

                for rank, idx in enumerate(temp_ranks):
                    scores[idx] = scores.get(idx, 0) + (len(temp_ranks) - rank)

                for rank, idx in enumerate(ext_ranks):
                    scores[idx] = scores.get(idx, 0) + ((len(ext_ranks) - rank) * 2)

                sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                merged_ranks.append([idx for idx, _ in sorted_indices])

            return merged_ranks

        ranks: List[List[int]] = borda_count_fusion(temporal_similarities, external_similarities)
        print("Borda count fusion completed")
        
        print("Loading ground truth for ranking evaluation...")
        ground_truth: List[List[int]] = []

        with open(benchmark_file_path, "r") as f:
            benchmark_data: List[dict] = json.load(f)

            for e in benchmark_data:
                ground_truth.append(e["answer"])

        print(f"Computing ranking metrics with top_k={top_k}, metric={metric}")
        print(compute_metrics_ranks(ground_truth, ranks, top_k, metric))

    else:
        print("Using score normalization and weighted fusion...")

        print("Normalizing similarity scores...")
        temporal_similarities = normalize_list(temporal_similarities)
        external_similarities = normalize_list(external_similarities)

        print("Merging similarities with weights (temporal: 1x, external: 2x)...")
        merged_list = [[((alpha*x) + ((1-alpha)*y)) for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(temporal_similarities, external_similarities)]

        merged_similarities: List[List[float]] = merged_list
        print("Score fusion completed")
        
        print("Loading ground truth for score evaluation...")
        ground_truth: List[List[int]] = []

        with open(benchmark_file_path, "r") as f:
            benchmark_data: List[dict] = json.load(f)

            for e in benchmark_data:
                ground_truth.append(e["answer"])

        print(f"Computing score metrics with top_k={top_k}, metric={metric}")
        print(compute_metrics(ground_truth, merged_similarities, top_k, metric))
    
    print("TemporalBERT Full evaluation completed successfully")
