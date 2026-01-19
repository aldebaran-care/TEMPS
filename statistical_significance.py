import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

import pandas as pd
from scipy import stats
import numpy as np

from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def normalize_list(values: List[float]) -> List[float]:
    """Normalize a list of values to [0, 1] range."""
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def load_similarities_and_ground_truth(similarities_path: Path, benchmark_file_path: Path) -> Tuple[pd.DataFrame, List[List[int]], List[dict]]:
    similarities_df = pd.read_pickle(similarities_path)
    
    with open(benchmark_file_path, "r") as f:
        benchmark_data = add_negative_samples(json.load(f), num_negatives=-1)
    
    ground_truth = [item["answer"] for item in benchmark_data]
    
    return similarities_df, ground_truth, benchmark_data

def filter_similarities_to_list(similarities_df: pd.DataFrame, benchmark_data: List[dict]) -> List[List[float]]:
    similarities_list = []
    for benchmark_item in benchmark_data:
        question_sims = similarities_df.loc[benchmark_item["question"]][benchmark_item["paragraphs"]].tolist()
        similarities_list.append(question_sims)
    
    return similarities_list

def compute_per_query_metrics(ground_truth: List[List[int]], similarities: List[List[float]], top_k: int, metric: str) -> List[float]:
    per_query_scores = []
    
    for gt, sims in zip(ground_truth, similarities):
        result = compute_metrics([gt], [sims], top_k, metric)
        
        if metric == "all":
            score = result.get(f"top_{top_k}_accuracy", 0.0)
        elif metric == "top":
            score = result.get(f"top_{top_k}_accuracy", 0.0)
        elif metric == "mrr":
            score = result.get("mrr", 0.0)
        elif metric == "ndcg":
            score = result.get(f"ndcg@{top_k}", 0.0)
        elif metric in ["precision", "recall", "f1"]:
            score = result.get(f"{metric}@{top_k}", 0.0)
        else:
            score = 0.0
        
        per_query_scores.append(score)
    
    return per_query_scores


def perform_statistical_tests(scores_model1: List[float], scores_model2: List[float]) -> Dict[str, float]:
    t_stat, t_pvalue = stats.ttest_rel(scores_model1, scores_model2)
    
    differences = np.array(scores_model1) - np.array(scores_model2)
    
    return {
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_pvalue),
        "mean_diff": float(np.mean(differences))
    }


def statistical_significance_test(temporal_similarities_path: Path, semantic_similarities_path: Path, benchmark_file_path: Path, top_k: int, metric: str, alpha: float) -> None:
    print("=" * 80)
    print("Statistical Significance Testing")
    print("=" * 80)
    print(f"Temporal Model: {temporal_similarities_path}")
    print(f"Semantic Model (for merge): {semantic_similarities_path}")
    print(f"Benchmark: {benchmark_file_path}")
    print(f"Merge Alpha: {alpha}, Significance Threshold: 0.05")
    print("=" * 80)
    
    print("\nLoading temporal similarities...")
    temporal_sims_df, ground_truth, benchmark_data = load_similarities_and_ground_truth(
        temporal_similarities_path,
        benchmark_file_path
    )
    
    print("Loading semantic similarities for merge...")
    semantic_sims_df, _, _ = load_similarities_and_ground_truth(
        semantic_similarities_path,
        benchmark_file_path
    )
    
    print(f"\nNumber of queries: {len(ground_truth)}")
    
    print(f"\nMerging temporal and semantic similarities with alpha={alpha}...")
    normalized_temporal = temporal_sims_df.apply(lambda row: normalize_list(row.tolist()), axis=1, result_type='expand')
    normalized_temporal.columns = temporal_sims_df.columns
    normalized_temporal.index = temporal_sims_df.index
    
    normalized_semantic = semantic_sims_df.apply(lambda row: normalize_list(row.tolist()), axis=1, result_type='expand')
    normalized_semantic.columns = semantic_sims_df.columns
    normalized_semantic.index = semantic_sims_df.index
    
    merged_similarities_df = (alpha * normalized_temporal) + ((1 - alpha) * normalized_semantic)
    
    print("Filtering similarities to candidate paragraphs...")
    merged_similarities = filter_similarities_to_list(merged_similarities_df, benchmark_data)
    semantic_similarities = filter_similarities_to_list(semantic_sims_df, benchmark_data)
    
    print("\nComputing per-query metrics...")
    merged_scores = compute_per_query_metrics(ground_truth, merged_similarities, top_k, metric)
    semantic_scores = compute_per_query_metrics(ground_truth, semantic_similarities, top_k, metric)
    
    print(f"Merged (Temporal+Semantic) Model - Mean: {np.mean(merged_scores):.4f}, Std: {np.std(merged_scores):.4f}")
    print(f"Semantic Only Model - Mean: {np.mean(semantic_scores):.4f}, Std: {np.std(semantic_scores):.4f}")
    
    print("\nPerforming paired t-test...")
    test_results = perform_statistical_tests(merged_scores, semantic_scores)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPaired T-Test:")
    print(f"  T-statistic: {test_results['paired_t_statistic']:.4f}")
    print(f"  P-value: {test_results['paired_t_pvalue']:.6f}")
    print(f"  Mean difference: {test_results['mean_diff']:.4f}")
    print(f"  Significant at α=0.05: {'YES' if test_results['paired_t_pvalue'] < 0.05 else 'NO'}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if test_results['paired_t_pvalue'] < 0.05:
        direction = "better" if test_results['mean_diff'] > 0 else "worse"
        print(f"✓ Merged (Temporal+Semantic) Model performs significantly {direction} than Semantic Only Model")
        print(f"  (p = {test_results['paired_t_pvalue']:.6f} < 0.05)")
    else:
        print(f"✗ No significant difference detected between models")
        print(f"  (p = {test_results['paired_t_pvalue']:.6f} >= 0.05)")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compute statistical significance between merged temporal+semantic and semantic-only models")
    parser.add_argument("--temporal_similarities", type=str, required=True, help="Path to temporal model similarities pickle file")
    parser.add_argument("--semantic_similarities", type=str, required=True, help="Path to semantic model similarities pickle file (for merging with temporal)")
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to benchmark JSON file")
    parser.add_argument("--top_k", type=int, default=1, help="Value of k for top-k metrics")
    parser.add_argument("--metric", type=str, default="top", choices=["all", "top", "mrr", "ndcg", "precision", "recall", "f1"], help="Metric to use for comparison")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for merging temporal and semantic similarities")
    
    args = parser.parse_args()
    
    statistical_significance_test(
        temporal_similarities_path=Path(args.temporal_similarities),
        semantic_similarities_path=Path(args.semantic_similarities),
        benchmark_file_path=Path(args.benchmark_file),
        top_k=args.top_k,
        metric=args.metric,
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()
