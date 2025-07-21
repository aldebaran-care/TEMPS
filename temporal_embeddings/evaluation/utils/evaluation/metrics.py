from typing import List, Literal, Dict, Union
import numpy as np

def compute_accuracy(
    first_list: List[List[int]],  # list of relevant indices per query
    second_list: List[List[float]],  # similarity scores per query
    top_k: int,
    metric: Literal['top', 'mrr', 'ndcg', 'precision', 'recall', 'f1', 'all']
) -> Dict[str, float]:
    if metric == 'all':
        metrics = ['top', 'mrr', 'ndcg', 'precision', 'recall', 'f1']
        results = {m: compute_single_metric(m) for m in metrics}
        return results
    else:
        return {metric: compute_single_metric(first_list, second_list, top_k, metric)}

def compute_single_metric(first_list: List[List[int]], second_list: List[List[float]], top_k: int, metric: Literal['top', 'mrr', 'ndcg', 'precision', 'recall', 'f1']) -> float:
        scores = []
        
        for gt_indices, sim_scores in zip(first_list, second_list):
            ranked_indices = np.argsort(sim_scores)[::-1]
            top_k_indices = ranked_indices[:top_k]
            relevant_set = set(gt_indices)

            if metric == 'top':
                scores.append(1 if len(relevant_set & set(top_k_indices)) > 0 else 0)
            
            elif metric == 'mrr':
                rr = 0
                for rank, idx in enumerate(ranked_indices, 1):
                    if idx in relevant_set:
                        rr = 1.0 / rank
                        break
                scores.append(rr)
            
            elif metric == 'ndcg':
                dcg = 0.0
                for i, idx in enumerate(ranked_indices[:top_k]):
                    if idx in relevant_set:
                        dcg += 1.0 / np.log2(i + 2)
                ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), top_k)))
                scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
            
            elif metric == 'precision':
                retrieved_relevant = len(relevant_set & set(top_k_indices))
                scores.append(retrieved_relevant / top_k if top_k > 0 else 0.0)
            
            elif metric == 'recall':
                retrieved_relevant = len(relevant_set & set(top_k_indices))
                scores.append(retrieved_relevant / len(relevant_set) if len(relevant_set) > 0 else 0.0)
            
            elif metric == 'f1':
                retrieved_relevant = len(relevant_set & set(top_k_indices))
                precision = retrieved_relevant / top_k if top_k > 0 else 0.0
                recall = retrieved_relevant / len(relevant_set) if len(relevant_set) > 0 else 0.0
                if precision + recall > 0:
                    scores.append(2 * precision * recall / (precision + recall))
                else:
                    scores.append(0.0)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return np.mean(scores) if scores else 0.0