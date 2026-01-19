import random
from typing import List, Dict, Any

from tqdm import tqdm
from rank_bm25 import BM25Okapi


def add_negative_samples(data: List[Dict[str, Any]], num_negatives: int, seed: int = 42) -> List[Dict[str, Any]]:
    if num_negatives == -1:
        random.seed(seed)

        all_paragraphs: List[str] = []
        for item in tqdm(data, desc="Collecting all paragraphs"):
            all_paragraphs.extend(item["paragraphs"])
        all_paragraphs = list(set(all_paragraphs))

        output_data: List[Dict[str, Any]] = []
        
        for item in tqdm(data, desc="Adding negative samples"):
            positive_paragraphs = [p for idx, p in enumerate(item["paragraphs"]) if idx in item["answer"]]
            item["paragraphs"] = positive_paragraphs + all_paragraphs
            random.shuffle(item["paragraphs"])
            new_answer_indices = [item["paragraphs"].index(p) for p in positive_paragraphs]
            item["answer"] = new_answer_indices
            output_data.append(item)
        
        return output_data
    
    elif num_negatives == 0:
        return data

    elif num_negatives > 0:
        random.seed(seed)
        
        # Collect all paragraphs from all items
        all_paragraphs: List[str] = []
        for item in tqdm(data, desc="Collecting all paragraphs"):
            all_paragraphs.extend(item["paragraphs"])
        all_paragraphs = list(set(all_paragraphs))
        
        # Tokenize all paragraphs for BM25
        tokenized_paragraphs = [p.lower().split() for p in all_paragraphs]
        bm25 = BM25Okapi(tokenized_paragraphs)
        
        output_data: List[Dict[str, Any]] = []
        
        for item in tqdm(data, desc="Adding BM25 negative samples"):
            positive_paragraphs = [p for idx, p in enumerate(item["paragraphs"]) if idx in item["answer"]]
            positive_set = set(positive_paragraphs)
            
            # Tokenize query
            query_tokens = item["question"].lower().split()
            
            # Get BM25 scores for all paragraphs
            scores = bm25.get_scores(query_tokens)
            
            # Create list of (paragraph, score) tuples, excluding positive paragraphs
            candidates = [(all_paragraphs[i], scores[i]) for i in range(len(all_paragraphs)) 
                         if all_paragraphs[i] not in positive_set]
            
            # Sort by score (descending) and take top num_negatives
            candidates.sort(key=lambda x: x[1], reverse=True)
            negative_paragraphs = [p for p, _ in candidates[:num_negatives]]
            
            # Combine positive and negative paragraphs
            item["paragraphs"] = positive_paragraphs + negative_paragraphs
            random.shuffle(item["paragraphs"])
            
            # Update answer indices
            new_answer_indices = [item["paragraphs"].index(p) for p in positive_paragraphs]
            item["answer"] = new_answer_indices
            output_data.append(item)
        
        return output_data

    else:
        raise ValueError("num_negatives must be -1 (for all paragraphs), 0 (no additional negatives), or > 0 (BM25-based negatives)")