import random
from typing import List, Dict, Any

from tqdm import tqdm


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

    else:
        raise ValueError("num_negatives must be -1 (for all paragraphs) or 0 (no additional negatives)")