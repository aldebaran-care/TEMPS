import random
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any


def add_negative_samples(
    data: List[Dict[str, Any]],
    num_negative_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    cache_path = Path("output/cache/negative_samples")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    data_str = json.dumps(data, sort_keys=True)
    data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    cache_file = cache_path / f"negative_samples_n{num_negative_samples}_seed{seed}_{data_hash}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_result = json.load(f)
                return cached_result
        except (json.JSONDecodeError, IOError):
            pass
    
    random.seed(seed)
    
    all_paragraphs = set()
    for item in data:
        for para in item['paragraphs']:
            all_paragraphs.add(para)
    
    all_paragraphs_list = list(all_paragraphs)
    
    result = []
    
    for item in data:
        original_paragraphs = item['paragraphs'].copy()
        original_answer_indices = item['answer'].copy() if item['answer'] else []
        
        correct_paragraphs = set()
        if original_answer_indices:
            correct_paragraphs = {original_paragraphs[idx] for idx in original_answer_indices if idx < len(original_paragraphs)}
        
        available_negatives = [p for p in all_paragraphs_list if p not in original_paragraphs]
        
        if num_negative_samples == -1:
            negative_samples = available_negatives
        else:
            num_to_sample = min(num_negative_samples, len(available_negatives))
            negative_samples = random.sample(available_negatives, num_to_sample)
        
        combined_paragraphs = original_paragraphs + negative_samples
        
        indices = list(range(len(combined_paragraphs)))
        random.shuffle(indices)
        
        shuffled_paragraphs = [combined_paragraphs[i] for i in indices]
        
        new_answer_indices = []
        for para in correct_paragraphs:
            if para in shuffled_paragraphs:
                new_answer_indices.append(shuffled_paragraphs.index(para))
        
        new_answer_indices.sort()
        
        new_item = {
            'question': item['question'],
            'paragraphs': shuffled_paragraphs,
            'answer': new_answer_indices
        }
        
        result.append(new_item)
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
    except IOError:
        pass
    
    return result