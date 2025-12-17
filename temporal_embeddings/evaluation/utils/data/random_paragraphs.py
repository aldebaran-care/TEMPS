import random
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
from rank_bm25 import BM25Okapi
import pandas as pd


def add_negative_samples(
    data: List[Dict[str, Any]],
    num_negatives: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    cache_path = Path("output/negative_samples")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    data_str = json.dumps(data, sort_keys=True)
    data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    cache_file = cache_path / f"negative_samples_bm25_seed{seed}_{data_hash}.json"
    
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
    
    print("Building BM25 index for candidate selection...")
    
    tokenized_paragraphs = [para.lower().split() for para in all_paragraphs_list]
    bm25 = BM25Okapi(tokenized_paragraphs)
    
    print("Computing BM25 scores cache...")
    bm25_cache_path = cache_path / f"bm25_scores_{data_hash}.json"
    
    if bm25_cache_path.exists():
        print("Loading BM25 scores from cache...")
        bm25_cache_df = pd.read_json(bm25_cache_path)
    else:
        questions = list({item['question'] for item in data})
        
        scores_data = []
        
        for question in tqdm(questions, desc="Computing BM25 scores"):
            tokenized_query = question.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            scores_dict = {'question': question}
            scores_dict.update({para: float(score) for para, score in zip(all_paragraphs_list, scores)})
            scores_data.append(scores_dict)
        
        bm25_cache_df = pd.DataFrame(scores_data).set_index('question')
        
        print("Saving BM25 scores cache...")
        bm25_cache_df.to_json(bm25_cache_path)
    
    result = []
    
    for item in tqdm(data, desc="Adding negative samples"):
        original_paragraphs = item['paragraphs'].copy()
        original_answer_indices = item['answer'].copy() if item['answer'] else []
        
        answer_paragraphs = []
        if original_answer_indices:
            answer_paragraphs = [original_paragraphs[idx] for idx in original_answer_indices if idx < len(original_paragraphs)]
        
        available_negatives = [p for p in all_paragraphs_list if p not in original_paragraphs]
        
        question = item['question']
        
        if question in bm25_cache_df.index:
            question_scores = bm25_cache_df.loc[question, available_negatives]
            
            negative_samples = question_scores.sort_values(ascending=False).index.tolist()
        else:
            negative_samples = available_negatives
        
        combined_paragraphs = answer_paragraphs + negative_samples[:num_negatives]
        
        indices = list(range(len(combined_paragraphs)))
        random.shuffle(indices)
        
        shuffled_paragraphs = [combined_paragraphs[i] for i in indices]
        
        new_answer_indices = []
        for para in answer_paragraphs:
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