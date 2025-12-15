import random
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from stanza.server import CoreNLPClient
from tqdm import tqdm

from temporal_embeddings.data_utils.utils.compute_similarity_expressions import compute_similarity_expressions


def extract_temporal_expressions(client: CoreNLPClient, text: str) -> List[Tuple[str, str]]:
    temporal_expressions = []
    ann = client.annotate(text)
    for sentence in ann.sentence:
        for token in sentence.token:
            if token.ner in ["DATE", "TIME", "DURATION", "SET"]:
                expr_text = token.timexValue.text
                expr_value = token.timexValue.value if token.timexValue.value else token.timexValue.altValue
                if expr_text not in [e[0] for e in temporal_expressions]:
                    temporal_expressions.append((expr_text, expr_value))
    return temporal_expressions


def add_negative_samples(
    data: List[Dict[str, Any]],
    num_negative_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if num_negative_samples == 0:
        return data

    cache_path = Path("output/negative_samples")
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
    
    print("Extracting temporal expressions for similarity-based sampling...")
    temporal_cache_path = cache_path / f"temporal_expressions_{data_hash}.json"
    
    temporal_cache = {}
    if temporal_cache_path.exists():
        with open(temporal_cache_path, 'r') as f:
            temporal_cache = json.load(f)
    else:
        unique_texts = set()
        for item in data:
            unique_texts.add(item['question'])
            unique_texts.update(item['paragraphs'])
        
        client = CoreNLPClient(annotators=['tokenize', 'ner'], be_quiet=True)
        for text in tqdm(list(unique_texts), desc="Extracting temporal expressions"):
            expressions = extract_temporal_expressions(client, text)
            temporal_cache[text] = expressions
        
        with open(temporal_cache_path, 'w') as f:
            json.dump(temporal_cache, f, indent=2)
    
    print("Computing similarity cache...")
    similarity_cache_path = cache_path / f"similarities_{data_hash}.json"
    
    similarity_cache = {}
    if similarity_cache_path.exists():
        with open(similarity_cache_path, 'r') as f:
            similarity_cache = json.load(f)
    else:
        current_date = '2025-11-13'
        questions = list({item['question'] for item in data})
        
        def compute_question_similarities(question: str) -> Dict[str, float]:
            question_expressions = temporal_cache.get(question, [])
            similarities = {}
            
            for paragraph in all_paragraphs_list:
                paragraph_expressions = temporal_cache.get(paragraph, [])
                max_similarity = 0.0
                
                if question_expressions and paragraph_expressions:
                    for q_expr_text, q_expr_value in question_expressions:
                        for p_expr_text, p_expr_value in paragraph_expressions:
                            similarity = compute_similarity_expressions(
                                q_expr_value if q_expr_value else q_expr_text,
                                p_expr_value if p_expr_value else p_expr_text,
                                current_date,
                                current_date
                            )
                            max_similarity = max(max_similarity, similarity)
                
                similarities[paragraph] = max_similarity
            
            return similarities
        
        for question in tqdm(questions, desc="Computing similarities"):
            similarity_cache[question] = compute_question_similarities(question)
        
        with open(similarity_cache_path, 'w') as f:
            json.dump(similarity_cache, f, indent=2)
    
    result = []
    
    for item in tqdm(data, desc="Adding negative samples"):
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
            
            question = item['question']
            question_similarities = similarity_cache.get(question, {})
            
            similarities = [(para, question_similarities.get(para, 0.0)) for para in available_negatives]
            similarities.sort(key=lambda x: x[1], reverse=True)
            negative_samples = [para for para, _ in similarities[:num_to_sample]]
        
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