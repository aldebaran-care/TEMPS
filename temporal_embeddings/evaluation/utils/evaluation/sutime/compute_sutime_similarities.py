from pathlib import Path
import json
from typing import Dict, List, Tuple

from stanza.server import CoreNLPClient
from tqdm import tqdm
import pandas as pd

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

def compute_sutime_similarities(benchmark_file_path: Path, cache_file_path: Path, similarities_file_path: Path) -> pd.DataFrame:
    print("Starting SUTime similarity computation...")
    
    print(f"Loading benchmark data from: {benchmark_file_path}")
    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data: List[Dict] = json.load(f)
        print(f"Loaded {len(benchmark_data)} benchmark items")

    all_paragraphs: List[str] = []
    for item in benchmark_data:
        all_paragraphs.extend(item["paragraphs"])
    
    all_paragraphs = sorted(list(set(all_paragraphs)))
    print(f"Found {len(all_paragraphs)} unique paragraphs")

    print("\n=== STAGE 1: Extracting Temporal Expressions ===")
    
    temporal_cache = pd.DataFrame(columns=["expressions"])
    
    if cache_file_path.exists():
        print(f"Cache file found at: {cache_file_path}")
        temporal_cache = pd.read_pickle(cache_file_path)
        print(f"Loaded cache with {len(temporal_cache)} texts")
    else:
        print(f"No cache file found - will create new cache")

        print("Collecting unique texts for temporal expression extraction...")
        unique_texts = set()
        for benchmark_item in benchmark_data:
            question = benchmark_item["question"]
            unique_texts.add(question)
            unique_texts.update(benchmark_item["paragraphs"])

        unique_texts = list(unique_texts)
        print(f"Found {len(unique_texts)} unique texts")
        
        texts_to_process = [t for t in unique_texts if t not in temporal_cache.index]
        
        if texts_to_process:
            print(f"Extracting temporal expressions from {len(texts_to_process)} new texts...")
            print("Initializing CoreNLP client with SUTime...")
            client = CoreNLPClient(annotators=['tokenize', 'ner'], be_quiet=True)
            print("CoreNLP client initialized successfully")
            
            for text in tqdm(texts_to_process, desc="Extracting temporal expressions"):
                expressions = extract_temporal_expressions(client, text)
                temporal_cache.loc[text] = {'expressions': expressions}
            
            print(f"Saving temporal expression cache to: {cache_file_path}")
            temporal_cache.to_pickle(cache_file_path)
            print(f"Cache saved with {len(temporal_cache)} texts")
        else:
            print("All texts found in cache - no new extraction needed")

    print("\n=== STAGE 2: Computing Similarities ===")
    
    output_similarities_cache: pd.DataFrame = pd.DataFrame(columns=all_paragraphs)

    if similarities_file_path.exists():
        print(f"Similarities file found at: {similarities_file_path}")
        output_similarities_cache = pd.read_pickle(similarities_file_path)
        print(f"Loaded similarities cache with {len(output_similarities_cache)} entries")
    else:
        print(f"No similarities file found - will create new similarities cache")
        print("Processing benchmark items for similarity computation...")
        
        question_current_date = '2025-11-13'
        
        def compute_question_similarities(question: str) -> pd.Series:
            question_expressions = temporal_cache.loc[question, "expressions"]
            similarities = []
            
            for paragraph in all_paragraphs:
                paragraph_expressions = temporal_cache.loc[paragraph, "expressions"]
                paragraph_current_date = question_current_date
                
                max_similarity = 0.0
                
                if question_expressions and paragraph_expressions:
                    for q_expr_text, q_expr_value in question_expressions:
                        for p_expr_text, p_expr_value in paragraph_expressions:
                            similarity = compute_similarity_expressions(
                                q_expr_value if q_expr_value else q_expr_text,
                                p_expr_value if p_expr_value else p_expr_text,
                                question_current_date,
                                paragraph_current_date
                            )
                            max_similarity = max(max_similarity, similarity)
                
                similarities.append(max_similarity)
            
            return pd.Series(similarities, index=all_paragraphs)
        
        questions = list({item["question"] for item in benchmark_data})
        
        print("Computing similarities using vectorized operations...")
        output_similarities_cache = pd.DataFrame([
            compute_question_similarities(question) 
            for question in tqdm(questions, desc="Computing similarities")
        ], index=questions)

        print(f"Saving SUTime similarities to: {similarities_file_path}")
        output_similarities_cache.to_pickle(similarities_file_path)
        print("SUTime similarities saved successfully")
    
    return output_similarities_cache
