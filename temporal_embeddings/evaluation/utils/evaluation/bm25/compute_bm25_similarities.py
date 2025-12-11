from pathlib import Path
import json
from typing import List

from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi

from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and converting to lowercase."""
    return text.lower().split()

def compute_bm25_similarities(bm25_model_name: str, benchmark_file_path: Path, bm25_cache_file_path: Path, bm25_similarities_file_path: Path, num_negative_samples: int) -> pd.DataFrame:
    print("Starting BM25 similarities computation...")
    print(f"Model: {bm25_model_name}")

    print(f"Loading benchmark data from: {benchmark_file_path}")
    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data = add_negative_samples(json.load(f), num_negative_samples=num_negative_samples)
        print(f"Loaded {len(benchmark_data)} benchmark items")

        all_paragraphs: List[str] = []
        for item in benchmark_data:
            all_paragraphs.extend(item["paragraphs"])
        
        all_paragraphs = sorted(list(set(all_paragraphs)))
        print(f"Total unique paragraphs: {len(all_paragraphs)} paragraphs")

        print("Initializing BM25 cache...")
        bm25_cache: pd.DataFrame = pd.DataFrame(columns=['tokenized'])
        
        if bm25_cache_file_path.exists():
            print(f"BM25 cache file found at: {bm25_cache_file_path}")
            bm25_cache = pd.read_pickle(bm25_cache_file_path)
            print(f"Loaded cache with {len(bm25_cache)} tokenized texts")
        else:
            print(f"No BM25 cache file found - will create new cache")

            print("\n=== STAGE 1: Tokenizing Texts ===")
            print("Tokenizing benchmark items...")

            for benchmark_item in tqdm(benchmark_data, desc="Processing questions"):
                question: str = benchmark_item["question"]

                if question not in bm25_cache.index:
                    question_tokens = tokenize(question)
                    bm25_cache.loc[question] = {'tokenized': question_tokens}

            for paragraph in tqdm(all_paragraphs, desc="Processing paragraphs"):
                if paragraph not in bm25_cache.index:
                    paragraph_tokens = tokenize(paragraph)
                    bm25_cache.loc[paragraph] = {'tokenized': paragraph_tokens}

            print(f"Saving BM25 cache to: {bm25_cache_file_path}")
            bm25_cache.to_pickle(bm25_cache_file_path)
            print(f"Cache saved with {len(bm25_cache)} tokenized texts")

        output_similarities_cache: pd.DataFrame = pd.DataFrame(columns=all_paragraphs)

        if bm25_similarities_file_path.exists():
            print(f"BM25 similarities file found at: {bm25_similarities_file_path}")
            output_similarities_cache = pd.read_pickle(bm25_similarities_file_path)
            print(f"Loaded similarities cache with {len(output_similarities_cache)} entries")
        else:
            print(f"No BM25 similarities file found - will create new similarities cache")

            print("\n=== STAGE 2: Computing BM25 Scores ===")
            
            print("Preparing BM25 corpus...")
            tokenized_paragraphs = [
                bm25_cache.loc[paragraph, 'tokenized'] 
                for paragraph in all_paragraphs
            ]
            
            print("Initializing BM25 model...")
            bm25 = BM25Okapi(tokenized_paragraphs)
            
            def compute_question_bm25_scores(question: str) -> pd.Series:
                question_tokens = bm25_cache.loc[question, 'tokenized']
                scores = bm25.get_scores(question_tokens)
                return pd.Series(scores, index=all_paragraphs)
            
            questions = list({item["question"] for item in benchmark_data})
            
            print("Computing BM25 scores...")
            output_similarities_cache = pd.DataFrame([
                compute_question_bm25_scores(question) 
                for question in tqdm(questions, desc="Computing BM25 scores")
            ], index=questions)

            print(f"Saving BM25 similarities to: {bm25_similarities_file_path}")
            output_similarities_cache.to_pickle(bm25_similarities_file_path)
            print("BM25 similarities saved successfully")

    return output_similarities_cache