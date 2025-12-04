from pathlib import Path
import json
from typing import List

from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def compute_semantic_similarities(semantic_model_name: str, max_seq_len: int, benchmark_file_path: Path, semantic_cache_file_path: Path, semantic_similarities_file_path: Path, use_all_paragraphs: bool = True) -> pd.DataFrame:
    print("Starting semantic model embeddings computation...")
    print(f"Model: {semantic_model_name}")

    output_similarities_cache: pd.DataFrame = None

    print(f"Loading benchmark data from: {benchmark_file_path}")
    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
        print(f"Loaded {len(benchmark_data)} benchmark items")

        if use_all_paragraphs:
            all_paragraphs: List[str] = []
            for item in benchmark_data:
                all_paragraphs.extend(item["paragraphs"])
            
            all_paragraphs = sorted(list(set(all_paragraphs)))
            print(f"Using all paragraphs mode: {len(all_paragraphs)} unique paragraphs")

        print("Loading SentenceTransformer model...")
        if "inf-retriever" in semantic_model_name:
            print("Using trust_remote_code=True for inf-retriever model")
            model: SentenceTransformer = SentenceTransformer(semantic_model_name, trust_remote_code=True)
        else:
            model: SentenceTransformer = SentenceTransformer(semantic_model_name)
        
        model.max_seq_length = max_seq_len
        print(f"Model loaded successfully with max_seq_length: {max_seq_len}")

        print("Initializing embedding cache...")
        embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])
        if semantic_cache_file_path.exists():
            print(f"Semantic cache file found at: {semantic_cache_file_path}")
            embedding_cache = pd.read_pickle(semantic_cache_file_path)
            print(f"Loaded cache with {len(embedding_cache)} embeddings")
        else:
            print(f"No semantic cache file found - will create new cache")

        print("Processing benchmark items with semantic model...")
        output_similarities_cache = pd.DataFrame(columns=all_paragraphs)

        for benchmark_item in tqdm(benchmark_data):
            question: str = benchmark_item["question"]

            if question not in embedding_cache.index:
                if "inf-retriever" in semantic_model_name:
                    question_emb = model.encode(question, convert_to_tensor=True, prompt_name="query")
                elif semantic_model_name != "BAAI/bge-large-en":
                    question_emb = model.encode(question, convert_to_tensor=True)
                else:
                    question_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
                
                embedding_cache.loc[question] = {'embedding': question_emb.cpu()}
            else:
                question_emb = embedding_cache.loc[question, 'embedding']

            paragraphs: List[str] = benchmark_item["paragraphs"] if not use_all_paragraphs else all_paragraphs

            similarities: List[float] = []
            
            for paragraph in paragraphs:
                if paragraph not in embedding_cache.index:
                    if "inf-retriever" in semantic_model_name:
                        paragraph_emb = model.encode(paragraph, convert_to_tensor=True)
                    elif semantic_model_name != "BAAI/bge-large-en":
                        paragraph_emb = model.encode(paragraph, convert_to_tensor=True)
                    else:
                        paragraph_emb = model.encode(paragraph, convert_to_tensor=True, normalize_embeddings=True)
                    
                    embedding_cache.loc[paragraph] = {'embedding': paragraph_emb.cpu()}
                else:
                    paragraph_emb = embedding_cache.loc[paragraph, 'embedding']

                similarities.append(float(util.cos_sim(torch.Tensor(question_emb).cpu(), torch.Tensor(paragraph_emb).cpu())[0].item()))

            row = {k: v for k, v in zip(all_paragraphs, similarities)}
            output_similarities_cache.loc[question] = row

    print(f"Saving embedding cache to: {semantic_cache_file_path}")
    embedding_cache.to_pickle(semantic_cache_file_path)
    print(f"Cache saved with {len(embedding_cache)} embeddings")

    print(f"Saving semantic model similarities to: {semantic_similarities_file_path}")
    output_similarities_cache.to_pickle(semantic_similarities_file_path)
    
    print("Semantic model similarities saved successfully")

    return output_similarities_cache
