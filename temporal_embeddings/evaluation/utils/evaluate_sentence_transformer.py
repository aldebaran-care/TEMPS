from typing import List, Dict
import json
from pathlib import Path

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def evaluate_sentence_transformer(model_name: str, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool) -> None:
    print(f"Starting SentenceTransformer evaluation for model: {model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    print(f"Max sequence length: {max_seq_len}")
    
    print("Loading SentenceTransformer model...")
    if "inf-retriever" in model_name:
        print("Using trust_remote_code=True for inf-retriever model")
        model: SentenceTransformer = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model: SentenceTransformer = SentenceTransformer(model_name)
    
    model.max_seq_length = max_seq_len
    print(f"Model loaded successfully with max_seq_length: {max_seq_len}")
    
    similarities_file_path: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{eval_id}_similarities.json")
    print(f"Similarities will be saved to: {similarities_file_path}")
    
    cache_file_path: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name}/{eval_id}_cache.pkl")
    print(f"Cache file path: {cache_file_path}")
    create_folders(cache_file_path.parent)
    print(f"Created cache directory: {cache_file_path.parent}")

    if not skip:
        print("Starting similarity computation phase...")
        
        # Load embedding cache if exists
        embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])
        if cache_file_path.exists():
            print(f"Cache file found at: {cache_file_path}")
            embedding_cache = pd.read_pickle(cache_file_path)
            print(f"Loaded cache with {len(embedding_cache)} embeddings")
        else:
            print(f"No cache file found - will create new cache at: {cache_file_path}")

        output_similarities: List[List[float]] = []

        data: List[Dict] = []
        ground_truth: List[int] = []

        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} benchmark items")

            print("Processing benchmark items...")
            for element in tqdm(data):
                ground_truth.append(element["answer"])

                question: str = element["question"]
                
                # Check cache for question embedding
                if question not in embedding_cache.index:
                    if "inf-retriever" in model_name:
                        question_emb = model.encode(question, convert_to_tensor=True, prompt_name="query")
                    elif model_name != "BAAI/bge-large-en":
                        question_emb = model.encode(question, convert_to_tensor=True)
                    else:
                        question_emb = model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
                    
                    embedding_cache.loc[question] = {'embedding': question_emb.cpu()}
                else:
                    question_emb = embedding_cache.loc[question, 'embedding']

                paragraphs: List[str] = element["paragraphs"]

                similarities: List[float] = []
                
                for paragraph in paragraphs:
                    # Check cache for paragraph embedding
                    if paragraph not in embedding_cache.index:
                        if "inf-retriever" in model_name:
                            paragraph_emb = model.encode(paragraph, convert_to_tensor=True)
                        elif model_name != "BAAI/bge-large-en":
                            paragraph_emb = model.encode(paragraph, convert_to_tensor=True)
                        else:
                            paragraph_emb = model.encode(paragraph, convert_to_tensor=True, normalize_embeddings=True)
                        
                        embedding_cache.loc[paragraph] = {'embedding': paragraph_emb.cpu()}
                    else:
                        paragraph_emb = embedding_cache.loc[paragraph, 'embedding']

                    similarities.append(float(util.cos_sim(torch.Tensor(question_emb).cpu(), torch.Tensor(paragraph_emb).cpu())[0].item()))

                output_similarities.append(similarities)

        # Save embedding cache as pandas DataFrame
        print(f"Saving embedding cache to: {cache_file_path}")
        embedding_cache.to_pickle(cache_file_path)
        print(f"Cache saved with {len(embedding_cache)} embeddings")

        create_folders(similarities_file_path.parent)
        print(f"Created similarities directory: {similarities_file_path.parent}")
        
        print(f"Saving similarities to: {similarities_file_path}")
        with similarities_file_path.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)
        print("Similarities saved successfully")
    else:
        print("Skipping similarity computation - using existing results")

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with benchmark_file_path.open("r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

        for element in data:
            ground_truth.append(element["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")

    print(f"Loading similarities from: {similarities_file_path}")
    output_similarities: List[List[float]] = []
    
    with similarities_file_path.open("r", encoding="utf-8") as f:
        output_similarities = json.load(f)
    print(f"Loaded {len(output_similarities)} similarity lists")

    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    print(compute_metrics(ground_truth, output_similarities, top_k, metric))
    print("Evaluation completed successfully")