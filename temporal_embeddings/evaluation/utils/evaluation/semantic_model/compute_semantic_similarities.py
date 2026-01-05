from pathlib import Path
import json
from typing import List
from math import ceil

from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def encode(model: SentenceTransformer, texts: List[str], prompt_name: str = None) -> List[List[float]]:
    model_name = model._first_module().__class__.__name__.lower()
    
    if "inf-retriever" in model_name:
        if prompt_name is not None:
            return model.encode(texts, convert_to_tensor=True, prompt_name=prompt_name)
        
        return model.encode(texts, convert_to_tensor=True)
    
    if model_name == "baai/bge-large-en":
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    
    return model.encode(texts, convert_to_tensor=True)

def batch_encode(model, texts_to_encode: List[str], batch_size: int = 64) -> List[List[float]]:
    """Encode texts in batches to optimize memory usage."""

    encoded_embeddings = []
    num_batches = ceil(len(texts_to_encode) / batch_size)
    for i in tqdm(range(num_batches), desc="Encoding batches"):
        batch = texts_to_encode[i * batch_size:(i + 1) * batch_size]
        batch = [b[:400] for b in batch]
        encoded_embeddings.extend(model.encode(batch, show_progress_bar=False))
    return encoded_embeddings

def compute_semantic_similarities(semantic_model_name: str, max_seq_len: int, benchmark_file_path: Path, semantic_cache_file_path: Path, semantic_similarities_file_path: Path, num_negative_samples: int = 0) -> pd.DataFrame:
    print("Starting semantic model embeddings computation...")
    print(f"Model: {semantic_model_name}")

    if semantic_model_name == "salesforce":
        return compute_salesforce_similarities(benchmark_file_path, semantic_cache_file_path, semantic_similarities_file_path, num_negative_samples)

    print(f"Loading benchmark data from: {benchmark_file_path}")
    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data = add_negative_samples(json.load(f), num_negatives=num_negative_samples)
        print(f"Loaded {len(benchmark_data)} benchmark items")

        all_paragraphs: List[str] = []
        for item in benchmark_data:
            all_paragraphs.extend(item["paragraphs"])
            
        all_paragraphs = sorted(list(set(all_paragraphs)))

        print("Initializing embedding cache...")
        embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])
        
        if semantic_cache_file_path.exists():
            print(f"Semantic cache file found at: {semantic_cache_file_path}")
            embedding_cache = pd.read_pickle(semantic_cache_file_path)
            print(f"Loaded cache with {len(embedding_cache)} embeddings")
        else:
            print(f"No semantic cache file found - will create new cache")

            print("Loading SentenceTransformer model...")
            if "inf-retriever" in semantic_model_name:
                print("Using trust_remote_code=True for inf-retriever model")
                model: SentenceTransformer = SentenceTransformer(semantic_model_name, trust_remote_code=True)
            else:
                model: SentenceTransformer = SentenceTransformer(semantic_model_name)
            
            model.max_seq_length = max_seq_len
            print(f"Model loaded successfully with max_seq_length: {max_seq_len}")

            print("\n=== STAGE 1: Computing Embeddings ===")
            print("Processing benchmark items with semantic model...")

            for benchmark_item in tqdm(benchmark_data):
                question: str = benchmark_item["question"]

                if question not in embedding_cache.index:
                    question_emb = encode(model, [question], prompt_name="question_prompt")[0]
                    embedding_cache.loc[question] = {'embedding': question_emb.cpu()}

                paragraphs: List[str] = benchmark_item["paragraphs"]
                
                for paragraph in paragraphs:
                    if paragraph not in embedding_cache.index:
                        paragraph_emb = encode(model, [paragraph], prompt_name="passage_prompt")[0]
                        embedding_cache.loc[paragraph] = {'embedding': paragraph_emb.cpu()}

            print(f"Saving embedding cache to: {semantic_cache_file_path}")
            embedding_cache.to_pickle(semantic_cache_file_path)
            print(f"Cache saved with {len(embedding_cache)} embeddings")

        output_similarities_cache: pd.DataFrame = pd.DataFrame(columns=all_paragraphs)

        if semantic_similarities_file_path.exists():
            print(f"Semantic similarities file found at: {semantic_similarities_file_path}")
            output_similarities_cache = pd.read_pickle(semantic_similarities_file_path)
            print(f"Loaded similarities cache with {len(output_similarities_cache)} entries")
        else:
            print(f"No semantic similarities file found - will create new similarities cache")

            print("\n=== STAGE 2: Computing Similarities ===")
            
            paragraph_embeddings = torch.stack([
                torch.Tensor(embedding_cache.loc[paragraph, 'embedding']) 
                for paragraph in all_paragraphs
            ])
            
            def compute_question_similarities(question: str) -> pd.Series:
                question_emb = torch.Tensor(embedding_cache.loc[question, 'embedding']).unsqueeze(0)
                similarities = util.cos_sim(question_emb, paragraph_embeddings)[0]
                return pd.Series(similarities.cpu().numpy(), index=all_paragraphs)
            
            questions = list({item["question"] for item in benchmark_data})
            
            print("Computing similarities using vectorized operations...")
            output_similarities_cache = pd.DataFrame([
                compute_question_similarities(question) 
                for question in tqdm(questions, desc="Computing similarities")
            ], index=questions)

            print(f"Saving semantic model similarities to: {semantic_similarities_file_path}")
            output_similarities_cache.to_pickle(semantic_similarities_file_path)
            print("Semantic model similarities saved successfully")

    return output_similarities_cache

def compute_salesforce_similarities(benchmark_file_path: Path, semantic_cache_file_path: Path, semantic_similarities_file_path: Path, num_negative_samples: int) -> pd.DataFrame:
    model_name = "Salesforce/SFR-Embedding-Mistral"
    task = 'Given a question with temporal constraints, retrieve relevant passages that answer the question with the correct temporal information.'
    
    print(f"Using Salesforce model: {model_name}")
    print(f"Task description: {task}")
    
    print("Processing benchmark items with Salesforce model...")
    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data = add_negative_samples(json.load(f), num_negatives=num_negative_samples)
        print(f"Loaded {len(benchmark_data)} benchmark items")
        
        all_paragraphs: List[str] = []
        for item in benchmark_data:
            all_paragraphs.extend(item["paragraphs"])
        
        all_paragraphs = sorted(list(set(all_paragraphs)))
    
    def get_detailed_instruction(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    print("Initializing embedding cache...")
    embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])
    
    if semantic_cache_file_path.exists():
        print(f"Semantic cache file found at: {semantic_cache_file_path}")
        embedding_cache = pd.read_pickle(semantic_cache_file_path)
        print(f"Loaded cache with {len(embedding_cache)} embeddings")
    else:
        print(f"No semantic cache file found - will create new cache")
    
        print("\n=== STAGE 1: Computing Embeddings ===")

        print("Loading Salesforce model...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        print("Salesforce model loaded successfully")
        
        questions_to_encode = []
        question_texts = []
        paragraphs_to_encode = []
        
        print("Collecting texts to encode...")
        for benchmark_item in benchmark_data:
            question: str = benchmark_item["question"]
            if question not in embedding_cache.index:
                questions_to_encode.append(question)
                question_texts.append(get_detailed_instruction(task, question))
            
            paragraphs: List[str] = benchmark_item["paragraphs"]
            for paragraph in paragraphs:
                if paragraph not in embedding_cache.index and paragraph not in paragraphs_to_encode:
                    paragraphs_to_encode.append(paragraph)
        
        print(f"Found {len(questions_to_encode)} questions and {len(paragraphs_to_encode)} paragraphs to encode")
        
        if questions_to_encode:
            print("Batch encoding questions...")
            question_embeddings = batch_encode(model, question_texts)
            for question, embedding in zip(questions_to_encode, question_embeddings):
                embedding_cache.loc[question] = {'embedding': torch.tensor(embedding).cpu()}
        
        if paragraphs_to_encode:
            print("Batch encoding paragraphs...")
            paragraph_embeddings = batch_encode(model, paragraphs_to_encode)
            for paragraph, embedding in zip(paragraphs_to_encode, paragraph_embeddings):
                embedding_cache.loc[paragraph] = {'embedding': torch.tensor(embedding).cpu()}
        
        print(f"Saving embedding cache to: {semantic_cache_file_path}")
        embedding_cache.to_pickle(semantic_cache_file_path)
        print(f"Cache saved with {len(embedding_cache)} embeddings")
    
    print("\n=== STAGE 2: Computing Similarities ===")
    output_similarities_cache: pd.DataFrame = pd.DataFrame(columns=all_paragraphs)

    if semantic_similarities_file_path.exists():
        print(f"Semantic similarities file found at: {semantic_similarities_file_path}")
        output_similarities_cache = pd.read_pickle(semantic_similarities_file_path)
        print(f"Loaded similarities cache with {len(output_similarities_cache)} entries")
    else:
        print(f"No semantic similarities file found - will create new similarities cache")
        
        paragraph_embeddings = torch.stack([
            torch.Tensor(embedding_cache.loc[paragraph, 'embedding']) 
            for paragraph in all_paragraphs
        ])
        
        def compute_question_similarities(question: str) -> pd.Series:
            question_emb = torch.Tensor(embedding_cache.loc[question, 'embedding']).unsqueeze(0)
            similarities = util.cos_sim(question_emb, paragraph_embeddings)[0]
            return pd.Series(similarities.cpu().numpy(), index=all_paragraphs)
        
        questions = list({item["question"] for item in benchmark_data})
        
        print("Computing similarities using vectorized operations...")
        output_similarities_cache = pd.DataFrame([
            compute_question_similarities(question) 
            for question in tqdm(questions, desc="Computing similarities")
        ], index=questions)
        
        print(f"Saving semantic model similarities to: {semantic_similarities_file_path}")
        output_similarities_cache.to_pickle(semantic_similarities_file_path)
        print("Semantic similarities saved successfully")
    
    return output_similarities_cache