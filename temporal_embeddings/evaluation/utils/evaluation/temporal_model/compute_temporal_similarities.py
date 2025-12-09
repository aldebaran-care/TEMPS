from pathlib import Path
import json
from typing import List

from tqdm import tqdm
import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.inference import Inference

def compute_temporal_similarities(temporal_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, temporal_cache_file_path: Path, temporal_similarities_file_path: Path, use_all_paragraphs: bool = True, reference_date: str = "09 august 2024") -> pd.DataFrame:
        print("Starting temporal model embeddings computation...")
        print(f"Using reference date: {reference_date}")

        print("Initializing temporal model inference...")
        inference: Inference = Inference(model_name=temporal_model_name, model_path=temporal_model_path, batch_size=batch_size, max_seq_len=max_seq_len)
        print("Temporal model inference initialized successfully")

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

            output_similarities_cache: pd.DataFrame = pd.DataFrame(columns=all_paragraphs)

            if temporal_similarities_file_path.exists():
                print(f"Temporal similarities file found at: {temporal_similarities_file_path}")
                output_similarities_cache = pd.read_pickle(temporal_similarities_file_path)
                print(f"Loaded similarities cache with {len(output_similarities_cache)} entries")
            else:
                print(f"No temporal similarities file found - will create new similarities cache")
                
                print("Initializing embedding cache...")
                embedding_cache: pd.DataFrame = pd.DataFrame(columns=['mu', 'std', 'dates'])
                
                if temporal_cache_file_path.exists():
                    print(f"Temporal cache file found at: {temporal_cache_file_path}")
                    embedding_cache = pd.read_pickle(temporal_cache_file_path)
                    print(f"Loaded cache with {len(embedding_cache)} embeddings")
                else:
                    print(f"No temporal cache file found - will create new cache")

                    print("\n=== STAGE 1: Computing Embeddings ===")
                    print("Collecting texts to encode...")
                    
                    questions_to_encode = []
                    paragraphs_to_encode = []
                    
                    for benchmark_item in tqdm(benchmark_data, desc="Collecting texts"):
                        question: str = benchmark_item["question"]
                        if question not in embedding_cache.index:
                            questions_to_encode.append(question)
                        
                        paragraphs: List[str] = benchmark_item["paragraphs"] if not use_all_paragraphs else all_paragraphs
                        for paragraph in paragraphs:
                            if paragraph not in embedding_cache.index and paragraph not in paragraphs_to_encode:
                                paragraphs_to_encode.append(paragraph)
                    
                    print(f"Found {len(questions_to_encode)} questions and {len(paragraphs_to_encode)} paragraphs to encode")
                    
                    if questions_to_encode:
                        print("Computing question embeddings...")
                        question_dates = [reference_date] * len(questions_to_encode)
                        question_embeddings = inference.compute_embeddings(questions_to_encode, question_dates)
                        embedding_cache = pd.concat([embedding_cache, question_embeddings])
                    
                    if paragraphs_to_encode:
                        print("Computing paragraph embeddings...")
                        paragraph_dates = [reference_date] * len(paragraphs_to_encode)
                        paragraph_embeddings = inference.compute_embeddings(paragraphs_to_encode, paragraph_dates)
                        embedding_cache = pd.concat([embedding_cache, paragraph_embeddings])
                    
                    print(f"Saving temporal cache to: {temporal_cache_file_path}")
                    embedding_cache.to_pickle(temporal_cache_file_path)
                    print(f"Cache saved with {len(embedding_cache)} embeddings")

                print("\n=== STAGE 2: Computing Similarities ===")
                print("Computing similarities using cached embeddings...")

                for benchmark_item in tqdm(benchmark_data, desc="Computing similarities"):
                    question: str = benchmark_item["question"]

                    paragraphs: List[str] = benchmark_item["paragraphs"] if not use_all_paragraphs else all_paragraphs
                    questions: List[str] = [question] * len(paragraphs)
                    reference_dates: List[str] = [reference_date] * len(paragraphs)
                    ground_truth: List[float] = [0.0] * len(paragraphs)

                    inference.set_sentences(questions, reference_dates, paragraphs, reference_dates, ground_truth)

                    output = inference.evaluate(embedding_cache)
                    
                    row = {k: v for k, v in zip(all_paragraphs, output["similarity"])}

                    output_similarities_cache.loc[question] = row

                print(f"Saving temporal model similarities to: {temporal_similarities_file_path}")
                output_similarities_cache.to_pickle(temporal_similarities_file_path)
                print("Temporal model similarities saved successfully")

        return output_similarities_cache