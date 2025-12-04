from pathlib import Path
import json
from typing import List

from tqdm import tqdm
import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.inference import Inference

def compute_temporal_similarities(temporal_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, temporal_cache_file_path: Path, temporal_similarities_file_path: Path, use_all_paragraphs: bool = True, reference_date: str = "09 august 2024") -> pd.DataFrame:
        print("Starting temporal model embeddings computation...")
        print(f"Using reference date: {reference_date}")

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

            print("Initializing TemporalBERT inference...")
            if temporal_cache_file_path.exists():
                print(f"Temporal cache file found at: {temporal_cache_file_path}")
            else:
                print(f"No temporal cache file found - will create new cache")
            
            inference: Inference = Inference(model_name=temporal_model_name, model_path=temporal_model_path, batch_size=batch_size, max_seq_len=max_seq_len, temporal_cache_file_path=temporal_cache_file_path)
            print("TemporalBERT inference initialized successfully")

            print("Processing benchmark items with temporal model...")
            output_similarities_cache = pd.DataFrame(columns=all_paragraphs)

            for benchmark_item in tqdm(benchmark_data):
                question: str = benchmark_item["question"]

                paragraphs: List[str] = benchmark_item["paragraphs"] if not use_all_paragraphs else all_paragraphs
                questions: List[str] = [question] * len(paragraphs)
                reference_dates: List[str] = [reference_date] * len(paragraphs)
                ground_truth: List[float] = [0.0] * len(paragraphs)

                inference.set_sentences(questions, reference_dates, paragraphs, reference_dates, ground_truth)

                output = inference.evaluate()
                
                row = {k: v for k, v in zip(all_paragraphs, output["similarity"])}

                output_similarities_cache.loc[question] = row

        print(f"Saving temporal model similarities to: {temporal_similarities_file_path}")
        output_similarities_cache.to_pickle(temporal_similarities_file_path)
        
        print("Temporal model similarities saved successfully")

        return output_similarities_cache