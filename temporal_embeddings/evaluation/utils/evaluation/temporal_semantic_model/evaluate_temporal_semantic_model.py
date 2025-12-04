from pathlib import Path
import json
from typing import List, Dict

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import torch

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.parameters import MAX_SEQ_LEN
from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics, compute_metrics_ranks
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import compute_temporal_similarities

def evaluate_temporal_semantic_model(temporal_model_name: str, semantic_model_name: str, temporal_model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False, use_ranking: bool = False, alpha: float = 0.5, use_all_paragraphs: bool = False) -> None:
    print(f"Starting temporal semantic model evaluation")
    print(f"Temporal model: {temporal_model_name} at {temporal_model_path}")
    print(f"Semantic model: {semantic_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    print(f"Use ranking fusion: {use_ranking}")
    print(f"Use all paragraphs: {use_all_paragraphs}")
    
    temporal_cache_path, temporal_similarities_path, semantic_cache_path, semantic_similarities_path = set_output_files(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name=semantic_model_name,
        benchmark_file_path=benchmark_file_path,
        eval_id=eval_id
    ).values()

    temporal_similarities: pd.DataFrame = compute_temporal_similarities(
        temporal_model_name=temporal_model_name,
        temporal_model_path=temporal_model_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        benchmark_file_path=benchmark_file_path,
        temporal_cache_file_path=temporal_cache_path,
        temporal_similarities_file_path=temporal_similarities_path,
        use_all_paragraphs=use_all_paragraphs
    )

    def run_external_model(model_name: str) -> None:
        print(f"Starting external model evaluation: {model_name}")
        model: SentenceTransformer = SentenceTransformer(model_name)
        model.max_seq_length = MAX_SEQ_LEN
        print(f"External model loaded with max_seq_length: {MAX_SEQ_LEN}")

        # Load embedding cache if exists
        embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])
        if EXTERNAL_CACHE_FILE_PATH.exists():
            print(f"External cache file found at: {EXTERNAL_CACHE_FILE_PATH}")
            embedding_cache = pd.read_pickle(EXTERNAL_CACHE_FILE_PATH)
            print(f"Loaded external cache with {len(embedding_cache)} embeddings")
        else:
            print(f"No external cache file found - will create new cache")

        output_similarities: List[List[float]] = []

        benchmark_data: List[Dict] = []
        ground_truth: List[List[int]] = []

        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
            print(f"Loaded {len(benchmark_data)} benchmark items for external model")

            if use_all_paragraphs:
                all_paragraphs: List[str] = []
                for item in benchmark_data:
                    all_paragraphs.extend(item["paragraphs"])
                all_paragraphs = sorted(list(set(all_paragraphs)))
                print(f"Using all paragraphs mode: {len(all_paragraphs)} unique paragraphs")

            print("Processing benchmark items with external model...")
            for benchmark_element in tqdm(benchmark_data):
                ground_truth.append(benchmark_element["answer"])

                question: str = benchmark_element["question"]
                
                # Check cache for question embedding
                if question not in embedding_cache.index:
                    question_emb = model.encode(question, convert_to_tensor=True).cpu().numpy()
                    embedding_cache.loc[question] = {'embedding': question_emb}
                else:
                    question_emb = embedding_cache.loc[question, 'embedding']

                paragraphs: List[str] = benchmark_element["paragraphs"] if not use_all_paragraphs else all_paragraphs

                similarities: List[float] = []
                
                for paragraph in paragraphs:
                    # Check cache for paragraph embedding
                    if paragraph not in embedding_cache.index:
                        paragraph_emb = model.encode(paragraph, convert_to_tensor=True).cpu().numpy()
                        embedding_cache.loc[paragraph] = {'embedding': paragraph_emb}
                    else:
                        paragraph_emb = embedding_cache.loc[paragraph, 'embedding']

                    similarities.append(float(util.cos_sim(torch.Tensor(question_emb).cpu(), torch.Tensor(paragraph_emb).cpu())[0].item()))

                output_similarities.append(similarities)

        # Save embedding cache as pandas DataFrame
        print(f"Saving external cache to: {EXTERNAL_CACHE_FILE_PATH}")
        embedding_cache.to_pickle(EXTERNAL_CACHE_FILE_PATH)
        print(f"External cache saved with {len(embedding_cache)} embeddings")

        print(f"Saving external similarities to: {EXTERNAL_SIMILARITIES_FILE_PATH}")
        with EXTERNAL_SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)
        print("External similarities saved successfully")

    if not skip:
        print("Running both models (not skipping)...")
        run_external_model(semantic_model_name)
        run_temporal_bert(temporal_model_name, temporal_model_path, batch_size, max_seq_len)
    else:
        print("Skipping model runs - using existing similarities")

    print("Loading similarities for fusion...")
    with TEMPORAL_SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f1, EXTERNAL_SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f2:
        temporal_similarities = json.load(f1)
        external_similarities = json.load(f2)
    
    print(f"Loaded {len(temporal_similarities)} temporal similarity lists")
    print(f"Loaded {len(external_similarities)} external similarity lists")

    if use_ranking:
        print("Using Borda count ranking fusion...")
        def borda_count_fusion(temporal_similarities: List[List[float]], external_similarities: List[List[float]]) -> List[List[int]]:
            merged_ranks = []

            for temp_sim, ext_sim in zip(temporal_similarities, external_similarities):
                scores = {}
                
                temp_ranks: List[float] = sorted(range(len(temp_sim)), key=lambda i: temp_sim[i], reverse=True)
                ext_ranks: List[float] = sorted(range(len(ext_sim)), key=lambda i: ext_sim[i], reverse=True)

                for rank, idx in enumerate(temp_ranks):
                    scores[idx] = scores.get(idx, 0) + (len(temp_ranks) - rank)

                for rank, idx in enumerate(ext_ranks):
                    scores[idx] = scores.get(idx, 0) + ((len(ext_ranks) - rank) * 2)

                sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                merged_ranks.append([idx for idx, _ in sorted_indices])

            return merged_ranks

        ranks: List[List[int]] = borda_count_fusion(temporal_similarities, external_similarities)
        print("Borda count fusion completed")
        
        print("Loading ground truth for ranking evaluation...")
        ground_truth: List[List[int]] = []

        with open(benchmark_file_path, "r") as f:
            benchmark_data: List[dict] = json.load(f)

            for e in benchmark_data:
                ground_truth.append(e["answer"])

        print(f"Computing ranking metrics with top_k={top_k}, metric={metric}")
        print(compute_metrics_ranks(ground_truth, ranks, top_k, metric))

    else:
        print("Using score normalization and weighted fusion...")
        def normalize_list(lst: List[List[float]]) -> List[List[float]]:
            normalized = []
            
            for sublist in lst:
                arr = np.array(sublist)
                if arr.max() - arr.min() == 0:
                    normalized.append([0.0 for _ in arr])
            
                else:
                    norm = (arr - arr.min()) / (arr.max() - arr.min())
                    normalized.append(norm.tolist())
            
            return normalized

        print("Normalizing similarity scores...")
        temporal_similarities = normalize_list(temporal_similarities)
        external_similarities = normalize_list(external_similarities)

        print("Merging similarities with weights (temporal: 1x, external: 2x)...")
        merged_list = [[((alpha*x) + ((1-alpha)*y)) for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(temporal_similarities, external_similarities)]

        merged_similarities: List[List[float]] = merged_list
        print("Score fusion completed")
        
        print("Loading ground truth for score evaluation...")
        ground_truth: List[List[int]] = []

        with open(benchmark_file_path, "r") as f:
            benchmark_data: List[dict] = json.load(f)

            for e in benchmark_data:
                ground_truth.append(e["answer"])

        print(f"Computing score metrics with top_k={top_k}, metric={metric}")
        print(compute_metrics(ground_truth, merged_similarities, top_k, metric))
    
    print("TemporalBERT Full evaluation completed successfully")
