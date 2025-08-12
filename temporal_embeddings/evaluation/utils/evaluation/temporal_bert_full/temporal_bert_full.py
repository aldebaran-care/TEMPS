from pathlib import Path
import json
from typing import List, Dict

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.inference import Inference
from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.parameters import MAX_SEQ_LEN
from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics, compute_metrics_ranks

def evaluate_temporal_bert_full(model_name: str, external_model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False, use_ranking: bool = False) -> None:
    print(f"Starting TemporalBERT Full evaluation")
    print(f"Temporal model: {model_name} at {model_path}")
    print(f"External model: {external_model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    print(f"Use ranking fusion: {use_ranking}")
    
    TEMPORAL_SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name.replace('-full', '')}/{model_path.stem}/{eval_id}_similarities.json")
    create_folders(TEMPORAL_SIMILARITIES_FILE_PATH.parent)
    print(f"Temporal similarities path: {TEMPORAL_SIMILARITIES_FILE_PATH}")

    TEMPORAL_CACHE_FILE_PATH: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name.replace('-full', '')}/{model_path.stem}/{eval_id}_cache.pkl")
    create_folders(TEMPORAL_CACHE_FILE_PATH.parent)
    print(f"Temporal cache path: {TEMPORAL_CACHE_FILE_PATH}")

    EXTERNAL_SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{external_model_name}/{eval_id}_similarities.json")
    create_folders(EXTERNAL_SIMILARITIES_FILE_PATH.parent)
    print(f"External similarities path: {EXTERNAL_SIMILARITIES_FILE_PATH}")

    EXTERNAL_CACHE_FILE_PATH: Path = Path(f"output/cache/{benchmark_file_path.stem}/{external_model_name}/{eval_id}_cache.pkl")
    create_folders(EXTERNAL_CACHE_FILE_PATH.parent)
    print(f"External cache path: {EXTERNAL_CACHE_FILE_PATH}")

    if "ts_retriever" in str(benchmark_file_path):
            print("Detected ts_retriever benchmark - loading document paragraphs...")
            ts_retriever_paragraphs: List[str] = []
            with Path("data/evaluation/ts_retriever/doc.json").open("r", encoding="utf-8") as f:
                ts_retriever_paragraphs = json.load(f)
            print(f"Loaded {len(ts_retriever_paragraphs)} paragraphs from ts_retriever document")

    def run_temporal_bert(model_name: str, model_path: Path, batch_size: int, max_seq_len: int) -> None:
        print("Starting TemporalBERT evaluation...")
        output_similarities: List[List[float]] = []

        reference_date: str = "09 august 2024"
        print(f"Using reference date: {reference_date}")

        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
            print(f"Loaded {len(benchmark_data)} benchmark items")

            print("Initializing TemporalBERT inference...")
            if TEMPORAL_CACHE_FILE_PATH.exists():
                print(f"Temporal cache file found at: {TEMPORAL_CACHE_FILE_PATH}")
            else:
                print(f"No temporal cache file found - will create new cache")
            
            inference: Inference = Inference(model_name=model_name, model_path=model_path, batch_size=batch_size, max_seq_len=max_seq_len, cache_file_path=TEMPORAL_CACHE_FILE_PATH)
            print("TemporalBERT inference initialized successfully")

            print("Processing benchmark items with TemporalBERT...")
            for benchmark_item in tqdm(benchmark_data):
                question: str = benchmark_item["question"]

                paragraphs: List[str] = benchmark_item["paragraphs"] if "ts_retriever" not in str(benchmark_file_path) else ts_retriever_paragraphs
                questions: List[str] = [question] * len(paragraphs)
                reference_dates: List[str] = [reference_date] * len(paragraphs)
                ground_truth: List[float] = [0.0] * len(paragraphs)

                inference.set_sentences(questions, reference_dates, paragraphs, reference_dates, ground_truth)

                output = inference.evaluate()

                output_similarities.append(output["similarity"])

        print(f"Saving TemporalBERT similarities to: {TEMPORAL_SIMILARITIES_FILE_PATH}")
        with TEMPORAL_SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)
        print("TemporalBERT similarities saved successfully")

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

            print("Processing benchmark items with external model...")
            for benchmark_element in tqdm(benchmark_data):
                ground_truth.append(benchmark_element["answer"])

                question: str = benchmark_element["question"]
                
                # Check cache for question embedding
                if question not in embedding_cache.index:
                    question_emb = model.encode(question, convert_to_tensor=True)
                    embedding_cache.loc[question] = {'embedding': question_emb.cpu().numpy()}
                else:
                    question_emb = embedding_cache.loc[question, 'embedding']

                paragraphs: List[str] = benchmark_element["paragraphs"] if "ts_retriever" not in str(benchmark_file_path) else ts_retriever_paragraphs

                similarities: List[float] = []
                
                for paragraph in paragraphs:
                    # Check cache for paragraph embedding
                    if paragraph not in embedding_cache.index:
                        paragraph_emb = model.encode(paragraph, convert_to_tensor=True)
                        embedding_cache.loc[paragraph] = {'embedding': paragraph_emb.cpu().numpy()}
                    else:
                        paragraph_emb = embedding_cache.loc[paragraph, 'embedding']

                    similarities.append(float(util.cos_sim(question_emb, paragraph_emb)[0].item()))

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
        run_external_model(external_model_name)
        run_temporal_bert(model_name, model_path, batch_size, max_seq_len)
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
        merged_list = [[(x + (2*y)) for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(temporal_similarities, external_similarities)]

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
