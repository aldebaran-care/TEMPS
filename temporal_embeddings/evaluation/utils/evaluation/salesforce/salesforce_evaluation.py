from typing import Dict, List
from pathlib import Path
import json
from math import ceil

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pandas as pd

from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def encode_in_batches(model, texts_to_encode: List[str], batch_size: int = 128) -> List[List[float]]:
    """Encode texts in batches to optimize memory usage."""

    encoded_embeddings = []
    num_batches = ceil(len(texts_to_encode) / batch_size)
    for i in tqdm(range(num_batches), desc="Encoding batches"):
        batch = texts_to_encode[i * batch_size:(i + 1) * batch_size]
        encoded_embeddings.extend(model.encode(batch, show_progress_bar=False))
    return encoded_embeddings

def evaluate_salesforce(benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool) -> None:
    model_name = "Salesforce/SFR-Embedding-Mistral"
    print(f"Starting Salesforce evaluation with model: {model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{eval_id}_similarities.json")
    create_folders([SIMILARITIES_FILE_PATH.parent])
    print(f"Similarities will be saved to: {SIMILARITIES_FILE_PATH}")
    print(f"Created similarities directory: {SIMILARITIES_FILE_PATH.parent}")

    CACHE_FILE_PATH: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name}/{eval_id}_cache.pkl")
    create_folders([CACHE_FILE_PATH.parent])
    print(f"Cache file path: {CACHE_FILE_PATH}")
    print(f"Created cache directory: {CACHE_FILE_PATH.parent}")

    # Load embedding cache if exists
    embeddings_cache = pd.DataFrame(columns=["text", "embedding"]).set_index("text")
    if CACHE_FILE_PATH.exists():
        print(f"Cache file found at: {CACHE_FILE_PATH}")
        embeddings_cache = pd.read_pickle(CACHE_FILE_PATH)
        print(f"Loaded cache with {len(embeddings_cache)} embeddings")
    else:
        print(f"No cache file found - will create new cache")

    if not skip:
        print("Starting similarity computation phase...")
        task = 'Given a question with temporal constraints, retrieve relevant passages that answer the question with the correct temporal information.'
        print(f"Using task description: {task}")

        print("Loading Salesforce model...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        print("Salesforce model loaded successfully")

        output_similarities: List[List[float]] = []

        benchmark_data: List[Dict] = []

        print(f"Evaluating model: {model_name}")
        print(f"Dataset file path: {benchmark_file_path}")

        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
            print(f"Loaded {len(benchmark_data)} benchmark items")

            print("Collecting unique texts for batch encoding...")
            unique_texts = set()

            for benchmark_item in tqdm(benchmark_data, desc="Collecting unique texts"):
                question = get_detailed_instruct(task, benchmark_item["question"])
                unique_texts.add(question)
                unique_texts.update(benchmark_item["paragraphs"])

            unique_texts = list(unique_texts)
            print(f"Found {len(unique_texts)} unique texts")
            
            texts_to_encode = [t for t in unique_texts if (t not in embeddings_cache.index)]
            
            if texts_to_encode:
                print(f"Encoding {len(texts_to_encode)} new unique texts...")
                new_embeddings = encode_in_batches(model, texts_to_encode, batch_size=32)
                print("Batch encoding completed - updating cache...")
                for t, emb in zip(texts_to_encode, new_embeddings):
                    embeddings_cache.loc[t] = [emb]
                print(f"Cache updated with {len(texts_to_encode)} new embeddings")
            else:
                print("All texts found in cache - no new encoding needed")

            print("Processing benchmark items for similarity computation...")
            for element in tqdm(benchmark_data, desc="Evaluating"):
                question: str = get_detailed_instruct(task, element["question"])
                paragraphs: List[str] = element["paragraphs"]

                texts = [question] + paragraphs
                embeddings = []

                for t in texts:
                    embeddings.append(embeddings_cache.loc[t, "embedding"])

                similarities: List[float] = []
                
                for i, _ in enumerate(paragraphs):
                    scores = util.cos_sim(embeddings[0], embeddings[i+1])
                    similarities.append(scores.tolist()[0][0])

                output_similarities.append(similarities)

        # Save embedding cache
        print(f"Saving embedding cache to: {CACHE_FILE_PATH}")
        embeddings_cache.to_pickle(CACHE_FILE_PATH)
        print(f"Cache saved with {len(embeddings_cache)} embeddings")

        print(f"Saving similarities to: {SIMILARITIES_FILE_PATH}")
        with SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)
        print("Similarities saved successfully")
    else:
        print("Skipping similarity computation - using existing results")

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data: List[Dict] = json.load(f)

        for element in benchmark_data:
            ground_truth.append(element["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")

    print(f"Loading similarities from: {SIMILARITIES_FILE_PATH}")
    output_similarities: List[List[float]] = []

    with SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f:
        output_similarities = json.load(f)
    print(f"Loaded {len(output_similarities)} similarity lists")

    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    print(compute_metrics(ground_truth, output_similarities, top_k, metric))
    print("Salesforce evaluation completed successfully")