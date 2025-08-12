from pathlib import Path
import json
from typing import List

from tqdm import tqdm

from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.inference import Inference
from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def evaluate_temporal_bert(model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False) -> None:
    print(f"Starting TemporalBERT evaluation for model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    OUTPUT_SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{model_path.stem}/{eval_id}_similarities.json")
    print(f"Output similarities will be saved to: {OUTPUT_SIMILARITIES_FILE_PATH}")
    create_folders(OUTPUT_SIMILARITIES_FILE_PATH.parent)
    print(f"Created output directory: {OUTPUT_SIMILARITIES_FILE_PATH.parent}")

    CACHE_FILE_PATH: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name}/{model_path.stem}/{eval_id}_cache.pkl")
    print(f"Cache file path: {CACHE_FILE_PATH}")
    create_folders(CACHE_FILE_PATH.parent)
    print(f"Created cache directory: {CACHE_FILE_PATH.parent}")

    if "ts_retriever" in str(benchmark_file_path):
            print("Detected ts_retriever benchmark - loading document paragraphs...")
            ts_retriever_paragraphs: List[str] = []
            with Path("data/evaluation/ts_retriever/doc.json").open("r", encoding="utf-8") as f:
                ts_retriever_paragraphs = json.load(f)
            print(f"Loaded {len(ts_retriever_paragraphs)} paragraphs from ts_retriever document")

    if not skip:
        print("Starting similarity computation phase...")
        if OUTPUT_SIMILARITIES_FILE_PATH.exists():
            print(f"Output similarities file already exists at: {OUTPUT_SIMILARITIES_FILE_PATH}")
        
        output_similarities: List[List[float]] = []

        reference_date: str = "09 august 2024"
        print(f"Using reference date: {reference_date}")

        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
            print(f"Loaded {len(benchmark_data)} benchmark items")

            print("Initializing inference model...")
            if CACHE_FILE_PATH.exists():
                print(f"Cache file found at: {CACHE_FILE_PATH}")
            else:
                print(f"No cache file found - will create new cache at: {CACHE_FILE_PATH}")
            
            inference: Inference = Inference(model_name=model_name, model_path=model_path, batch_size=batch_size, max_seq_len=max_seq_len, cache_file_path=CACHE_FILE_PATH)
            print("Inference model initialized successfully")

            print("Processing benchmark items...")
            for i, benchmark_item in enumerate(tqdm(benchmark_data)):
                question: str = benchmark_item["question"]

                paragraphs: List[str] = benchmark_item["paragraphs"] if "ts_retriever" not in str(benchmark_file_path) else ts_retriever_paragraphs
                questions: List[str] = [question] * len(paragraphs)
                reference_dates: List[str] = [reference_date] * len(paragraphs)
                ground_truth: List[float] = [0.0] * len(paragraphs)

                inference.set_sentences(questions, reference_dates, paragraphs, reference_dates, ground_truth)

                output = inference.evaluate()
                
                output_similarities.append(output["similarity"])

        print(f"Saving similarities to: {OUTPUT_SIMILARITIES_FILE_PATH}")
        with OUTPUT_SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)
        print("Similarities saved successfully")
    else:
        print("Skipping similarity computation - using existing results")

    print(f"Loading similarities from: {OUTPUT_SIMILARITIES_FILE_PATH}")
    with OUTPUT_SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f:
        similarities_list: List[List[float]] = json.load(f)
    print(f"Loaded {len(similarities_list)} similarity lists")
    
    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = json.load(f)

        for e in benchmark_data:
            ground_truth.append(e["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")
    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    print(compute_metrics(ground_truth, similarities_list, top_k, metric))
    print("Evaluation completed successfully")