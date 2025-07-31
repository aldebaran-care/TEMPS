from pathlib import Path
import json
from typing import List

from tqdm import tqdm

from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.inference import Inference
from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def evaluate_temporal_bert(model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False) -> None:
    OUTPUT_SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{model_path.stem}/{eval_id}_similarities.json")
    create_folders(OUTPUT_SIMILARITIES_FILE_PATH.parent)

    CACHE_FILE_PATH: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name}/{model_path.stem}/{eval_id}_cache.pkl")
    create_folders(CACHE_FILE_PATH.parent)

    if not skip:
        output_similarities: List[List[float]] = []

        reference_date: str = "09 august 2024"

        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

            inference: Inference = Inference(model_name=model_name, model_path=model_path, batch_size=batch_size, max_seq_len=max_seq_len, cache_file_path=CACHE_FILE_PATH)

            for benchmark_item in tqdm(benchmark_data):
                question: str = benchmark_item["question"]

                paragraphs: List[str] = benchmark_item["paragraphs"]
                questions: List[str] = [question] * len(paragraphs)
                reference_dates: List[str] = [reference_date] * len(paragraphs)
                ground_truth: List[float] = [0.0] * len(paragraphs)

                inference.set_sentences(questions, reference_dates, paragraphs, reference_dates, ground_truth)

                output = inference.evaluate()
                
                output_similarities.append(output["similarity"])

        with OUTPUT_SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)

    with OUTPUT_SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f:
        similarities_list: List[List[float]] = json.load(f)
    
    ground_truth: List[List[int]] = []

    with open(benchmark_file_path, "r") as f:
        benchmark_data: List[dict] = json.load(f)

        for e in benchmark_data:
            ground_truth.append(e["answer"])

    print(compute_metrics(ground_truth, similarities_list, top_k, metric))