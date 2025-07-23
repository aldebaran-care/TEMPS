from pathlib import Path
import json
from typing import List, Dict

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np

from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.inference import Inference
from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.parameters import MAX_SEQ_LEN
from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics, compute_metrics_ranks

def evaluate_temporal_bert_full(model_name: str, external_model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool = False, use_ranking: bool = False) -> None:
    TEMPORAL_SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{model_path.stem}/{eval_id}_similarities.json")
    create_folders(TEMPORAL_SIMILARITIES_FILE_PATH.parent)

    EXTERNAL_SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{external_model_name}/{model_path.stem}/{eval_id}_similarities.json")
    create_folders(EXTERNAL_SIMILARITIES_FILE_PATH.parent)

    def run_temporal_bert(model_name: str, model_path: Path, batch_size: int, max_seq_len: int) -> None:
        output_similarities: List[List[float]] = []

        reference_date: str = "09 august 2024"

        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

            inference: Inference = Inference(model_name=model_name, model_path=model_path, batch_size=batch_size, max_seq_len=max_seq_len)

            for benchmark_item in tqdm(benchmark_data):
                question: str = benchmark_item["question"]

                paragraphs: List[str] = benchmark_item["paragraphs"]
                questions: List[str] = [question] * len(paragraphs)
                reference_dates: List[str] = [reference_date] * len(paragraphs)
                ground_truth: List[float] = [0.0] * len(paragraphs)

                inference.set_sentences(questions, reference_dates, paragraphs, reference_dates, ground_truth)

                output = inference.evaluate()

                output_similarities.append(output["similarity"])

        create_folders(TEMPORAL_SIMILARITIES_FILE_PATH.parent)
        with TEMPORAL_SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)

    def run_external_model(model_name: str) -> None:
        model: SentenceTransformer = SentenceTransformer(model_name)
        model.max_seq_length = MAX_SEQ_LEN

        output_similarities: List[List[float]] = []

        benchmark_data: List[Dict] = []
        ground_truth: List[List[int]] = []

        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

            for benchmark_element in tqdm(benchmark_data):
                ground_truth.append(benchmark_element["answer"])

                question: str = benchmark_element["question"]
                question_emb = model.encode(question, convert_to_tensor=True)

                paragraphs: List[str] = benchmark_element["paragraphs"]

                similarities: List[float] = []
                
                for paragraph in paragraphs:
                    paragraph_emb = model.encode(paragraph, convert_to_tensor=True)

                    similarities.append(float(util.cos_sim(question_emb, paragraph_emb)[0].item()))

                output_similarities.append(similarities)

        create_folders(EXTERNAL_SIMILARITIES_FILE_PATH.parent)

        with EXTERNAL_SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)

    if not skip:
        run_external_model(external_model_name)
        run_temporal_bert(model_name, model_path, batch_size, max_seq_len)

    with TEMPORAL_SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f1, EXTERNAL_SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f2:
        temporal_similarities = json.load(f1)
        external_similarities = json.load(f2)

    if use_ranking:
        def borda_count_fusion(temporal_similarities: List[List[float]], external_similarities: List[List[float]]) -> List[List[int]]:
            merged_ranks = []

            for temp_sim, ext_sim in zip(temporal_similarities, external_similarities):
                scores = {}
                
                temp_ranks: List[float] = sorted(range(len(temp_sim)), key=lambda i: temp_sim[i], reverse=True)
                ext_ranks: List[float] = sorted(range(len(ext_sim)), key=lambda i: ext_sim[i], reverse=True)

                for rank, idx in enumerate(temp_ranks):
                    scores[idx] = scores.get(idx, 0) + (len(temp_ranks) - rank)

                for rank, idx in enumerate(ext_ranks):
                    scores[idx] = scores.get(idx, 0) + (len(ext_ranks) - rank)

                sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                merged_ranks.append([idx for idx, _ in sorted_indices])

            return merged_ranks

        ranks: List[List[int]] = borda_count_fusion(temporal_similarities, external_similarities)
        
        ground_truth: List[List[int]] = []

        with open(benchmark_file_path, "r") as f:
            benchmark_data: List[dict] = json.load(f)

            for e in benchmark_data:
                ground_truth.append(e["answer"])

        print(compute_metrics_ranks(ground_truth, ranks, top_k, metric))

    else:
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

        temporal_similarities = normalize_list(temporal_similarities)
        external_similarities = normalize_list(external_similarities)

        merged_list = [[(x + (10*y)) for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(temporal_similarities, external_similarities)]

        merged_similarities: List[List[float]] = merged_list
        ground_truth: List[List[int]] = []

        with open(benchmark_file_path, "r") as f:
            benchmark_data: List[dict] = json.load(f)

            for e in benchmark_data:
                ground_truth.append(e["answer"])

        print(compute_metrics(ground_truth, merged_similarities, top_k, metric))
