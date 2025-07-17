from typing import List, Dict
import json
from pathlib import Path

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_accuracy

def evaluate_sentence_transformer(model_name: str, max_seq_len: int, dataset_file_path: Path, eval_id: int, top_k: int) -> None:
    model: SentenceTransformer = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len

    output_similarities: List[List[float]] = []

    data: List[Dict] = []
    ground_truth: List[int] = []

    with dataset_file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

        for element in tqdm(data):
            ground_truth.append(element["answer"])

            question: str = element["question"]
            question_emb = model.encode(question, convert_to_tensor=True) if model_name != "BAAI/bge-large-en" else model.encode(question, convert_to_tensor=True, normalize_embeddings=True)

            paragraphs: List[str] = element["paragraphs"]

            similarities: List[float] = []
            
            for paragraph in paragraphs:
                paragraph_emb = model.encode(paragraph, convert_to_tensor=True) if model_name != "BAAI/bge-large-en" else model.encode(paragraph, convert_to_tensor=True, normalize_embeddings=True)

                similarities.append(float(util.cos_sim(question_emb, paragraph_emb)[0].item()))

            output_similarities.append(similarities)

    similarities_file_path: Path = Path(f"output/similarities/{model_name}/{eval_id}_{model_name}_similarities.json")
    create_folders(similarities_file_path.parent)
    
    with similarities_file_path.open("w", encoding="utf-8") as g:
        json.dump(output_similarities, g, indent=4, ensure_ascii=False)

    print(compute_accuracy(ground_truth, output_similarities, top_k))