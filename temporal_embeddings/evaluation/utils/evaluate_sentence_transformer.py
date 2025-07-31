from typing import List, Dict
import json
from pathlib import Path

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import pandas as pd

from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics

def evaluate_sentence_transformer(model_name: str, max_seq_len: int, benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool) -> None:
    model: SentenceTransformer = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len
    similarities_file_path: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{eval_id}_similarities.json")
    
    cache_file_path: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name}/{eval_id}_cache.pkl")
    create_folders(cache_file_path.parent)

    if "ts_retriever" in str(benchmark_file_path):
            ts_retriever_paragraphs: List[str] = []
            with Path("data/evaluation/ts_retriever/doc.json").open("r", encoding="utf-8") as f:
                ts_retriever_paragraphs = json.load(f)

    if not skip:
        # Load embedding cache if exists
        embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])
        if cache_file_path.exists():
            embedding_cache = pd.read_pickle(cache_file_path)

        output_similarities: List[List[float]] = []

        data: List[Dict] = []
        ground_truth: List[int] = []

        with benchmark_file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

            for element in tqdm(data):
                ground_truth.append(element["answer"])

                question: str = element["question"]
                
                # Check cache for question embedding
                if question not in embedding_cache.index:
                    question_emb = model.encode(question, convert_to_tensor=True) if model_name != "BAAI/bge-large-en" else model.encode(question, convert_to_tensor=True, normalize_embeddings=True)
                    embedding_cache.loc[question] = {'embedding': question_emb.cpu()}
                else:
                    question_emb = embedding_cache.loc[question, 'embedding']

                paragraphs: List[str] = element["paragraphs"] if "ts_retriever" not in str(benchmark_file_path) else ts_retriever_paragraphs

                similarities: List[float] = []
                
                for paragraph in paragraphs:
                    # Check cache for paragraph embedding
                    if paragraph not in embedding_cache.index:
                        paragraph_emb = model.encode(paragraph, convert_to_tensor=True) if model_name != "BAAI/bge-large-en" else model.encode(paragraph, convert_to_tensor=True, normalize_embeddings=True)
                        embedding_cache.loc[paragraph] = {'embedding': paragraph_emb.cpu()}
                    else:
                        paragraph_emb = embedding_cache.loc[paragraph, 'embedding']

                    similarities.append(float(util.cos_sim(question_emb, paragraph_emb)[0].item()))

                output_similarities.append(similarities)

        # Save embedding cache as pandas DataFrame
        embedding_cache.to_pickle(cache_file_path)

        create_folders(similarities_file_path.parent)
        
        with similarities_file_path.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)

    ground_truth: List[List[int]] = []

    with benchmark_file_path.open("r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

        for element in data:
            ground_truth.append(element["answer"])

    output_similarities: List[List[float]] = []
    
    with similarities_file_path.open("r", encoding="utf-8") as f:
        output_similarities = json.load(f)

    print(compute_metrics(ground_truth, output_similarities, top_k, metric))