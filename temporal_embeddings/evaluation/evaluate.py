from pathlib import Path

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.evaluate_temporal_model import evaluate_temporal_model
from temporal_embeddings.evaluation.utils.evaluation.temporal_semantic_model.evaluate_temporal_semantic_model import evaluate_temporal_semantic_model
from temporal_embeddings.evaluation.utils.dataset_file_path import get_dataset_file_path
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.evaluate_semantic_model import evaluate_semantic_model
from temporal_embeddings.evaluation.utils.evaluation.bm25.evaluate_bm25 import evaluate_bm25
from temporal_embeddings.evaluation.utils.evaluation.sutime.evaluate_sutime import evaluate_sutime

def evaluate_model(model_name: str, external_model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark: str, eval_id: str, top_k: int, metric: str, alpha: float, num_negative_samples: int = 0) -> None:
    benchmark_file_path: Path = get_dataset_file_path(benchmark)

    if model_name in ["temporal_bert", "all-minilm-l6-v2", "prajjwal1/bert-tiny"]:
        evaluate_temporal_model(model_name, model_path, batch_size, max_seq_len, benchmark, benchmark_file_path, eval_id, top_k, metric, num_negative_samples)
    
    elif model_name in ["temporal_bert_full", "all-minilm-l6-v2-full", "prajjwal1/bert-tiny-full"]:
        evaluate_temporal_semantic_model(model_name, external_model_name, model_path, batch_size, max_seq_len, benchmark, benchmark_file_path, eval_id, top_k, metric, alpha, num_negative_samples)

    elif model_name == "bm25":
        evaluate_bm25(model_name, benchmark, benchmark_file_path, eval_id, top_k, metric, num_negative_samples)

    elif model_name == "sutime":
        evaluate_sutime(model_name, benchmark, benchmark_file_path, eval_id, top_k, metric, num_negative_samples)

    else:
        evaluate_semantic_model(model_name, max_seq_len, benchmark, benchmark_file_path, eval_id, top_k, metric, num_negative_samples)