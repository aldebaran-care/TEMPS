from pathlib import Path

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.evaluate_temporal_model import evaluate_temporal_model
from temporal_embeddings.evaluation.utils.evaluation.temporal_semantic_model.evaluate_temporal_semantic_model import evaluate_temporal_semantic_model
from temporal_embeddings.evaluation.utils.evaluation.mistral.mistral_evaluation import evaluate_mistral
from temporal_embeddings.evaluation.utils.evaluation.alibaba.alibaba_evaluation import evaluate_alibaba
from temporal_embeddings.evaluation.utils.evaluation.salesforce.salesforce_evaluation import evaluate_salesforce
from temporal_embeddings.evaluation.utils.dataset_file_path import get_dataset_file_path
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.evaluate_semantic_model import evaluate_semantic_model

def evaluate_model(model_name: str, external_model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark: str, eval_id: int, top_k: int, metric: str, alpha: float = 0.5, use_all_paragraphs: bool = False) -> None:
    benchmark_file_path: Path = get_dataset_file_path(benchmark)

    if model_name in ["temporal_bert", "all-minilm-l6-v2"]:
        evaluate_temporal_model(model_name, model_path, batch_size, max_seq_len, benchmark, benchmark_file_path, eval_id, top_k, metric, use_all_paragraphs)
    
    elif model_name in ["temporal_bert_full", "all-minilm-l6-v2-full"]:
        evaluate_temporal_semantic_model(model_name, external_model_name, model_path, batch_size, max_seq_len, benchmark, benchmark_file_path, eval_id, top_k, metric, alpha, use_all_paragraphs)

    elif model_name == "mistral":
        evaluate_mistral()
    
    elif model_name == "alibaba":
        evaluate_alibaba()

    else:
        evaluate_semantic_model(model_name, max_seq_len, benchmark, benchmark_file_path, eval_id, top_k, metric, use_all_paragraphs)