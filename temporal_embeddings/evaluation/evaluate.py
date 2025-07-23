from pathlib import Path

from temporal_embeddings.evaluation.utils.evaluation.temporal_bert.temporal_bert_evaluation import evaluate_temporal_bert
from temporal_embeddings.evaluation.utils.evaluation.temporal_bert_full.temporal_bert_full import evaluate_temporal_bert_full
from temporal_embeddings.evaluation.utils.evaluation.mistral.mistral_evaluation import evaluate_mistral
from temporal_embeddings.evaluation.utils.evaluation.alibaba.alibaba_evaluation import evaluate_alibaba
from temporal_embeddings.evaluation.utils.evaluation.salesforce.salesforce_evaluation import evaluate_salesforce
from temporal_embeddings.evaluation.utils.dataset_file_path import get_dataset_file_path
from temporal_embeddings.evaluation.utils.evaluate_sentence_transformer import evaluate_sentence_transformer

def evaluate_model(model_name: str, external_model_name: str, model_path: Path, batch_size: int, max_seq_len: int, benchmark: str, eval_id: int, top_k: int, metric: str, skip: bool = False) -> None:
    dataset_file_path: Path = get_dataset_file_path(benchmark)

    if model_name in ["temporal_bert", "all-minilm-l6-v2"]:
        evaluate_temporal_bert(model_name, model_path, batch_size, max_seq_len, dataset_file_path, eval_id, top_k, metric, skip)
    
    elif model_name in ["temporal_bert_full", "all-minilm-l6-v2-full"]:
        evaluate_temporal_bert_full(model_name, external_model_name, model_path, batch_size, max_seq_len, dataset_file_path, eval_id, top_k, metric, skip)

    elif model_name == "mistral":
        evaluate_mistral()
    
    elif model_name == "alibaba":
        evaluate_alibaba()
    
    elif model_name == "salesforce":
        evaluate_salesforce(dataset_file_path, eval_id, top_k, metric, skip)

    else:
        evaluate_sentence_transformer(model_name, max_seq_len, dataset_file_path, eval_id, top_k, metric, skip)
