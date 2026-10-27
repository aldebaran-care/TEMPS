import argparse

from temporal_embeddings.evaluation.benchmarks.time_sensitive_qa import create_time_sensitive_qa_benchmark
from temporal_embeddings.evaluation.benchmarks.ts_retriever import create_ts_retriever_benchmark
from temporal_embeddings.evaluation.benchmarks.temp_reason_l1 import create_temp_reason_l1_benchmark
from temporal_embeddings.evaluation.benchmarks.temp_reason_l2 import create_temp_reason_l2_benchmark
from temporal_embeddings.evaluation.benchmarks.temp_reason import create_temp_reason_benchmark
from temporal_embeddings.evaluation.benchmarks.menat_qa import create_menat_qa_benchmark
from temporal_embeddings.evaluation.benchmarks.temp_rag_eval import create_temp_rag_eval_dataset

def create_evaluation_dataset(dataset_name):
    if dataset_name.lower().startswith("time_sensitive_qa"):
        create_time_sensitive_qa_benchmark(add_negative_samples=False)
    
    elif dataset_name.lower().startswith("menat_qa"):
        create_menat_qa_benchmark(dataset_name)

    elif dataset_name.lower().startswith("ts_retriever"):
        create_ts_retriever_benchmark()

    elif dataset_name.lower().startswith("temp_reason_l1"):
        create_temp_reason_l1_benchmark()

    elif dataset_name.lower().startswith("temp_reason_l2"):
        create_temp_reason_l2_benchmark()

    elif dataset_name.lower().startswith("temp_reason"):
        create_temp_reason_benchmark()

    elif dataset_name.lower().startswith("temp_rag_eval"):
        create_temp_rag_eval_dataset()
        
    else:
        raise ValueError("Unsupported dataset name.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation dataset.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to process")
    args = parser.parse_args()
    create_evaluation_dataset(args.dataset_name)