from pathlib import Path

def get_dataset_file_path(benchmark: str) -> Path:
    """
    Returns the file path for a given benchmark dataset.

    Args:
        benchmark (str): The name of the benchmark.

    Returns:
        str: The full file path for the specified dataset.
    """
    
    if benchmark == "time_sensitive_qa":
        dataset_file_path: Path = Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json")

    elif benchmark == "menat_qa":
        dataset_file_path: Path = Path("data/evaluation/menat_qa/processed_menat_qa.json")

    elif benchmark == "menat_qa_granularity":
        dataset_file_path = Path("data/evaluation/menat_qa/processed_menat_qa_granularity.json")

    elif benchmark == "menat_qa_counterfactual":
        dataset_file_path = Path("data/evaluation/menat_qa/processed_menat_qa_counterfactual.json")

    elif benchmark == "menat_qa_expand":
        dataset_file_path = Path("data/evaluation/menat_qa/processed_menat_qa_expand.json")

    elif benchmark == "menat_qa_narrow":
        dataset_file_path = Path("data/evaluation/menat_qa/processed_menat_qa_narrow.json")

    elif benchmark == "ts_retriever":
        dataset_file_path = Path("data/evaluation/ts_retriever/processed_ts_retriever.json")

    elif benchmark == "temp_reason_l1":
        dataset_file_path = Path("data/evaluation/temp_reason_l1/processed_data.json")
    
    elif benchmark == "chronoqa":
        dataset_file_path = Path("data/evaluation/chronoqa/chronoqa.json")

    else:
        dataset_file_path = Path(benchmark)

    return dataset_file_path