from pathlib import Path
from typing import Dict
import json

from temporal_embeddings.utils.os.folder_management import create_folders

def set_output_files(temporal_model_name: str, temporal_model_path: Path, semantic_model_name: str, benchmark_file_path: Path, eval_id: int) -> Dict:
    """
    Set the output file paths in the configuration dictionary based on the provided parameters.

    Args:
        benchmark_file_path (Path): Path to the benchmark file.
        model_name (str): Name of the temporal model.
        model_stem (str): Stem of the temporal model file.
        external_model_name (str): Name of the external model.
        eval_id (int): Evaluation ID to identify the experiment.
        config (dict): Configuration dictionary to update.

    Returns:
        dict: Updated configuration dictionary with output file paths set.
    """
    with open('temporal_embeddings/config/output_config.json', 'r') as f:
        config = json.load(f)

    benchmark_stem = benchmark_file_path.stem

    output_paths = config.get("output_paths", {})

    temporal_similarities_path = output_paths.get["temporal_similarities_path"].format(
        benchmark_stem=benchmark_stem,
        model_name=temporal_model_name,
        model_stem=temporal_model_path.stem,
        eval_id=eval_id
    )

    semantic_similarities_path = output_paths.get["semantic_similarities_path"].format(
        benchmark_stem=benchmark_stem,
        external_model_name=semantic_model_name,
        eval_id=eval_id
    )

    temporal_cache_path = output_paths.get["temporal_cache_path"].format(
        benchmark_stem=benchmark_stem,
        model_name=temporal_model_name,
        model_stem=temporal_model_path.stem,
        eval_id=eval_id
    )

    semantic_cache_path = output_paths.get["semantic_cache_path"].format(
        benchmark_stem=benchmark_stem,
        external_model_name=semantic_model_name,
        eval_id=eval_id
    )

    create_folders([Path(temporal_similarities_path).parent,
                    Path(semantic_similarities_path).parent,
                    Path(temporal_cache_path).parent,
                    Path(semantic_cache_path).parent])

    return {
        "temporal_cache_path": temporal_cache_path,
        "temporal_similarities_path": temporal_similarities_path,
        "semantic_cache_path": semantic_cache_path,
        "semantic_similarities_path": semantic_similarities_path
    }