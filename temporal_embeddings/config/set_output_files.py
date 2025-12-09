from pathlib import Path
from typing import Dict
import json

from temporal_embeddings.utils.os.folder_management import create_folders

def set_output_files(temporal_model_name: str, temporal_model_path: Path, semantic_model_name: str, benchmark: str, eval_id: int) -> Dict:
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
    if temporal_model_name.endswith("-full"):
        temporal_model_name = temporal_model_name[:-5]

    with open('temporal_embeddings/config/output_config.json', 'r') as f:
        config = json.load(f)

    output_paths = config.get("output_paths", {})

    temporal_similarities_path = Path(output_paths["temporal_similarities_path"]["default"].format(
        benchmark_stem=benchmark,
        model_name=temporal_model_name,
        model_stem=temporal_model_path.stem,
        eval_id=eval_id
    ))

    semantic_similarities_path = Path(output_paths["semantic_similarities_path"]["default"].format(
        benchmark_stem=benchmark,
        external_model_name=semantic_model_name,
        eval_id=eval_id
    ))

    temporal_cache_path = Path(output_paths["temporal_cache_path"]["default"].format(
        benchmark_stem=benchmark,
        model_name=temporal_model_name,
        model_stem=temporal_model_path.stem,
        eval_id=eval_id
    ))

    semantic_cache_path = Path(output_paths["semantic_cache_path"]["default"].format(
        benchmark_stem=benchmark,
        external_model_name=semantic_model_name,
        eval_id=eval_id
    ))

    create_folders([temporal_similarities_path.parent,
                    semantic_similarities_path.parent,
                    temporal_cache_path.parent,
                    semantic_cache_path.parent])

    return {
        "temporal_cache_path": temporal_cache_path,
        "temporal_similarities_path": temporal_similarities_path,
        "semantic_cache_path": semantic_cache_path,
        "semantic_similarities_path": semantic_similarities_path
    }