import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import (
    compute_semantic_similarities,
)
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import (
    compute_temporal_similarities,
)
from temporal_embeddings.utils.math.normalize import normalize_list
from temporal_embeddings.utils.os.folder_management import create_folders


BENCHMARK_PATHS: Dict[str, Path] = {
    "time_sensitive_qa": Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json"),
    "temp_reason": Path("data/evaluation/temp_reason/processed_data.json"),
    "ts_retriever": Path("data/evaluation/ts_retriever/processed_ts_retriever.json"),
}

METRIC_COLUMNS: List[Tuple[str, str]] = [
    ("mrr", "MRR"),
    ("ndcg", "NDCG@5"),
    ("recall", "Recall@5"),
    ("precision", "Precision@5"),
]


@dataclass
class RunSpec:
    run_id: str
    run_type: str
    temporal_model_name: str
    semantic_model_name: str
    alpha: float


def _validate_config(config: Dict[str, Any], config_path: Path) -> None:
    required_sections = ["global", "benchmarks", "temporal", "semantic_baselines", "hybrid", "report"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing} in {config_path}")

    top_k = int(config["global"].get("top_k", 5))
    if top_k != 5:
        raise ValueError("Only top_k=5 is supported for this paper runner to keep fixed report columns.")

    temporal = config["temporal"]
    needs_temporal_checkpoint = temporal.get("enabled", False) or config["hybrid"].get("enabled", False)
    if needs_temporal_checkpoint:
        model_path = temporal.get("model_path", "")
        if not model_path:
            raise ValueError("temporal.model_path is required when temporal.enabled=true or hybrid.enabled=true")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Temporal checkpoint not found: {model_path}")

    enabled_benchmarks = [
        benchmark_name
        for benchmark_name, benchmark_cfg in config["benchmarks"].items()
        if benchmark_cfg.get("enabled", False)
    ]
    if not enabled_benchmarks:
        raise ValueError("At least one benchmark must be enabled in benchmarks.*.enabled")

    invalid_benchmarks = [benchmark for benchmark in enabled_benchmarks if benchmark not in BENCHMARK_PATHS]
    if invalid_benchmarks:
        raise ValueError(
            f"Unsupported benchmark ids in config: {invalid_benchmarks}. "
            f"Supported ids: {list(BENCHMARK_PATHS.keys())}"
        )
    for benchmark in enabled_benchmarks:
        benchmark_path = BENCHMARK_PATHS[benchmark]
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found for {benchmark}: {benchmark_path}")

    if (
        not config["temporal"].get("enabled", False)
        and not config["semantic_baselines"].get("enabled", False)
        and not config["hybrid"].get("enabled", False)
    ):
        raise ValueError("Enable at least one run family: temporal, semantic_baselines, or hybrid.")

    if config["semantic_baselines"].get("enabled", False):
        semantic_models = config["semantic_baselines"].get("models", [])
        if not semantic_models:
            raise ValueError("semantic_baselines.models must be non-empty when semantic_baselines.enabled=true")

    if config["hybrid"].get("enabled", False):
        hybrid_cfg = config["hybrid"]
        semantic_models = hybrid_cfg.get("semantic_models", [])
        alphas = hybrid_cfg.get("alphas", [0.5])
        if not semantic_models:
            raise ValueError("hybrid.semantic_models must be non-empty when hybrid.enabled=true")
        if not alphas:
            raise ValueError("hybrid.alphas must be non-empty when hybrid.enabled=true")

    report_path = config["report"].get("output_path", "")
    if not report_path:
        raise ValueError("report.output_path is required")


def _get_enabled_benchmarks(config: Dict[str, Any]) -> List[str]:
    benchmarks: List[str] = []
    for benchmark_name, benchmark_cfg in config["benchmarks"].items():
        if benchmark_cfg.get("enabled", False):
            benchmarks.append(benchmark_name)
    return benchmarks


def _build_run_matrix(config: Dict[str, Any]) -> List[RunSpec]:
    runs: List[RunSpec] = []

    temporal_cfg = config["temporal"]
    semantic_cfg = config["semantic_baselines"]
    hybrid_cfg = config["hybrid"]

    if temporal_cfg.get("enabled", False):
        temporal_model_name = temporal_cfg.get("model_name", "all-minilm-l6-v2")
        runs.append(
            RunSpec(
                run_id=f"temporal:{temporal_model_name}",
                run_type="temporal",
                temporal_model_name=temporal_model_name,
                semantic_model_name="",
                alpha=0.0,
            )
        )

    if semantic_cfg.get("enabled", False):
        for semantic_model_name in semantic_cfg.get("models", []):
            runs.append(
                RunSpec(
                    run_id=f"semantic:{semantic_model_name}",
                    run_type="semantic",
                    temporal_model_name="",
                    semantic_model_name=semantic_model_name,
                    alpha=0.0,
                )
            )

    if hybrid_cfg.get("enabled", False):
        temporal_model_name = hybrid_cfg.get("temporal_model_name", "all-minilm-l6-v2-full")
        for semantic_model_name in hybrid_cfg.get("semantic_models", []):
            for alpha in hybrid_cfg.get("alphas", [0.5]):
                runs.append(
                    RunSpec(
                        run_id=f"hybrid:{temporal_model_name}+{semantic_model_name}:alpha={alpha}",
                        run_type="hybrid",
                        temporal_model_name=temporal_model_name,
                        semantic_model_name=semantic_model_name,
                        alpha=float(alpha),
                    )
                )

    return runs


def _load_benchmark_data(
    benchmark_name: str,
    benchmark_path: Path,
    num_negative_samples: int,
    benchmark_cache: Dict[Tuple[str, int], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    cache_key = (benchmark_name, num_negative_samples)
    if cache_key in benchmark_cache:
        return benchmark_cache[cache_key]

    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark_data: List[Dict[str, Any]] = add_negative_samples(
            json.load(f),
            num_negatives=num_negative_samples,
        )
    benchmark_cache[cache_key] = benchmark_data
    return benchmark_data


def _get_temporal_similarities(
    run_spec: RunSpec,
    benchmark_name: str,
    benchmark_path: Path,
    config: Dict[str, Any],
    temporal_model_path: Path,
    similarity_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output_paths = set_output_files(
        temporal_model_name=run_spec.temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name="",
        benchmark=benchmark_name,
    )
    similarities_path = output_paths["temporal_similarities_path"]
    cache_key = str(similarities_path)

    if cache_key in similarity_cache:
        return similarity_cache[cache_key]

    if similarities_path.exists():
        temporal_similarities = pd.read_pickle(similarities_path)
    else:
        temporal_similarities = compute_temporal_similarities(
            temporal_model_name=run_spec.temporal_model_name,
            temporal_model_path=temporal_model_path,
            batch_size=int(config["global"].get("batch_size", 128)),
            max_seq_len=int(config["global"].get("max_seq_len", 512)),
            benchmark_file_path=benchmark_path,
            temporal_cache_file_path=output_paths["temporal_cache_path"],
            temporal_similarities_file_path=similarities_path,
            num_negative_samples=int(config["global"].get("num_negative_samples", 0)),
            reference_date=str(config["global"].get("reference_date", "2021-11-09")),
        )
    similarity_cache[cache_key] = temporal_similarities
    return temporal_similarities


def _get_semantic_similarities(
    semantic_model_name: str,
    temporal_model_path: Path,
    benchmark_name: str,
    benchmark_path: Path,
    config: Dict[str, Any],
    similarity_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output_paths = set_output_files(
        temporal_model_name="",
        temporal_model_path=temporal_model_path,
        semantic_model_name=semantic_model_name,
        benchmark=benchmark_name,
    )
    similarities_path = output_paths["semantic_similarities_path"]
    cache_key = str(similarities_path)

    if cache_key in similarity_cache:
        return similarity_cache[cache_key]

    if similarities_path.exists():
        semantic_similarities = pd.read_pickle(similarities_path)
    else:
        semantic_similarities = compute_semantic_similarities(
            semantic_model_name=semantic_model_name,
            max_seq_len=int(config["global"].get("max_seq_len", 512)),
            benchmark_file_path=benchmark_path,
            semantic_cache_file_path=output_paths["semantic_cache_path"],
            semantic_similarities_file_path=similarities_path,
            num_negative_samples=int(config["global"].get("num_negative_samples", 0)),
        )

    similarity_cache[cache_key] = semantic_similarities
    return semantic_similarities


def _merge_hybrid_similarities(
    temporal_similarities: pd.DataFrame,
    semantic_similarities: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    return (alpha * temporal_similarities) + ((1.0 - alpha) * semantic_similarities)


def _normalize_similarity_df(
    similarities_df: pd.DataFrame,
    normalized_cache: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    cache_key = id(similarities_df)
    if cache_key in normalized_cache:
        return normalized_cache[cache_key]

    normalized_df = similarities_df.apply(lambda row: normalize_list(row.tolist()), axis=1, result_type="expand")
    normalized_df.columns = similarities_df.columns
    normalized_df.index = similarities_df.index
    normalized_cache[cache_key] = normalized_df
    return normalized_df


def _compute_similarity_lists(
    similarities_df: pd.DataFrame,
    benchmark_data: List[Dict[str, Any]],
) -> List[List[float]]:
    similarities_list: List[List[float]] = []
    for benchmark_item in benchmark_data:
        question = benchmark_item["question"]
        paragraphs = benchmark_item["paragraphs"]
        question_similarities = similarities_df.loc[question][paragraphs].tolist()
        similarities_list.append(question_similarities)
    return similarities_list


def _compute_report_metrics(
    ground_truth: List[List[int]],
    similarities_list: List[List[float]],
    top_k: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for metric_name, _ in METRIC_COLUMNS:
        result = compute_metrics(
            first_list=ground_truth,
            second_list=similarities_list,
            top_k=top_k,
            metric=metric_name,
        )
        metrics[metric_name] = float(result[metric_name])
    return metrics


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _write_markdown_report(
    report_path: Path,
    rows: List[Dict[str, Any]],
    include_macro_average: bool,
) -> None:
    create_folders([report_path.parent])

    headers = [
        "Run ID",
        "Type",
        "Temporal Model",
        "Semantic Model",
        "Alpha",
        "Benchmark",
        "Queries",
        "MRR",
        "NDCG@5",
        "Recall@5",
        "Precision@5",
    ]

    markdown_lines: List[str] = []
    markdown_lines.append("# Paper Evaluation Report")
    markdown_lines.append("")
    markdown_lines.append("| " + " | ".join(headers) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    row["run_id"],
                    row["run_type"],
                    row["temporal_model_name"],
                    row["semantic_model_name"],
                    row["alpha"],
                    row["benchmark"],
                    str(row["num_queries"]),
                    _format_float(row["mrr"]),
                    _format_float(row["ndcg"]),
                    _format_float(row["recall"]),
                    _format_float(row["precision"]),
                ]
            )
            + " |"
        )

    if include_macro_average:
        markdown_lines.append("")
        markdown_lines.append("## Macro Averages")
        markdown_lines.append("")
        markdown_lines.append("| Run ID | MRR | NDCG@5 | Recall@5 | Precision@5 |")
        markdown_lines.append("| --- | --- | --- | --- | --- |")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["run_id"], []).append(row)

        for run_id, run_rows in grouped.items():
            macro_mrr = sum(r["mrr"] for r in run_rows) / len(run_rows)
            macro_ndcg = sum(r["ndcg"] for r in run_rows) / len(run_rows)
            macro_recall = sum(r["recall"] for r in run_rows) / len(run_rows)
            macro_precision = sum(r["precision"] for r in run_rows) / len(run_rows)

            markdown_lines.append(
                f"| {run_id} | {_format_float(macro_mrr)} | {_format_float(macro_ndcg)} | "
                f"{_format_float(macro_recall)} | {_format_float(macro_precision)} |"
            )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def run_paper_evaluations(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    _validate_config(config, config_path)

    top_k = int(config["global"].get("top_k", 5))
    num_negative_samples = int(config["global"].get("num_negative_samples", 0))
    temporal_model_path = Path(config["temporal"].get("model_path", ""))
    enabled_benchmarks = _get_enabled_benchmarks(config)
    run_matrix = _build_run_matrix(config)

    benchmark_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    similarity_cache: Dict[str, pd.DataFrame] = {}
    normalized_similarity_cache: Dict[int, pd.DataFrame] = {}
    report_rows: List[Dict[str, Any]] = []

    print(f"Running {len(run_matrix)} run configurations over {len(enabled_benchmarks)} benchmarks.")

    for run_spec in run_matrix:
        print(f"\n=== Run: {run_spec.run_id} ===")
        for benchmark_name in enabled_benchmarks:
            benchmark_path = BENCHMARK_PATHS[benchmark_name]
            print(f"[{run_spec.run_id}] Benchmark: {benchmark_name}")

            benchmark_data = _load_benchmark_data(
                benchmark_name=benchmark_name,
                benchmark_path=benchmark_path,
                num_negative_samples=num_negative_samples,
                benchmark_cache=benchmark_cache,
            )
            ground_truth = [item["answer"] for item in benchmark_data]

            if run_spec.run_type == "temporal":
                similarities_df = _get_temporal_similarities(
                    run_spec=run_spec,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    temporal_model_path=temporal_model_path,
                    similarity_cache=similarity_cache,
                )
            elif run_spec.run_type == "semantic":
                similarities_df = _get_semantic_similarities(
                    semantic_model_name=run_spec.semantic_model_name,
                    temporal_model_path=temporal_model_path,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    similarity_cache=similarity_cache,
                )
            else:
                temporal_df = _get_temporal_similarities(
                    run_spec=run_spec,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    temporal_model_path=temporal_model_path,
                    similarity_cache=similarity_cache,
                )
                semantic_df = _get_semantic_similarities(
                    semantic_model_name=run_spec.semantic_model_name,
                    temporal_model_path=temporal_model_path,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    similarity_cache=similarity_cache,
                )
                normalized_temporal_df = _normalize_similarity_df(temporal_df, normalized_similarity_cache)
                normalized_semantic_df = _normalize_similarity_df(semantic_df, normalized_similarity_cache)
                similarities_df = _merge_hybrid_similarities(
                    temporal_similarities=normalized_temporal_df,
                    semantic_similarities=normalized_semantic_df,
                    alpha=run_spec.alpha,
                )

            similarities_list = _compute_similarity_lists(similarities_df, benchmark_data)
            metrics = _compute_report_metrics(ground_truth, similarities_list, top_k=top_k)

            report_rows.append(
                {
                    "run_id": run_spec.run_id,
                    "run_type": run_spec.run_type,
                    "temporal_model_name": run_spec.temporal_model_name,
                    "semantic_model_name": run_spec.semantic_model_name,
                    "alpha": f"{run_spec.alpha:.2f}" if run_spec.run_type == "hybrid" else "-",
                    "benchmark": benchmark_name,
                    "num_queries": len(benchmark_data),
                    "mrr": metrics["mrr"],
                    "ndcg": metrics["ndcg"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                }
            )

            print(
                "Metrics: "
                f"MRR={metrics['mrr']:.6f}, "
                f"NDCG@5={metrics['ndcg']:.6f}, "
                f"Recall@5={metrics['recall']:.6f}, "
                f"Precision@5={metrics['precision']:.6f}"
            )

    report_path = Path(config["report"]["output_path"])
    include_macro_average = bool(config["report"].get("include_macro_average", True))
    _write_markdown_report(report_path, report_rows, include_macro_average=include_macro_average)
    print(f"\nEvaluation report written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper evaluations with one config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="temporal_embeddings/config/paper_evaluation_config.json",
        help="Path to evaluation config JSON file.",
    )
    args = parser.parse_args()

    run_paper_evaluations(Path(args.config))


if __name__ == "__main__":
    main()
