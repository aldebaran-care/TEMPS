import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples
from temporal_embeddings.evaluation.utils.data.splits import (
    VALID_SPLITS,
    select_split_items as _select_split_items,
)
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import (
    compute_semantic_similarities,
)
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import (
    compute_temporal_similarities,
)
from temporal_embeddings.utils.math.normalize import normalize_list
from temporal_embeddings.utils.os.folder_management import create_folders


Similarities = Dict[str, Dict[str, float]]
TEMP_REASON_CHILD_BENCHMARKS = {"temp_reason_l1", "temp_reason_l2"}


BENCHMARK_PATHS: Dict[str, Path] = {
    "time_sensitive_qa": Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json"),
    "temp_reason_l1": Path("data/evaluation/temp_reason_l1/processed_data.json"),
    "temp_reason_l2": Path("data/evaluation/temp_reason_l2/processed_data.json"),
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
    sutime_similarity_mode: str = "merge"


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
        and not config.get("sutime_hybrid", {}).get("enabled", False)
    ):
        raise ValueError(
            "Enable at least one run family: temporal, semantic_baselines, hybrid, or sutime_hybrid."
        )

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

    if config.get("sutime_hybrid", {}).get("enabled", False):
        sutime_cfg = config["sutime_hybrid"]
        if not sutime_cfg.get("semantic_models", []):
            raise ValueError("sutime_hybrid.semantic_models must be non-empty when sutime_hybrid.enabled=true")
        if not sutime_cfg.get("alphas", []):
            raise ValueError("sutime_hybrid.alphas must be non-empty when sutime_hybrid.enabled=true")
        similarity_mode = sutime_cfg.get("similarity_mode", "merge")
        if similarity_mode not in ("merge", "union", "max", "mean", "product"):
            raise ValueError("sutime_hybrid.similarity_mode must be 'merge', 'union', 'max', 'mean', or 'product'")

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

    # Ablation: rule-based grounding + KL (SUTime) fused with a semantic backbone,
    # the same way TEMPS is fused. Isolates the learned temporal module against
    # pure grounding + KL when both are combined with semantics.
    sutime_cfg = config.get("sutime_hybrid", {})
    if sutime_cfg.get("enabled", False):
        similarity_mode = sutime_cfg.get("similarity_mode", "merge")
        for semantic_model_name in sutime_cfg.get("semantic_models", []):
            for alpha in sutime_cfg.get("alphas", [0.5]):
                runs.append(
                    RunSpec(
                        run_id=f"sutime_hybrid:{similarity_mode}:{semantic_model_name}:alpha={alpha}",
                        run_type="sutime_hybrid",
                        temporal_model_name="sutime",
                        semantic_model_name=semantic_model_name,
                        alpha=float(alpha),
                        sutime_similarity_mode=similarity_mode,
                    )
                )

    return runs


def _load_benchmark_data(
    benchmark_path: Path,
    num_negative_samples: int,
) -> List[Dict[str, Any]]:
    with benchmark_path.open("r", encoding="utf-8") as f:
        return add_negative_samples(
            json.load(f),
            num_negatives=num_negative_samples,
        )


def _load_nested_similarity_cache(path: Path) -> Optional[Similarities]:
    if not path.exists():
        return None

    import pandas as pd

    obj = pd.read_pickle(path)
    if isinstance(obj, dict):
        return obj

    print(
        f"Ignoring legacy similarities cache at {path} "
        f"(type {type(obj).__name__}); cannot reuse for TempReason subset."
    )
    return None


def _similarities_cover_benchmark(
    similarities: Similarities,
    benchmark_data: List[Dict[str, Any]],
) -> bool:
    for item in benchmark_data:
        question = item["question"]
        if question not in similarities:
            return False
        paragraph_sims = similarities[question]
        for paragraph in item["paragraphs"]:
            if paragraph not in paragraph_sims:
                return False
    return True


def _get_temp_reason_parent_temporal_similarities(
    run_spec: RunSpec,
    benchmark_name: str,
    temporal_model_path: Path,
    similarity_cache: Dict[str, Similarities],
    benchmark_data: Optional[List[Dict[str, Any]]],
) -> Optional[Similarities]:
    if benchmark_name not in TEMP_REASON_CHILD_BENCHMARKS or benchmark_data is None:
        return None

    parent_paths = set_output_files(
        temporal_model_name=run_spec.temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name="",
        benchmark="temp_reason",
    )
    parent_path = parent_paths["temporal_similarities_path"]
    parent_cache_key = str(parent_path)
    parent_similarities = similarity_cache.get(parent_cache_key)
    if parent_similarities is None:
        parent_similarities = _load_nested_similarity_cache(parent_path)

    if parent_similarities is None:
        return None

    if not _similarities_cover_benchmark(parent_similarities, benchmark_data):
        print(
            f"Parent TempReason temporal cache does not cover {benchmark_name}; "
            f"using the benchmark-specific cache instead."
        )
        return None

    print(f"Reusing parent TempReason temporal similarities for {benchmark_name}: {parent_path}")
    similarity_cache[parent_cache_key] = parent_similarities
    return parent_similarities


def _get_temp_reason_parent_semantic_similarities(
    semantic_model_name: str,
    temporal_model_path: Path,
    benchmark_name: str,
    similarity_cache: Dict[str, Similarities],
    benchmark_data: Optional[List[Dict[str, Any]]],
) -> Optional[Similarities]:
    if benchmark_name not in TEMP_REASON_CHILD_BENCHMARKS or benchmark_data is None:
        return None

    parent_paths = set_output_files(
        temporal_model_name="",
        temporal_model_path=temporal_model_path,
        semantic_model_name=semantic_model_name,
        benchmark="temp_reason",
    )
    parent_path = parent_paths["semantic_similarities_path"]
    parent_cache_key = str(parent_path)
    parent_similarities = similarity_cache.get(parent_cache_key)
    if parent_similarities is None:
        parent_similarities = _load_nested_similarity_cache(parent_path)

    if parent_similarities is None:
        return None

    if not _similarities_cover_benchmark(parent_similarities, benchmark_data):
        print(
            f"Parent TempReason semantic cache does not cover {benchmark_name}; "
            f"using the benchmark-specific cache instead."
        )
        return None

    print(f"Reusing parent TempReason semantic similarities for {benchmark_name}: {parent_path}")
    similarity_cache[parent_cache_key] = parent_similarities
    return parent_similarities


def _get_temporal_similarities(
    run_spec: RunSpec,
    benchmark_name: str,
    benchmark_path: Path,
    config: Dict[str, Any],
    temporal_model_path: Path,
    similarity_cache: Dict[str, Similarities],
    benchmark_data: Optional[List[Dict[str, Any]]] = None,
) -> Similarities:
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

    parent_similarities = _get_temp_reason_parent_temporal_similarities(
        run_spec=run_spec,
        benchmark_name=benchmark_name,
        temporal_model_path=temporal_model_path,
        similarity_cache=similarity_cache,
        benchmark_data=benchmark_data,
    )
    if parent_similarities is not None:
        similarity_cache[cache_key] = parent_similarities
        return parent_similarities

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
        benchmark_data=benchmark_data,
    )
    similarity_cache[cache_key] = temporal_similarities
    return temporal_similarities


def _get_semantic_similarities(
    semantic_model_name: str,
    temporal_model_path: Path,
    benchmark_name: str,
    benchmark_path: Path,
    config: Dict[str, Any],
    similarity_cache: Dict[str, Similarities],
    benchmark_data: Optional[List[Dict[str, Any]]] = None,
) -> Similarities:
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

    parent_similarities = _get_temp_reason_parent_semantic_similarities(
        semantic_model_name=semantic_model_name,
        temporal_model_path=temporal_model_path,
        benchmark_name=benchmark_name,
        similarity_cache=similarity_cache,
        benchmark_data=benchmark_data,
    )
    if parent_similarities is not None:
        similarity_cache[cache_key] = parent_similarities
        return parent_similarities

    semantic_similarities = compute_semantic_similarities(
        semantic_model_name=semantic_model_name,
        max_seq_len=int(config["global"].get("max_seq_len", 512)),
        benchmark_file_path=benchmark_path,
        semantic_cache_file_path=output_paths["semantic_cache_path"],
        semantic_similarities_file_path=similarities_path,
        num_negative_samples=int(config["global"].get("num_negative_samples", 0)),
        benchmark_data=benchmark_data,
    )
    similarity_cache[cache_key] = semantic_similarities
    return semantic_similarities


def _load_sutime_similarities(
    benchmark_name: str,
    benchmark_data: List[Dict[str, Any]],
    similarity_cache: Dict[str, Similarities],
    similarity_mode: str = "merge",
) -> Similarities:
    """Load precomputed SUTime similarities as the
    same nested-dict format used for the TEMPS temporal branch, restricted to
    each question's candidate paragraphs.

    SUTime similarities are produced by `evaluate.py --model_name sutime` and
    stored as a dense (question x paragraph) DataFrame. This drives the
    grounding ablation (run_type "sutime_hybrid"); no SUTime extraction is run
    here — the cache must already exist.
    """
    import pandas as pd

    if similarity_mode not in ("merge", "union", "max", "mean", "product"):
        raise ValueError("similarity_mode must be 'merge', 'union', 'max', 'mean', or 'product'")

    output_paths = set_output_files(
        temporal_model_name="sutime",
        temporal_model_path=Path(""),
        semantic_model_name="",
        benchmark=benchmark_name,
    )
    similarities_path = output_paths["temporal_similarities_path"]
    if similarity_mode in ("union", "max", "mean", "product"):
        similarities_path = similarities_path.with_name(f"{similarities_path.stem}_{similarity_mode}{similarities_path.suffix}")
    cache_key = str(similarities_path)
    if cache_key in similarity_cache:
        return similarity_cache[cache_key]

    if not similarities_path.exists():
        raise FileNotFoundError(
            f"SUTime {similarity_mode} similarities cache not found for '{benchmark_name}': "
            f"{similarities_path}\n"
            f"Run `python evaluate.py --model_name sutime --benchmark {benchmark_name} "
            f"--sutime_similarity_mode {similarity_mode}` "
            f"to populate it."
        )

    obj = pd.read_pickle(similarities_path)

    # Per-question union of candidate paragraphs, mirroring the work-item build
    # in the temporal/semantic branches so normalization sees the same buckets.
    question_to_paragraphs: Dict[str, List[str]] = {}
    for item in benchmark_data:
        bucket = question_to_paragraphs.setdefault(item["question"], [])
        seen = set(bucket)
        for paragraph in item["paragraphs"]:
            if paragraph not in seen:
                bucket.append(paragraph)
                seen.add(paragraph)

    similarities: Similarities = {}
    if isinstance(obj, dict):
        for question, paragraphs in question_to_paragraphs.items():
            source = obj.get(question, {})
            similarities[question] = {p: float(source.get(p, 0.0)) for p in paragraphs}
    else:
        # pandas DataFrame: question index, paragraph columns.
        column_set = set(obj.columns)
        index_set = set(obj.index)
        for question, paragraphs in question_to_paragraphs.items():
            if question not in index_set:
                similarities[question] = {p: 0.0 for p in paragraphs}
                continue
            similarities[question] = {
                p: (float(obj.at[question, p]) if p in column_set else 0.0)
                for p in paragraphs
            }

    similarity_cache[cache_key] = similarities
    return similarities


def _merge_hybrid_similarities(
    temporal_similarities: Similarities,
    semantic_similarities: Similarities,
    alpha: float,
) -> Similarities:
    merged: Similarities = {}
    for question, paragraphs in temporal_similarities.items():
        semantic_paragraphs = semantic_similarities.get(question, {})
        merged[question] = {
            paragraph: alpha * sim + (1.0 - alpha) * semantic_paragraphs.get(paragraph, 0.0)
            for paragraph, sim in paragraphs.items()
        }
    return merged


def _normalize_similarities(
    similarities: Similarities,
    normalized_cache: Dict[int, Similarities],
) -> Similarities:
    # Per-question min-max normalization over the candidate set. Differs from the
    # original dense-DataFrame implementation which normalized over the union of
    # paragraphs across all questions; that variant required the O(Q*P) matrix
    # which is what blew up RAM on TempReason / TSRetriever.
    cache_key = id(similarities)
    if cache_key in normalized_cache:
        return normalized_cache[cache_key]

    normalized: Similarities = {}
    for question, paragraph_sims in similarities.items():
        keys = list(paragraph_sims.keys())
        values = [paragraph_sims[k] for k in keys]
        normalized_values = normalize_list(values)
        normalized[question] = dict(zip(keys, normalized_values))

    normalized_cache[cache_key] = normalized
    return normalized


def _compute_similarity_lists(
    similarities: Similarities,
    benchmark_data: List[Dict[str, Any]],
) -> List[List[float]]:
    similarities_list: List[List[float]] = []
    for benchmark_item in benchmark_data:
        question = benchmark_item["question"]
        paragraphs = benchmark_item["paragraphs"]
        question_bucket = similarities[question]
        question_similarities = [question_bucket[paragraph] for paragraph in paragraphs]
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


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_paper_evaluations(config_path: Path, split: str = "test") -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    _validate_config(config, config_path)

    top_k = int(config["global"].get("top_k", 5))
    num_negative_samples = int(config["global"].get("num_negative_samples", 0))
    temporal_model_path = Path(config["temporal"].get("model_path", ""))
    enabled_benchmarks = _get_enabled_benchmarks(config)
    run_matrix = _build_run_matrix(config)

    report_rows: List[Dict[str, Any]] = []
    run_order = {run.run_id: idx for idx, run in enumerate(run_matrix)}
    benchmark_order = {benchmark_name: idx for idx, benchmark_name in enumerate(enabled_benchmarks)}

    print(
        f"Running {len(run_matrix)} run configurations over {len(enabled_benchmarks)} "
        f"benchmarks on the '{split}' split."
    )

    for benchmark_name in enabled_benchmarks:
        benchmark_path = BENCHMARK_PATHS[benchmark_name]
        print(f"\n=== Benchmark: {benchmark_name} ===")

        # Similarity caches are built over the *full* benchmark; metrics are
        # computed only on the requested split (test by default).
        benchmark_data = _load_benchmark_data(
            benchmark_path=benchmark_path,
            num_negative_samples=num_negative_samples,
        )
        eval_data = _select_split_items(benchmark_data, split)
        ground_truth = [item["answer"] for item in eval_data]
        similarity_cache: Dict[str, Similarities] = {}
        normalized_temporal_cache: Dict[int, Similarities] = {}

        for run_spec in run_matrix:
            print(f"[{benchmark_name}] Run: {run_spec.run_id}")

            if run_spec.run_type == "temporal":
                similarities = _get_temporal_similarities(
                    run_spec=run_spec,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    temporal_model_path=temporal_model_path,
                    similarity_cache=similarity_cache,
                    benchmark_data=benchmark_data,
                )
            elif run_spec.run_type == "semantic":
                similarities = _get_semantic_similarities(
                    semantic_model_name=run_spec.semantic_model_name,
                    temporal_model_path=temporal_model_path,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    similarity_cache=similarity_cache,
                    benchmark_data=benchmark_data,
                )
            elif run_spec.run_type == "sutime_hybrid":
                sutime_similarities = _load_sutime_similarities(
                    benchmark_name=benchmark_name,
                    benchmark_data=benchmark_data,
                    similarity_cache=similarity_cache,
                    similarity_mode=run_spec.sutime_similarity_mode,
                )
                semantic_similarities = _get_semantic_similarities(
                    semantic_model_name=run_spec.semantic_model_name,
                    temporal_model_path=temporal_model_path,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    similarity_cache=similarity_cache,
                    benchmark_data=benchmark_data,
                )
                normalized_sutime = _normalize_similarities(sutime_similarities, normalized_temporal_cache)
                normalized_semantic = _normalize_similarities(semantic_similarities, {})
                similarities = _merge_hybrid_similarities(
                    temporal_similarities=normalized_sutime,
                    semantic_similarities=normalized_semantic,
                    alpha=run_spec.alpha,
                )
            else:
                temporal_similarities = _get_temporal_similarities(
                    run_spec=run_spec,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    temporal_model_path=temporal_model_path,
                    similarity_cache=similarity_cache,
                    benchmark_data=benchmark_data,
                )
                semantic_similarities = _get_semantic_similarities(
                    semantic_model_name=run_spec.semantic_model_name,
                    temporal_model_path=temporal_model_path,
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    config=config,
                    similarity_cache=similarity_cache,
                    benchmark_data=benchmark_data,
                )
                normalized_temporal = _normalize_similarities(temporal_similarities, normalized_temporal_cache)
                normalized_semantic = _normalize_similarities(semantic_similarities, {})
                similarities = _merge_hybrid_similarities(
                    temporal_similarities=normalized_temporal,
                    semantic_similarities=normalized_semantic,
                    alpha=run_spec.alpha,
                )

            similarities_list = _compute_similarity_lists(similarities, eval_data)
            metrics = _compute_report_metrics(ground_truth, similarities_list, top_k=top_k)

            report_rows.append(
                {
                    "run_id": run_spec.run_id,
                    "run_type": run_spec.run_type,
                    "temporal_model_name": (
                        f"{run_spec.temporal_model_name}:{run_spec.sutime_similarity_mode}"
                        if run_spec.run_type == "sutime_hybrid"
                        else run_spec.temporal_model_name
                    ),
                    "semantic_model_name": run_spec.semantic_model_name,
                    "alpha": f"{run_spec.alpha:.2f}" if run_spec.run_type in ("hybrid", "sutime_hybrid") else "-",
                    "benchmark": benchmark_name,
                    "num_queries": len(eval_data),
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

            del similarities_list
            del metrics

        similarity_cache.clear()
        normalized_temporal_cache.clear()
        del similarity_cache
        del normalized_temporal_cache
        del ground_truth
        del eval_data
        del benchmark_data
        _release_memory()

    report_rows.sort(
        key=lambda row: (run_order[row["run_id"]], benchmark_order[row["benchmark"]]),
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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=list(VALID_SPLITS),
        help=(
            "Which split to compute metrics on. Defaults to 'test'. "
            "Splits are created with split_and_tune_alpha.py; benchmarks without "
            "split tags fall back to evaluating on all items."
        ),
    )
    args = parser.parse_args()

    run_paper_evaluations(Path(args.config), split=args.split)


if __name__ == "__main__":
    main()
