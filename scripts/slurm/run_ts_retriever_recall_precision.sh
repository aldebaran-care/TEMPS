#!/bin/bash
# -----------------------------------------------------------------------------
# TS-Retriever recall/precision sweep.
#
# Default config:
#   temporal_embeddings/config/paper_evaluation_config.json
#
# Default report:
#   output/metrics/ts_retriever_recall_precision_report.md
#
# Override when submitting:
#   EVAL_CONFIG=/abs/path/to/config.json \
#   TS_RETRIEVER_REPORT=/abs/path/to/report.md \
#   sbatch scripts/slurm/run_ts_retriever_recall_precision.sh
#
# Example SLURM job. Adjust the commented-out ##SBATCH lines (account,
# partition, constraint) and module loads to your cluster, then submit
# from the repository root.
# -----------------------------------------------------------------------------
#SBATCH --job-name=ts-ret-rp
#SBATCH --output=logs/slurm/ts-ret-rp-%j.out
#SBATCH --error=logs/slurm/ts-ret-rp-%j.err
##SBATCH --account=<your_account>
##SBATCH --partition=<your_gpu_partition>
##SBATCH --constraint=<your_gpu_constraint>
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Stage-2 similarity computation is CPU-parallel. If your scheduler allots a
# fixed number of cores per GPU, --exclusive gets the full node (it also
# reserves every GPU, so the node is billed in full).
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

set -euo pipefail

# Only source the bash rc on hosts that actually have one. Cluster setups
# rely on it (module load lines, uv on PATH, etc.); macOS/zsh users don't
# have a ~/.bashrc and `source` would abort the script.
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

mkdir -p logs/slurm

# Under SLURM run from the submit directory (the repository root); locally
# fall back to the repository root relative to this script.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

uv sync

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
# Compute nodes may be offline — pin HF to the local cache under SLURM.
# Locally we want HF to be free to download missing weights.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

EVAL_CONFIG="${EVAL_CONFIG:-temporal_embeddings/config/paper_evaluation_config.json}"
TS_RETRIEVER_REPORT="${TS_RETRIEVER_REPORT:-output/metrics/ts_retriever_recall_precision_report.md}"

echo "Using evaluation config: ${EVAL_CONFIG}"
echo "Writing TS-Retriever recall/precision report to: ${TS_RETRIEVER_REPORT}"

uv run python - "${EVAL_CONFIG}" "${TS_RETRIEVER_REPORT}" <<'PY'
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from run_paper_evaluations import (
    BENCHMARK_PATHS,
    Similarities,
    _build_run_matrix,
    _compute_similarity_lists,
    _get_semantic_similarities,
    _get_temporal_similarities,
    _load_benchmark_data,
    _load_sutime_similarities,
    _merge_hybrid_similarities,
    _normalize_similarities,
    _validate_config,
)
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.utils.os.folder_management import create_folders


TOP_K_VALUES = [1, 5, 10, 20, 50, 100]
BENCHMARK_NAME = "ts_retriever"


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_ranking_metrics(
    ground_truth: List[List[int]],
    similarities_list: List[List[float]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # MRR walks the full ranked list, so it is independent of top_k. Compute
    # once with an arbitrary top_k argument.
    mrr = compute_metrics(
        first_list=ground_truth,
        second_list=similarities_list,
        top_k=1,
        metric="mrr",
    )["mrr"]
    metrics["MRR"] = float(mrr)

    for top_k in TOP_K_VALUES:
        recall = compute_metrics(
            first_list=ground_truth,
            second_list=similarities_list,
            top_k=top_k,
            metric="recall",
        )["recall"]
        precision = compute_metrics(
            first_list=ground_truth,
            second_list=similarities_list,
            top_k=top_k,
            metric="precision",
        )["precision"]
        ndcg = compute_metrics(
            first_list=ground_truth,
            second_list=similarities_list,
            top_k=top_k,
            metric="ndcg",
        )["ndcg"]
        metrics[f"R@{top_k}"] = float(recall)
        metrics[f"P@{top_k}"] = float(precision)
        metrics[f"NDCG@{top_k}"] = float(ndcg)
    return metrics


def _write_report(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    create_folders([report_path.parent])

    recall_headers = [f"R@{top_k}" for top_k in TOP_K_VALUES]
    precision_headers = [f"P@{top_k}" for top_k in TOP_K_VALUES]
    ndcg_headers = [f"NDCG@{top_k}" for top_k in TOP_K_VALUES]
    headers = [
        "Run ID",
        "Type",
        "Temporal Model",
        "Semantic Model",
        "Alpha",
        "Queries",
        "MRR",
        *recall_headers,
        *precision_headers,
        *ndcg_headers,
    ]

    markdown_lines = [
        "# TS-Retriever Ranking Metrics Report",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [
            row["run_id"],
            row["run_type"],
            row["temporal_model_name"],
            row["semantic_model_name"],
            row["alpha"],
            str(row["num_queries"]),
            _format_float(row["MRR"]),
        ]
        values.extend(_format_float(row[header]) for header in recall_headers)
        values.extend(_format_float(row[header]) for header in precision_headers)
        values.extend(_format_float(row[header]) for header in ndcg_headers)
        markdown_lines.append("| " + " | ".join(values) + " |")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def main() -> None:
    config_path = Path(sys.argv[1])
    report_path = Path(sys.argv[2])

    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    for benchmark_cfg in config["benchmarks"].values():
        benchmark_cfg["enabled"] = False
    config["benchmarks"][BENCHMARK_NAME]["enabled"] = True

    _validate_config(config, config_path)

    benchmark_path = BENCHMARK_PATHS[BENCHMARK_NAME]
    temporal_model_path = Path(config["temporal"].get("model_path", ""))
    num_negative_samples = int(config["global"].get("num_negative_samples", 0))
    run_matrix = _build_run_matrix(config)

    benchmark_data = _load_benchmark_data(
        benchmark_path=benchmark_path,
        num_negative_samples=num_negative_samples,
    )
    ground_truth = [item["answer"] for item in benchmark_data]
    similarity_cache: Dict[str, Similarities] = {}
    normalized_temporal_cache: Dict[int, Similarities] = {}
    rows: List[Dict[str, Any]] = []

    print(
        f"Running {len(run_matrix)} run configurations on {BENCHMARK_NAME} "
        f"({len(benchmark_data)} queries)."
    )

    for run_spec in run_matrix:
        print(f"[{BENCHMARK_NAME}] Run: {run_spec.run_id}")

        if run_spec.run_type == "temporal":
            similarities = _get_temporal_similarities(
                run_spec=run_spec,
                benchmark_name=BENCHMARK_NAME,
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
                benchmark_name=BENCHMARK_NAME,
                benchmark_path=benchmark_path,
                config=config,
                similarity_cache=similarity_cache,
                benchmark_data=benchmark_data,
            )
        elif run_spec.run_type == "sutime_hybrid":
            temporal_similarities = _load_sutime_similarities(
                benchmark_name=BENCHMARK_NAME,
                benchmark_data=benchmark_data,
                similarity_cache=similarity_cache,
                similarity_mode=run_spec.sutime_similarity_mode,
            )
            semantic_similarities = _get_semantic_similarities(
                semantic_model_name=run_spec.semantic_model_name,
                temporal_model_path=temporal_model_path,
                benchmark_name=BENCHMARK_NAME,
                benchmark_path=benchmark_path,
                config=config,
                similarity_cache=similarity_cache,
                benchmark_data=benchmark_data,
            )
            normalized_temporal = _normalize_similarities(
                temporal_similarities,
                normalized_temporal_cache,
            )
            normalized_semantic = _normalize_similarities(semantic_similarities, {})
            similarities = _merge_hybrid_similarities(
                temporal_similarities=normalized_temporal,
                semantic_similarities=normalized_semantic,
                alpha=run_spec.alpha,
            )
        else:
            temporal_similarities = _get_temporal_similarities(
                run_spec=run_spec,
                benchmark_name=BENCHMARK_NAME,
                benchmark_path=benchmark_path,
                config=config,
                temporal_model_path=temporal_model_path,
                similarity_cache=similarity_cache,
                benchmark_data=benchmark_data,
            )
            semantic_similarities = _get_semantic_similarities(
                semantic_model_name=run_spec.semantic_model_name,
                temporal_model_path=temporal_model_path,
                benchmark_name=BENCHMARK_NAME,
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

        similarities_list = _compute_similarity_lists(similarities, benchmark_data)
        metrics = _compute_ranking_metrics(ground_truth, similarities_list)

        rows.append(
            {
                "run_id": run_spec.run_id,
                "run_type": run_spec.run_type,
                "temporal_model_name": (
                    f"{run_spec.temporal_model_name}:{run_spec.sutime_similarity_mode}"
                    if run_spec.run_type == "sutime_hybrid"
                    else run_spec.temporal_model_name
                ),
                "semantic_model_name": run_spec.semantic_model_name,
                "alpha": (
                    f"{run_spec.alpha:.2f}"
                    if run_spec.run_type in ("hybrid", "sutime_hybrid")
                    else "-"
                ),
                "num_queries": len(benchmark_data),
                **metrics,
            }
        )

        print(
            "Metrics: "
            + ", ".join(
                f"{metric_name}={metric_value:.6f}"
                for metric_name, metric_value in metrics.items()
            )
        )

        del similarities_list
        del metrics
        _release_memory()

    _write_report(report_path, rows)
    print(f"TS-Retriever recall/precision report written to: {report_path}")


if __name__ == "__main__":
    main()
PY
