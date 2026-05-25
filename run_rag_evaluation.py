"""End-to-end RAG evaluation on Time-Sensitive QA.

For each (question, candidate paragraphs) item in
`processed_human_annotated_test.json`:

  1. Retrieve top-k paragraphs using one of:
       - temporal (the trained TEMPS model)
       - semantic (a SentenceTransformer baseline)
       - hybrid   (alpha * temporal + (1 - alpha) * semantic, per-question
                   min-max normalized — same scheme as run_paper_evaluations.py)
  2. Build a chat-style prompt with the top-k passages and call the
     generator LLM (default Qwen2.5-7B-Instruct, configurable).
  3. Score:
       - Retrieval: Recall@k, MRR, NDCG@k
       - QA:        SQuAD-style EM, token F1, containment

The script reuses the existing similarity computation pipeline so cached
embeddings/similarities from `run_paper_evaluations.py` are picked up
automatically.

Usage:
    uv run python run_rag_evaluation.py \\
        --config temporal_embeddings/config/rag_evaluation_config.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.evaluation.utils.evaluation.rag.rag_pipeline import (
    Generator,
    RagItem,
    aggregate_qa_metrics,
    build_messages,
    load_rag_benchmark,
    select_top_k,
)
from temporal_embeddings.evaluation.utils.evaluation.semantic_model.compute_semantic_similarities import (
    compute_semantic_similarities,
)
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.compute_temporal_similarities import (
    compute_temporal_similarities,
)
from temporal_embeddings.utils.math.normalize import normalize_list
from temporal_embeddings.utils.os.folder_management import create_folders


Similarities = Dict[str, Dict[str, float]]


PROCESSED_BENCHMARK_PATH = Path(
    "data/evaluation/time_sensitive_qa/processed_human_annotated_test.json"
)
RAW_BENCHMARK_PATH = Path(
    "data/evaluation/time_sensitive_qa/human_annotated_test.json"
)
BENCHMARK_NAME = "time_sensitive_qa"


@dataclass
class RetrievalSpec:
    """One row of the evaluation table.

    `display_label` is what shows up in the markdown report — paired so the
    baseline and `baseline + temporal` rows sit next to each other.
    """

    run_id: str
    display_label: str
    run_type: str  # "temporal" | "semantic" | "hybrid"
    temporal_model_name: str
    semantic_model_name: str
    alpha: float


# ---------------------------------------------------------------------------
# Config schema:
#
#   retrieval:
#     temporal_model_name: "all-minilm-l6-v2"
#     temporal_model_path: "output/trained_models/....pth"
#     include_temporal_only: true            # adds a "temporal" reference row
#     baselines:
#       - name: "e5-base-v2"
#         model: "intfloat/e5-base-v2"
#         alpha: 0.5                         # hybrid mix weight (temporal share)
#       - name: "bge-large"
#         model: "BAAI/bge-large-en-v1.5"
#         alpha: 0.5
#
# Run order produced:
#   [temporal alone]            (only if include_temporal_only)
#   e5-base-v2
#   e5-base-v2 + temporal
#   bge-large
#   bge-large + temporal
# ---------------------------------------------------------------------------


def _validate_config(config: Dict[str, Any], config_path: Path) -> None:
    for section in ("global", "retrieval", "generator", "report"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section} in {config_path}")

    retrieval = config["retrieval"]
    baselines = retrieval.get("baselines", [])
    include_temporal_only = bool(retrieval.get("include_temporal_only", False))

    if not baselines and not include_temporal_only:
        raise ValueError(
            "retrieval.baselines must be non-empty, or set "
            "retrieval.include_temporal_only=true."
        )

    needs_temporal = include_temporal_only or bool(baselines)
    if needs_temporal:
        model_path = retrieval.get("temporal_model_path", "")
        if not model_path:
            raise ValueError("retrieval.temporal_model_path is required.")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Temporal checkpoint not found: {model_path}")
        if not retrieval.get("temporal_model_name"):
            raise ValueError("retrieval.temporal_model_name is required.")

    seen_names: set = set()
    for i, baseline in enumerate(baselines):
        for required in ("name", "model", "alpha"):
            if required not in baseline:
                raise ValueError(
                    f"retrieval.baselines[{i}] missing required field '{required}'."
                )
        name = baseline["name"]
        if name in seen_names:
            raise ValueError(f"retrieval.baselines: duplicate baseline name '{name}'.")
        seen_names.add(name)
        alpha = float(baseline["alpha"])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"retrieval.baselines[{i}].alpha must be in [0, 1], got {alpha}."
            )

    if not PROCESSED_BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {PROCESSED_BENCHMARK_PATH}")
    if not RAW_BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Raw benchmark file not found: {RAW_BENCHMARK_PATH}")

    if not config["report"].get("output_dir"):
        raise ValueError("report.output_dir is required")
    if not config["generator"].get("model_name_or_path"):
        raise ValueError("generator.model_name_or_path is required")


def _build_retrieval_matrix(config: Dict[str, Any]) -> List[RetrievalSpec]:
    """Produce paired runs: each baseline contributes a (semantic-only,
    semantic+temporal) pair, in declaration order. Optional leading row for
    temporal alone."""

    runs: List[RetrievalSpec] = []
    retrieval = config["retrieval"]
    temporal_name = retrieval.get("temporal_model_name", "temporal")

    if retrieval.get("include_temporal_only", False):
        runs.append(
            RetrievalSpec(
                run_id=f"temporal:{temporal_name}",
                display_label=f"temporal ({temporal_name})",
                run_type="temporal",
                temporal_model_name=temporal_name,
                semantic_model_name="",
                alpha=1.0,
            )
        )

    for baseline in retrieval.get("baselines", []):
        baseline_name = baseline["name"]
        semantic_model = baseline["model"]
        alpha = float(baseline["alpha"])

        # Baseline alone.
        runs.append(
            RetrievalSpec(
                run_id=f"baseline:{baseline_name}",
                display_label=baseline_name,
                run_type="semantic",
                temporal_model_name="",
                semantic_model_name=semantic_model,
                alpha=0.0,
            )
        )
        # Baseline + temporal.
        runs.append(
            RetrievalSpec(
                run_id=f"baseline+temporal:{baseline_name}:alpha={alpha}",
                display_label=f"{baseline_name} + temporal",
                run_type="hybrid",
                temporal_model_name=temporal_name,
                semantic_model_name=semantic_model,
                alpha=alpha,
            )
        )

    return runs


def _temporal_model_path(config: Dict[str, Any]) -> Path:
    return Path(config["retrieval"]["temporal_model_path"])


def _benchmark_data_from_items(items: List[RagItem]) -> List[Dict[str, Any]]:
    return [
        {
            "question": item.question,
            "paragraphs": item.paragraphs,
            "answer": item.gold_paragraph_indices,
        }
        for item in items
    ]


def _get_temporal_similarities(
    spec: RetrievalSpec,
    config: Dict[str, Any],
    benchmark_data: List[Dict[str, Any]],
    cache: Dict[str, Similarities],
) -> Similarities:
    temporal_model_path = _temporal_model_path(config)
    output_paths = set_output_files(
        temporal_model_name=spec.temporal_model_name,
        temporal_model_path=temporal_model_path,
        semantic_model_name="",
        benchmark=BENCHMARK_NAME,
    )
    key = str(output_paths["temporal_similarities_path"])
    if key in cache:
        return cache[key]

    similarities = compute_temporal_similarities(
        temporal_model_name=spec.temporal_model_name,
        temporal_model_path=temporal_model_path,
        batch_size=int(config["global"].get("batch_size", 128)),
        max_seq_len=int(config["global"].get("max_seq_len", 512)),
        benchmark_file_path=PROCESSED_BENCHMARK_PATH,
        temporal_cache_file_path=output_paths["temporal_cache_path"],
        temporal_similarities_file_path=output_paths["temporal_similarities_path"],
        num_negative_samples=0,
        reference_date=str(config["global"].get("reference_date", "2021-11-09")),
        benchmark_data=benchmark_data,
    )
    cache[key] = similarities
    return similarities


def _get_semantic_similarities(
    spec: RetrievalSpec,
    config: Dict[str, Any],
    benchmark_data: List[Dict[str, Any]],
    cache: Dict[str, Similarities],
) -> Similarities:
    temporal_model_path = _temporal_model_path(config)
    output_paths = set_output_files(
        temporal_model_name="",
        temporal_model_path=temporal_model_path,
        semantic_model_name=spec.semantic_model_name,
        benchmark=BENCHMARK_NAME,
    )
    key = str(output_paths["semantic_similarities_path"])
    if key in cache:
        return cache[key]

    similarities = compute_semantic_similarities(
        semantic_model_name=spec.semantic_model_name,
        max_seq_len=int(config["global"].get("max_seq_len", 512)),
        benchmark_file_path=PROCESSED_BENCHMARK_PATH,
        semantic_cache_file_path=output_paths["semantic_cache_path"],
        semantic_similarities_file_path=output_paths["semantic_similarities_path"],
        num_negative_samples=0,
        benchmark_data=benchmark_data,
    )
    cache[key] = similarities
    return similarities


def _normalize(similarities: Similarities) -> Similarities:
    normalized: Similarities = {}
    for question, paragraph_sims in similarities.items():
        keys = list(paragraph_sims.keys())
        values = [paragraph_sims[k] for k in keys]
        normalized[question] = dict(zip(keys, normalize_list(values)))
    return normalized


def _merge_hybrid(
    temporal: Similarities,
    semantic: Similarities,
    alpha: float,
) -> Similarities:
    merged: Similarities = {}
    for question, paragraphs in temporal.items():
        semantic_paragraphs = semantic.get(question, {})
        merged[question] = {
            paragraph: alpha * sim + (1.0 - alpha) * semantic_paragraphs.get(paragraph, 0.0)
            for paragraph, sim in paragraphs.items()
        }
    return merged


def _similarities_for_run(
    spec: RetrievalSpec,
    config: Dict[str, Any],
    benchmark_data: List[Dict[str, Any]],
    raw_cache: Dict[str, Similarities],
) -> Similarities:
    if spec.run_type == "temporal":
        return _get_temporal_similarities(spec, config, benchmark_data, raw_cache)
    if spec.run_type == "semantic":
        return _get_semantic_similarities(spec, config, benchmark_data, raw_cache)

    temporal_similarities = _get_temporal_similarities(spec, config, benchmark_data, raw_cache)
    semantic_similarities = _get_semantic_similarities(spec, config, benchmark_data, raw_cache)
    return _merge_hybrid(
        _normalize(temporal_similarities),
        _normalize(semantic_similarities),
        spec.alpha,
    )


# ---------------------------------------------------------------------------
# Metrics + report.
# ---------------------------------------------------------------------------


def _retrieval_metrics(
    ground_truth: List[List[int]],
    similarities_list: List[List[float]],
    top_k: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for metric in ("mrr", "ndcg", "recall", "precision"):
        metrics[metric] = float(
            compute_metrics(ground_truth, similarities_list, top_k, metric)[metric]
        )
    return metrics


def _format_float(value: float) -> str:
    return f"{value:.6f}"


_SAFE_RUN_ID_RE = re.compile(r"[^A-Za-z0-9._+-]+")


def _safe_run_id(run_id: str) -> str:
    """Filesystem-safe slug for per-run artifact filenames."""

    return _SAFE_RUN_ID_RE.sub("_", run_id).strip("_") or "run"


def _write_markdown(
    report_path: Path,
    rows: List[Dict[str, Any]],
    generator_model: str,
    top_k: int,
) -> None:
    create_folders([report_path.parent])

    headers = [
        "Method",
        "Alpha",
        "Queries",
        "Scored",
        f"Recall@{top_k}",
        "MRR",
        f"NDCG@{top_k}",
        "EM",
        "F1",
        "Containment",
    ]

    lines: List[str] = [
        "# RAG Evaluation Report — Time-Sensitive QA",
        "",
        f"- Benchmark: `{PROCESSED_BENCHMARK_PATH}`",
        f"- Generator: `{generator_model}`",
        f"- Top-k: {top_k}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display_label"],
                    row["alpha"],
                    str(row["num_queries"]),
                    str(row["num_scored"]),
                    _format_float(row["recall"]),
                    _format_float(row["mrr"]),
                    _format_float(row["ndcg"]),
                    _format_float(row["em"]),
                    _format_float(row["f1"]),
                    _format_float(row["containment"]),
                ]
            )
            + " |"
        )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_predictions(
    predictions_path: Path,
    rows_payload: List[Dict[str, Any]],
) -> None:
    create_folders([predictions_path.parent])
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(rows_payload, f, indent=2, ensure_ascii=False)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


def run_rag_evaluation(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)
    _validate_config(config, config_path)

    top_k = int(config["global"].get("top_k", 5))
    report_cfg = config["report"]
    output_dir = Path(report_cfg["output_dir"])
    report_path = output_dir / report_cfg.get("report_filename", "rag_evaluation_report.md")
    predictions_path = output_dir / report_cfg.get(
        "predictions_filename", "rag_evaluation_predictions.json"
    )
    runs_dir = output_dir / "runs"
    create_folders([output_dir, runs_dir])

    # Snapshot config + run metadata for reproducibility.
    config_snapshot_path = output_dir / "config.snapshot.json"
    shutil.copyfile(config_path, config_snapshot_path)
    run_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_source": str(config_path),
        "processed_benchmark": str(PROCESSED_BENCHMARK_PATH),
        "raw_benchmark": str(RAW_BENCHMARK_PATH),
        "top_k": top_k,
        "generator": config["generator"],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False)
    )

    items = load_rag_benchmark(PROCESSED_BENCHMARK_PATH, RAW_BENCHMARK_PATH)
    benchmark_data = _benchmark_data_from_items(items)
    print(f"Loaded {len(items)} RAG items")

    runs = _build_retrieval_matrix(config)
    print(f"Retrieval runs: {len(runs)}")

    # --- Stage 1: compute all retrievals first, defer generator load. ---
    raw_sim_cache: Dict[str, Similarities] = {}
    per_run_topk: Dict[str, List[Tuple[List[int], List[str]]]] = {}
    per_run_sim_lists: Dict[str, List[List[float]]] = {}

    for spec in runs:
        print(f"\n=== Retrieval: {spec.run_id} ===")
        similarities = _similarities_for_run(spec, config, benchmark_data, raw_sim_cache)

        topk_per_item: List[Tuple[List[int], List[str]]] = []
        sim_lists: List[List[float]] = []
        for item in items:
            bucket = similarities.get(item.question, {})
            top_indices, top_paragraphs = select_top_k(item.paragraphs, bucket, top_k)
            topk_per_item.append((top_indices, top_paragraphs))
            sim_lists.append([float(bucket.get(p, 0.0)) for p in item.paragraphs])
        per_run_topk[spec.run_id] = topk_per_item
        per_run_sim_lists[spec.run_id] = sim_lists

    raw_sim_cache.clear()
    _release_memory()

    # --- Stage 2: load generator once, run all prompts. ---
    generator_cfg = config["generator"]
    generator = Generator(
        model_name_or_path=generator_cfg["model_name_or_path"],
        max_new_tokens=int(generator_cfg.get("max_new_tokens", 64)),
        batch_size=int(generator_cfg.get("batch_size", 4)),
        dtype=str(generator_cfg.get("dtype", "bfloat16")),
    )
    system_prompt: Optional[str] = generator_cfg.get("system_prompt")

    report_rows: List[Dict[str, Any]] = []
    predictions_payload: List[Dict[str, Any]] = []

    for spec in runs:
        print(f"\n=== Generation: {spec.run_id} ===")
        topk_per_item = per_run_topk[spec.run_id]

        prompts = []
        for item, (_, top_paragraphs) in zip(items, topk_per_item):
            messages = (
                build_messages(item.question, top_paragraphs, system_prompt)
                if system_prompt
                else build_messages(item.question, top_paragraphs)
            )
            prompts.append(messages)

        predictions = generator.generate(prompts)

        ground_truth = [item.gold_paragraph_indices for item in items]
        sim_lists = per_run_sim_lists[spec.run_id]
        retrieval = _retrieval_metrics(ground_truth, sim_lists, top_k)

        gold_answers_list = [item.gold_answers for item in items]
        qa_metrics = aggregate_qa_metrics(predictions, gold_answers_list)

        num_scored = sum(1 for golds in gold_answers_list if golds)

        report_rows.append(
            {
                "run_id": spec.run_id,
                "display_label": spec.display_label,
                "run_type": spec.run_type,
                "temporal_model_name": spec.temporal_model_name,
                "semantic_model_name": spec.semantic_model_name,
                "alpha": f"{spec.alpha:.2f}" if spec.run_type == "hybrid" else "-",
                "num_queries": len(items),
                "num_scored": num_scored,
                "recall": retrieval["recall"],
                "mrr": retrieval["mrr"],
                "ndcg": retrieval["ndcg"],
                "em": qa_metrics["em"],
                "f1": qa_metrics["f1"],
                "containment": qa_metrics["containment"],
            }
        )

        run_predictions: List[Dict[str, Any]] = []
        for item, (top_indices, top_paragraphs), prediction in zip(
            items, topk_per_item, predictions
        ):
            entry = {
                "run_id": spec.run_id,
                "question": item.question,
                "gold_paragraph_indices": item.gold_paragraph_indices,
                "gold_answers": item.gold_answers,
                "retrieved_paragraph_indices": top_indices,
                "retrieved_paragraphs": top_paragraphs,
                "prediction": prediction,
            }
            run_predictions.append(entry)
            predictions_payload.append(entry)

        run_slug = _safe_run_id(spec.run_id)
        run_predictions_path = runs_dir / f"{run_slug}.predictions.jsonl"
        with run_predictions_path.open("w", encoding="utf-8") as f:
            for entry in run_predictions:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        run_metrics_path = runs_dir / f"{run_slug}.metrics.json"
        run_metrics_path.write_text(
            json.dumps(
                {
                    "run_id": spec.run_id,
                    "run_type": spec.run_type,
                    "temporal_model_name": spec.temporal_model_name,
                    "semantic_model_name": spec.semantic_model_name,
                    "alpha": spec.alpha,
                    "top_k": top_k,
                    "num_queries": len(items),
                    "num_scored": num_scored,
                    "retrieval": retrieval,
                    "qa": qa_metrics,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "Retrieval: "
            f"Recall@{top_k}={retrieval['recall']:.4f}, "
            f"MRR={retrieval['mrr']:.4f}, "
            f"NDCG@{top_k}={retrieval['ndcg']:.4f} | "
            f"QA: EM={qa_metrics['em']:.4f}, "
            f"F1={qa_metrics['f1']:.4f}, "
            f"Containment={qa_metrics['containment']:.4f} "
            f"(scored {num_scored}/{len(items)})"
        )

    _write_markdown(report_path, report_rows, generator_cfg["model_name_or_path"], top_k)
    _write_predictions(predictions_path, predictions_payload)
    print(f"\nReport written to: {report_path}")
    print(f"Predictions written to: {predictions_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation on Time-Sensitive QA.")
    parser.add_argument(
        "--config",
        type=str,
        default="temporal_embeddings/config/rag_evaluation_config.json",
        help="Path to RAG evaluation config JSON file.",
    )
    args = parser.parse_args()
    run_rag_evaluation(Path(args.config))


if __name__ == "__main__":
    main()
