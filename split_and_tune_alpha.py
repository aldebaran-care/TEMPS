"""Split benchmarks into val/test and tune the hybrid alpha on the val split.

Two subcommands:

    tag         Randomly tag every datapoint in each enabled benchmark as
                "val" or "test" (20/80 by default). The tag is written back as
                a "split" field on each item, in place, so it travels with the
                data. Deterministic given --seed, so val/test selection and the
                final test evaluation always see the same split.

    tune-alpha  Run the hybrid (temporal + semantic) pipeline on the *val*
                split only, sweeping alpha over 0.1..0.9 (step 0.1 by default),
                and select the best alpha per benchmark. Reuses the exact
                similarity computation, normalization, and hybrid merge from
                run_paper_evaluations, so the tuned alpha matches how the test
                evaluation combines the two models.

Similarity caches are split-agnostic: they are built over the full benchmark
(as they already are on the SLURM server), and only the metric computation is
restricted to a split. So `tune-alpha` reuses the ready caches without
recomputing anything.

Typical flow on the remote server (caches already present):

    python split_and_tune_alpha.py tag
    python split_and_tune_alpha.py tune-alpha
    python run_paper_evaluations.py            # evaluates on the test split
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from run_paper_evaluations import (
    BENCHMARK_PATHS,
    METRIC_COLUMNS,
    RunSpec,
    Similarities,
    _compute_report_metrics,
    _get_enabled_benchmarks,
    _get_semantic_similarities,
    _get_temporal_similarities,
    _load_benchmark_data,
    _normalize_similarities,
)
from temporal_embeddings.evaluation.utils.data.splits import (
    SPLIT_KEY,
    select_split_items as _select_split_items,
)

DEFAULT_CONFIG = "temporal_embeddings/config/paper_evaluation_config.json"
DEFAULT_ALPHAS: List[float] = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 .. 0.9
DEFAULT_SELECTION_METRIC = "mrr"


# ---------------------------------------------------------------------------
# tag: assign val/test splits
# ---------------------------------------------------------------------------
def _assign_splits(
    benchmark_data: List[Dict[str, Any]],
    benchmark_name: str,
    val_fraction: float,
    seed: int,
) -> Dict[str, int]:
    """Tag each item in place with SPLIT_KEY = "val" or "test".

    Uses a per-benchmark RNG seeded from (seed, benchmark_name) so each
    benchmark gets an independent, reproducible split whose assignment does not
    depend on the other benchmarks' sizes or ordering.
    """
    n = len(benchmark_data)
    rng = random.Random(f"{seed}:{benchmark_name}")

    indices = list(range(n))
    rng.shuffle(indices)
    num_val = round(n * val_fraction)
    val_indices = set(indices[:num_val])

    for idx, item in enumerate(benchmark_data):
        item[SPLIT_KEY] = "val" if idx in val_indices else "test"

    return {"val": len(val_indices), "test": n - len(val_indices)}


def tag_benchmarks(
    config_path: Path,
    val_fraction: float,
    seed: int,
) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in (0, 1), got {val_fraction}")

    enabled_benchmarks = _get_enabled_benchmarks(config)
    print(
        f"Tagging {len(enabled_benchmarks)} benchmarks "
        f"(val={val_fraction:.0%} / test={1 - val_fraction:.0%}, seed={seed})."
    )

    for benchmark_name in enabled_benchmarks:
        benchmark_path = BENCHMARK_PATHS[benchmark_name]
        with benchmark_path.open("r", encoding="utf-8") as f:
            benchmark_data: List[Dict[str, Any]] = json.load(f)

        counts = _assign_splits(benchmark_data, benchmark_name, val_fraction, seed)

        with benchmark_path.open("w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, ensure_ascii=False)

        print(
            f"  {benchmark_name}: {len(benchmark_data)} items -> "
            f"val={counts['val']}, test={counts['test']}  ({benchmark_path})"
        )

    print("\nDone. Re-run with the same --seed to reproduce the identical split.")


# ---------------------------------------------------------------------------
# tune-alpha: select best alpha on the val split
# ---------------------------------------------------------------------------
def _hybrid_similarity_lists(
    normalized_temporal: Similarities,
    normalized_semantic: Similarities,
    alpha: float,
    items: List[Dict[str, Any]],
) -> List[List[float]]:
    """Build per-item hybrid similarity lists directly for `items`.

    Mirrors `_merge_hybrid_similarities` + `_compute_similarity_lists` from the
    main runner, but only over the items we score (the val split) instead of
    materializing the full-corpus merged dict for every alpha.
    """
    similarities_list: List[List[float]] = []
    for item in items:
        question = item["question"]
        temporal_bucket = normalized_temporal[question]
        semantic_bucket = normalized_semantic.get(question, {})
        similarities_list.append(
            [
                alpha * temporal_bucket[paragraph]
                + (1.0 - alpha) * semantic_bucket.get(paragraph, 0.0)
                for paragraph in item["paragraphs"]
            ]
        )
    return similarities_list


def tune_alpha(
    config_path: Path,
    alphas: List[float],
    selection_metric: str,
    output_path: Path,
) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    hybrid_cfg = config.get("hybrid", {})
    if not hybrid_cfg.get("enabled", False):
        raise ValueError("hybrid.enabled must be true in the config to tune alpha.")
    semantic_models = hybrid_cfg.get("semantic_models", [])
    if not semantic_models:
        raise ValueError("hybrid.semantic_models must be non-empty to tune alpha.")
    temporal_model_name = hybrid_cfg.get("temporal_model_name", "all-minilm-l6-v2-full")

    temporal_model_path = Path(config["temporal"].get("model_path", ""))
    if not temporal_model_path or not temporal_model_path.exists():
        raise FileNotFoundError(
            f"temporal.model_path is required and must exist to tune alpha: "
            f"{temporal_model_path}"
        )

    metric_keys = [key for key, _ in METRIC_COLUMNS]
    if selection_metric not in metric_keys:
        raise ValueError(
            f"--metric must be one of {metric_keys}, got '{selection_metric}'."
        )

    top_k = int(config["global"].get("top_k", 5))
    num_negative_samples = int(config["global"].get("num_negative_samples", 0))
    enabled_benchmarks = _get_enabled_benchmarks(config)

    print(
        f"Tuning alpha on the val split for {len(enabled_benchmarks)} benchmarks.\n"
        f"  temporal model : {temporal_model_name}\n"
        f"  semantic models: {semantic_models}\n"
        f"  alphas         : {alphas}\n"
        f"  selection      : mean {selection_metric.upper()} across semantic models\n"
    )

    results: Dict[str, Any] = {}

    for benchmark_name in enabled_benchmarks:
        benchmark_path = BENCHMARK_PATHS[benchmark_name]
        print(f"\n=== Benchmark: {benchmark_name} ===")

        benchmark_data = _load_benchmark_data(
            benchmark_path=benchmark_path,
            num_negative_samples=num_negative_samples,
        )
        val_data = _select_split_items(benchmark_data, "val")
        ground_truth = [item["answer"] for item in val_data]

        similarity_cache: Dict[str, Similarities] = {}
        normalized_temporal_cache: Dict[int, Similarities] = {}

        temporal_run = RunSpec(
            run_id=f"tune-alpha:temporal:{temporal_model_name}",
            run_type="hybrid",
            temporal_model_name=temporal_model_name,
            semantic_model_name="",
            alpha=0.0,
        )
        temporal_similarities = _get_temporal_similarities(
            run_spec=temporal_run,
            benchmark_name=benchmark_name,
            benchmark_path=benchmark_path,
            config=config,
            temporal_model_path=temporal_model_path,
            similarity_cache=similarity_cache,
            benchmark_data=benchmark_data,
        )
        normalized_temporal = _normalize_similarities(
            temporal_similarities, normalized_temporal_cache
        )

        # grid[alpha][model] = {metric: value, ...} on the val split
        grid: Dict[float, Dict[str, Dict[str, float]]] = {alpha: {} for alpha in alphas}

        for semantic_model_name in semantic_models:
            print(f"[{benchmark_name}] semantic model: {semantic_model_name}")
            semantic_similarities = _get_semantic_similarities(
                semantic_model_name=semantic_model_name,
                temporal_model_path=temporal_model_path,
                benchmark_name=benchmark_name,
                benchmark_path=benchmark_path,
                config=config,
                similarity_cache=similarity_cache,
                benchmark_data=benchmark_data,
            )
            normalized_semantic = _normalize_similarities(semantic_similarities, {})

            for alpha in alphas:
                similarities_list = _hybrid_similarity_lists(
                    normalized_temporal, normalized_semantic, alpha, val_data
                )
                metrics = _compute_report_metrics(ground_truth, similarities_list, top_k=top_k)
                grid[alpha][semantic_model_name] = metrics
                print(
                    f"  alpha={alpha:.1f}: "
                    + ", ".join(f"{key.upper()}={metrics[key]:.4f}" for key in metric_keys)
                )

        mean_by_alpha = {
            alpha: mean(grid[alpha][model][selection_metric] for model in semantic_models)
            for alpha in alphas
        }
        best_alpha = max(alphas, key=lambda a: mean_by_alpha[a])

        results[benchmark_name] = {
            "best_alpha": best_alpha,
            "selection_metric": selection_metric,
            "num_val_queries": len(val_data),
            "mean_by_alpha": {f"{a:.1f}": mean_by_alpha[a] for a in alphas},
            "grid": {
                f"{a:.1f}": {model: grid[a][model] for model in semantic_models}
                for a in alphas
            },
        }

        print(
            f"\n[{benchmark_name}] best alpha = {best_alpha:.1f} "
            f"(mean {selection_metric.upper()}={mean_by_alpha[best_alpha]:.4f} over "
            f"{len(semantic_models)} semantic models on {len(val_data)} val queries)"
        )

        del benchmark_data, val_data, ground_truth, similarity_cache
        del normalized_temporal_cache, normalized_temporal

    _write_tuning_report(results, alphas, selection_metric, output_path)


def _write_tuning_report(
    results: Dict[str, Any],
    alphas: List[float],
    selection_metric: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON artifact with the full grid + selected alphas.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary alongside the JSON.
    lines: List[str] = []
    lines.append("# Alpha tuning on the val split\n")
    lines.append(
        f"Selection: alpha maximizing mean **{selection_metric.upper()}** across "
        f"semantic models on the val split.\n"
    )
    header = ["Benchmark"] + [f"a={a:.1f}" for a in alphas] + ["Best alpha", "Val queries"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for benchmark_name, entry in results.items():
        best_alpha = entry["best_alpha"]
        cells = [benchmark_name]
        for a in alphas:
            value = entry["mean_by_alpha"][f"{a:.1f}"]
            cell = f"{value:.4f}"
            if abs(a - best_alpha) < 1e-9:
                cell = f"**{cell}**"
            cells.append(cell)
        cells.append(f"**{best_alpha:.1f}**")
        cells.append(str(entry["num_val_queries"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Selected alphas\n")
    for benchmark_name, entry in results.items():
        lines.append(f"- {benchmark_name}: alpha = {entry['best_alpha']:.1f}")

    report_md_path = output_path.with_suffix(".md")
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nBest alpha per benchmark:")
    for benchmark_name, entry in results.items():
        print(f"  {benchmark_name}: {entry['best_alpha']:.1f}")
    print(f"\nWrote tuning grid to: {output_path}")
    print(f"Wrote summary table to: {report_md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag benchmarks into val/test and tune the hybrid alpha on val."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tag_parser = subparsers.add_parser(
        "tag", help="Randomly tag benchmark datapoints as val/test, in place."
    )
    tag_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    tag_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of datapoints assigned to val (default 0.2 -> 20%% val / 80%% test).",
    )
    tag_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic val/test assignment.",
    )

    tune_parser = subparsers.add_parser(
        "tune-alpha", help="Select the best hybrid alpha per benchmark on the val split."
    )
    tune_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    tune_parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Alpha values to test (default: 0.1 0.2 ... 0.9).",
    )
    tune_parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_SELECTION_METRIC,
        help="Metric to maximize when selecting alpha (default: mrr).",
    )
    tune_parser.add_argument(
        "--output",
        type=str,
        default="output/metrics/alpha_tuning_val.json",
        help="Where to write the tuning grid (a .md summary is written alongside).",
    )

    args = parser.parse_args()

    if args.command == "tag":
        tag_benchmarks(
            config_path=Path(args.config),
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    elif args.command == "tune-alpha":
        tune_alpha(
            config_path=Path(args.config),
            alphas=[round(a, 4) for a in args.alphas],
            selection_metric=args.metric,
            output_path=Path(args.output),
        )


if __name__ == "__main__":
    main()
