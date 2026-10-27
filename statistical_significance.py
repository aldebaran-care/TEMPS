"""Paper statistical significance: baseline vs baseline+TEMPS.

For every (benchmark, baseline, metric) cell of the paper results table, this
script computes per-query metric values for (a) the baseline alone and (b) the
baseline fused with the trained TEMPS temporal model at the per-benchmark
alpha used in the paper, then runs paired tests on the per-query differences:

  * Paired Wilcoxon signed-rank test  (primary — non-parametric, robust to
    the heavy-tailed / quasi-binary distributions IR metrics typically have).
  * Paired t-test                      (secondary — standard, normal-theory).
  * Bootstrap 95% CI on the mean diff  (effect-size visualization).

The script reuses the similarity pickles that `run_paper_evaluations.py`
already wrote to `output/similarities/...` — no embeddings are recomputed.
The "Mistral" baseline in the paper corresponds to `salesforce` on disk
(SFR-Embedding-Mistral); the config maps the display name to the model id.

Output: a single markdown report under `output/metrics/`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples
from temporal_embeddings.evaluation.utils.data.splits import VALID_SPLITS, select_split_items
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.utils.math.normalize import normalize_list
from temporal_embeddings.utils.os.folder_management import create_folders


Similarities = Dict[str, Dict[str, float]]


# Mirrors run_paper_evaluations.BENCHMARK_PATHS — kept inline to avoid pulling
# in the (heavy) imports at the top of that module.
BENCHMARK_PATHS: Dict[str, Path] = {
    "time_sensitive_qa": Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json"),
    "temp_reason": Path("data/evaluation/temp_reason/processed_data.json"),
    "ts_retriever": Path("data/evaluation/ts_retriever/processed_ts_retriever.json"),
}


METRIC_DISPLAY: Dict[str, str] = {
    "mrr": "MRR",
    "ndcg": "NDCG@{k}",
    "recall": "Recall@{k}",
    "precision": "Precision@{k}",
}


@dataclass
class PairResult:
    benchmark_display: str
    baseline_display: str
    metric: str
    n: int
    mean_baseline: float
    mean_hybrid: float
    mean_diff: float
    wilcoxon_stat: float
    wilcoxon_p: float
    t_stat: float
    t_p: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    n_better: int  # queries where hybrid > baseline
    n_worse: int   # queries where hybrid < baseline
    n_tied: int


# ---------------------------------------------------------------------------
# Loading + merging similarities (matches run_paper_evaluations.py semantics).
# ---------------------------------------------------------------------------


def _load_similarities_pickle(path: Path) -> Similarities:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing similarity cache: {path}\n"
            f"Run run_paper_evaluations.py first so the cache is populated."
        )
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected nested-dict similarity cache at {path}, got "
            f"{type(obj).__name__}. Re-run run_paper_evaluations.py to "
            f"regenerate in the current format."
        )
    return obj


def _normalize_similarities(similarities: Similarities) -> Similarities:
    """Per-query min-max normalization. Same scheme as
    `_normalize_similarities` in run_paper_evaluations.py — required before
    blending two heterogeneous score families."""

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
    """alpha * normalized_temporal + (1 - alpha) * normalized_semantic.

    Convention matches run_paper_evaluations.py: `alpha` is the temporal
    share, so alpha=0.6 means 60% temporal, 40% semantic.
    """

    merged: Similarities = {}
    for question, paragraphs in temporal.items():
        semantic_paragraphs = semantic.get(question, {})
        merged[question] = {
            paragraph: alpha * sim + (1.0 - alpha) * semantic_paragraphs.get(paragraph, 0.0)
            for paragraph, sim in paragraphs.items()
        }
    return merged


# ---------------------------------------------------------------------------
# Per-query metric computation.
# ---------------------------------------------------------------------------


def _benchmark_data(benchmark_path: Path) -> List[Dict[str, Any]]:
    with benchmark_path.open("r", encoding="utf-8") as f:
        return add_negative_samples(json.load(f), num_negatives=0)


def _per_query_similarity_lists(
    similarities: Similarities,
    benchmark_data: List[Dict[str, Any]],
) -> List[List[float]]:
    out: List[List[float]] = []
    for item in benchmark_data:
        bucket = similarities[item["question"]]
        out.append([float(bucket[paragraph]) for paragraph in item["paragraphs"]])
    return out


def _per_query_metric(
    ground_truth: List[List[int]],
    similarities_list: List[List[float]],
    top_k: int,
    metric: str,
) -> np.ndarray:
    """Compute the requested metric one query at a time, returning a vector of
    per-query scores (length = number of queries)."""

    scores = np.empty(len(ground_truth), dtype=np.float64)
    for i, (gt, sims) in enumerate(zip(ground_truth, similarities_list)):
        result = compute_metrics([gt], [sims], top_k=top_k, metric=metric)
        scores[i] = float(result[metric])
    return scores


# ---------------------------------------------------------------------------
# Statistical tests.
# ---------------------------------------------------------------------------


def _bootstrap_mean_diff_ci(
    differences: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    if differences.size == 0:
        return float("nan"), float("nan")
    n = differences.size
    idx = rng.integers(0, n, size=(iterations, n))
    samples = differences[idx].mean(axis=1)
    low = float(np.percentile(samples, (1.0 - confidence) / 2.0 * 100.0))
    high = float(np.percentile(samples, (1.0 + confidence) / 2.0 * 100.0))
    return low, high


def _paired_tests(
    baseline_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    assert baseline_scores.shape == hybrid_scores.shape
    differences = hybrid_scores - baseline_scores

    # Wilcoxon: zero differences must be handled — `zero_method="wilcox"`
    # drops them (the most common choice, e.g. used in trec_eval).
    if np.all(differences == 0):
        wilcoxon_stat, wilcoxon_p = float("nan"), 1.0
    else:
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                hybrid_scores,
                baseline_scores,
                zero_method="wilcox",
                alternative="two-sided",
            )
        except ValueError:
            wilcoxon_stat, wilcoxon_p = float("nan"), 1.0

    if np.all(differences == 0):
        t_stat, t_p = 0.0, 1.0
    else:
        t_stat, t_p = stats.ttest_rel(hybrid_scores, baseline_scores)

    ci_low, ci_high = _bootstrap_mean_diff_ci(differences, iterations, rng)

    return {
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_better": int(np.sum(differences > 0)),
        "n_worse": int(np.sum(differences < 0)),
        "n_tied": int(np.sum(differences == 0)),
    }


def _significance_marker(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    # A literal 0.0 from SciPy still means "vanishingly small" — treat it as
    # the most significant bin.
    if p == 0.0:
        return "***"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Pipeline.
# ---------------------------------------------------------------------------


def _resolve_cache_paths(
    config: Dict[str, Any],
    benchmark_name: str,
    semantic_model: str,
) -> Tuple[Path, Path]:
    output_paths = set_output_files(
        temporal_model_name=config["temporal_model_name"],
        temporal_model_path=Path(config["temporal_model_path"]),
        semantic_model_name=semantic_model,
        benchmark=benchmark_name,
    )
    return (
        output_paths["temporal_similarities_path"],
        output_paths["semantic_similarities_path"],
    )


def _run_pairs_for_benchmark(
    config: Dict[str, Any],
    benchmark_name: str,
    benchmark_cfg: Dict[str, Any],
    rng: np.random.Generator,
    split: str,
) -> List[PairResult]:
    top_k = int(config["top_k"])
    metrics = list(config["metrics"])
    iterations = int(config["bootstrap_iterations"])
    alpha = float(benchmark_cfg["alpha"])
    benchmark_display = str(benchmark_cfg["display"])

    benchmark_path = BENCHMARK_PATHS[benchmark_name]
    # Similarity caches are split-agnostic (built over the full benchmark); only
    # the per-query significance test is restricted to the requested split.
    data = _benchmark_data(benchmark_path)
    data = select_split_items(data, split)
    ground_truth = [item["answer"] for item in data]
    print(f"\n[{benchmark_display}] alpha={alpha}  split={split}  queries={len(data)}")

    # The temporal cache is shared across all baselines on this benchmark.
    temporal_sims_path, _ = _resolve_cache_paths(config, benchmark_name, semantic_model="")
    temporal_sims = _load_similarities_pickle(temporal_sims_path)
    normalized_temporal = _normalize_similarities(temporal_sims)

    rows: List[PairResult] = []
    for baseline in config["baselines"]:
        baseline_display = str(baseline["display"])
        semantic_model = str(baseline["model"])

        _, semantic_sims_path = _resolve_cache_paths(config, benchmark_name, semantic_model)
        print(
            f"  - {baseline_display:25s} (model={semantic_model})\n"
            f"      temporal  cache: {temporal_sims_path}\n"
            f"      semantic  cache: {semantic_sims_path}"
        )
        semantic_sims = _load_similarities_pickle(semantic_sims_path)
        normalized_semantic = _normalize_similarities(semantic_sims)

        hybrid_sims = _merge_hybrid(normalized_temporal, normalized_semantic, alpha)

        baseline_sim_lists = _per_query_similarity_lists(semantic_sims, data)
        hybrid_sim_lists = _per_query_similarity_lists(hybrid_sims, data)

        for metric in metrics:
            baseline_scores = _per_query_metric(ground_truth, baseline_sim_lists, top_k, metric)
            hybrid_scores = _per_query_metric(ground_truth, hybrid_sim_lists, top_k, metric)

            tests = _paired_tests(baseline_scores, hybrid_scores, iterations, rng)

            rows.append(
                PairResult(
                    benchmark_display=benchmark_display,
                    baseline_display=baseline_display,
                    metric=metric,
                    n=int(baseline_scores.size),
                    mean_baseline=float(baseline_scores.mean()),
                    mean_hybrid=float(hybrid_scores.mean()),
                    mean_diff=float(hybrid_scores.mean() - baseline_scores.mean()),
                    wilcoxon_stat=tests["wilcoxon_stat"],
                    wilcoxon_p=tests["wilcoxon_p"],
                    t_stat=tests["t_stat"],
                    t_p=tests["t_p"],
                    bootstrap_ci_low=tests["ci_low"],
                    bootstrap_ci_high=tests["ci_high"],
                    n_better=tests["n_better"],
                    n_worse=tests["n_worse"],
                    n_tied=tests["n_tied"],
                )
            )

    return rows


# ---------------------------------------------------------------------------
# Markdown report.
# ---------------------------------------------------------------------------


def _fmt(x: float, prec: int = 4) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{prec}f}"


def _fmt_p(p: float) -> str:
    """Emit the literal p-value as returned by SciPy. If SciPy underflowed
    to 0.0 we print `0` — no analytical recovery is performed."""

    if not np.isfinite(p):
        return "—"
    if p == 0.0:
        return "0"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4g}"


def _metric_label(metric: str, top_k: int) -> str:
    return METRIC_DISPLAY[metric].format(k=top_k)


def _write_markdown(
    rows: List[PairResult],
    config: Dict[str, Any],
    report_path: Path,
    split: str,
) -> None:
    create_folders([report_path.parent])
    top_k = int(config["top_k"])
    metrics = list(config["metrics"])
    iterations = int(config["bootstrap_iterations"])

    lines: List[str] = []
    lines.append("# Statistical Significance — Baseline vs. Baseline + TEMPS")
    lines.append("")
    lines.append(
        "For each (benchmark, baseline) pair we compare the semantic baseline "
        "alone against the same baseline fused with the trained TEMPS temporal "
        "model at the per-benchmark $\\alpha$ used in the paper. Per-query "
        "metric scores are produced from cached similarity matrices "
        "(`output/similarities/...`) and the hybrid follows the "
        "min-max-normalized merge `alpha * temporal + (1 - alpha) * semantic` "
        "used in `run_paper_evaluations.py`."
    )
    lines.append("")
    lines.append(
        f"All per-query metrics are computed on the held-out **{split}** split; "
        "the similarity caches themselves are split-agnostic (built over the full "
        "benchmark), so no similarities are recomputed."
    )
    lines.append("")
    lines.append("**Tests reported:**")
    lines.append("")
    lines.append(
        "- **Wilcoxon signed-rank** (paired, two-sided) — primary test; "
        "non-parametric and appropriate for the heavy-tailed, quasi-binary "
        "distributions IR metrics typically produce."
    )
    lines.append("- **Paired t-test** (two-sided) — secondary, normal-theory.")
    lines.append(
        f"- **Bootstrap 95% CI** on the mean per-query difference "
        f"(B={iterations:,} resamples, seed={config['random_seed']})."
    )
    lines.append("")
    lines.append(
        "**Significance markers** (Wilcoxon p): `***` p<0.001, `**` p<0.01, "
        "`*` p<0.05, `ns` not significant, `n/a` undefined. A literal `0` in "
        "the p-value columns is SciPy's float64 underflow — the true value is "
        "below ~$2.2\\times 10^{-308}$ and is reported verbatim as returned."
    )
    lines.append("")

    # Per-benchmark grouping
    benchmarks_in_order = [
        cfg["display"] for cfg in config["benchmarks"].values()
    ]
    by_benchmark: Dict[str, List[PairResult]] = {b: [] for b in benchmarks_in_order}
    for row in rows:
        by_benchmark.setdefault(row.benchmark_display, []).append(row)

    alpha_lookup = {
        cfg["display"]: cfg["alpha"] for cfg in config["benchmarks"].values()
    }

    for benchmark_display, benchmark_rows in by_benchmark.items():
        if not benchmark_rows:
            continue
        alpha = alpha_lookup.get(benchmark_display, float("nan"))
        n_queries = benchmark_rows[0].n
        lines.append(f"## {benchmark_display}  ($\\alpha = {alpha}$, $n = {n_queries}$)")
        lines.append("")
        header = [
            "Baseline",
            "Metric",
            "Baseline",
            "+TEMPS",
            "Δ (mean)",
            "95% CI (bootstrap)",
            "Wilcoxon p",
            "t-test p",
            "Win / Loss / Tie",
            "Sig",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Sort: baselines in config declaration order, metrics in config order.
        baseline_order = {b["display"]: i for i, b in enumerate(config["baselines"])}
        metric_order = {m: i for i, m in enumerate(metrics)}
        benchmark_rows.sort(
            key=lambda r: (baseline_order[r.baseline_display], metric_order[r.metric])
        )

        for row in benchmark_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.baseline_display,
                        _metric_label(row.metric, top_k),
                        _fmt(row.mean_baseline),
                        _fmt(row.mean_hybrid),
                        ("+" if row.mean_diff >= 0 else "") + _fmt(row.mean_diff),
                        f"[{_fmt(row.bootstrap_ci_low)}, {_fmt(row.bootstrap_ci_high)}]",
                        _fmt_p(row.wilcoxon_p),
                        _fmt_p(row.t_p),
                        f"{row.n_better} / {row.n_worse} / {row.n_tied}",
                        _significance_marker(row.wilcoxon_p),
                    ]
                )
                + " |"
            )
        lines.append("")

    # Compact omnibus summary: one row per (benchmark, baseline), Wilcoxon p
    # on the headline metric (MRR) for at-a-glance reading.
    lines.append("## Headline summary (MRR, Wilcoxon)")
    lines.append("")
    lines.append("| Benchmark | Baseline | MRR (base) | MRR (+TEMPS) | Δ | Wilcoxon p | Sig |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    headline_rows = [r for r in rows if r.metric == "mrr"]
    benchmark_index = {cfg["display"]: i for i, cfg in enumerate(config["benchmarks"].values())}
    baseline_index = {b["display"]: i for i, b in enumerate(config["baselines"])}
    headline_rows.sort(
        key=lambda r: (benchmark_index[r.benchmark_display], baseline_index[r.baseline_display])
    )
    for row in headline_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.benchmark_display,
                    row.baseline_display,
                    _fmt(row.mean_baseline),
                    _fmt(row.mean_hybrid),
                    ("+" if row.mean_diff >= 0 else "") + _fmt(row.mean_diff),
                    _fmt_p(row.wilcoxon_p),
                    _significance_marker(row.wilcoxon_p),
                ]
            )
            + " |"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def run_statistical_significance(config_path: Path, split: str = "test") -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    for required in (
        "temporal_model_name",
        "temporal_model_path",
        "benchmarks",
        "baselines",
        "metrics",
        "top_k",
        "bootstrap_iterations",
        "random_seed",
        "report_path",
    ):
        if required not in config:
            raise ValueError(f"Missing config key: {required}")

    for benchmark_name in config["benchmarks"]:
        if benchmark_name not in BENCHMARK_PATHS:
            raise ValueError(
                f"Unknown benchmark '{benchmark_name}'. "
                f"Supported: {list(BENCHMARK_PATHS)}"
            )

    if split not in VALID_SPLITS:
        raise ValueError(f"--split must be one of {VALID_SPLITS}, got '{split}'.")

    rng = np.random.default_rng(int(config["random_seed"]))
    all_rows: List[PairResult] = []
    for benchmark_name, benchmark_cfg in config["benchmarks"].items():
        rows = _run_pairs_for_benchmark(config, benchmark_name, benchmark_cfg, rng, split)
        all_rows.extend(rows)

    report_path = Path(config["report_path"])
    _write_markdown(all_rows, config, report_path, split)
    print(f"\nReport written to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired statistical significance for baseline vs baseline+TEMPS."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="temporal_embeddings/config/statistical_significance_config.json",
        help="Path to the statistical-significance config JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=list(VALID_SPLITS),
        help=(
            "Which split to run the significance tests on (default: test). "
            "Splits are created with split_and_tune_alpha.py; untagged benchmarks "
            "fall back to all queries with a warning."
        ),
    )
    args = parser.parse_args()
    run_statistical_significance(Path(args.config), split=args.split)


if __name__ == "__main__":
    main()
