"""Qualitative analysis for TimeQA (time_sensitive_qa): how TEMPS changes the
retrieved documents relative to a semantic-only baseline.

Reuses the *exact* pipeline arithmetic from run_paper_evaluations.py:
  - per-question min-max normalization over the full candidate bucket
    (temporal_embeddings.utils.math.normalize.normalize_list),
  - hybrid score = alpha * temporal_norm + (1 - alpha) * semantic_norm,
  - ranking = argsort(score) descending (temporal_embeddings ... metrics).

It loads the precomputed similarity pickles (no model inference) and mines the
cases where the semantic-only retriever ranks the gold paragraph low while the
hybrid TEMPS retriever pulls it to the top. Emits a markdown report and a JSON
dump of the cases for the rebuttal.

Run:  python qualitative_analysis_timeqa.py --baseline intfloat/e5-base-v2 --alpha 0.6
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.data.splits import select_split_items
from temporal_embeddings.utils.math.normalize import normalize_list


BENCHMARK = "time_sensitive_qa"
BENCHMARK_PATH = Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json")
TEMPORAL_MODEL_NAME = "all-minilm-l6-v2"
TEMPORAL_MODEL_PATH = Path(
    "output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2026-05-24_00-00-51.pth"
)

Similarities = Dict[str, Dict[str, float]]

# Human-facing labels for the baselines we can feature.
BASELINE_LABELS = {
    "intfloat/e5-base-v2": "E5-base-v2",
    "salesforce": "SFR-Embedding-Mistral",
    "all-mpnet-base-v2": "MPNet-base-v2",
    "BAAI/bge-large-en-v1.5": "BGE-large-v1.5",
}

YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def load_similarities(path: Path) -> Similarities:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected nested-dict similarities at {path}, got {type(obj)}")
    return obj


def normalize_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    keys = list(bucket.keys())
    values = [bucket[k] for k in keys]
    return dict(zip(keys, normalize_list(values)))


def rank_of_gold(scores: List[float], gold_idx: int) -> int:
    """1-indexed rank of the gold candidate under descending sort (matches
    compute_single_metric: np.argsort(scores)[::-1])."""
    ranked = np.argsort(scores)[::-1]
    return int(np.where(ranked == gold_idx)[0][0]) + 1


def top_k_indices(scores: List[float], k: int) -> List[int]:
    return list(np.argsort(scores)[::-1][:k])


def years_in(text: str) -> List[str]:
    return YEAR_RE.findall(text)


def snippet(text: str, width: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width].rstrip() + " …"


def build_cases(
    eval_data: List[Dict[str, Any]],
    temporal_sims: Similarities,
    semantic_sims: Similarities,
    alpha: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # Precompute normalized buckets per question (pipeline normalizes over the
    # full question bucket, then extracts the item's candidate paragraphs).
    norm_t_cache: Dict[str, Dict[str, float]] = {}
    norm_s_cache: Dict[str, Dict[str, float]] = {}

    for item in eval_data:
        question = item["question"]
        paragraphs = item["paragraphs"]
        gold_idx = item["answer"][0]

        if question not in temporal_sims or question not in semantic_sims:
            continue

        if question not in norm_t_cache:
            norm_t_cache[question] = normalize_bucket(temporal_sims[question])
            norm_s_cache[question] = normalize_bucket(semantic_sims[question])
        norm_t = norm_t_cache[question]
        norm_s = norm_s_cache[question]

        semantic_scores = [semantic_sims[question][p] for p in paragraphs]
        temporal_scores = [temporal_sims[question][p] for p in paragraphs]
        hybrid_scores = [
            alpha * norm_t[p] + (1.0 - alpha) * norm_s[p] for p in paragraphs
        ]

        rank_sem = rank_of_gold(semantic_scores, gold_idx)
        rank_hyb = rank_of_gold(hybrid_scores, gold_idx)
        rank_tmp = rank_of_gold(temporal_scores, gold_idx)

        cases.append(
            {
                "question": question,
                "paragraphs": paragraphs,
                "gold_idx": gold_idx,
                "gold_text": paragraphs[gold_idx],
                "question_years": years_in(question),
                "rank_semantic": rank_sem,
                "rank_hybrid": rank_hyb,
                "rank_temporal": rank_tmp,
                "num_candidates": len(paragraphs),
                "semantic_scores": semantic_scores,
                "hybrid_scores": hybrid_scores,
                "temporal_scores": temporal_scores,
            }
        )
    return cases


def aggregate_stats(cases: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    n = len(cases)
    sem_top1 = sum(c["rank_semantic"] == 1 for c in cases)
    hyb_top1 = sum(c["rank_hybrid"] == 1 for c in cases)
    sem_top5 = sum(c["rank_semantic"] <= top_k for c in cases)
    hyb_top5 = sum(c["rank_hybrid"] <= top_k for c in cases)

    # Directional flips.
    sem_fail_hyb_fix_top1 = sum(
        c["rank_semantic"] > 1 and c["rank_hybrid"] == 1 for c in cases
    )
    sem_ok_hyb_break_top1 = sum(
        c["rank_semantic"] == 1 and c["rank_hybrid"] > 1 for c in cases
    )
    sem_fail_hyb_fix_top5 = sum(
        c["rank_semantic"] > top_k and c["rank_hybrid"] <= top_k for c in cases
    )
    sem_ok_hyb_break_top5 = sum(
        c["rank_semantic"] <= top_k and c["rank_hybrid"] > top_k for c in cases
    )
    return {
        "n": n,
        "sem_top1": sem_top1,
        "hyb_top1": hyb_top1,
        "sem_top1_acc": sem_top1 / n,
        "hyb_top1_acc": hyb_top1 / n,
        "sem_top5": sem_top5,
        "hyb_top5": hyb_top5,
        "sem_top5_acc": sem_top5 / n,
        "hyb_top5_acc": hyb_top5 / n,
        "sem_fail_hyb_fix_top1": sem_fail_hyb_fix_top1,
        "sem_ok_hyb_break_top1": sem_ok_hyb_break_top1,
        "sem_fail_hyb_fix_top5": sem_fail_hyb_fix_top5,
        "sem_ok_hyb_break_top5": sem_ok_hyb_break_top5,
    }


def select_showcase(cases: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Rank 'baseline fails -> TEMPS fixes' cases for presentation.

    Priority: the query carries an explicit year, the semantic top-1 is
    temporally wrong (does not contain a query year) while gold contains it,
    and the improvement in gold rank is large.
    """
    winners = [
        c
        for c in cases
        if c["rank_hybrid"] == 1 and c["rank_semantic"] > 1 and c["question_years"]
    ]

    for c in winners:
        paragraphs = c["paragraphs"]
        sem_top1_idx = top_k_indices(c["semantic_scores"], 1)[0]
        sem_top1_text = paragraphs[sem_top1_idx]
        qyears = set(c["question_years"])
        gold_years = set(years_in(c["gold_text"]))
        sem_top1_years = set(years_in(sem_top1_text))

        c["sem_top1_idx"] = sem_top1_idx
        c["sem_top1_text"] = sem_top1_text
        c["gold_has_query_year"] = bool(qyears & gold_years)
        c["sem_top1_has_query_year"] = bool(qyears & sem_top1_years)
        # A clean "temporal distractor": baseline picks a passage that is topically
        # close but carries the wrong year, while gold carries the queried year.
        c["clean_temporal_distractor"] = (
            c["gold_has_query_year"] and not c["sem_top1_has_query_year"]
        )
        c["rank_gain"] = c["rank_semantic"] - c["rank_hybrid"]

    winners.sort(
        key=lambda c: (
            c["clean_temporal_distractor"],
            c["gold_has_query_year"],
            c["rank_gain"],
            -c["num_candidates"],
        ),
        reverse=True,
    )
    return winners


def format_ranklist(
    case: Dict[str, Any], score_key: str, top_k: int
) -> List[str]:
    paragraphs = case["paragraphs"]
    scores = case[score_key]
    gold_idx = case["gold_idx"]
    qyears = set(case["question_years"])
    lines: List[str] = []
    for rank, idx in enumerate(top_k_indices(scores, top_k), 1):
        is_gold = idx == gold_idx
        text = paragraphs[idx]
        pyears = years_in(text)
        year_hits = sorted(set(pyears) & qyears)
        marker = " ✅ **GOLD**" if is_gold else ""
        year_note = f" _(years: {', '.join(pyears)})_" if pyears else " _(no year)_"
        lines.append(
            f"{rank}. [{score_key.split('_')[0]} {scores[idx]:.3f}]{marker} "
            f"{snippet(text)}{year_note}"
        )
    return lines


def write_report(
    report_path: Path,
    baseline_label: str,
    alpha: float,
    top_k: int,
    stats: Dict[str, Any],
    showcase: List[Dict[str, Any]],
    n_examples: int,
) -> None:
    L: List[str] = []
    L.append(f"# TimeQA qualitative analysis — {baseline_label} vs. TEMPS (α={alpha})")
    L.append("")
    L.append(
        f"Retrieval on the TimeQA (`time_sensitive_qa`) **test** split "
        f"({stats['n']} queries, 1 gold paragraph each). "
        f"**Baseline** = semantic-only {baseline_label}. "
        f"**TEMPS** = per-query normalized hybrid, "
        f"`{alpha:.1f}·temporal + {1 - alpha:.1f}·{baseline_label}`."
    )
    L.append("")
    L.append("## Aggregate effect of adding the temporal signal")
    L.append("")
    L.append("| Metric | Baseline | TEMPS |")
    L.append("| --- | --- | --- |")
    L.append(
        f"| Gold@1 (top-1 accuracy) | {stats['sem_top1_acc']:.3f} "
        f"({stats['sem_top1']}/{stats['n']}) | {stats['hyb_top1_acc']:.3f} "
        f"({stats['hyb_top1']}/{stats['n']}) |"
    )
    L.append(
        f"| Gold@{top_k} (recall@{top_k}) | {stats['sem_top5_acc']:.3f} "
        f"({stats['sem_top5']}/{stats['n']}) | {stats['hyb_top5_acc']:.3f} "
        f"({stats['hyb_top5']}/{stats['n']}) |"
    )
    L.append("")
    L.append(
        f"- **{stats['sem_fail_hyb_fix_top1']}** queries: baseline ranks gold below #1, "
        f"TEMPS moves it to #1. (Regressions the other way: "
        f"{stats['sem_ok_hyb_break_top1']}.)"
    )
    L.append(
        f"- **{stats['sem_fail_hyb_fix_top5']}** queries: gold is outside the baseline's "
        f"top-{top_k} but inside TEMPS's top-{top_k}. (Regressions: "
        f"{stats['sem_ok_hyb_break_top5']}.)"
    )
    L.append("")
    L.append("## Qualitative examples (baseline fails → TEMPS fixes)")
    L.append("")
    L.append(
        "Each block shows the query (with its temporal constraint), the baseline's "
        f"top-{top_k}, and TEMPS's top-{top_k}. All candidates are paragraphs from the "
        "same Wikipedia page, so they are near-identical topically — the only thing "
        "that separates them is temporal scope."
    )
    L.append("")

    for i, case in enumerate(showcase[:n_examples], 1):
        L.append(f"### Example {i}")
        L.append("")
        L.append(f"**Q:** {case['question']}")
        yrs = ", ".join(case["question_years"]) or "—"
        L.append(f"_Temporal constraint in query:_ **{yrs}**")
        L.append("")
        L.append(
            f"_Gold rank — baseline: **#{case['rank_semantic']}** / "
            f"{case['num_candidates']}; temporal-only: #{case['rank_temporal']}; "
            f"TEMPS: **#{case['rank_hybrid']}**._"
        )
        L.append("")
        L.append(f"**Baseline ({baseline_label}) top-{top_k}:**")
        L.append("")
        for line in format_ranklist(case, "semantic_scores", top_k):
            L.append(f"- {line}")
        L.append("")
        L.append(f"**TEMPS (α={alpha}) top-{top_k}:**")
        L.append("")
        for line in format_ranklist(case, "hybrid_scores", top_k):
            L.append(f"- {line}")
        L.append("")
        L.append("---")
        L.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="intfloat/e5-base-v2",
                        help="Semantic baseline model id (must have a cached similarities pickle).")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-examples", type=int, default=12)
    parser.add_argument("--out-dir", default="output/qualitative")
    args = parser.parse_args()

    baseline_label = BASELINE_LABELS.get(args.baseline, args.baseline)

    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        benchmark_data = json.load(f)  # num_negatives=0 -> data unchanged
    eval_data = select_split_items(benchmark_data, args.split)

    temporal_paths = set_output_files(
        temporal_model_name=TEMPORAL_MODEL_NAME,
        temporal_model_path=TEMPORAL_MODEL_PATH,
        semantic_model_name="",
        benchmark=BENCHMARK,
    )
    semantic_paths = set_output_files(
        temporal_model_name="",
        temporal_model_path=TEMPORAL_MODEL_PATH,
        semantic_model_name=args.baseline,
        benchmark=BENCHMARK,
    )
    temporal_sims = load_similarities(temporal_paths["temporal_similarities_path"])
    semantic_sims = load_similarities(semantic_paths["semantic_similarities_path"])
    print(f"Loaded temporal sims ({len(temporal_sims)} q) and "
          f"{baseline_label} sims ({len(semantic_sims)} q).")

    cases = build_cases(eval_data, temporal_sims, semantic_sims, args.alpha, args.top_k)
    print(f"Scored {len(cases)} evaluable queries on split='{args.split}'.")

    stats = aggregate_stats(cases, args.top_k)
    print(json.dumps(stats, indent=2))

    showcase = select_showcase(cases, args.top_k)
    print(f"{len(showcase)} 'baseline>1 -> TEMPS=1 with a dated query' cases; "
          f"{sum(c['clean_temporal_distractor'] for c in showcase)} are clean "
          f"temporal-distractor cases.")

    out_dir = Path(args.out_dir)
    slug = baseline_label.lower().replace(" ", "_")
    report_path = out_dir / f"timeqa_qualitative_{slug}_alpha{args.alpha}.md"
    write_report(report_path, baseline_label, args.alpha, args.top_k, stats,
                 showcase, args.n_examples)
    print(f"Report written to: {report_path}")

    # JSON dump of the showcase (trimmed) for downstream use.
    json_path = out_dir / f"timeqa_qualitative_{slug}_alpha{args.alpha}.json"
    dump = {
        "baseline": args.baseline,
        "baseline_label": baseline_label,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "split": args.split,
        "stats": stats,
        "examples": [
            {
                "question": c["question"],
                "question_years": c["question_years"],
                "gold_text": c["gold_text"],
                "rank_semantic": c["rank_semantic"],
                "rank_temporal": c["rank_temporal"],
                "rank_hybrid": c["rank_hybrid"],
                "num_candidates": c["num_candidates"],
                "clean_temporal_distractor": c.get("clean_temporal_distractor", False),
                "semantic_top1_text": c.get("sem_top1_text", ""),
            }
            for c in showcase[: args.n_examples]
        ],
    }
    json_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON written to: {json_path}")


if __name__ == "__main__":
    main()
