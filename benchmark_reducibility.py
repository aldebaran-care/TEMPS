"""Reproduce the 'timestamp-matching reducibility' table for the rebuttal.

For each benchmark (and, for TempReason, each reasoning subtype) it reports, on
the *test* split, using the cached similarity pickles (no model inference):

  1. % golds carrying the query timestamp
       fraction of queries for which at least one gold passage literally
       contains a year token that appears in the query. This is the necessary
       condition for an explicit-timestamp matcher to be able to find the gold.

  2. Timestamp matcher (MRR)
       a *pure* explicit-timestamp retriever: each candidate scores 1 if it
       carries a query year token, else 0; ranking = matches first, ties broken
       uniformly at random. We report the EXACT expected reciprocal rank under
       that random tie-breaking (closed form, deterministic — no seed), which is
       the honest ceiling of "surface-timestamp matching".

  3. Temporal-only (MRR)  — ranking by the trained temporal head's similarity.
  4. Semantic (MRR)       — ranking by the semantic baseline's cosine similarity.

Ranking / MRR follow the repo convention exactly
(temporal_embeddings/.../metrics.py: np.argsort(scores)[::-1], reciprocal rank
of the first relevant candidate), so columns 3-4 reproduce the numbers in
output/metrics/paper_evaluation_report.md (printed as a sanity check).

Run:
    python benchmark_reducibility.py
    python benchmark_reducibility.py --semantic intfloat/e5-base-v2
"""

import argparse
import json
import re
from math import comb
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.data.splits import select_split_items

# The trained temporal checkpoint used for the paper (same across benchmarks).
TEMPORAL_MODEL_NAME = "all-minilm-l6-v2"
TEMPORAL_MODEL_PATH = Path(
    "output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2026-05-24_00-00-51.pth"
)
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def query_years(text: str) -> Set[int]:
    return {int(y) for y in YEAR_RE.findall(text)}


def load_similarities(path: Path) -> Dict[str, Dict[str, float]]:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected a nested-dict similarities cache at {path}, got {type(obj).__name__}. "
            f"(Legacy DataFrame caches are not supported by this script.)"
        )
    return obj


def mrr_from_scores(scores: List[float], golds: Set[int]) -> float:
    """Reciprocal rank of the first relevant candidate (repo convention:
    np.argsort(scores)[::-1])."""
    ranked = np.argsort(scores)[::-1]
    for rank, idx in enumerate(ranked, 1):
        if idx in golds:
            return 1.0 / rank
    return 0.0


def _expected_inv_rank(n_tier: int, g_tier: int, offset: int) -> float:
    """E[1 / (offset + R)] where R is the rank of the FIRST of `g_tier` gold
    items placed uniformly at random among `n_tier` tied items.

    P(R = r) = C(n_tier - r, g_tier - 1) / C(n_tier, g_tier), for r = 1..n_tier-g_tier+1.
    """
    if g_tier <= 0 or n_tier <= 0:
        return 0.0
    if g_tier == 1:  # fast path (the common single-gold case)
        return sum(1.0 / (offset + r) for r in range(1, n_tier + 1)) / n_tier
    denom = comb(n_tier, g_tier)
    total = 0.0
    for r in range(1, n_tier - g_tier + 2):
        total += (comb(n_tier - r, g_tier - 1) / denom) * (1.0 / (offset + r))
    return total


def timestamp_matcher_mrr(match: List[bool], golds: Set[int]) -> float:
    """Exact expected reciprocal rank of a pure timestamp matcher.

    Candidates with a query-year token (match=True) form the top tier, the rest
    the bottom tier; within a tier order is uniformly random. Returns the exact
    expectation over that randomness (no sampling, fully deterministic)."""
    top = [i for i in range(len(match)) if match[i]]
    bot = [i for i in range(len(match)) if not match[i]]
    g_top = sum(1 for i in top if i in golds)
    g_bot = sum(1 for i in bot if i in golds)
    if g_top > 0:
        return _expected_inv_rank(len(top), g_top, offset=0)
    if g_bot > 0:
        return _expected_inv_rank(len(bot), g_bot, offset=len(top))
    return 0.0


def temp_reason_subtype(question: str) -> str:
    # L1: pure date arithmetic ("What is the time X after/before DATE").
    # L2: time-event ("Which <relation> did X ... in DATE?").
    return "L1_arithmetic" if question.lower().startswith("what is the time") else "L2_time_event"


def analyze_benchmark(
    benchmark: str,
    data_path: Path,
    semantic_model: str,
    subtype_fn: Optional[Callable[[str], str]] = None,
    split: str = "test",
) -> List[Dict]:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # num_negatives=0 -> data used as-is
    eval_data = select_split_items(data, split)

    temporal_paths = set_output_files(TEMPORAL_MODEL_NAME, TEMPORAL_MODEL_PATH, "", benchmark)
    semantic_paths = set_output_files("", TEMPORAL_MODEL_PATH, semantic_model, benchmark)
    temporal_sims = load_similarities(temporal_paths["temporal_similarities_path"])
    semantic_sims = load_similarities(semantic_paths["semantic_similarities_path"])

    rows: List[Dict] = []
    for item in eval_data:
        question = item["question"]
        paragraphs = item["paragraphs"]
        golds = set(item["answer"])
        if question not in temporal_sims or question not in semantic_sims:
            continue
        qyears = query_years(question)
        match = [bool(query_years(p) & qyears) for p in paragraphs]
        gold_has_qyear = any(match[i] for i in golds)

        rows.append(
            {
                "subset": subtype_fn(question) if subtype_fn else "all",
                "mrr_semantic": mrr_from_scores([semantic_sims[question][p] for p in paragraphs], golds),
                "mrr_temporal": mrr_from_scores([temporal_sims[question][p] for p in paragraphs], golds),
                "mrr_timestamp": timestamp_matcher_mrr(match, golds),
                "gold_has_qyear": gold_has_qyear,
            }
        )
    return rows


def summarize(label: str, rows: List[Dict], subsets: List[str]) -> List[Dict]:
    out = []
    for subset in subsets:
        sub = rows if subset == "all" else [r for r in rows if r["subset"] == subset]
        if not sub:
            continue
        out.append(
            {
                "benchmark": label if subset == "all" else f"  – {subset}",
                "n": len(sub),
                "pct_gold_ts": 100.0 * np.mean([r["gold_has_qyear"] for r in sub]),
                "mrr_timestamp": float(np.mean([r["mrr_timestamp"] for r in sub])),
                "mrr_temporal": float(np.mean([r["mrr_temporal"] for r in sub])),
                "mrr_semantic": float(np.mean([r["mrr_semantic"] for r in sub])),
            }
        )
    return out


def print_markdown_table(table: List[Dict]) -> None:
    header = (
        "| Benchmark (subset) | n | % golds carrying query timestamp | "
        "Timestamp matcher | Temporal-only | Semantic |"
    )
    print(header)
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in table:
        print(
            f"| {r['benchmark']} | {r['n']} | {r['pct_gold_ts']:.1f}% | "
            f"{r['mrr_timestamp']:.3f} | {r['mrr_temporal']:.3f} | {r['mrr_semantic']:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", default="intfloat/e5-base-v2",
                        help="Semantic baseline id (needs a cached similarities pickle).")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    specs = [
        ("TempReason", Path("data/evaluation/temp_reason/processed_data.json"),
         temp_reason_subtype, ["all", "L1_arithmetic", "L2_time_event"]),
        ("TimeQA", Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json"),
         None, ["all"]),
        ("TS-Retriever", Path("data/evaluation/ts_retriever/processed_ts_retriever.json"),
         None, ["all"]),
    ]

    # The benchmark id must match the pickle folder name, not the display label.
    benchmark_ids = {"TempReason": "temp_reason", "TimeQA": "time_sensitive_qa",
                     "TS-Retriever": "ts_retriever"}

    table: List[Dict] = []
    for label, path, subtype_fn, subsets in specs:
        rows = analyze_benchmark(benchmark_ids[label], path, args.semantic, subtype_fn, args.split)
        table.extend(summarize(label, rows, subsets))

    print(f"\nSemantic baseline: {args.semantic} | split: {args.split}\n")
    print_markdown_table(table)
    print(
        "\nSanity check: the 'Temporal-only' and 'Semantic' MRR columns reproduce "
        "output/metrics/paper_evaluation_report.md\n(e.g. TempReason temporal-only "
        "0.908, E5 semantic 0.819; TimeQA temporal-only 0.459, E5 0.333)."
    )


if __name__ == "__main__":
    main()
