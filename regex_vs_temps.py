"""Rebuttal analysis: is the TEMPS gain just explicit-year regex?

Builds progressively stronger *date-matching oracles* and compares them to the
semantic baseline and to TEMPS (hybrid, alpha=0.6), then mines the cases that a
regex cannot solve:

  (1) questions with NO explicit year (ordinal / relative temporal reasoning),
  (2) questions WITH a year whose gold passage contains none of the query's
      years/interval (regex-on-passage fails: granularity, ranges, "since"/
      "until", paraphrase), yet TEMPS ranks the gold #1,
  (3) cases where a regex would be actively misled: a distractor carries the
      query year while the gold does not.

Everything reuses the exact pipeline arithmetic (per-query min-max normalization
+ alpha*temporal + (1-alpha)*semantic + descending-argsort ranking) and the
precomputed similarity caches. No model inference.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from temporal_embeddings.config.set_output_files import set_output_files
from temporal_embeddings.evaluation.utils.data.splits import select_split_items
from temporal_embeddings.utils.math.normalize import normalize_list


BENCHMARK_PATHS = {
    "time_sensitive_qa": Path("data/evaluation/time_sensitive_qa/processed_human_annotated_test.json"),
    "temp_reason_l1": Path("data/evaluation/temp_reason_l1/processed_data.json"),
    "temp_reason_l2": Path("data/evaluation/temp_reason_l2/processed_data.json"),
    "temp_reason": Path("data/evaluation/temp_reason/processed_data_emnlp.json"),
    "ts_retriever": Path("data/evaluation/ts_retriever/processed_ts_retriever.json"),
}
BENCHMARK_LABELS = {
    "time_sensitive_qa": "TimeQA",
    "temp_reason_l1": "TempReason L1",
    "temp_reason_l2": "TempReason L2",
    "temp_reason": "TempReason",
    "ts_retriever": "TSRetriever",
}
TEMP_REASON_CHILD_BENCHMARKS = {"temp_reason_l1", "temp_reason_l2"}
TEMPORAL_MODEL_NAME = "all-minilm-l6-v2"
TEMPORAL_MODEL_PATH = Path(
    "output/trained_models/model_sentence-transformers_all-MiniLM-L6-v2_2026-05-24_00-00-51.pth"
)
BASELINE_LABELS = {
    "intfloat/e5-base-v2": "E5-base-v2",
    "salesforce": "SFR-Embedding-Mistral",
    "all-mpnet-base-v2": "MPNet-base-v2",
    "BAAI/bge-large-en-v1.5": "BGE-large-v1.5",
}
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

Similarities = Dict[str, Dict[str, float]]
RunStats = Dict[str, Any]
RetrieverSpec = Dict[str, str]


def pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def load_similarities(path: Path) -> Similarities:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected nested-dict similarities at {path}, got {type(obj)}")
    return obj


def similarities_cover_benchmark(
    similarities: Similarities,
    eval_data: List[Dict[str, Any]],
) -> bool:
    for item in eval_data:
        question = item["question"]
        if question not in similarities:
            return False
        paragraph_sims = similarities[question]
        for paragraph in item["paragraphs"]:
            if paragraph not in paragraph_sims:
                return False
    return True


def load_temporal_similarities(benchmark: str, eval_data: List[Dict[str, Any]]) -> Similarities:
    paths = set_output_files(TEMPORAL_MODEL_NAME, TEMPORAL_MODEL_PATH, "", benchmark)
    path = paths["temporal_similarities_path"]
    if path.exists():
        return load_similarities(path)

    if benchmark in TEMP_REASON_CHILD_BENCHMARKS:
        parent_paths = set_output_files(TEMPORAL_MODEL_NAME, TEMPORAL_MODEL_PATH, "", "temp_reason")
        parent_path = parent_paths["temporal_similarities_path"]
        if parent_path.exists():
            parent_similarities = load_similarities(parent_path)
            if similarities_cover_benchmark(parent_similarities, eval_data):
                print(f"Reusing parent TempReason temporal similarities for {benchmark}: {parent_path}")
                return parent_similarities

    raise FileNotFoundError(f"Temporal similarities not found for {benchmark}: {path}")


def load_semantic_similarities(
    benchmark: str,
    eval_data: List[Dict[str, Any]],
    baseline: str,
) -> Similarities:
    paths = set_output_files("", TEMPORAL_MODEL_PATH, baseline, benchmark)
    path = paths["semantic_similarities_path"]
    if path.exists():
        return load_similarities(path)

    if benchmark in TEMP_REASON_CHILD_BENCHMARKS:
        parent_paths = set_output_files("", TEMPORAL_MODEL_PATH, baseline, "temp_reason")
        parent_path = parent_paths["semantic_similarities_path"]
        if parent_path.exists():
            parent_similarities = load_similarities(parent_path)
            if similarities_cover_benchmark(parent_similarities, eval_data):
                print(f"Reusing parent TempReason semantic similarities for {benchmark}: {parent_path}")
                return parent_similarities

    raise FileNotFoundError(f"Semantic similarities not found for {benchmark}: {path}")


def normalize_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    keys = list(bucket.keys())
    return dict(zip(keys, normalize_list([bucket[k] for k in keys])))


def year_ints(text: str) -> Set[int]:
    return {int(y) for y in YEAR_RE.findall(text)}


def query_interval(question: str) -> Set[int]:
    """Generous reading of the query's temporal constraint: the full closed
    interval spanned by the query's year tokens (a single year -> {year})."""
    ys = sorted(year_ints(question))
    if not ys:
        return set()
    if len(ys) == 1:
        return {ys[0]}
    return set(range(ys[0], ys[-1] + 1))


def rank_of(scores: List[float], gold_indices: List[int]) -> int:
    ranked = np.argsort(scores)[::-1]
    ranks = [
        int(np.where(ranked == gold_idx)[0][0]) + 1
        for gold_idx in gold_indices
        if gold_idx in ranked
    ]
    if not ranks:
        raise ValueError(f"No valid gold index found in ranked list: {gold_indices}")
    return min(ranks)


def argsort_desc(scores: List[float]) -> List[int]:
    return list(np.argsort(scores)[::-1])


def snippet(text: str, width: int = 230) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width].rstrip() + " …"


def build(
    eval_data: List[Dict[str, Any]],
    temporal_sims: Similarities,
    semantic_sims: Similarities,
    alpha: float,
) -> List[Dict[str, Any]]:
    norm_t: Dict[str, Dict[str, float]] = {}
    norm_s: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []

    for item in eval_data:
        q = item["question"]
        paras = item["paragraphs"]
        gold_indices = [
            int(idx)
            for idx in item["answer"]
            if 0 <= int(idx) < len(paras)
        ]
        if not gold_indices:
            continue
        if q not in temporal_sims or q not in semantic_sims:
            continue
        if any(p not in temporal_sims[q] or p not in semantic_sims[q] for p in paras):
            continue
        if q not in norm_t:
            norm_t[q] = normalize_bucket(temporal_sims[q])
            norm_s[q] = normalize_bucket(semantic_sims[q])

        sem = [semantic_sims[q][p] for p in paras]
        sem_n = [norm_s[q][p] for p in paras]
        tmp = [temporal_sims[q][p] for p in paras]
        hyb = [alpha * norm_t[q][p] + (1.0 - alpha) * norm_s[q][p] for p in paras]

        q_int = query_interval(q)
        # Oracle A: strict token match — passage shares an exact year *token*
        # with the query, ties broken by the semantic score.
        q_tok = year_ints(q)
        regex_tok = [
            (1.0 if (year_ints(p) & q_tok) else 0.0) * 10.0 + sem_n[i]
            for i, p in enumerate(paras)
        ]
        # Oracle B (most generous): passage mentions any year inside the query's
        # full interval; ties broken by the semantic score. This is the strongest
        # "year-filter then semantic rerank" heuristic.
        regex_int = [
            (1.0 if (year_ints(p) & q_int) else 0.0) * 10.0 + sem_n[i]
            for i, p in enumerate(paras)
        ]

        gold_years: Set[int] = set()
        for gold_idx in gold_indices:
            gold_years.update(year_ints(paras[gold_idx]))
        rows.append(
            {
                "question": q,
                "paragraphs": paras,
                "gold_idx": gold_indices[0],
                "gold_indices": gold_indices,
                "gold_text": paras[gold_indices[0]],
                "q_years": sorted(q_tok),
                "q_interval": q_int,
                "has_q_year": bool(q_tok),
                "gold_year_ints": sorted(gold_years),
                "gold_has_q_token": bool(gold_years & q_tok),
                "gold_has_q_interval": bool(gold_years & q_int),
                "sem": sem, "tmp": tmp, "hyb": hyb,
                "regex_tok": regex_tok, "regex_int": regex_int,
                "n": len(paras),
                "rank_sem": rank_of(sem, gold_indices),
                "rank_tmp": rank_of(tmp, gold_indices),
                "rank_hyb": rank_of(hyb, gold_indices),
                "rank_regex_tok": rank_of(regex_tok, gold_indices),
                "rank_regex_int": rank_of(regex_int, gold_indices),
            }
        )
    return rows


def acc(rows: List[Dict[str, Any]], key: str, k: int) -> float:
    return sum(r[key] <= k for r in rows) / len(rows)


def mrr(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(1.0 / r[key] for r in rows) / len(rows)


def top1(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(r[key] == 1 for r in rows) / len(rows)


def collect_stats(rows: List[Dict[str, Any]]) -> RunStats:
    with_year = [r for r in rows if r["has_q_year"]]
    no_year = [r for r in rows if not r["has_q_year"]]
    gold_missing_token = [r for r in with_year if not r["gold_has_q_token"]]
    gold_missing_interval = [r for r in with_year if not r["gold_has_q_interval"]]
    temps_wins = [r for r in rows if r["rank_hyb"] == 1 and r["rank_sem"] > 1]
    wins_regex_int_fails = [r for r in temps_wins if r["rank_regex_int"] > 1]
    wins_gold_no_interval = [
        r for r in temps_wins
        if r["has_q_year"] and not r["gold_has_q_interval"]
    ]
    return {
        "with_year": with_year,
        "no_year": no_year,
        "gold_missing_token": gold_missing_token,
        "gold_missing_interval": gold_missing_interval,
        "temps_wins": temps_wins,
        "wins_regex_int_fails": wins_regex_int_fails,
        "wins_gold_no_interval": wins_gold_no_interval,
    }


def retriever_specs(baseline_label: str, alpha: float) -> List[RetrieverSpec]:
    return [
        {"name": f"semantic baseline ({baseline_label})", "report_name": f"Semantic baseline ({baseline_label})", "key": "rank_sem"},
        {"name": "regex: exact year-token + rerank", "report_name": "Regex oracle: exact year-token, then semantic rerank", "key": "rank_regex_tok"},
        {"name": "regex: query-interval + rerank", "report_name": "Regex oracle: query **interval**, then semantic rerank", "key": "rank_regex_int"},
        {"name": "temporal-only (TEMPS temporal head)", "report_name": "Temporal-only (our temporal head)", "key": "rank_tmp"},
        {"name": f"TEMPS hybrid (a={alpha})", "report_name": f"**TEMPS** hybrid (α={alpha})", "key": "rank_hyb"},
    ]


def format_block(r: Dict[str, Any], baseline_label: str, alpha: float, k: int) -> List[str]:
    L: List[str] = []
    L.append(f"**Q:** {r['question']}")
    qc = ", ".join(str(y) for y in r["q_years"]) if r["q_years"] else "— (no explicit year)"
    L.append(f"_Query years:_ {qc} · _Gold passage years:_ "
             f"{', '.join(str(y) for y in r['gold_year_ints']) or '—'}")
    L.append(f"_Gold rank — baseline {baseline_label}: **#{r['rank_sem']}**/{r['n']} · "
             f"year-filter+semantic oracle: **#{r['rank_regex_int']}** · "
             f"temporal-only: #{r['rank_tmp']} · TEMPS: **#{r['rank_hyb']}**._")
    L.append("")
    L.append(f"Baseline ({baseline_label}) top-{k}:")
    for rank, idx in enumerate(argsort_desc(r["sem"])[:k], 1):
        p = r["paragraphs"][idx]
        yrs = sorted(year_ints(p))
        tag = " ✅GOLD" if idx in r["gold_indices"] else ""
        yr = f" _(yrs: {', '.join(str(y) for y in yrs)})_" if yrs else " _(no year)_"
        L.append(f"  {rank}.{tag} {snippet(p)}{yr}")
    L.append("")
    L.append(f"TEMPS (α={alpha}) top-{k}:")
    for rank, idx in enumerate(argsort_desc(r["hyb"])[:k], 1):
        p = r["paragraphs"][idx]
        yrs = sorted(year_ints(p))
        tag = " ✅GOLD" if idx in r["gold_indices"] else ""
        yr = f" _(yrs: {', '.join(str(y) for y in yrs)})_" if yrs else " _(no year)_"
        L.append(f"  {rank}.{tag} {snippet(p)}{yr}")
    L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="time_sensitive_qa", choices=sorted(BENCHMARK_PATHS))
    ap.add_argument("--baseline", default="intfloat/e5-base-v2")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="output/qualitative")
    ap.add_argument("--n-examples", type=int, default=8)
    args = ap.parse_args()

    label = BASELINE_LABELS.get(args.baseline, args.baseline)
    benchmark_label = BENCHMARK_LABELS.get(args.benchmark, args.benchmark)
    k = args.top_k

    data = json.load(open(BENCHMARK_PATHS[args.benchmark], encoding="utf-8"))
    eval_data = select_split_items(data, args.split)

    temporal_sims = load_temporal_similarities(args.benchmark, eval_data)
    semantic_sims = load_semantic_similarities(args.benchmark, eval_data, args.baseline)

    rows = build(eval_data, temporal_sims, semantic_sims, args.alpha)
    n = len(rows)
    if not rows:
        raise ValueError(
            f"No evaluable {args.split} queries for {args.benchmark}. "
            "Check that the similarity caches cover the selected benchmark."
        )
    stats = collect_stats(rows)
    with_year = stats["with_year"]
    no_year = stats["no_year"]
    gold_missing_token = stats["gold_missing_token"]
    gold_missing_interval = stats["gold_missing_interval"]
    temps_wins = stats["temps_wins"]
    wins_regex_int_fails = stats["wins_regex_int_fails"]
    wins_gold_no_interval = stats["wins_gold_no_interval"]
    specs = retriever_specs(label, args.alpha)

    print(f"{benchmark_label} / {label}: {n} evaluable {args.split} queries\n")

    # --- Retriever comparison table ---
    print(f"{'retriever':<34} {'MRR':>8} {'Gold@1':>8} {'Recall@'+str(k):>9}")
    for spec in specs:
        key = spec["key"]
        print(f"{spec['name']:<34} {mrr(rows, key):>8.4f} {top1(rows, key):>8.3f} {acc(rows, key, k):>9.3f}")
    print()

    # --- Why regex is insufficient: structure of the data ---
    print(f"queries with an explicit year: {len(with_year)}/{n} "
          f"({pct(len(with_year), n):.1%});  without: {len(no_year)}")
    print(f"  of those, GOLD passage lacks the query's exact year token: "
          f"{len(gold_missing_token)} ({pct(len(gold_missing_token), len(with_year)):.1%})")
    print(f"  of those, GOLD passage lacks ANY year in the query interval: "
          f"{len(gold_missing_interval)} ({pct(len(gold_missing_interval), len(with_year)):.1%})")
    print()

    # TEMPS wins (baseline ranks gold below 1, TEMPS = 1) that regex cannot get.
    print(f"TEMPS pulls gold to #1 from a worse baseline rank on {len(temps_wins)} queries.")
    print(f"  the year-filter+semantic oracle still fails (gold not #1) on "
          f"{len(wins_regex_int_fails)} of them "
          f"({pct(len(wins_regex_int_fails), len(temps_wins)):.1%}).")
    print(f"  {len(wins_gold_no_interval)} of the wins have a dated query whose GOLD "
          f"passage contains no query-interval year (pure containment/paraphrase).")
    print()

    # ---- Build example report ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = label.lower().replace(" ", "_")
    L: List[str] = []
    L.append(f"# {benchmark_label} — is the TEMPS gain just explicit-year regex? ({label}, α={args.alpha})")
    L.append("")
    L.append(f"_Split:_ `{args.split}` · _Evaluable queries:_ **{n}**")
    L.append("")
    L.append("| Retriever | MRR | Gold@1 | Recall@%d |" % k)
    L.append("| --- | --- | --- | --- |")
    for spec in specs:
        key = spec["key"]
        L.append(f"| {spec['report_name']} | {mrr(rows, key):.4f} | {top1(rows, key):.3f} | {acc(rows, key, k):.3f} |")
    L.append("")
    L.append("## Reviewer-facing explicit-cue statistics")
    L.append("")
    if with_year:
        L.append(f"- {len(with_year)}/{n} queries carry an explicit year, **but the gold "
                 f"passage lacks any year in the query's interval in "
                 f"{len(gold_missing_interval)} of them ({pct(len(gold_missing_interval), len(with_year)):.0%})** — "
                 f"so passage-level date matching cannot retrieve the answer.")
        L.append(f"- The gold passage lacks the exact query-year token in "
                 f"{len(gold_missing_token)}/{len(with_year)} dated queries "
                 f"({pct(len(gold_missing_token), len(with_year)):.1%}).")
    else:
        L.append(f"- 0/{n} queries carry an explicit year, so the year-filter oracle has no query-side date signal.")
    L.append(f"- Of the {len(temps_wins)} queries where TEMPS lifts the gold to #1 from a "
             f"lower baseline rank, the strongest regex oracle (interval-filter + semantic "
             f"rerank) still fails on **{len(wins_regex_int_fails)}** "
             f"({pct(len(wins_regex_int_fails), len(temps_wins)):.1%}).")
    L.append(f"- {len(wins_gold_no_interval)} TEMPS top-1 corrections have a dated query whose "
             f"gold passage contains no query-interval year.")
    L.append("")

    # Category 1: no explicit year at all.
    L.append("## Category 1 — questions with no explicit year (ordinal / relative reasoning)")
    L.append("")
    L.append("A regex has nothing to match on; the temporal constraint is implicit "
             "(*second*, *third*, *prior to*, *upon expanding*).")
    L.append("")
    for r in no_year[: args.n_examples]:
        L += format_block(r, label, args.alpha, k)
        L.append("---"); L.append("")

    # Category 2: dated query, gold has no interval-year, TEMPS ranks it #1.
    cat2 = [r for r in wins_gold_no_interval if r["rank_regex_int"] > 1]
    cat2.sort(key=lambda r: (r["rank_sem"] - r["rank_hyb"], -r["n"]), reverse=True)
    L.append("## Category 2 — dated query, but the gold passage never states a query year")
    L.append("")
    L.append("The query names a year/range; the gold answer expresses the time implicitly "
             "(different granularity, an open interval such as *since*/*until*, or paraphrase), "
             "so it contains **no** query-interval year. A regex/date-filter ranks it below "
             "distractors that do mention the year; TEMPS still ranks it #1.")
    L.append("")
    for r in cat2[: args.n_examples]:
        L += format_block(r, label, args.alpha, k)
        L.append("---"); L.append("")

    report = out / f"regex_vs_temps_{args.benchmark}_{slug}_alpha{args.alpha}.md"
    report.write_text("\n".join(L), encoding="utf-8")
    print(f"Report written to: {report}")
    print(f"Category 1 (no-year) examples: {len(no_year)} total, {min(len(no_year), args.n_examples)} shown;  "
          f"Category 2 (gold lacks query year, regex-oracle fails, TEMPS #1): {len(cat2)}")


if __name__ == "__main__":
    main()
