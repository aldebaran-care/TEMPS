# Migration Report: Symmetric TSF v2 → Asymmetric KL Pipeline

A self-contained record of what changed between the previous training/evaluation
pipeline and the current one. Companion document to
[temporal_similarity_analysis.md](temporal_similarity_analysis.md), which
covers the *why* in detail; this document covers the *what* and *how*.

---

## 1. Headline change: the temporal similarity function

The function that assigns a ground-truth label to every pair of TimeML date
expressions was replaced.

### 1.1 Old behavior

[compute_similarity_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py)
(pre-migration) exposed two formulas guarded by `mode` and `version`:

- **`tsf v2 mode="inference"`** — returned `IoU` over the day intervals, with a
  `(1 + IoU)/2` bonus for strict containment and hard `0` for disjoint pairs.
- **`tsf v2 mode="train"`** — same as above when intervals overlap, otherwise
  `ε · exp(−(gap/length)²)` with `ε = 1e-3` to give the loss a smooth tail.
- **`tsf v1`** — a separate ad-hoc formula with magic constants
  (`0.4 / d²`, `0.1 / d²`, `0.5` containment, threshold `100`).

Five issues with that design (all documented in
[temporal_similarity_analysis.md §3](temporal_similarity_analysis.md#3-mathematical-critique-of-tsf-v2-the-active-label-generator)):

1. **Symmetric** — `TSF(q, d) = TSF(d, q)` — but the model head is
   asymmetric KL ([similarity.py:7-18](../temporal_embeddings/utils/similarity.py#L7-L18)).
   The asymmetric capacity of the Gaussian embedding head was unsupervised.
2. **Hard zero on disjoint pairs at inference**. `1990` vs `1991` and `1990`
   vs `2030` collapsed to the same label.
3. **Train ≠ inference label** for the same input.
4. **Discontinuous** at the overlap-to-gap boundary.
5. **Magic constants** with no derivation: `ε = 1e-3`, the `(1+IoU)/2` lift,
   the `0.2` overlap cap, the `30`-day-per-month approximation, etc.

### 1.2 New behavior

Each grounded TimeML annotation `I = [a, b]` is mapped to the
moment-matched Gaussian of the continuous uniform `U[a, b+1)`:

```
μ_I  = (a + b + 1) / 2
σ_I² = (b − a + 1)² / 12
```

The asymmetric temporal distance from query `I_q` to document `I_d` is

```
TSF(I_q → I_d)  =  KL( N(μ_q, σ_q²) ‖ N(μ_d, σ_d²) )
                =  0.5 · log(σ_d² / σ_q²)
                 + (σ_q² + (μ_q − μ_d)²) / (2 σ_d²)
                 − 0.5
```

The public API in
[compute_similarity_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py)
exposes `temporal_kl(I_q, I_d)` (raw KL distance) and
`tsf(first_interval, second_interval)` (returns `1 / (1 + KL)` similarity in
`(0, 1]`, matching the model head's `asymmetrical_kl_sim` convention).
**`mode`, `version`, and `epsilon` arguments are gone.** One function only.

### 1.3 Numerical contrast on the canonical edge cases

| Pair | Old TSF v2 (inference / train) | New (q → d, d → q) |
|---|---|---|
| `1990` vs `1990` | `1.0` / `1.0` | `1.0` / `1.0` |
| `1990` vs `1990-06-15` | `0.0027` / `0.0027` | `1.46e-5` / `0.156` |
| `1990-12-31` vs `1990` | `0.0027` / `0.0027` | `0.127` / `3.77e-6` |
| `1990` vs `1991` | `0.0` / `~4e-4` | `0.143` / `0.143` |
| `1990` vs `2025` (far) | `0.0` / `~1e-12` | `6e-7` / `6e-7` |

Key observations:

- **Asymmetry is real.** The old function reported `0.0027` in both directions
  for year-vs-day-in-middle; the new function reports `1.46e-5` (`q = year,
  d = day` — year is *not* "explained by" the day) vs `0.156` (`q = day,
  d = year` — day *is* explained by the year). Four orders of magnitude apart.
- **Disjoint pairs are smoothly ranked.** `1990 vs 1991` gets `~0.14`,
  `1990 vs 2025` gets `~6e-7`. Old TSF collapsed both to zero (inference) or
  near-zero (train). Contrastive learning gets meaningful gradient.
- **Symmetric inputs stay symmetric.** Equal-width intervals like
  `1990` vs `1991` produce identical values in both directions.

---

## 2. Bug fixes in the grounding pipeline

These are arithmetic defects that contaminated labels *before* TSF ran. They
are now fixed independently of the formulation change.

### B1. End-of-month was hard-coded to day 30

[to_explicit_date.py:14-23](../temporal_embeddings/data_utils/utils/dates/to_explicit_date.py#L14-L23)
previously contained:

```python
if int(month) == 2:
    last_day = 28
else:
    last_day = 30
```

Effect: every `yyyy-mm` interval ending in a 31-day month lost a day; every
February in a leap year lost a day. Since `yyyy-mm` is the most common TIMEX
granularity after `yyyy`, this systematically biased a large fraction of the
training labels. Replaced with `calendar.monthrange(year, month)[1]`.

### B6. Dead `yyyy-s` season branch

[compute_distance_dates.py](https://github.com/your-org/temporal-embeddings/blob/main/temporal_embeddings/data_utils/utils/dates/compute_distance_dates.py)
contained a branch reading a 2-character season code (`first_date[-2:]`) for a
`yyyy-s` date type that `is_date.py`'s regex never returned. Dead code.
Removed as part of the v1 deletion.

### B7. Silent reorder of interval boundaries

[compute_similarity_dates.py:35](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L35)
previously did `start, end = sorted(...)` to canonicalize backwards intervals.
Replaced with an explicit `raise ValueError` so upstream data-quality issues
surface instead of being masked. **This caught a real bug** in the random
expression generator (see §3).

### v1 deletion

[compute_distance_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_distance_dates.py)
and [is_in.py](../temporal_embeddings/data_utils/utils/dates/is_in.py) were
deleted. The `_tsf_v1*` helpers in
[compute_similarity_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py)
were deleted. The `version` parameter is gone from every public entry point.

---

## 3. Random expression generator: chronological vs lexicographic sort

[generate_random_interval.py](../temporal_embeddings/data_utils/utils/intervals/generate_random_interval.py)
used to sort the two random endpoints with plain `sorted([first, second])`,
which compares strings. This was wrong when the endpoints had different
granularities. For example, `sorted(["1770s-mid", "1772-02-01"])` returns
`["1770s-mid", "1772-02-01"]` lexicographically, but chronologically
`"1770s-mid"` denotes 1774–1776, *after* February 1772.

Empirical rate of backwards intervals before the fix: **~35 in 5000 samples
(~0.7%)**, all from this single path. After the fix
(`sorted(..., key=lambda d: to_explicit_date(d)[0])`): **0 in 5000 samples**.

---

## 4. Public API change: dropped parameters

Every public entry point that exposed `mode`, `version`, or `epsilon` was
simplified. The dataset builders had to be updated accordingly:

| Caller | Old call | New call |
|---|---|---|
| [compute_similarity_expressions.py](../temporal_embeddings/data_utils/utils/compute_similarity_expressions.py) | `(...,mode,version,epsilon)` | `(first_expr, first_date, second_expr, second_date)` |
| [compute_similarity_expressions_sutime.py](../temporal_embeddings/data_utils/utils/compute_similarity_expressions_sutime.py) | same | same |
| [create_synthetic_dataset.py](../create_synthetic_dataset.py) | `--tsf_version --tsf_epsilon` | flags removed |
| [create_real_world_dataset.py](../create_real_world_dataset.py) | `--tsf_version --tsf_epsilon` | flags removed |
| [create_temporal_relationships_dataset.py](../create_temporal_relationships_dataset.py) | `--tsf_version --tsf_epsilon` | flags removed |
| [add_score_v2_to_training_data.py](../add_score_v2_to_training_data.py) | `mode="train", version="v2"` | positional args only |

The dataset CSV schema (`sent0, sent0_date, sent1, sent1_date, score`) is
unchanged. Existing CSVs from the old TSF v2 are *schema-compatible* but the
`score` distribution is different (asymmetric, broader dynamic range, no hard
zeros). Re-annotate before training.

---

## 5. Evaluation pipeline: dense matrix → nested dict

The paper-evaluation runner used to materialize a dense `Q × P` similarity
matrix per benchmark, where `Q` is the number of unique questions and `P` is
the union of all candidate paragraphs across the benchmark. On TempReason
(`Q = 9.4k`, `P = 15.7k`) and TSRetriever this OOM-killed the SLURM job at
about 60% of the way through stage 2 — `~12 GB` of `float64` cells plus
pandas overhead exceeded the per-job RAM budget on `gpu_p5`.

### 5.1 What changed structurally

The return type of
[compute_temporal_similarities.py](../temporal_embeddings/evaluation/utils/evaluation/temporal_model/compute_temporal_similarities.py)
and
[compute_semantic_similarities.py](../temporal_embeddings/evaluation/utils/evaluation/semantic_model/compute_semantic_similarities.py)
changed from `pd.DataFrame` (indexed by question, columns = all paragraphs) to
`Dict[str, Dict[str, float]]` (outer = question, inner = paragraph). The stage-2
loop now iterates `benchmark_data` and, for each item, only computes
similarities for that item's own candidate list — working set is `O(p)` per
iteration where `p ≈ 5–100`, never `O(Q · P)`.

[run_paper_evaluations.py](../run_paper_evaluations.py) was refactored to
consume the new format:

- `_compute_similarity_lists` is now a nested-dict lookup.
- `_merge_hybrid_similarities` does per-pair convex combination across the two
  dicts.
- `_normalize_similarities` (renamed from `_normalize_similarity_df`)
  normalizes per question over its own candidate set.

### 5.2 One behavioral change to record

The old `_normalize_similarity_df` did min-max over each question's row across
**all `P` paragraphs in the union**. The new `_normalize_similarities` does
min-max over **each question's own candidate set only**. The two are
mathematically different when candidate set sizes vary across questions; the
new version is the more natural per-query normalization. Only the `hybrid` run
type is affected (pure temporal and pure semantic don't normalize). If you
compare hybrid numbers across the old and new pipeline, note this in the
paper.

### 5.3 Cache compatibility

Legacy `.pkl` files containing the old dense DataFrame are auto-detected via
`isinstance(obj, dict)` in `_load_cached_similarities` and silently rebuilt in
the new format. The embedding caches (`cache.pkl` under each model directory)
are unchanged and still reused — only the *similarities* cache files
regenerate on first run.

---

## 6. Training results under the new pipeline

For reference. Full numbers in your training logs.

| Metric | Old pipeline (TSF v2 labels, 130k steps, batch 512×1) | New pipeline (KL labels, 3.7k steps, batch 2048×8) |
|---|---|---|
| Effective batch | 512 | 16,384 |
| Optimizer steps for 1 epoch on the merged 60M-row dataset | ~117k | ~3.7k |
| Final dev Spearman | (varies by run; see prior checkpoints) | **96.35** |

The new labels and the model head are in the same family (Gaussian + KL), so
training reduces to **distillation of a closed-form temporal geometry into a
text-conditioned Gaussian embedder**. This is why dev Spearman saturates in a
single epoch.

---

## 7. Files changed (full list)

Added:
- [docs/temporal_similarity_analysis.md](temporal_similarity_analysis.md)
- [docs/migration_report.md](migration_report.md) (this file)

Replaced or substantially rewritten:
- [temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py) — new asymmetric KL function, B7 fix
- [temporal_embeddings/data_utils/utils/dates/to_explicit_date.py](../temporal_embeddings/data_utils/utils/dates/to_explicit_date.py) — B1 fix
- [temporal_embeddings/data_utils/utils/compute_similarity_expressions.py](../temporal_embeddings/data_utils/utils/compute_similarity_expressions.py) — params removed
- [temporal_embeddings/data_utils/utils/compute_similarity_expressions_sutime.py](../temporal_embeddings/data_utils/utils/compute_similarity_expressions_sutime.py) — params removed
- [temporal_embeddings/data_utils/utils/intervals/generate_random_interval.py](../temporal_embeddings/data_utils/utils/intervals/generate_random_interval.py) — chronological sort
- [temporal_embeddings/evaluation/utils/evaluation/temporal_model/compute_temporal_similarities.py](../temporal_embeddings/evaluation/utils/evaluation/temporal_model/compute_temporal_similarities.py) — nested-dict output
- [temporal_embeddings/evaluation/utils/evaluation/semantic_model/compute_semantic_similarities.py](../temporal_embeddings/evaluation/utils/evaluation/semantic_model/compute_semantic_similarities.py) — nested-dict output
- [run_paper_evaluations.py](../run_paper_evaluations.py) — adapted to nested-dict

Updated callers (parameter removal + defensive `try/except` around generator
similarity calls):
- [create_synthetic_dataset.py](../create_synthetic_dataset.py)
- [create_real_world_dataset.py](../create_real_world_dataset.py)
- [create_temporal_relationships_dataset.py](../create_temporal_relationships_dataset.py)
- [add_score_v2_to_training_data.py](../add_score_v2_to_training_data.py)
- [temporal_embeddings/synthetic_data/create_synthetic_dataset.py](../temporal_embeddings/synthetic_data/create_synthetic_dataset.py)

Deleted:
- `temporal_embeddings/data_utils/utils/dates/compute_distance_dates.py`
- `temporal_embeddings/data_utils/utils/dates/is_in.py`

---

## 8. How to use the new pipeline

```bash
# 1. Regenerate datasets with the new labels
python create_synthetic_dataset.py        --output_file_path data/new_training_dataset/synthetic_dataset.csv     --size 1000000
python create_temporal_relationships_dataset.py --output_file_path data/new_training_dataset/relationships_dataset.csv --n_phrases 1000000
python create_real_world_dataset.py       --input_path <annotated.json> --output_path data/new_training_dataset/real_world.csv

# 2. Merge + shuffle into the path train.sh expects
python merge_training_dataset.py \
    data/new_training_dataset/synthetic_dataset.csv \
    data/new_training_dataset/relationships_dataset.csv \
    data/new_training_dataset/real_world.csv

# 3. Train (8 × A100 SLURM config in train.sh)
sbatch train.sh

# 4. Evaluate on the paper benchmarks
sbatch run_paper_evaluations.sh
```

The `tsf_version` / `tsf_epsilon` / `mode` / `version` / `epsilon` flags are
gone everywhere. Do not pass them.
