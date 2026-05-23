# Temporal Similarity Function: Critique and Replacement

A mathematical analysis of the temporal similarity function (TSF) used to
annotate training data for the temporal embedding model in this repository,
together with a proposed replacement: **asymmetric KL divergence between
moment-matched Gaussians on interval supports**.

The goal of this document is to specify a single, principled, constant-free,
asymmetric distance over TimeML date expressions that can serve as the
ground-truth label for contrastive training of a Gaussian temporal
embedding model.

---

## 1. The function being replaced

The function under analysis is `tsf` in
[compute_similarity_dates.py:119-136](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L119-L136),
reached through
[compute_similarity_expressions.py:13-77](../temporal_embeddings/data_utils/utils/compute_similarity_expressions.py#L13-L77).
The expression dispatcher converts each TimeML expression (an explicit date,
an offset, a reference, or an interval) into a pair of grounded inclusive
day intervals `[a, b]`. TSF v2 then computes a similarity in `[0, 1]`.

Two modes coexist:

- **Inference mode**
  ([compute_similarity_dates.py:84-103](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L84-L103))
  returns Intersection-over-Union, with a `(1 + IoU)/2` bonus for strict
  containment and `0` for disjoint intervals.
- **Train mode**
  ([compute_similarity_dates.py:106-116](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L106-L116))
  reuses the inference score when intervals overlap and otherwise adds a
  smoothing tail `ε · exp(−(gap/length)²)` with `ε = 1e-3`.

The model head consuming these labels is **asymmetric** by design:
`asymmetrical_kl_sim(μ₁, σ₁, μ₂, σ₂)` in
[similarity.py:7-18](../temporal_embeddings/utils/similarity.py#L7-L18)
returns `1 / (1 + KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²))) / T`. The encoder
[gauss_model.py](../temporal_embeddings/model/gauss_model.py) emits a
Gaussian `(μ̂, σ̂)` per input. The training loss
([cosent_loss.py:17-29](../temporal_embeddings/utils/loss/cosent_loss.py#L17-L29))
is a rank-based pairwise objective on these similarities.

TSF v1 ([compute_similarity_dates.py:39-81](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L39-L81))
is treated as deprecated throughout this document.

---

## 2. Bugs in the current grounding pipeline

These are arithmetic defects that contaminate labels **before TSF runs**.
They can and should be fixed independently of the formulation change.

### B1. `to_explicit_date` truncates months to 30 days

[to_explicit_date.py:14-23](../temporal_embeddings/data_utils/utils/dates/to_explicit_date.py#L14-L23):

```python
if int(month) == 2:
    last_day = 28
else:
    last_day = 30
```

Every `yyyy-mm` interval ending in January, March, May, July, August,
October, or December loses one day; February loses three days in non-leap
years and four days in leap years (since the code never branches on leap
year either). Since `yyyy-mm` is the most common TIMEX granularity after
`yyyy`, this defect contaminates a substantial fraction of the training
labels.

**Fix.** Use the standard library:

```python
from calendar import monthrange
last_day = monthrange(int(year), int(month))[1]
```

### B2–B5. v1-only arithmetic bugs

The v1 helpers in
[compute_interval_distance.py](../temporal_embeddings/data_utils/utils/compute_interval_distance.py)
and
[compute_distance_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_distance_dates.py)
exhibit several further defects (containment hard-coded to `0.9`,
partial-overlap cap of `0.2`, 30-day month approximation in distance
computation, forced minimum distance of 1). They are listed here for
completeness. These code paths are removed entirely by the v1 deprecation
in section 8.

### B6. Dead-or-inconsistent `yyyy-s` season format

[is_date.py:40](../temporal_embeddings/data_utils/utils/dates/is_date.py#L40)
matches the regex `^\d{4}s(?:-(early|mid|late))?$` — decades like `1980s`
or `1980s-early`. There is **no dash before the `s`**, so `2020-WI` and
similar TIMEX season-coded values are rejected by `is_valid_date`.

However,
[compute_distance_dates.py:8](../temporal_embeddings/data_utils/utils/dates/compute_distance_dates.py#L8)
branches on `date_type == "yyyy-s"` and reads `first_date[-2:]` as a
two-character season code. Either the branch is dead code, or it relies on
some bypass of `is_valid_date` from a caller not visible to the static
graph.

**Fix.** Confirm what your upstream parser (HeidelTime / Stanza) emits.
If TIMEX seasons (`2020-WI`, `2020-SP`) ever appear in your sources,
extend the regex in `is_valid_date` and `to_explicit_date` to ground
them as `[YYYY-MM-DD, YYYY-MM-DD]` over the relevant months. If they do
not, delete the `yyyy-s` branch.

### B7. Silent reorder of interval boundaries

[compute_similarity_dates.py:35](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L35):

```python
start, end = sorted((days_since_base(dates[0]), days_since_base(dates[1])))
```

Reordering masks a data-quality bug in whatever upstream produced
`start > end`. The cost of a silent fix-up is hard-to-trace label noise.

**Fix.** Raise `ValueError(...)` and require the parser to be the
authoritative source.

---

## 3. Mathematical critique of TSF v2

### C0. The label is symmetric; the model head is asymmetric

This is the headline issue. The model
([similarity.py:7-18](../temporal_embeddings/utils/similarity.py#L7-L18))
computes

```
score(q → d) = 1 / (1 + KL(N(μ̂_q, σ̂²_q) ‖ N(μ̂_d, σ̂²_d))) / T,
```

which is asymmetric in `(q, d)` — that is precisely why the Gaussian
parameterization was chosen: to represent directional relations such as
*q is during d*. The label produced by TSF v2 is IoU (plus an
ε-smoothed gap term), which is **symmetric**: `TSF(I_q, I_d) = TSF(I_d, I_q)`
for every input. CoSentLoss
([cosent_loss.py:17-29](../temporal_embeddings/utils/loss/cosent_loss.py#L17-L29))
treats the (q, d) and (d, q) entries of the similarity matrix as
independent training points, and asks the network to predict different
values for the two directions while supervising both with the same target.
The asymmetric capacity of the head is unsupervised.

### C1. Hard zero on disjoint intervals at inference

In inference mode,
[compute_similarity_dates.py:92-93](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L92-L93)
returns `0` for every non-overlapping pair. The pair `(1990, 1991)` and the
pair `(1990, 2500)` receive identical labels. For a contrastive ranking
loss this collapses an entire half of the temporal ordering into a single
equivalence class.

### C2. Train and inference modes disagree

Train mode adds an `ε · exp(−(gap/length)²)` tail
([lines 106-116](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L106-L116))
with `ε = 1e-3`; inference mode returns hard zero. The model is fit to one
target and evaluated against another for the same `(I_q, I_d)` input.

### C3. Discontinuity at the overlap/gap boundary

Slide two short intervals across the boundary from "one-day overlap" to
"one-day gap". The inference label jumps from `IoU = 1 / (|I_q| + |I_d| − 1)`
to `0`. The train label jumps to `ε · exp(0) = 1e-3`. For typical lengths
this is a downward step of one or two orders of magnitude — the temporal
geometry has no such discontinuity.

### C4. Magic constants without justification

- `ε = 1e-3` smoothing floor.
- `(1 + IoU) / 2` containment bonus
  ([line 101](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L101)).
- `length = |I_q| + |I_d| + 1` normalization in the train tail.

None of these values are derived from a principle; tuning them shifts
the loss landscape arbitrarily.

### C5. Granularity asymmetry under IoU

For a year `Y` containing a day `d` inside it, `IoU(Y, d) = 1/365`. The
`(1 + IoU)/2` patch lifts this to `0.5014`, but only when containment is
strict and only for that one configuration: a day at the year boundary
(no containment) drops back to `1/365 ≈ 0.003`. A coarse expression
covering a fine one is *temporally consistent* in a way IoU cannot
express, and the patch covers only one case.

### C6. Smoothness and asymmetry are both needed by the loss stack

CoSentLoss + KL similarity work as a ranker over **smooth, asymmetric**
scores. Symmetric labels (C0) and degenerate labels on disjoint pairs (C1)
deprive the rank loss of the information it is designed to consume.

---

## 4. Related literature

### Asymmetric Gaussian embeddings — the model's lineage

- **Vilnis, L. & McCallum, A. (2014).** *Word Representations via Gaussian
  Embedding.* ICLR. Each object is embedded as `N(μ, Σ)` and similarity is
  asymmetric KL; demonstrates that this geometry captures entailment and
  hierarchical inclusion. This is the architectural ancestor of
  [gauss_model.py](../temporal_embeddings/model/gauss_model.py).
- **Athiwaratkun, B. & Wilson, A. G. (2017).** *Multimodal Word
  Distributions.* ACL. Mixtures of Gaussians retain the same asymmetric
  KL machinery.
- **Bojchevski, A. & Günnemann, S. (2018).** *Deep Gaussian Embedding of
  Graphs: Unsupervised Inductive Learning via Ranking.* ICLR. Pairs
  asymmetric KL between learned Gaussians with a personalized ranking
  loss — closest precedent for the recipe proposed here, with TimeML
  intervals playing the role of graph neighborhoods.
- **Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016).** *Order
  Embeddings of Images and Language.* ICLR. An alternative asymmetric
  geometry (partial-order cones); cited as a contrast — KL is preferred
  here because the model head is already KL-based.

### Temporal IR and interval similarity — the data side

- **Allen, J. F. (1983).** *Maintaining knowledge about temporal
  intervals.* CACM. The 13 qualitative relations
  `{=, <, >, m, mi, o, oi, d, di, s, si, f, fi}`. Used in section 6 as a
  qualitative sanity check.
- **Berberich, K., Bedathur, S., Alonso, O., Weikum, G. (2010).** *A
  language modeling approach for temporal information needs.* ECIR.
  Models each TIMEX as a uniform distribution on its interval and uses
  the resulting probabilistic geometry to score temporal relevance. We
  retain Berberich's uniform-distribution premise but moment-match it
  to a Gaussian to obtain a closed-form, smooth, asymmetric distance
  compatible with the model head.
- **Kanhabua, N. & Anand, A. (2016).** *Temporal Information Retrieval.*
  Foundations & Trends / SIGIR tutorial. Survey identifying KL- and
  overlap-based scoring as the principled options.
- **Mani, I., Verhagen, M., Wellner, B., Lee, C., Pustejovsky, J. (2006).**
  *Machine Learning of Temporal Relations.* ACL. Background on how TimeML
  relations are operationalized in ML.

---

## 5. Proposed function — asymmetric KL on interval Gaussians

### 5.1 Construction

Each grounded TimeML annotation is a closed integer day interval
`I = [a, b]` with `a ≤ b`. Treat the underlying temporal extent as the
continuous uniform distribution `U[a, b + 1)`. Its moments are

```
mean(U[a, b+1)) = (a + b + 1) / 2,
var(U[a, b+1)) = (b − a + 1)² / 12.
```

The **moment-matched Gaussian** of the interval is

```
p_I = N(μ_I, σ_I²) with  μ_I = (a + b + 1) / 2,  σ_I² = (b − a + 1)² / 12.
```

Both parameters are determined entirely by `(a, b)`: **no global constants
appear**. The uniform → Gaussian moment match is the standard projection
of a uniform onto the Gaussian family in IR (Berberich et al., 2010) and
in graph embedding (Bojchevski & Günnemann, 2018).

### 5.2 The function

The **temporal distance** from query interval `I_q` to document interval
`I_d` is

```
TSF(I_q → I_d)  :=  KL( p_{I_q}  ‖  p_{I_d} )
                =   log(σ_d / σ_q)
                  + (σ_q² + (μ_q − μ_d)²) / (2 σ_d²)
                  − 1/2.
```

This is the single function used to annotate training pairs. It returns a
non-negative distance; smaller means *q is more temporally consistent
with d*. A similarity (if needed) is any monotone-decreasing transform —
the model already applies `1 / (1 + KL)`
([similarity.py:14-17](../temporal_embeddings/utils/similarity.py#L14-L17)).

### 5.3 Reference implementation

```python
import math
from typing import Tuple

Interval = Tuple[int, int]  # (a, b) inclusive day indices

def _gaussian_of_interval(I: Interval) -> Tuple[float, float]:
    a, b = I
    length = b - a + 1                      # always >= 1
    mu = (a + b + 1) / 2.0
    sigma2 = (length ** 2) / 12.0
    return mu, sigma2

def temporal_kl(I_q: Interval, I_d: Interval) -> float:
    mu_q, var_q = _gaussian_of_interval(I_q)
    mu_d, var_d = _gaussian_of_interval(I_d)
    return (
        0.5 * math.log(var_d / var_q)
        + (var_q + (mu_q - mu_d) ** 2) / (2.0 * var_d)
        - 0.5
    )
```

It replaces TSF v2 as the entry point reached through
[compute_similarity_expressions.py:67-74](../temporal_embeddings/data_utils/utils/compute_similarity_expressions.py#L67-L74).
Grounding (`to_explicit_date`, `days_since_base`) is unchanged. The
function returns a scalar per ordered pair; to obtain the
`(q→d, d→q)` pair simply call it twice.

### 5.4 Properties — point by point against the critique

- **Asymmetric** (resolves C0).
  `TSF(q → d) ≠ TSF(d → q)` whenever `σ_q ≠ σ_d`. The variance term
  `σ_q² / (2 σ_d²)` is small when the query is narrower than the document
  and large in the reverse direction — exactly the asymmetry the Gaussian
  head learns to emit.
- **Continuous and smooth everywhere** (resolves C3).
  KL between two strictly-positive-σ Gaussians is `C^∞` in `(μ_q, σ_q,
  μ_d, σ_d)`. The overlap/disjoint boundary is not visible to the
  function.
- **Non-degenerate on disjoint pairs** (resolves C1).
  The mean-difference term `(μ_q − μ_d)² / (2 σ_d²)` is unbounded above;
  far-apart pairs receive larger distances than nearby disjoint pairs.
- **One function** (resolves C2). No train/inference split.
- **No tunable constants** (resolves C4). `(μ, σ²)` depend only on
  `(a, b)`. No `ε`, no thresholds, no `(1 + IoU)/2` patch.
- **Granularity-aware in the right direction** (resolves C5). For a day
  inside its containing year, `TSF(day → year)` is moderate (the day is
  consistent with the year, modulo the log-σ penalty for the precision
  mismatch); `TSF(year → day)` is large (the year is not "explained" by
  a single day). The asymmetry direction matches the Allen relation.
- **Distillation-shaped for the model head.** Both the labels and the
  model output live in the same family (Gaussian + KL). The training
  task becomes: *learn to map `(text, date)` to a Gaussian `(μ̂, σ̂)`
  such that the predicted KL matches the closed-form temporal KL of the
  underlying intervals.* This is the same setup as Bojchevski &
  Günnemann (2018).

### 5.5 Consumption by the existing loss

CoSentLoss
([cosent_loss.py:17-29](../temporal_embeddings/utils/loss/cosent_loss.py#L17-L29))
uses only pairwise comparisons `labels[i] < labels[j]` — its objective is
invariant under any strictly monotone transform of the labels. The raw KL
can therefore be stored and consumed directly. The only practical caveat
is the large dynamic range: KL values can span many orders of magnitude
(see the table in section 6). If labels are inspected, plotted, or
combined with other losses, prefer storing `log(1 + KL)` for numerical
ergonomics — the ordering is preserved.

A small change to the existing pipeline is suggested: store the label
**as a distance, not as a similarity**, and flip the comparison direction
inside `CoSentLoss` accordingly. Storing distance avoids the
information-destroying `1 / (1 + KL)` compression at the label
boundary.

---

## 6. Edge cases worked numerically

All intervals are expressed in day indices with day 0 at the start of the
relevant decade. Values are computed from the closed form in section 5.2.

| Case | `I_q` | `I_d` | `TSF v2` (inference, train) | `TSF(q → d)` | `TSF(d → q)` |
|---|---|---|---|---|---|
| Same interval | `1990` | `1990` | 1.0, 1.0 | 0.0 | 0.0 |
| Year vs day-in-middle | `1990` | `1990-06-15` | 0.0027, 0.0027 | **68340** | **5.41** |
| Day-in-middle vs year | `1990-06-15` | `1990` | 0.0027, 0.0027 | **5.41** | **68340** |
| Day at year boundary | `1990-12-31` | `1990` | 0.0027, 0.0027 | 6.89 | 265350 |
| Decade contains year | `1985` | `1980s` | 0.10, 0.10 | 1.82 | 48.78 |
| Adjacent years | `1990` | `1991` | 0.0, ε·… ≈ 4e-4 | 6.00 | 6.00 |
| Far disjoint | `1990` | `2500` | 0.0, ε·… ≈ 0 | 1.56 × 10⁶ | 1.56 × 10⁶ |
| One-day overlap | `[0, 9]` | `[9, 18]` | 0.053, 0.053 | 4.86 | 4.86 |
| One-day gap | `[0, 9]` | `[10, 19]` | 0.0, ~1e-3 | 6.00 | 6.00 |

Key observations:

- **Asymmetry is real and substantial** for the containment cases:
  `TSF(year → day-in-year)` and `TSF(day-in-year → year)` differ by four
  orders of magnitude. The current v2 collapses both to the same `0.0027`.
- **Far-disjoint pairs receive a label that is six orders of magnitude
  larger than near-disjoint pairs**, and the labels are correctly ordered.
  Under v2, the train-mode tail compresses both to `O(10⁻⁴)` and the
  inference mode collapses both to zero.
- **The overlap/gap transition is smooth.** The label moves from `4.86`
  to `6.00` as the intervals slide from one-day overlap to one-day gap.
  Under v2 the corresponding transition is `0.053 → 0.001`, a 50×
  downward step.
- **Symmetric cases remain symmetric.** When `σ_q = σ_d`, the variance
  term reduces to `(μ_q − μ_d)² / (2 σ²) + 1/2 − 1/2`, which is symmetric
  in `μ_q, μ_d`. No asymmetry is introduced where none exists.

### Qualitative check against Allen's 13 relations

Picking a canonical pair from each Allen class and computing
`(TSF(X → Y), TSF(Y → X))`:

| Relation | `(X → Y, Y → X)` qualitative |
|---|---|
| `X = Y` (equal) | `(0, 0)` |
| `X < Y` (before) | both large, growing with gap; symmetric when `|X| = |Y|` |
| `X m Y` (meets, gap = 0) | both moderate; same direction as gap = 1 |
| `X o Y` (overlaps) | both moderate; small asymmetry only from σ mismatch |
| `X d Y` (during, X ⊂ Y) | `(small, large)` — **strong asymmetry** |
| `X di Y` (contains, X ⊃ Y) | `(large, small)` — **strong asymmetry** |
| `X s Y` / `X si Y` (starts, σ different) | asymmetric, scaled by σ ratio |
| `X f Y` / `X fi Y` (finishes, σ different) | asymmetric, scaled by σ ratio |

The asymmetry direction is correctly aligned with the containment
relations the Gaussian head is designed to represent.

---

## 7. Verification

How to validate without committing to an implementation yet.

### 7.1 Closed-form unit examples

Reproduce the table in section 6 from the reference implementation in
section 5.3 against the hand-computed values. Confirm:

- `TSF(I, I) = 0` for any interval `I`.
- `TSF(q → d) ≥ 0` everywhere.
- Monotone-increasing in `|μ_q − μ_d|` for fixed `(σ_q, σ_d)`.
- Asymmetric for `σ_q ≠ σ_d`; symmetric for `σ_q = σ_d`.

### 7.2 Label-distribution audit on existing training data

For a representative dataset CSV (e.g. one of those produced by
[create_temporal_relationships_dataset.py](../create_temporal_relationships_dataset.py),
[create_real_world_dataset.py](../create_real_world_dataset.py),
[create_synthetic_dataset.py](../create_synthetic_dataset.py)),
regenerate labels under the new function and compare histograms:

- The v2 spike at exactly `0.0` (the disjoint collapse predicted by C1)
  should disappear under the new function.
- The v2 spike at exactly `1.0` (identical-date pairs) should map to
  exactly `0.0` under the new function.
- The new function should produce a fat-tailed positive distribution
  rather than a bimodal `{0, IoU}` mass.

### 7.3 Pairwise ranking agreement

On a held-out set of TIMEX pairs with a defensible reference ordering —
the dataset from
[create_temporal_relationships_dataset.py](../create_temporal_relationships_dataset.py)
is a natural choice — compute Spearman ρ between v2 and the new function
both globally and split by Allen relation. Large disagreements localize
where the v2 magic constants distort the label geometry; identical
rankings on overlapping pairs confirm that the new function is not
spuriously reordering the cases the old function did get right.

### 7.4 Downstream retraining (larger investment)

Re-annotate the training data, retrain
[gauss_model.py](../temporal_embeddings/model/gauss_model.py) with the
new labels using the existing
[train.py](../temporal_embeddings/train.py) loop, and compare on the
evaluation harness in
[temporal_embeddings/evaluation](../temporal_embeddings/evaluation).
Two metrics are especially informative:

- **Directional retrieval accuracy** on pairs whose Allen relation
  involves containment (`d`, `di`, `s`, `si`, `f`, `fi`) — the cases
  where the old labels were symmetric and the new labels are not.
- **Spearman ρ between predicted asymmetric KL and label asymmetric KL**
  on held-out pairs, as a direct distillation-quality measure.

---

## 8. v1 deprecation

The replacement also makes the entire v1 code path obsolete. The following
can be removed once the new function is in place:

- `_tsf_v1`, `_tsf_v1_dates` in
  [compute_similarity_dates.py:39-81](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L39-L81).
- `compute_interval_distance`, `compute_interval_distance_date` in
  [compute_interval_distance.py](../temporal_embeddings/data_utils/utils/compute_interval_distance.py)
  (keep `days_since_base`).
- `compute_distance_dates`, `compute_distance_dates_same_type` in
  [compute_distance_dates.py](../temporal_embeddings/data_utils/utils/dates/compute_distance_dates.py).
- `is_in` in
  [compute_similarity_dates.py:20](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L20)
  (used only by v1).
- The `version` parameter on the public TSF entry points
  ([compute_similarity_dates.py:119, 139, 165](../temporal_embeddings/data_utils/utils/dates/compute_similarity_dates.py#L119-L173))
  and on `compute_similarity_expressions`
  ([compute_similarity_expressions.py:20](../temporal_embeddings/data_utils/utils/compute_similarity_expressions.py#L20)).

The grounding stack (`to_explicit_date`, `is_valid_date`, `days_since_base`,
`offset_to_date`, `ref_to_date`, `interval_to_date`) is unchanged — the
new function consumes the same `[a, b]` representation.
