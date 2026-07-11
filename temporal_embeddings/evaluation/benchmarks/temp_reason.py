import json
import random
import re
from pathlib import Path

# Month abbreviation -> ordinal (1-12), used to turn a "Mon, YYYY" string into a
# single absolute month index so that two dates can be compared / subtracted.
MONTH_TO_NUM = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}


def _parse_month_year(text: str):
    """Parse a ``"Mon, YYYY"`` date string into an absolute month index.

    The index is ``year * 12 + (month - 1)`` so that the temporal distance
    between two dates is simply the absolute difference of their indices,
    expressed in months. Returns ``None`` when the string is not a
    recognisable ``Mon, YYYY`` date.
    """
    match = re.match(r"\s*([A-Za-z]{3}),\s*(\d{4})", text.strip())
    if not match:
        return None
    month = MONTH_TO_NUM.get(match.group(1).capitalize())
    if month is None:
        return None
    return int(match.group(2)) * 12 + (month - 1)


def create_temp_reason_benchmark(
    l1_path: str = "data/evaluation/temp_reason_l1/processed_data.json",
    l2_path: str = "data/evaluation/temp_reason_l2/processed_data.json",
    output_path: str = "data/evaluation/temp_reason/processed_data.json",
    num_negative_samples: int = 5,
    hard_negative_pool: int = 20,
    seed: int = 42,
) -> None:
    """
    Read and merge TempReason L1 and L2 datasets into a single benchmark file.

    For L1 (date-arithmetic questions whose gold answer is a single ``Mon, YYYY``
    date), negatives are drawn with **temporal hard-negative sampling**: instead
    of sampling random dates from the whole pool -- which are almost always
    centuries away from the gold answer and therefore trivial to reject -- the
    distractors are sampled from the answer dates that lie *closest in time* to
    the gold. These near-miss dates (a few months / years off) force the model
    to reason about the exact temporal offset asked by the question rather than
    matching the rough era, making the benchmark considerably more challenging.

    L2 already provides its own hard negatives by construction (several
    statements about the same entity over overlapping time spans), so it is
    merged unchanged.

    Args:
        l1_path: Path to the L1 processed_data.json file
        l2_path: Path to the L2 processed_data.json file
        output_path: Path where the merged data will be saved
        num_negative_samples: Number of negative paragraphs to sample for L1
        hard_negative_pool: Size of the temporally-nearest candidate pool the
            negatives are sampled from. Smaller values yield harder (closer)
            distractors; must be >= num_negative_samples.
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    hard_negative_pool = max(hard_negative_pool, num_negative_samples)

    # Get the project root directory (assumes script is in temporal_embeddings/evaluation/benchmarks/)
    project_root = Path(__file__).parent.parent.parent.parent

    # Construct full paths
    l1_full_path = project_root / l1_path
    l2_full_path = project_root / l2_path
    output_full_path = project_root / output_path

    # Read L1 data
    print(f"Reading L1 data from: {l1_full_path}")
    with open(l1_full_path, "r", encoding="utf-8") as f:
        l1_data = json.load(f)
    print(f"Loaded {len(l1_data)} items from L1")

    # Collect all paragraphs from L1 for negative sampling
    all_l1_paragraphs = []
    for item in l1_data:
        if "paragraphs" in item and item["paragraphs"]:
            all_l1_paragraphs.extend(item["paragraphs"])

    # Build the pool of unique, temporally-parseable candidate dates once.
    # Each entry is (month_index, date_string); duplicates are dropped so a
    # distractor is never a repeated string.
    seen_texts = set()
    dated_candidates = []  # list[tuple[int, str]]
    for paragraph in all_l1_paragraphs:
        if paragraph in seen_texts:
            continue
        value = _parse_month_year(paragraph)
        if value is not None:
            dated_candidates.append((value, paragraph))
            seen_texts.add(paragraph)

    print(
        f"Total paragraphs available for negative sampling: {len(all_l1_paragraphs)} "
        f"({len(dated_candidates)} unique dated candidates)"
    )

    hard_neg_count = 0
    fallback_count = 0

    # Add temporal hard negatives to L1 data
    for item in l1_data:
        # Paragraphs already attached to the item must never be sampled as
        # negatives (the first one is the gold answer).
        current_paragraphs = set(item.get("paragraphs", []))
        gold_text = item["paragraphs"][0] if item.get("paragraphs") else None
        gold_value = _parse_month_year(gold_text) if gold_text else None

        if gold_value is None or not dated_candidates:
            # Fallback: gold date is unparseable -> keep the original random
            # strategy so the item still gets distractors.
            available_negatives = [
                p for p in all_l1_paragraphs if p not in current_paragraphs
            ]
            num_samples = min(num_negative_samples, len(available_negatives))
            negative_paragraphs = random.sample(available_negatives, num_samples)
            fallback_count += 1
        else:
            # Rank every candidate by how close its date is to the gold answer
            # (ties broken by the earlier date for determinism), skipping the
            # gold date itself and anything already on the item.
            ranked = sorted(
                dated_candidates, key=lambda c: (abs(c[0] - gold_value), c[0])
            )
            nearest = []
            for value, text in ranked:
                if value == gold_value or text in current_paragraphs:
                    continue
                nearest.append(text)
                if len(nearest) >= hard_negative_pool:
                    break
            # Sample the distractors from the temporally-nearest window so the
            # negatives are hard yet not identical across similar questions.
            num_samples = min(num_negative_samples, len(nearest))
            negative_paragraphs = random.sample(nearest, num_samples)
            hard_neg_count += 1

        item["paragraphs"].extend(negative_paragraphs)

    print(
        f"Added temporal hard negatives to {hard_neg_count} L1 items "
        f"({fallback_count} fell back to random sampling)"
    )

    # Read L2 data
    print(f"Reading L2 data from: {l2_full_path}")
    with open(l2_full_path, "r", encoding="utf-8") as f:
        l2_data = json.load(f)
    print(f"Loaded {len(l2_data)} items from L2")

    # Merge the datasets
    merged_data = l1_data + l2_data
    print(f"Total merged items: {len(merged_data)}")

    # Create output directory if it doesn't exist
    output_full_path.parent.mkdir(parents=True, exist_ok=True)

    # Save merged data
    print(f"Saving merged data to: {output_full_path}")
    with open(output_full_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print("Merge completed successfully!")
    print(f"L1 items now have up to {num_negative_samples} temporal hard negatives each")
