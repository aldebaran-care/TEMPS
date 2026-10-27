import json
from pathlib import Path

from temporal_embeddings.evaluation.benchmarks.temp_reason_utils import add_query_anchored_l1_negatives


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
    centuries away and therefore trivial to reject -- the distractors are sampled
    from the answer dates that lie *closest in time* to the reference date in the
    query. For example, for ``"What is the time 10 year and 10 month before Sep,
    1272"``, negatives are sampled near September 1272 rather than near the gold
    answer. This forces the model to reason about the requested temporal offset
    instead of only matching the rough era of the answer.

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

    stats = add_query_anchored_l1_negatives(
        l1_data,
        num_negative_samples=num_negative_samples,
        hard_negative_pool=hard_negative_pool,
        seed=seed,
    )
    print(
        f"Total paragraphs available for negative sampling: {stats['all_paragraphs']} "
        f"({stats['dated_candidates']} unique dated candidates)"
    )
    print(
        f"Added query-anchored temporal hard negatives to {stats['query_anchor_count']} L1 items "
        f"({stats['fallback_count']} fell back to random sampling)"
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
