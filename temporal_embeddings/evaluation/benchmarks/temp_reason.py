import json
import random
from pathlib import Path
from typing import List, Dict, Any


def create_temp_reason_benchmark(
    l1_path: str = "data/evaluation/temp_reason_l1/processed_data.json",
    l2_path: str = "data/evaluation/temp_reason_l2/processed_data.json",
    output_path: str = "data/evaluation/temp_reason/processed_data.json",
) -> None:
    """
    Read and merge TempReason L1 and L2 datasets into a single benchmark file.
    For L1 data, randomly sample negative paragraphs from the same file.

    Args:
        l1_path: Path to the L1 processed_data.json file
        l2_path: Path to the L2 processed_data.json file
        output_path: Path where the merged data will be saved
        num_negative_samples: Number of negative paragraphs to sample for L1
        seed: Random seed for reproducibility
    """
    random.seed(42)

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

    print(f"Total paragraphs available for negative sampling: {len(all_l1_paragraphs)}")

    # Add negative samples to L1 data
    for item in l1_data:
        # Get current paragraphs to exclude from negative sampling
        current_paragraphs = set(item.get("paragraphs", []))

        # Sample negative paragraphs
        available_negatives = [
            p for p in all_l1_paragraphs if p not in current_paragraphs]

        num_samples = min(5, len(available_negatives))
        negative_paragraphs = random.sample(available_negatives, num_samples)
        item["paragraphs"].extend(negative_paragraphs)

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
    print(f"L1 items now have {5} negative samples each")