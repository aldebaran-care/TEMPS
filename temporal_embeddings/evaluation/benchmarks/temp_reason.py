import json
from pathlib import Path
from typing import List, Dict, Any


def create_temp_reason_benchmark(
    l1_path: str = "data/evaluation/temp_reason_l1/processed_data.json",
    l2_path: str = "data/evaluation/temp_reason_l2/processed_data.json",
    output_path: str = "data/evaluation/temp_reason/processed_data.json",
) -> None:
    """
    Read and merge TempReason L1 and L2 datasets into a single benchmark file.

    Args:
        l1_path: Path to the L1 processed_data.json file
        l2_path: Path to the L2 processed_data.json file
        output_path: Path where the merged data will be saved
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