import json
import csv
import random
from pathlib import Path
from typing import List, Dict
import os


def read_csv_data(filepath: str) -> List[Dict]:
    """Read data from CSV file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def merge_and_shuffle_datasets(
    synthetic_json_path: str,
    temporal_relationships_path: str,
    real_dataset_path: str,
    output_dir: str,
    chunk_size: int = 1_000_000
) -> None:
    """
    Merge datasets from multiple sources, shuffle, and split into chunks.
    
    Args:
        synthetic_json_path: Path to synthetic_dataset.json
        temporal_relationships_path: Path to temporal_relationships.csv
        real_dataset_path: Path to dataset.csv
        output_dir: Directory to save output files
        chunk_size: Number of items per output file (default: 1M)
    """
    print("Reading synthetic data from JSON...")
    synthetic_data = read_csv_data(synthetic_json_path)
    print(f"Loaded {len(synthetic_data)} synthetic data points")
    
    print("Reading temporal relationships...")
    temporal_data = read_csv_data(temporal_relationships_path)
    print(f"Loaded {len(temporal_data)} temporal relationship points")
    
    print("Reading real-world dataset...")
    real_data = read_csv_data(real_dataset_path)
    print(f"Loaded {len(real_data)} real-world data points")
    
    # Combine all datasets
    print("Combining datasets...")
    all_data = synthetic_data + temporal_data + real_data
    total_items = len(all_data)
    print(f"Total data points: {total_items}")
    
    # Shuffle the combined dataset
    print("Shuffling data...")
    random.shuffle(all_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into chunks and save
    num_chunks = (total_items + chunk_size - 1) // chunk_size
    print(f"Splitting into {num_chunks} chunks of up to {chunk_size:,} items each...")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_items)
        chunk_data = all_data[start_idx:end_idx]
        
        output_file = os.path.join(output_dir, f"training_data_chunk_{chunk_idx + 1}.csv")
        
        print(f"Writing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_data):,} items) to {output_file}...")
        
        # Get all unique keys from the chunk
        if chunk_data:
            fieldnames = list(chunk_data[0].keys())
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(chunk_data)
    
    print(f"Successfully created {num_chunks} training dataset files in {output_dir}")


if __name__ == "__main__":
    # Define paths
    base_dir = Path("data/new_training_dataset/")
    synthetic_json = base_dir / Path("synthetic_dataset/synthetic_dataset.csv")
    temporal_relationships = base_dir / Path("synthetic_dataset/temporal_relationships.csv")
    real_dataset = base_dir / Path("real_world_dataset/dataset.csv")
    output_directory = base_dir / Path("training_datasets")
    
    # Run the merge and shuffle
    merge_and_shuffle_datasets(
        synthetic_json_path=str(synthetic_json),
        temporal_relationships_path=str(temporal_relationships),
        real_dataset_path=str(real_dataset),
        output_dir=str(output_directory),
        chunk_size=1_000_000
    )
