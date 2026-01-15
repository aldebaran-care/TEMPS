import sys
import argparse
from pathlib import Path

import pandas as pd

def merge_csv_files(start_idx: int, end_idx: int) -> None:
    dataframes = []

    base_dir: Path = Path("data/new_training_dataset/training_datasets")
    
    for i in range(start_idx, end_idx + 1):
        filename = base_dir / f"training_data_chunk_{i}.csv"
        
        if not filename.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        print(f"Reading {filename}...")
        df = pd.read_csv(filename)
        dataframes.append(df)
    
    if not dataframes:
        print("No files found to merge!")
        return
    
    print(f"Concatenating {len(dataframes)} files...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    output_filename = base_dir / f"training_data_chunk_{start_idx}_{end_idx}.csv"
    print(f"Saving to {output_filename}...")
    merged_df.to_csv(output_filename, index=False)
    
    print(f"Successfully merged {len(dataframes)} files into {output_filename}")
    print(f"Total rows: {len(merged_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV training dataset chunks.")
    parser.add_argument("start_idx", type=int, help="Starting chunk index (inclusive)")
    parser.add_argument("end_idx", type=int, help="Ending chunk index (inclusive)")
    
    args = parser.parse_args()
    
    merge_csv_files(args.start_idx, args.end_idx)
