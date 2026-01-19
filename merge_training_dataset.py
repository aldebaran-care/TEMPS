import argparse
from pathlib import Path
from typing import List

import pandas as pd

def merge_csv_files(filenames: List[str]) -> None:
    dataframes = []

    base_dir: Path = Path("data/new_training_dataset/training_datasets")
    
    for filename in filenames:
        filepath = base_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        print(f"Reading {filepath}...")
        df = pd.read_csv(filepath)
        dataframes.append(df)
    
    if not dataframes:
        print("No files found to merge!")
        return
    
    print(f"Concatenating {len(dataframes)} files...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    output_filename = base_dir / "merged_training_data.csv"
    print(f"Saving to {output_filename}...")
    merged_df.to_csv(output_filename, index=False)
    
    print(f"Successfully merged {len(dataframes)} files into {output_filename}")
    print(f"Total rows: {len(merged_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV training dataset files.")
    parser.add_argument("filenames", nargs='+', help="Names of CSV files to merge (e.g., training_data_chunk_1.csv training_data_chunk_2.csv)")
    
    args = parser.parse_args()
    
    merge_csv_files(args.filenames)
