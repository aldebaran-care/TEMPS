import argparse

import pandas as pd

def create_final_dataset(real_dataset_path="original_temporal_dataset.csv", synthetic_dataset_path="synthetic_temporal_dataset.csv", synthetic_rel_dataset_path="synthetic_relative_temporal_dataset.csv", output_file_path="training_dataset.csv"):
    """
    Create the final dataset by combining the synthetic dataset with the original dataset.
    """
    # Load the synthetic dataset
    synthetic_df = pd.read_csv(synthetic_dataset_path)
    synthetic_rel_df = pd.read_csv(synthetic_rel_dataset_path)
    
    # Load the original dataset
    original_df = pd.read_csv(real_dataset_path)

    # Combine the datasets
    final_df = pd.concat([synthetic_df, synthetic_rel_df, original_df], ignore_index=True)

    # Shuffle the final dataset
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    # Save the final dataset
    final_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the final dataset by combining synthetic and original datasets.")
    parser.add_argument('--real_dataset_path', type=str, default="original_temporal_dataset.csv", help="Path to the original temporal dataset.")
    parser.add_argument('--synthetic_dataset_path', type=str, default="synthetic_temporal_dataset.csv", help="Path to the synthetic temporal dataset.")
    parser.add_argument('--synthetic_rel_dataset_path', type=str, default="synthetic_relative_temporal_dataset.csv", help="Path to the synthetic relative temporal dataset.")
    parser.add_argument('--output_file_path', type=str, default="training_dataset.csv", help="Path to save the final combined dataset.")
    args = parser.parse_args()

    create_final_dataset(real_dataset_path=args.real_dataset_path, synthetic_dataset_path=args.synthetic_dataset_path, synthetic_rel_dataset_path=args.synthetic_rel_dataset_path, output_file_path=args.output_file_path)