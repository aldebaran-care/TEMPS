import argparse
from pathlib import Path

from temporal_embeddings.synthetic_data.create_synthetic_dataset import create_synthetic_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="data/synthetic_data/synthetic_dataset.json",
        help="Path to the output synthetic dataset file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=400000,
        help="Number of synthetic data samples to generate.",
    )
    parser.add_argument(
        "--tsf_version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Temporal similarity function version to use for labels.",
    )
    parser.add_argument(
        "--tsf_epsilon",
        type=float,
        default=1e-3,
        help="Numerical floor for the TSF V2 disjoint training tail.",
    )
    args = parser.parse_args()

    create_synthetic_dataset(
        output_file_path=Path(args.output_file_path),
        size=args.size,
        tsf_version=args.tsf_version,
        tsf_epsilon=args.tsf_epsilon,
    )
