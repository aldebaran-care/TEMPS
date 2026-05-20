import argparse
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


CHUNK_SIZE = 100_000
NUM_BUCKETS = 64
SEED = 42


def merge_csv_files(
    filenames: List[str],
    num_buckets: int = NUM_BUCKETS,
    chunk_size: int = CHUNK_SIZE,
    seed: int = SEED,
) -> None:
    base_dir: Path = Path("./")
    output_path = base_dir / "merged_training_data.csv"

    existing: List[Path] = []
    for filename in filenames:
        filepath = base_dir / filename
        if filepath.exists():
            existing.append(filepath)
        else:
            print(f"Warning: {filepath} not found, skipping...")

    if not existing:
        print("No files found to merge!")
        return

    rng = np.random.default_rng(seed)

    # Temp buckets live next to the inputs so they share the same filesystem
    # (avoids filling a small /tmp on SLURM nodes).
    with tempfile.TemporaryDirectory(dir=base_dir, prefix=".merge_buckets_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bucket_paths = [tmpdir / f"bucket_{i:04d}.csv" for i in range(num_buckets)]
        bucket_has_data = [False] * num_buckets
        total_rows = 0

        # Phase 1: stream every input file in chunks and scatter each row
        # into a uniformly random bucket. Peak memory ~= one chunk.
        for filepath in existing:
            print(f"Reading {filepath} in chunks of {chunk_size:,}...")
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                bucket_ids = rng.integers(0, num_buckets, size=len(chunk))
                for b in range(num_buckets):
                    mask = bucket_ids == b
                    if not mask.any():
                        continue
                    chunk.loc[mask].to_csv(
                        bucket_paths[b],
                        mode="a",
                        header=not bucket_has_data[b],
                        index=False,
                    )
                    bucket_has_data[b] = True
                total_rows += len(chunk)
            print(f"  Scattered {total_rows:,} rows so far")

        print(f"Scattered {total_rows:,} rows into {sum(bucket_has_data)} buckets")

        # Phase 2: visit buckets in a random order, shuffle each bucket in
        # memory, and append to the final output. Peak memory ~= one bucket
        # (~total_rows / num_buckets).
        bucket_order = list(range(num_buckets))
        rng.shuffle(bucket_order)

        print(f"Shuffling buckets and writing to {output_path}...")
        first = True
        written = 0
        for b in bucket_order:
            if not bucket_has_data[b]:
                continue
            df = pd.read_csv(bucket_paths[b])
            df = df.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
            df.to_csv(
                output_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            written += len(df)
            first = False
            print(f"  Bucket {b}: {len(df):,} rows ({written:,}/{total_rows:,})")

    print(f"Successfully merged {len(existing)} files into {output_path}")
    print(f"Total rows: {written:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge and shuffle CSV training datasets with bounded memory.",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="Names of CSV files to merge (e.g. training_data_chunk_1.csv training_data_chunk_2.csv)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Rows per streaming read chunk (default {CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=NUM_BUCKETS,
        help=f"Number of shuffle buckets; larger = lower peak memory (default {NUM_BUCKETS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for the shuffle",
    )
    args = parser.parse_args()

    merge_csv_files(
        filenames=args.filenames,
        num_buckets=args.num_buckets,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
