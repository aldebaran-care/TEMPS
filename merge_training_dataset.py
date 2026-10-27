import argparse
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


CHUNK_SIZE = 100_000
NUM_BUCKETS = 64
SEED = 42
PRINT_INTERVAL_SEC = 5.0


def _fmt_rate(count: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "n/a"
    rate = count / elapsed
    if rate >= 1e6:
        return f"{rate / 1e6:.2f}M rows/s"
    if rate >= 1e3:
        return f"{rate / 1e3:.1f}k rows/s"
    return f"{rate:.0f} rows/s"


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
            _log(f"WARNING: {filepath} not found, skipping...")

    if not existing:
        _log("No files found to merge!")
        return

    _log(f"Merging {len(existing)} file(s) -> {output_path}")
    _log(f"Config: chunk_size={chunk_size:,} | num_buckets={num_buckets} | seed={seed}")
    for i, fp in enumerate(existing, 1):
        size_mb = fp.stat().st_size / (1024 * 1024)
        _log(f"  input {i}/{len(existing)}: {fp} ({size_mb:,.1f} MiB)")

    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory(dir=base_dir, prefix=".merge_buckets_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        bucket_paths = [tmpdir / f"bucket_{i:04d}.csv" for i in range(num_buckets)]
        bucket_has_data = [False] * num_buckets

        # ---------- Phase 1: scatter ----------
        _log("Phase 1/2: scattering rows into random buckets")
        phase1_start = time.monotonic()
        total_rows = 0
        last_print = phase1_start

        for file_idx, filepath in enumerate(existing, start=1):
            _log(f"  -> file {file_idx}/{len(existing)}: reading {filepath}")
            file_rows = 0
            file_start = time.monotonic()
            chunk_idx = 0

            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                chunk_idx += 1
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
                file_rows += len(chunk)

                now = time.monotonic()
                if now - last_print >= PRINT_INTERVAL_SEC:
                    overall_elapsed = now - phase1_start
                    _log(
                        f"     file {file_idx}/{len(existing)} chunk {chunk_idx} "
                        f"| file_rows={file_rows:,} total_rows={total_rows:,} "
                        f"| {_fmt_rate(total_rows, overall_elapsed)} "
                        f"| elapsed {_fmt_duration(overall_elapsed)}"
                    )
                    last_print = now

            file_elapsed = time.monotonic() - file_start
            _log(
                f"  <- file {file_idx}/{len(existing)} done: {file_rows:,} rows "
                f"in {_fmt_duration(file_elapsed)} ({_fmt_rate(file_rows, file_elapsed)})"
            )

        phase1_elapsed = time.monotonic() - phase1_start
        active_buckets = sum(bucket_has_data)
        _log(
            f"Phase 1 done: {total_rows:,} rows -> {active_buckets}/{num_buckets} buckets "
            f"in {_fmt_duration(phase1_elapsed)} ({_fmt_rate(total_rows, phase1_elapsed)})"
        )

        # ---------- Phase 2: shuffle & write ----------
        _log(f"Phase 2/2: shuffling buckets and writing {output_path}")
        phase2_start = time.monotonic()
        bucket_order = list(range(num_buckets))
        rng.shuffle(bucket_order)

        first = True
        written = 0
        active_idx = 0
        last_print = phase2_start

        for b in bucket_order:
            if not bucket_has_data[b]:
                continue
            active_idx += 1
            df = pd.read_csv(bucket_paths[b])
            df = df.sample(
                frac=1,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).reset_index(drop=True)
            df.to_csv(
                output_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            written += len(df)
            first = False

            now = time.monotonic()
            if now - last_print >= PRINT_INTERVAL_SEC or active_idx == active_buckets:
                elapsed = now - phase2_start
                pct = 100.0 * written / total_rows if total_rows else 0.0
                eta = (elapsed / written * (total_rows - written)) if written else 0.0
                _log(
                    f"  bucket {active_idx}/{active_buckets} (id {b:>3}): {len(df):,} rows "
                    f"| {written:,}/{total_rows:,} ({pct:.1f}%) "
                    f"| {_fmt_rate(written, elapsed)} "
                    f"| elapsed {_fmt_duration(elapsed)} | eta {_fmt_duration(eta)}"
                )
                last_print = now

        phase2_elapsed = time.monotonic() - phase2_start
        _log(
            f"Phase 2 done: {written:,} rows written in {_fmt_duration(phase2_elapsed)} "
            f"({_fmt_rate(written, phase2_elapsed)})"
        )

    total_elapsed = time.monotonic() - phase1_start
    out_size_mb = output_path.stat().st_size / (1024 * 1024)
    _log(
        f"Success: merged {len(existing)} file(s) -> {output_path} "
        f"({out_size_mb:,.1f} MiB, {written:,} rows) in {_fmt_duration(total_elapsed)}"
    )


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
