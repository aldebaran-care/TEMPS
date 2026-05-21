"""Pre-download a HuggingFace model into the project-local cache.

Intended to be run on a Jean Zay login node (which has internet) before
`sbatch train.sh`, so the offline compute nodes can read the model from
`$PWD/.hf_cache/`.

Usage:
    uv run python scripts/prefetch_model.py sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".hf_cache"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))

from transformers import AutoModel, AutoTokenizer  # noqa: E402  (after env setup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", help="HuggingFace model id (e.g. sentence-transformers/all-MiniLM-L6-v2)")
    args = parser.parse_args()

    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"Fetching tokenizer for {args.model_name}...")
    AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print(f"Fetching model weights for {args.model_name}...")
    AutoModel.from_pretrained(args.model_name)
    print(f"Done. Cache populated under {CACHE_DIR}")


if __name__ == "__main__":
    main()
