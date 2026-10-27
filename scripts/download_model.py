"""Download the trained TEMPS temporal module from the Hugging Face Hub.

The checkpoint is saved under `output/trained_models/` with the file name that
`temporal_embeddings/config/paper_evaluation_config.json` expects, so the paper
evaluation runs without editing the config.

Usage:
    uv run python scripts/download_model.py
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "aldebaran-research/temps"
HUB_FILENAME = "temps-weights.pth"
# Local name expected by paper_evaluation_config.json
FILENAME = "model_sentence-transformers_all-MiniLM-L6-v2_2026-05-24_00-00-51.pth"
OUTPUT_DIR = PROJECT_ROOT / "output" / "trained_models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default=REPO_ID, help=f"Hugging Face model repo (default: {REPO_ID})")
    parser.add_argument("--revision", default=None, help="Branch, tag or commit to download (default: main)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HUB_FILENAME} from {args.repo_id}...")
    cached = hf_hub_download(repo_id=args.repo_id, filename=HUB_FILENAME, revision=args.revision)
    target = OUTPUT_DIR / FILENAME
    shutil.copyfile(cached, target)
    print(f"Saved to {target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
