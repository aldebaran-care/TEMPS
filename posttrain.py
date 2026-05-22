"""Post-train the temporal embeddings model on a question-answering dataset.

The QA dataset (`data/new_training_dataset/synthetic_qa_dataset/synthetic_qa_dataset.csv`)
shares the schema of the main pre-training corpus
(`sent0,sent0_date,sent1,sent1_date,score`), so we reuse the `Execution` /
`GaussData` pipeline. The only differences vs. `train.py`:

  * `--model_path` is REQUIRED — post-training starts from a pre-trained
    Gaussian temporal model.
  * Defaults are tuned for fine-tuning (lower LR, QA dataset path, shorter
    warmup) so the QA loss can't blow up the pre-trained representations.
  * `continue_training=True` is forced so the checkpoint's model + optimizer
    state are loaded.
"""

import argparse
from pathlib import Path

from temporal_embeddings.parameters.parameters import (
    BATCH_SIZE,
    MODEL_NAME,
    NUM_EVAL_STEPS,
    OUTPUT_DIRECTORY_PATH,
    TEMPERATURE,
    WEIGHT_DECAY,
)
from train import main as train_main

# Fine-tuning defaults: small LR + short warmup so the QA loss only nudges
# the pre-trained Gaussian embeddings rather than retraining them.
QA_INPUT_FILE_PATH: Path = Path("data/new_training_dataset/synthetic_qa_dataset/synthetic_qa_dataset.csv")
POSTTRAIN_LR: float = 2e-5
POSTTRAIN_EPOCHS: int = 1
POSTTRAIN_WARMUP_RATIO: float = 0.02


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-train the temporal embeddings model on a QA dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained checkpoint (.pth produced by train.py).",
    )
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=POSTTRAIN_LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=POSTTRAIN_EPOCHS)
    parser.add_argument("--num_warmup_ratio", type=float, default=POSTTRAIN_WARMUP_RATIO)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--num_eval_steps", type=int, default=NUM_EVAL_STEPS)
    parser.add_argument("--input_file_path", type=str, default=str(QA_INPUT_FILE_PATH))
    parser.add_argument("--output_directory_path", type=str, default=str(OUTPUT_DIRECTORY_PATH))
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"--model_path does not exist: {args.model_path}. "
            "Post-training requires a pre-trained checkpoint from train.py."
        )
    if not Path(args.input_file_path).exists():
        raise FileNotFoundError(
            f"--input_file_path does not exist: {args.input_file_path}"
        )

    train_main(
        data_fraction=args.data_fraction,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        num_warmup_ratio=args.num_warmup_ratio,
        temperature=args.temperature,
        num_eval_steps=args.num_eval_steps,
        input_file_path=args.input_file_path,
        output_directory_path=args.output_directory_path,
        continue_training=True,
        model_path=args.model_path,
    )
