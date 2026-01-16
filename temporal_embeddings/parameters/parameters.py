from pathlib import Path
import torch

# Model Configuration
MODEL_NAME: str = "prajjwal1/bert-tiny"
MAX_SEQ_LEN: int = 512
POSITIONAL_ENCODING_DIM: int = 32
SPECIAL_TOKENS: bool = True

# Training Hyperparameters
LR: float = 3e-4
WEIGHT_DECAY: float = 0.01
EPOCHS: int = 1
NUM_WARMUP_RATIO: float = 0.05
TEMPERATURE: float = 0.05
NUM_EVAL_STEPS: int = 1000

# Data Loading Configuration
BATCH_SIZE: int = 64
SHUFFLE: bool = False
NUM_WORKERS: int = 2
DROP_LAST: bool = True

# Device and Performance
DEVICE: str = "cuda:0"
INFERENCE_DEVICE: str = "cuda:0"
DTYPE: torch.dtype = torch.float16
SEED: int = 42

# File Paths
INPUT_FILE_PATH: Path = Path("data/dataset/dataset.csv")
OUTPUT_DIRECTORY_PATH: Path = Path("output")