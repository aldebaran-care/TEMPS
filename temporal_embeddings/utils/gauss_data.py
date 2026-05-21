from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset

from temporal_embeddings.parameters.parameters import (
    SHUFFLE, NUM_WORKERS, DROP_LAST, SPECIAL_TOKENS, MAX_SEQ_LEN,
)
from temporal_embeddings.utils.positional_encoding import positional_encoding


class GaussData:
    """Train/val/test splits backed by a memory-mapped Arrow dataset.

    The CSV is loaded once via `datasets.load_dataset("csv", ...)`, which
    parses it into Arrow files under `$HF_HOME/datasets/` and memory-maps
    them. Peak resident memory is therefore bounded by the rows currently
    in flight in the DataLoader workers, not by the file size — this is
    what makes 100%-data training viable on a single A100 node.
    """

    def __init__(
        self,
        file_path: Path,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        data_fraction: float = 1.0,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.batch_size: int = batch_size
        self.rank = rank
        self.world_size = world_size

        if rank == 0:
            print(f"Loading dataset from {file_path} (data_fraction={data_fraction})", flush=True)

        # Arrow-backed, memory-mapped. First call on a given CSV builds the
        # cache (~minutes for tens of GB); subsequent calls are instant.
        ds = load_dataset("csv", data_files=str(file_path), split="train")

        # Drop overly long sentences (>= 100 whitespace tokens). `num_proc`
        # parallelises across CPUs; the filtered table is cached by `datasets`
        # so repeat runs skip this pass entirely.
        filter_workers = max(1, num_workers)
        ds = ds.filter(
            lambda ex: (
                len(ex["sent0"].split()) < 100
                and len(ex["sent1"].split()) < 100
            ),
            num_proc=filter_workers,
            desc="filtering long sentences",
        )

        total_after_filter = len(ds)

        if data_fraction < 1.0:
            n_keep = max(1, int(data_fraction * total_after_filter))
            ds = ds.select(range(n_keep))

        n = len(ds)
        if rank == 0:
            print(
                f"Dataset length: {n:,} (post-filter: {total_after_filter:,}, "
                f"fraction={data_fraction})",
                flush=True,
            )

        # Deterministic split — merge_training_dataset.py already shuffles
        # the merged CSV, so contiguous slices are representative.
        n_train = int(0.9 * n)
        n_val_lo = int(0.98 * n)
        n_val_hi = int(0.99 * n)

        self.train_dataset = ds.select(range(0, n_train))
        self.val_dataset = ds.select(range(n_val_lo, n_val_hi))
        self.test_dataset = ds.select(range(n_val_hi, n))

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=SHUFFLE,
            drop_last=DROP_LAST,
        )

        # Common loader kwargs.
        # - `pin_memory=True`: faster host→device copy in collate.
        # - `persistent_workers`: avoid worker respawn cost between epochs
        #   (only valid when num_workers > 0).
        # - `prefetch_factor=4`: each worker keeps 4 batches queued ahead of
        #   the GPU so dataloading doesn't bottleneck the step loop.
        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4

        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            sampler=train_sampler,
            drop_last=DROP_LAST,
            **loader_kwargs,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            add_special_tokens=SPECIAL_TOKENS,
        )

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        """Merge a list of samples into a mini-batch of tensors."""
        return BatchEncoding(
            {
                "sent0": self.tokenize([d["sent0"] for d in data_list]),
                "sent0_date": positional_encoding([d["sent0_date"] for d in data_list]),
                "sent1": self.tokenize([d["sent1"] for d in data_list]),
                "sent1_date": positional_encoding([d["sent1_date"] for d in data_list]),
                "score": torch.FloatTensor([float(d["score"]) for d in data_list]),
            }
        )

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_val_dataloader(self) -> DataLoader:
        return self.val_dataloader

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader
