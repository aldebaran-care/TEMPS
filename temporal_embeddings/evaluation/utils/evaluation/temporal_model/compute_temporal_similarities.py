import gc
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import pandas as pd
import torch

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.inference import Inference
from temporal_embeddings.evaluation.utils.evaluation.temporal_model.similarity import asymmetrical_kl_sim
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples


Similarities = Dict[str, Dict[str, float]]


# Module-level state read by the Pool workers. Populated once per worker
# process via `_init_worker` (called either as the Pool initializer or directly
# in single-process mode).
_WORKER_MU: Dict[str, torch.Tensor] = {}
_WORKER_STD: Dict[str, torch.Tensor] = {}


def _init_worker(mu_cache: Dict[str, torch.Tensor], std_cache: Dict[str, torch.Tensor]) -> None:
    global _WORKER_MU, _WORKER_STD
    _WORKER_MU = mu_cache
    _WORKER_STD = std_cache
    # PyTorch's intra-op threadpool fights with the multiprocessing pool —
    # each worker should run single-threaded. Keep this idempotent.
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def _compute_for_question(
    args: Tuple[str, Tuple[str, ...]],
) -> Tuple[str, Dict[str, float]]:
    question, paragraphs = args
    q_mu = _WORKER_MU[question]
    q_std = _WORKER_STD[question]

    para_mu = torch.stack([_WORKER_MU[p] for p in paragraphs])
    para_std = torch.stack([_WORKER_STD[p] for p in paragraphs])

    n = len(paragraphs)
    q_mu_b = q_mu.unsqueeze(0).expand(n, -1)
    q_std_b = q_std.unsqueeze(0).expand(n, -1)

    sims = asymmetrical_kl_sim(q_mu_b, q_std_b, para_mu, para_std).tolist()
    return question, dict(zip(paragraphs, sims))


def _available_cpus() -> int:
    """Number of CPUs this process is actually allowed to use.

    On Linux/SLURM, `os.sched_getaffinity(0)` reflects the cgroup-imposed CPU
    set, which matches `--cpus-per-task`. Falls back to `os.cpu_count()` on
    platforms without sched_getaffinity (macOS, Windows).
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _load_cached_similarities(path: Path) -> Optional[Similarities]:
    """Return a nested-dict similarities cache, or None if the file is absent or
    in the legacy dense-DataFrame format (which is no longer supported)."""
    if not path.exists():
        return None
    obj = pd.read_pickle(path)
    if isinstance(obj, dict):
        return obj
    print(
        f"Ignoring legacy similarities cache at {path} "
        f"(type {type(obj).__name__}); recomputing in nested-dict format."
    )
    return None


def _build_work_items(
    benchmark_data: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Per-question union of paragraphs across all benchmark items. Deduplicates
    work so each (question, paragraph) pair is computed at most once."""
    question_to_paragraphs: Dict[str, List[str]] = {}
    for item in benchmark_data:
        question = item["question"]
        bucket = question_to_paragraphs.setdefault(question, [])
        seen = set(bucket)
        for paragraph in item["paragraphs"]:
            if paragraph not in seen:
                bucket.append(paragraph)
                seen.add(paragraph)
    return question_to_paragraphs


def _materialize_embedding_caches(
    embedding_cache: pd.DataFrame,
    texts: set,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Convert the slow pandas-indexed embedding cache into plain dicts of
    tensors, keyed by text. Lookup goes from O(log n) pandas indexing to O(1)
    dict access, and the dicts can be shared with worker processes via the
    Pool initializer."""
    mu_cache: Dict[str, torch.Tensor] = {}
    std_cache: Dict[str, torch.Tensor] = {}
    for text in texts:
        mu_cache[text] = torch.FloatTensor(embedding_cache.loc[text, 'mu'])
        std_cache[text] = torch.FloatTensor(embedding_cache.loc[text, 'std'])
    return mu_cache, std_cache


def compute_temporal_similarities(
    temporal_model_name: str,
    temporal_model_path: Path,
    batch_size: int,
    max_seq_len: int,
    benchmark_file_path: Path,
    temporal_cache_file_path: Path,
    temporal_similarities_file_path: Path,
    num_negative_samples: int = 0,
    reference_date: str = "2021-11-09",
    benchmark_data: Optional[List[Dict[str, Any]]] = None,
) -> Similarities:
    print("Starting temporal model embeddings computation...")
    print(f"Using reference date: {reference_date}")

    cached = _load_cached_similarities(temporal_similarities_file_path)
    if cached is not None:
        print(f"Loaded similarities cache with {len(cached)} questions")
        return cached

    if benchmark_data is None:
        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = add_negative_samples(json.load(f), num_negatives=num_negative_samples)
    else:
        print("Using preloaded benchmark data")

    print(f"Loaded {len(benchmark_data)} benchmark items")

    # ===== STAGE 1: Embeddings =====
    print("Initializing embedding cache...")
    embedding_cache: pd.DataFrame = pd.DataFrame(columns=['mu', 'std', 'dates'])

    if temporal_cache_file_path.exists():
        print(f"Temporal cache file found at: {temporal_cache_file_path}")
        embedding_cache = pd.read_pickle(temporal_cache_file_path)
        print(f"Loaded cache with {len(embedding_cache)} embeddings")
    else:
        print("No temporal cache file found - will create new cache")

        print("\n=== STAGE 1: Computing Embeddings ===")
        print("Initializing temporal model inference...")
        inference: Inference = Inference(
            model_name=temporal_model_name,
            model_path=temporal_model_path,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        print("Collecting texts to encode...")

        questions_to_encode: List[str] = []
        question_reference_dates: List[str] = []
        paragraphs_to_encode: List[str] = []
        seen_questions: set = set()
        seen_paragraphs: set = set()

        for benchmark_item in tqdm(benchmark_data, desc="Collecting texts"):
            question: str = benchmark_item["question"]
            if question not in seen_questions:
                questions_to_encode.append(question)
                question_reference_dates.append(benchmark_item.get("reference_date", reference_date))
                seen_questions.add(question)

            for paragraph in benchmark_item["paragraphs"]:
                if paragraph not in seen_paragraphs:
                    paragraphs_to_encode.append(paragraph)
                    seen_paragraphs.add(paragraph)

        print(f"Found {len(questions_to_encode)} questions and {len(paragraphs_to_encode)} paragraphs to encode")

        if questions_to_encode:
            print("Computing question embeddings...")
            question_embeddings = inference.compute_embeddings(questions_to_encode, question_reference_dates)
            embedding_cache = pd.concat([embedding_cache, question_embeddings])

        if paragraphs_to_encode:
            print("Computing paragraph embeddings...")
            paragraph_dates = [reference_date] * len(paragraphs_to_encode)
            paragraph_embeddings = inference.compute_embeddings(paragraphs_to_encode, paragraph_dates)
            embedding_cache = pd.concat([embedding_cache, paragraph_embeddings])

        print(f"Saving temporal cache to: {temporal_cache_file_path}")
        temporal_cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_cache.to_pickle(temporal_cache_file_path)
        print(f"Cache saved with {len(embedding_cache)} embeddings")

        # Free model + GPU memory before forking workers — CUDA contexts and
        # `fork` do not mix well.
        del inference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===== STAGE 2: Similarities (parallel, CPU-bound) =====
    print("\n=== STAGE 2: Computing Similarities ===")

    question_to_paragraphs = _build_work_items(benchmark_data)

    all_texts: set = set(question_to_paragraphs.keys())
    for paragraphs in question_to_paragraphs.values():
        all_texts.update(paragraphs)

    print(f"Materializing tensor caches for {len(all_texts)} unique texts...")
    mu_cache, std_cache = _materialize_embedding_caches(embedding_cache, all_texts)

    work_items: List[Tuple[str, Tuple[str, ...]]] = [
        (question, tuple(paragraphs)) for question, paragraphs in question_to_paragraphs.items()
    ]

    n_workers = _available_cpus()
    similarities: Similarities = {}

    if n_workers <= 1 or len(work_items) <= 1:
        print(f"Computing similarities sequentially ({len(work_items)} questions)")
        _init_worker(mu_cache, std_cache)
        for args in tqdm(work_items, desc="Computing similarities"):
            question, bucket = _compute_for_question(args)
            similarities[question] = bucket
    else:
        chunksize = max(1, len(work_items) // (n_workers * 8))
        print(
            f"Computing similarities in parallel: {n_workers} workers, "
            f"{len(work_items)} questions, chunksize={chunksize}"
        )
        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(mu_cache, std_cache),
        ) as pool:
            for question, bucket in tqdm(
                pool.imap_unordered(_compute_for_question, work_items, chunksize=chunksize),
                total=len(work_items),
                desc=f"Computing similarities ({n_workers} workers)",
            ):
                similarities[question] = bucket

    print(f"Saving temporal model similarities to: {temporal_similarities_file_path}")
    temporal_similarities_file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(similarities, temporal_similarities_file_path)
    print(f"Temporal model similarities saved successfully ({len(similarities)} questions)")

    return similarities
