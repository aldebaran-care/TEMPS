from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import pandas as pd
import torch

from temporal_embeddings.evaluation.utils.evaluation.temporal_model.inference import Inference
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples


Similarities = Dict[str, Dict[str, float]]


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

    print("Initializing temporal model inference...")
    inference: Inference = Inference(
        model_name=temporal_model_name,
        model_path=temporal_model_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    print("Temporal model inference initialized successfully")

    if benchmark_data is None:
        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = add_negative_samples(json.load(f), num_negatives=num_negative_samples)
    else:
        print("Using preloaded benchmark data")

    print(f"Loaded {len(benchmark_data)} benchmark items")

    print("Initializing embedding cache...")
    embedding_cache: pd.DataFrame = pd.DataFrame(columns=['mu', 'std', 'dates'])

    if temporal_cache_file_path.exists():
        print(f"Temporal cache file found at: {temporal_cache_file_path}")
        embedding_cache = pd.read_pickle(temporal_cache_file_path)
        print(f"Loaded cache with {len(embedding_cache)} embeddings")
    else:
        print("No temporal cache file found - will create new cache")

        print("\n=== STAGE 1: Computing Embeddings ===")
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
        embedding_cache.to_pickle(temporal_cache_file_path)
        print(f"Cache saved with {len(embedding_cache)} embeddings")

    print("\n=== STAGE 2: Computing Similarities ===")
    print("Computing per-benchmark-item similarities (memory-bounded)...")

    similarities: Similarities = {}

    for benchmark_item in tqdm(benchmark_data, desc="Computing similarities"):
        question: str = benchmark_item["question"]
        paragraphs: List[str] = benchmark_item["paragraphs"]

        existing = similarities.get(question)
        if existing is not None:
            new_paragraphs = [p for p in paragraphs if p not in existing]
        else:
            new_paragraphs = list(paragraphs)

        if not new_paragraphs:
            continue

        para_mu = torch.stack([
            torch.FloatTensor(embedding_cache.loc[paragraph, 'mu'])
            for paragraph in new_paragraphs
        ])
        para_std = torch.stack([
            torch.FloatTensor(embedding_cache.loc[paragraph, 'std'])
            for paragraph in new_paragraphs
        ])

        q_mu = torch.FloatTensor(embedding_cache.loc[question, 'mu']).unsqueeze(0).expand(len(new_paragraphs), -1)
        q_std = torch.FloatTensor(embedding_cache.loc[question, 'std']).unsqueeze(0).expand(len(new_paragraphs), -1)

        question_emb = type('GaussOutput', (), {'mu': q_mu, 'std': q_std})()
        paragraph_emb = type('GaussOutput', (), {'mu': para_mu, 'std': para_std})()

        sims = inference.sim_fn(question_emb, paragraph_emb).cpu().tolist()

        bucket = similarities.setdefault(question, {})
        for paragraph, sim in zip(new_paragraphs, sims):
            bucket[paragraph] = float(sim)

    print(f"Saving temporal model similarities to: {temporal_similarities_file_path}")
    temporal_similarities_file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(similarities, temporal_similarities_file_path)
    print(f"Temporal model similarities saved successfully ({len(similarities)} questions)")

    return similarities
