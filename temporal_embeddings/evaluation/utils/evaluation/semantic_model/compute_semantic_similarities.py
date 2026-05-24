import gc
import json
import os
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import Normalize, Pooling, Transformer
import torch

from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples


Similarities = Dict[str, Dict[str, float]]


# Module-level state used by Pool workers, populated by `_init_worker`.
_WORKER_EMB: Dict[str, torch.Tensor] = {}


def _init_worker(emb_cache: Dict[str, torch.Tensor]) -> None:
    global _WORKER_EMB
    _WORKER_EMB = emb_cache
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def _compute_for_question(
    args: Tuple[str, Tuple[str, ...]],
) -> Tuple[str, Dict[str, float]]:
    question, paragraphs = args
    q_emb = _WORKER_EMB[question].unsqueeze(0)
    p_emb = torch.stack([_WORKER_EMB[p] for p in paragraphs])
    sims = util.cos_sim(q_emb, p_emb)[0].tolist()
    return question, dict(zip(paragraphs, sims))


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _load_cached_similarities(path: Path) -> Optional[Similarities]:
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


def _materialize_embedding_cache(
    embedding_cache: pd.DataFrame,
    texts: set,
) -> Dict[str, torch.Tensor]:
    return {
        text: torch.as_tensor(embedding_cache.loc[text, 'embedding']).float()
        for text in texts
    }


def _compute_similarities_parallel(
    embedding_cache: pd.DataFrame,
    benchmark_data: List[Dict[str, Any]],
) -> Similarities:
    question_to_paragraphs = _build_work_items(benchmark_data)

    all_texts: set = set(question_to_paragraphs.keys())
    for paragraphs in question_to_paragraphs.values():
        all_texts.update(paragraphs)

    print(f"Materializing tensor cache for {len(all_texts)} unique texts...")
    emb_cache = _materialize_embedding_cache(embedding_cache, all_texts)

    work_items: List[Tuple[str, Tuple[str, ...]]] = [
        (question, tuple(paragraphs)) for question, paragraphs in question_to_paragraphs.items()
    ]

    n_workers = _available_cpus()
    similarities: Similarities = {}

    if n_workers <= 1 or len(work_items) <= 1:
        print(f"Computing similarities sequentially ({len(work_items)} questions)")
        _init_worker(emb_cache)
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
            initargs=(emb_cache,),
        ) as pool:
            for question, bucket in tqdm(
                pool.imap_unordered(_compute_for_question, work_items, chunksize=chunksize),
                total=len(work_items),
                desc=f"Computing similarities ({n_workers} workers)",
            ):
                similarities[question] = bucket

    return similarities


def encode(model: SentenceTransformer, texts: List[str], prompt_name: str = None) -> List[List[float]]:
    model_name = model._first_module().__class__.__name__.lower()

    if "inf-retriever" in model_name:
        if prompt_name is not None:
            return model.encode(texts, convert_to_tensor=True, prompt_name=prompt_name)

        return model.encode(texts, convert_to_tensor=True)

    if model_name == "baai/bge-large-en":
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    return model.encode(texts, convert_to_tensor=True)


def batch_encode(model, texts_to_encode: List[str], batch_size: int = 64) -> List[List[float]]:
    """Encode texts in batches to optimize memory usage."""

    encoded_embeddings = []
    num_batches = ceil(len(texts_to_encode) / batch_size)
    for i in tqdm(range(num_batches), desc="Encoding batches"):
        batch = texts_to_encode[i * batch_size:(i + 1) * batch_size]
        batch = [b[:400] for b in batch]
        encoded_embeddings.extend(model.encode(batch, show_progress_bar=False))
    return encoded_embeddings


def compute_semantic_similarities(
    semantic_model_name: str,
    max_seq_len: int,
    benchmark_file_path: Path,
    semantic_cache_file_path: Path,
    semantic_similarities_file_path: Path,
    num_negative_samples: int = 0,
    benchmark_data: Optional[List[Dict[str, Any]]] = None,
) -> Similarities:
    print("Starting semantic model embeddings computation...")
    print(f"Model: {semantic_model_name}")

    if semantic_model_name == "salesforce":
        return compute_salesforce_similarities(
            benchmark_file_path=benchmark_file_path,
            semantic_cache_file_path=semantic_cache_file_path,
            semantic_similarities_file_path=semantic_similarities_file_path,
            num_negative_samples=num_negative_samples,
            benchmark_data=benchmark_data,
        )

    cached = _load_cached_similarities(semantic_similarities_file_path)
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

    print("Initializing embedding cache...")
    embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])

    if semantic_cache_file_path.exists():
        print(f"Semantic cache file found at: {semantic_cache_file_path}")
        embedding_cache = pd.read_pickle(semantic_cache_file_path)
        print(f"Loaded cache with {len(embedding_cache)} embeddings")
    else:
        print(f"No semantic cache file found - will create new cache")

        print("Loading SentenceTransformer model...")
        if "inf-retriever" in semantic_model_name:
            print("Using trust_remote_code=True for inf-retriever model")
            model: SentenceTransformer = SentenceTransformer(semantic_model_name, trust_remote_code=True)
        else:
            model: SentenceTransformer = SentenceTransformer(semantic_model_name)

        model.max_seq_length = max_seq_len
        print(f"Model loaded successfully with max_seq_length: {max_seq_len}")

        print("\n=== STAGE 1: Computing Embeddings ===")
        print("Processing benchmark items with semantic model...")

        for benchmark_item in tqdm(benchmark_data):
            question: str = benchmark_item["question"]

            if question not in embedding_cache.index:
                question_emb = encode(model, [question], prompt_name="question_prompt")[0]
                embedding_cache.loc[question] = {'embedding': question_emb.cpu()}

            paragraphs: List[str] = benchmark_item["paragraphs"]

            for paragraph in paragraphs:
                if paragraph not in embedding_cache.index:
                    paragraph_emb = encode(model, [paragraph], prompt_name="passage_prompt")[0]
                    embedding_cache.loc[paragraph] = {'embedding': paragraph_emb.cpu()}

        print(f"Saving embedding cache to: {semantic_cache_file_path}")
        embedding_cache.to_pickle(semantic_cache_file_path)
        print(f"Cache saved with {len(embedding_cache)} embeddings")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== STAGE 2: Computing Similarities ===")
    similarities = _compute_similarities_parallel(embedding_cache, benchmark_data)

    print(f"Saving semantic model similarities to: {semantic_similarities_file_path}")
    semantic_similarities_file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(similarities, semantic_similarities_file_path)
    print(f"Semantic model similarities saved successfully ({len(similarities)} questions)")

    return similarities


def compute_salesforce_similarities(
    benchmark_file_path: Path,
    semantic_cache_file_path: Path,
    semantic_similarities_file_path: Path,
    num_negative_samples: int,
    benchmark_data: Optional[List[Dict[str, Any]]] = None,
) -> Similarities:
    model_name = "Salesforce/SFR-Embedding-Mistral"
    task = 'Given a question with temporal constraints, retrieve relevant passages that answer the question with the correct temporal information.'

    print(f"Using Salesforce model: {model_name}")
    print(f"Task description: {task}")

    cached = _load_cached_similarities(semantic_similarities_file_path)
    if cached is not None:
        print(f"Loaded similarities cache with {len(cached)} questions")
        return cached

    if benchmark_data is None:
        print("Processing benchmark items with Salesforce model...")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = add_negative_samples(json.load(f), num_negatives=num_negative_samples)
    else:
        print("Using preloaded benchmark data for Salesforce model...")
    print(f"Loaded {len(benchmark_data)} benchmark items")

    def get_detailed_instruction(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    print("Initializing embedding cache...")
    embedding_cache: pd.DataFrame = pd.DataFrame(columns=['embedding'])

    if semantic_cache_file_path.exists():
        print(f"Semantic cache file found at: {semantic_cache_file_path}")
        embedding_cache = pd.read_pickle(semantic_cache_file_path)
        print(f"Loaded cache with {len(embedding_cache)} embeddings")
    else:
        print(f"No semantic cache file found - will create new cache")

        print("\n=== STAGE 1: Computing Embeddings ===")

        print("Loading Salesforce model...")
        # The offline HF cache for SFR-Embedding-Mistral on Jean Zay only ships
        # the base AutoModel files (no modules.json / 1_Pooling / 2_Normalize).
        # SentenceTransformer(name, ...) silently falls back to mean pooling
        # when modules.json is missing — catastrophic for a decoder-only
        # Mistral, since the shared instruction prefix dominates the mean and
        # every query embeds to nearly the same vector. Build the pipeline
        # explicitly with the last-token pooling the model card specifies.
        word_emb = Transformer(model_name, model_args={"trust_remote_code": True})
        pooling = Pooling(
            word_emb.get_word_embedding_dimension(),
            pooling_mode_lasttoken=True,
        )
        normalize = Normalize()
        model = SentenceTransformer(modules=[word_emb, pooling, normalize])
        print("Salesforce model loaded successfully")

        questions_to_encode = []
        question_texts = []
        paragraphs_to_encode = []

        print("Collecting texts to encode...")
        for benchmark_item in benchmark_data:
            question: str = benchmark_item["question"]
            if question not in embedding_cache.index:
                questions_to_encode.append(question)
                question_texts.append(get_detailed_instruction(task, question))

            paragraphs: List[str] = benchmark_item["paragraphs"]
            for paragraph in paragraphs:
                if paragraph not in embedding_cache.index and paragraph not in paragraphs_to_encode:
                    paragraphs_to_encode.append(paragraph)

        print(f"Found {len(questions_to_encode)} questions and {len(paragraphs_to_encode)} paragraphs to encode")

        if questions_to_encode:
            print("Batch encoding questions...")
            question_embeddings = batch_encode(model, question_texts)
            for question, embedding in zip(questions_to_encode, question_embeddings):
                embedding_cache.loc[question] = {'embedding': torch.tensor(embedding).cpu()}

        if paragraphs_to_encode:
            print("Batch encoding paragraphs...")
            paragraph_embeddings = batch_encode(model, paragraphs_to_encode)
            for paragraph, embedding in zip(paragraphs_to_encode, paragraph_embeddings):
                embedding_cache.loc[paragraph] = {'embedding': torch.tensor(embedding).cpu()}

        print(f"Saving embedding cache to: {semantic_cache_file_path}")
        embedding_cache.to_pickle(semantic_cache_file_path)
        print(f"Cache saved with {len(embedding_cache)} embeddings")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== STAGE 2: Computing Similarities ===")
    similarities = _compute_similarities_parallel(embedding_cache, benchmark_data)

    print(f"Saving semantic model similarities to: {semantic_similarities_file_path}")
    semantic_similarities_file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(similarities, semantic_similarities_file_path)
    print(f"Salesforce similarities saved successfully ({len(similarities)} questions)")

    return similarities
