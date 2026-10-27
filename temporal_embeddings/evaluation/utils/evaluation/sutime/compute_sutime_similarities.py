from pathlib import Path
import json
import math
from typing import Dict, List, Literal, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm
import pandas as pd

from temporal_embeddings.data_utils.utils.compute_similarity_expressions_sutime import _to_explicit, compute_similarity_expressions_sutime
from temporal_embeddings.data_utils.utils.dates.compute_similarity_dates import _date_list_to_interval, _gaussian_of_interval, temporal_kl
from temporal_embeddings.evaluation.utils.data.random_paragraphs import add_negative_samples

def extract_temporal_expressions(client: "CoreNLPClient", text: str) -> List[Tuple[str, str]]:
    temporal_expressions = []
    ann = client.annotate(text)
    for sentence in ann.sentence:
        for token in sentence.token:
            if token.ner in ["DATE", "TIME", "DURATION", "SET"]:
                expr_text = token.timexValue.text
                expr_value = token.timexValue.value if token.timexValue.value else token.timexValue.altValue
                if expr_text not in [e[0] for e in temporal_expressions]:
                    temporal_expressions.append((expr_text, expr_value))
    return temporal_expressions

Gaussian = Tuple[float, float]  # (mean, variance), in day units
Interval = Tuple[int, int]
SutimeSimilarityMode = Literal["merge", "union", "max", "mean", "product"]
PairwiseSimilarityMode = Literal["max", "mean", "product"]


def sutime_similarities_path_for_mode(similarities_file_path: Path, similarity_mode: SutimeSimilarityMode) -> Path:
    if similarity_mode == "merge":
        return similarities_file_path
    if similarity_mode in ("union", "max", "mean", "product"):
        return similarities_file_path.with_name(f"{similarities_file_path.stem}_{similarity_mode}{similarities_file_path.suffix}")
    raise ValueError("similarity_mode must be 'merge', 'union', 'max', 'mean', or 'product'")


def _merge_text_gaussian(expressions: List[Tuple[str, str]], reference_date: str) -> Optional[Gaussian]:
    """Collapse all temporal expressions of a text into a single Gaussian.

    Each expression is grounded to a day interval and moment-matched to a
    Gaussian (the paper's uniform->Gaussian projection). The per-expression
    Gaussians are then merged by moment-matching their equal-weight mixture
    (law of total variance):

        mu  = mean_i(mu_i)
        var = mean_i(var_i) + mean_i((mu_i - mu)^2)

    This yields one Gaussian per text — the same single-distribution
    representation TEMPS emits — so SUTime is compared to TEMPS like for like,
    rather than scoring individual expression pairs and pooling them. A text
    that mentions several far-apart dates therefore gets a wide, blended
    Gaussian and cannot isolate a single best-matching date. Returns None if no
    expression in the text resolves to an interval.
    """
    means: List[float] = []
    variances: List[float] = []
    for expr_text, expr_value in expressions:
        expression = expr_value if expr_value else expr_text
        explicit = _to_explicit(expression, reference_date)
        if not explicit:
            continue
        try:
            interval = _date_list_to_interval(explicit)
        except ValueError:
            continue
        mu, var = _gaussian_of_interval(interval)
        means.append(mu)
        variances.append(var)

    if not means:
        return None

    n = len(means)
    merged_mean = sum(means) / n
    merged_var = sum(variances) / n + sum((m - merged_mean) ** 2 for m in means) / n
    return merged_mean, merged_var


def _merge_text_interval_union(expressions: List[Tuple[str, str]], reference_date: str) -> Optional[Interval]:
    """Collapse all resolvable temporal expressions of a text into one covering interval.

    This is a deliberately simple interval-level merge: each SUTime expression
    is grounded to an inclusive day interval, then the text is represented by
    the union span from the earliest start to the latest end. The resulting
    interval can be scored by the same KL-based TSF used for ordinary temporal
    intervals.
    """
    intervals: List[Interval] = []
    for expr_text, expr_value in expressions:
        expression = expr_value if expr_value else expr_text
        explicit = _to_explicit(expression, reference_date)
        if not explicit:
            continue
        try:
            intervals.append(_date_list_to_interval(explicit))
        except ValueError:
            continue

    if not intervals:
        return None

    return min(start for start, _ in intervals), max(end for _, end in intervals)


def _gaussian_kl_similarity(g_query: Gaussian, g_doc: Gaussian) -> float:
    """Asymmetric temporal similarity 1 / (1 + KL(q || d)) between two Gaussians,
    matching TEMPS's inference-time score and the TSF label form."""
    mu_q, var_q = g_query
    mu_d, var_d = g_doc
    kl = 0.5 * math.log(var_d / var_q) + (var_q + (mu_q - mu_d) ** 2) / (2.0 * var_d) - 0.5
    return 1.0 / (1.0 + max(kl, 0.0))


def _interval_kl_similarity(query_interval: Interval, doc_interval: Interval) -> float:
    return 1.0 / (1.0 + temporal_kl(query_interval, doc_interval))


def compute_question_similarities_merge(question: str, text_to_gaussian: Dict[str, Optional[Gaussian]], all_paragraphs: List[str]) -> pd.Series:
    """Temporal similarity between the question's single merged Gaussian and
    each paragraph's single merged Gaussian. Scores 0.0 when either text has no
    resolvable temporal expression."""
    question_gaussian = text_to_gaussian.get(question)
    similarities = []
    for paragraph in all_paragraphs:
        paragraph_gaussian = text_to_gaussian.get(paragraph)
        if question_gaussian is None or paragraph_gaussian is None:
            similarities.append(0.0)
        else:
            similarities.append(_gaussian_kl_similarity(question_gaussian, paragraph_gaussian))
    return pd.Series(similarities, index=all_paragraphs)


def compute_question_similarities_union(question: str, text_to_interval: Dict[str, Optional[Interval]], all_paragraphs: List[str]) -> pd.Series:
    """Temporal similarity between each text's merged union interval.

    Scores 0.0 when either side has no resolvable temporal expression.
    """
    question_interval = text_to_interval.get(question)
    similarities = []
    for paragraph in all_paragraphs:
        paragraph_interval = text_to_interval.get(paragraph)
        if question_interval is None or paragraph_interval is None:
            similarities.append(0.0)
        else:
            similarities.append(_interval_kl_similarity(question_interval, paragraph_interval))
    return pd.Series(similarities, index=all_paragraphs)


def compute_question_similarities_pairwise(question: str, temporal_cache: pd.DataFrame, all_paragraphs: List[str], reference_date: str = "2021-11-09", similarity_mode: PairwiseSimilarityMode = "max") -> pd.Series:
    """Temporal similarity as an aggregate over all pairwise expression scores."""
    question_expressions = temporal_cache.loc[question, "expressions"]
    similarities = []

    for paragraph in all_paragraphs:
        paragraph_expressions = temporal_cache.loc[paragraph, "expressions"]

        pairwise_similarities = []
        if question_expressions and paragraph_expressions:
            for q_expr_text, q_expr_value in question_expressions:
                for p_expr_text, p_expr_value in paragraph_expressions:
                    similarity = compute_similarity_expressions_sutime(
                        q_expr_value if q_expr_value else q_expr_text,
                        reference_date,
                        p_expr_value if p_expr_value else p_expr_text,
                        reference_date,
                    )
                    pairwise_similarities.append(similarity)

        if not pairwise_similarities:
            similarities.append(0.0)
        elif similarity_mode == "max":
            similarities.append(max(pairwise_similarities))
        elif similarity_mode == "mean":
            similarities.append(sum(pairwise_similarities) / len(pairwise_similarities))
        elif similarity_mode == "product":
            product = 1.0
            for similarity in pairwise_similarities:
                product *= similarity
            similarities.append(product)
        else:
            raise ValueError("similarity_mode must be 'max', 'mean', or 'product'")

    return pd.Series(similarities, index=all_paragraphs)


def compute_sutime_similarities(benchmark_file_path: Path, cache_file_path: Path, similarities_file_path: Path, num_negative_samples: int, reference_date: str = "2021-11-09", similarity_mode: SutimeSimilarityMode = "merge") -> pd.DataFrame:
    print("Starting SUTime similarity computation...")
    similarities_file_path = sutime_similarities_path_for_mode(similarities_file_path, similarity_mode)
    if similarity_mode == "merge":
        print("Similarity: one merged Gaussian per text, asymmetric KL (TEMPS-equivalent representation)")
    elif similarity_mode == "union":
        print("Similarity: one union interval per text, asymmetric KL")
    elif similarity_mode == "max":
        print("Similarity: max pairwise SUTime expression similarity")
    elif similarity_mode == "mean":
        print("Similarity: mean pairwise SUTime expression similarity")
    elif similarity_mode == "product":
        print("Similarity: product pairwise SUTime expression similarity")
    else:
        raise ValueError("similarity_mode must be 'merge', 'union', 'max', 'mean', or 'product'")
    
    print(f"Loading benchmark data from: {benchmark_file_path}")
    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data: List[Dict] = add_negative_samples(json.load(f), num_negatives=num_negative_samples)
        print(f"Loaded {len(benchmark_data)} benchmark items")

    all_paragraphs: List[str] = []
    for item in benchmark_data:
        all_paragraphs.extend(item["paragraphs"])
    
    all_paragraphs = sorted(list(set(all_paragraphs)))
    print(f"Found {len(all_paragraphs)} unique paragraphs")

    print("\n=== STAGE 1: Extracting Temporal Expressions ===")
    
    temporal_cache = pd.DataFrame(columns=["expressions"])
    
    if cache_file_path.exists():
        print(f"Cache file found at: {cache_file_path}")
        temporal_cache = pd.read_pickle(cache_file_path)
        print(f"Loaded cache with {len(temporal_cache)} texts")
    else:
        print(f"No cache file found - will create new cache")

        print("Collecting unique texts for temporal expression extraction...")
        unique_texts = set()
        for benchmark_item in benchmark_data:
            question = benchmark_item["question"]
            unique_texts.add(question)
            unique_texts.update(benchmark_item["paragraphs"])

        unique_texts = list(unique_texts)
        print(f"Found {len(unique_texts)} unique texts")
        
        texts_to_process = [t for t in unique_texts if t not in temporal_cache.index]
        
        if texts_to_process:
            print(f"Extracting temporal expressions from {len(texts_to_process)} new texts...")
            print("Initializing CoreNLP client with SUTime...")
            try:
                from stanza.server import CoreNLPClient
            except ImportError as exc:
                raise ImportError(
                    "stanza is required to extract SUTime temporal expressions. "
                    "Install the requirements.txt environment or reuse an existing SUTime cache."
                ) from exc
            client = CoreNLPClient(annotators=['tokenize', 'ner'], be_quiet=True)
            print("CoreNLP client initialized successfully")
            
            for text in tqdm(texts_to_process, desc="Extracting temporal expressions"):
                expressions = extract_temporal_expressions(client, text)
                temporal_cache.loc[text] = {'expressions': expressions}
            
            print(f"Saving temporal expression cache to: {cache_file_path}")
            temporal_cache.to_pickle(cache_file_path)
            print(f"Cache saved with {len(temporal_cache)} texts")
        else:
            print("All texts found in cache - no new extraction needed")

    print("\n=== STAGE 2: Computing Similarities ===")
    
    output_similarities_cache: pd.DataFrame = pd.DataFrame(columns=all_paragraphs)

    if similarities_file_path.exists():
        print(f"Similarities file found at: {similarities_file_path}")
        output_similarities_cache = pd.read_pickle(similarities_file_path)
        print(f"Loaded similarities cache with {len(output_similarities_cache)} entries")
    else:
        print(f"No similarities file found - will create new similarities cache")
        print("Processing benchmark items for similarity computation...")
                
        questions = list({item["question"] for item in benchmark_data})

        if similarity_mode == "merge":
            print("Merging each text's temporal expressions into a single Gaussian...")
            text_to_gaussian: Dict[str, Optional[Gaussian]] = {
                text: _merge_text_gaussian(temporal_cache.loc[text, "expressions"], reference_date)
                for text in temporal_cache.index
            }
            resolved = sum(1 for g in text_to_gaussian.values() if g is not None)
            print(f"Merged Gaussians: {resolved}/{len(text_to_gaussian)} texts have a resolvable temporal expression")

            worker_func = partial(compute_question_similarities_merge,
                                text_to_gaussian=text_to_gaussian,
                                all_paragraphs=all_paragraphs)
        elif similarity_mode == "union":
            print("Merging each text's temporal expressions into one covering interval...")
            text_to_interval: Dict[str, Optional[Interval]] = {
                text: _merge_text_interval_union(temporal_cache.loc[text, "expressions"], reference_date)
                for text in temporal_cache.index
            }
            resolved = sum(1 for interval in text_to_interval.values() if interval is not None)
            print(f"Merged intervals: {resolved}/{len(text_to_interval)} texts have a resolvable temporal expression")

            worker_func = partial(compute_question_similarities_union,
                                text_to_interval=text_to_interval,
                                all_paragraphs=all_paragraphs)
        else:
            worker_func = partial(compute_question_similarities_pairwise,
                                temporal_cache=temporal_cache,
                                all_paragraphs=all_paragraphs,
                                reference_date=reference_date,
                                similarity_mode=similarity_mode)

        print("Computing similarities using parallel processing...")
        
        with Pool(processes=cpu_count() - 1) as pool:
            results = list(tqdm(
                pool.imap(worker_func, questions),
                total=len(questions),
                desc="Computing similarities"
            ))
        output_similarities_cache = pd.DataFrame(results, index=questions)

        print(f"Saving SUTime similarities to: {similarities_file_path}")
        output_similarities_cache.to_pickle(similarities_file_path)
        print("SUTime similarities saved successfully")
    
    return output_similarities_cache
