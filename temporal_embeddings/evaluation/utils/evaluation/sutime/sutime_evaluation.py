from pathlib import Path
import json
from typing import Dict, List, Tuple

from stanza.server import CoreNLPClient
from tqdm import tqdm
import pandas as pd

from temporal_embeddings.utils.os.folder_management import create_folders
from temporal_embeddings.evaluation.utils.evaluation.metrics import compute_metrics
from temporal_embeddings.data_utils.utils.compute_similarity_expressions import compute_similarity_expressions

def extract_temporal_expressions(client: CoreNLPClient, text: str) -> List[Tuple[str, str]]:
    """Extract temporal expressions from text using SUTime.
    
    Returns:
        List of tuples (text, value) for each temporal expression
    """
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

def evaluate_sutime(benchmark_file_path: Path, eval_id: int, top_k: int, metric: str, skip: bool) -> None:
    model_name = "SUTime"
    print(f"Starting SUTime evaluation with model: {model_name}")
    print(f"Benchmark file: {benchmark_file_path}")
    
    SIMILARITIES_FILE_PATH: Path = Path(f"output/similarities/{benchmark_file_path.stem}/{model_name}/{eval_id}_similarities.json")
    create_folders(SIMILARITIES_FILE_PATH.parent)
    print(f"Similarities will be saved to: {SIMILARITIES_FILE_PATH}")
    print(f"Created similarities directory: {SIMILARITIES_FILE_PATH.parent}")

    CACHE_FILE_PATH: Path = Path(f"output/cache/{benchmark_file_path.stem}/{model_name}/{eval_id}_cache.pkl")
    create_folders(CACHE_FILE_PATH.parent)
    print(f"Cache file path: {CACHE_FILE_PATH}")
    print(f"Created cache directory: {CACHE_FILE_PATH.parent}")

    temporal_cache = pd.DataFrame(columns=["text", "expressions"]).set_index("text")
    if CACHE_FILE_PATH.exists():
        print(f"Cache file found at: {CACHE_FILE_PATH}")
        temporal_cache = pd.read_pickle(CACHE_FILE_PATH)
        print(f"Loaded cache with {len(temporal_cache)} texts")
    else:
        print(f"No cache file found - will create new cache")

    if not skip:
        print("Starting similarity computation phase...")
        
        print("Initializing CoreNLP client with SUTime...")
        client = CoreNLPClient(annotators=['tokenize', 'ner'], be_quiet=True)
        print("CoreNLP client initialized successfully")

        output_similarities: List[List[float]] = []
        benchmark_data: List[Dict] = []

        print(f"Loading benchmark data from: {benchmark_file_path}")
        with benchmark_file_path.open("r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
            print(f"Loaded {len(benchmark_data)} benchmark items")

            print("Collecting unique texts for temporal expression extraction...")
            unique_texts = set()

            for benchmark_item in tqdm(benchmark_data, desc="Collecting unique texts"):
                question = benchmark_item["question"]
                unique_texts.add(question)
                unique_texts.update(benchmark_item["paragraphs"])

            unique_texts = list(unique_texts)
            print(f"Found {len(unique_texts)} unique texts")
            
            texts_to_process = [t for t in unique_texts if t not in temporal_cache.index]
            
            if texts_to_process:
                print(f"Extracting temporal expressions from {len(texts_to_process)} new texts...")
                for text in tqdm(texts_to_process, desc="Extracting temporal expressions"):
                    expressions = extract_temporal_expressions(client, text)
                    temporal_cache.loc[text] = [expressions]
                print(f"Cache updated with {len(texts_to_process)} new texts")
            else:
                print("All texts found in cache - no new extraction needed")

            print("Processing benchmark items for similarity computation...")
            for element in tqdm(benchmark_data, desc="Evaluating"):
                question: str = element["question"]
                paragraphs: List[str] = element["paragraphs"]
                
                # Get current date from element if available, otherwise use None
                question_current_date = '2025-11-13'
                
                # Extract temporal expressions from question
                question_expressions = temporal_cache.loc[question, "expressions"]

                similarities: List[float] = []
                
                for paragraph in paragraphs:
                    # Extract temporal expressions from paragraph
                    paragraph_expressions = temporal_cache.loc[paragraph, "expressions"]
                    paragraph_current_date = question_current_date  # Use same current date
                    
                    # Compute similarity between question and paragraph temporal expressions
                    max_similarity = 0.0
                    
                    if question_expressions and paragraph_expressions:
                        for q_expr_text, q_expr_value in question_expressions:
                            for p_expr_text, p_expr_value in paragraph_expressions:
                                # Use the normalized value (TIMEX3 value) for comparison
                                similarity = compute_similarity_expressions(
                                    q_expr_value if q_expr_value else q_expr_text,
                                    p_expr_value if p_expr_value else p_expr_text,
                                    question_current_date,
                                    paragraph_current_date
                                )
                                max_similarity = max(max_similarity, similarity)
                    
                    similarities.append(max_similarity)

                output_similarities.append(similarities)

        # Save temporal expression cache
        print(f"Saving temporal expression cache to: {CACHE_FILE_PATH}")
        temporal_cache.to_pickle(CACHE_FILE_PATH)
        print(f"Cache saved with {len(temporal_cache)} texts")

        print(f"Saving similarities to: {SIMILARITIES_FILE_PATH}")
        with SIMILARITIES_FILE_PATH.open("w", encoding="utf-8") as g:
            json.dump(output_similarities, g, indent=4, ensure_ascii=False)
        print("Similarities saved successfully")
    else:
        print("Skipping similarity computation - using existing results")

    print("Loading ground truth data...")
    ground_truth: List[List[int]] = []

    with benchmark_file_path.open("r", encoding="utf-8") as f:
        benchmark_data: List[Dict] = json.load(f)

        for element in benchmark_data:
            ground_truth.append(element["answer"])
    
    print(f"Loaded ground truth for {len(ground_truth)} items")

    print(f"Loading similarities from: {SIMILARITIES_FILE_PATH}")
    output_similarities: List[List[float]] = []

    with SIMILARITIES_FILE_PATH.open("r", encoding="utf-8") as f:
        output_similarities = json.load(f)
    print(f"Loaded {len(output_similarities)} similarity lists")

    print(f"Computing metrics with top_k={top_k}, metric={metric}")
    print(compute_metrics(ground_truth, output_similarities, top_k, metric))
    print("SUTime evaluation completed successfully")
