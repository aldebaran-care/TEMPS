import random
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from temporal_embeddings.data_utils.utils.dates.dates_settings import START_DATE, END_DATE
from temporal_embeddings.data_utils.utils.offsets.offset_to_date import offset_to_date
from temporal_embeddings.data_utils.utils.refs.ref_to_date import ref_to_date
from temporal_embeddings.data_utils.utils.intervals.interval_to_date import interval_to_date
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date
from temporal_embeddings.data_utils.utils.offsets.is_offset import is_offset
from temporal_embeddings.data_utils.utils.refs.is_ref import is_ref
from temporal_embeddings.data_utils.utils.intervals.is_interval import is_interval
from temporal_embeddings.data_utils.utils.generate_random_temporal_expression import generate_random_temporal_expression
from temporal_embeddings.data_utils.utils.compute_similarity_expressions import compute_similarity_expressions
from temporal_embeddings.synthetic_data.utils.mappings.expression_to_text import expression_to_text
from temporal_embeddings.data_utils.utils.dates.generate_random_date import generate_random_date
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date

def generate_single_sample(seed_offset: int) -> List[Tuple[str, str, str, str, float]]:
    random.seed(42 + seed_offset)
    output_data: List[Tuple[str, str, str, str, float]] = []
    
    first_random_temporal_expression: str = generate_random_temporal_expression(probabilities=[0.3, 0.04, 0.01, 0.65], close=False, expression="", current_date="")
    first_random_temporal_expression_text: str = expression_to_text(first_random_temporal_expression)
    
    first_reference_date: str = generate_random_date(START_DATE, END_DATE, granularity_probs=[0.0, 0.0, 0.0, 1.0])
    second_reference_date: str = generate_random_date(START_DATE, END_DATE, granularity_probs=[0.0, 0.0, 0.0, 1.0])
    
    for _ in range(4):
        second_random_temporal_expression: str = generate_random_temporal_expression(probabilities=[0.3, 0.04, 0.01, 0.65], close=False, expression="", current_date="")
        second_random_temporal_expression_text: str = expression_to_text(second_random_temporal_expression)
        
        try:
            similarity: float = compute_similarity_expressions(first_random_temporal_expression, first_reference_date, second_random_temporal_expression, second_reference_date)
        except ValueError as e:
            print(f"Error computing similarity: {e}")
            continue

        output_data.append((first_random_temporal_expression_text, first_reference_date, second_random_temporal_expression_text, second_reference_date, similarity))
    
    for _ in range(1):
        dates: List[str] = []
        
        if is_valid_date(first_random_temporal_expression)[0]:
            dates = to_explicit_date(first_random_temporal_expression)
        
        elif is_offset(first_random_temporal_expression)[0]:
            dates = offset_to_date(first_random_temporal_expression, first_reference_date)
        
        elif is_ref(first_random_temporal_expression)[0]:
            dates = ref_to_date(first_random_temporal_expression, first_reference_date)
        
        elif is_interval(first_random_temporal_expression)[0]:
            dates = interval_to_date(first_random_temporal_expression)
        
        if len(dates) > 0:
            if len(dates) > 1:
                second_random_temporal_expression: str = f"{dates[0]},{dates[1]}"
            
            else:
                second_random_temporal_expression: str = dates[0]
        
        else:
            raise ValueError(f"Cannot derive dates from expression: {first_random_temporal_expression}")
        
        try:
            second_random_temporal_expression_text: str = expression_to_text(second_random_temporal_expression)
        except ValueError as e:
            print(f"Error converting expression to text: {e}")
            continue
        
        similarity: float = compute_similarity_expressions(first_random_temporal_expression, first_reference_date, second_random_temporal_expression, second_reference_date)
        
        output_data.append((first_random_temporal_expression_text, first_reference_date, second_random_temporal_expression_text, second_reference_date, similarity))
    
    return output_data

def create_synthetic_dataset(output_file_path: Path, size: int) -> None:
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel generation")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(generate_single_sample, range(size)),
            total=size,
            desc="Generating synthetic data"
        ))
    
    output_data: List[Tuple[str, str, str, str, float]] = []
    for result in results:
        output_data.extend(result)

    df = pd.DataFrame(output_data, columns=[
        "sent0", "sent0_date", "sent1", "sent1_date", "score"
    ])

    if output_file_path:
        csv_path = output_file_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

    else:
        raise ValueError("Output file path must be provided.")

    count = (df["score"] > 0.9).sum()
    print(f"Number of pairs with similarity > 0.9: {count} out of {len(df)}")