import json
from typing import List, Dict, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from temporal_embeddings.data_utils.utils.compute_similarity_expressions import compute_similarity_expressions
from temporal_embeddings.data_utils.utils.offsets.is_offset import is_offset
from temporal_embeddings.data_utils.utils.refs.is_ref import is_ref
from temporal_embeddings.data_utils.utils.intervals.is_interval import is_interval
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.generate_random_temporal_expression import generate_random_temporal_expression
from temporal_embeddings.synthetic_data.utils.mappings.expression_to_text import expression_to_text
from temporal_embeddings.data_utils.utils.offsets.offset_to_date import offset_to_date
from temporal_embeddings.data_utils.utils.refs.ref_to_date import ref_to_date
from temporal_embeddings.data_utils.utils.intervals.interval_to_date import interval_to_date
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date

def is_valid_temporal_expression(expression: str) -> bool:
    return (is_valid_date(expression)[0] or
            is_interval(expression)[0])

def generate_sample_from_entry(entry: Dict) -> List[Tuple[str, str, str, str, float]]:
    output_data: List[Tuple[str, str, str, str, float]] = []
    
    first_text = entry['text'].replace('\n', ' ').strip()
    first_ref_date = entry['ref_date']
    first_values = entry['values']
    
    if not first_values:
        return output_data
    
    for value in first_values:
        for _ in range(4):
            second_random_temporal_expression = generate_random_temporal_expression(probabilities=[0.3, 0.04, 0.01, 0.65], close=False, expression="", current_date="")
            
            try:
                second_random_temporal_expression_text = expression_to_text(second_random_temporal_expression)
            
            except ValueError as e:
                print(f"Error converting expression to text: {e}")
                
                continue
            
            try:
                similarity = compute_similarity_expressions(value, first_ref_date, second_random_temporal_expression, first_ref_date)
                
                output_data.append((first_text, first_ref_date, second_random_temporal_expression_text, first_ref_date, similarity))
            
            except ValueError as e:
                print(f"Error computing similarity: {e}")
                continue
        
        for _ in range(1):
            dates: List[str] = []
            
            if is_valid_date(value)[0]:
                dates = to_explicit_date(value)
            
            elif is_offset(value)[0]:
                dates = offset_to_date(value, first_ref_date)
            
            elif is_ref(value)[0]:
                dates = ref_to_date(value, first_ref_date)
            
            elif is_interval(value)[0]:
                dates = interval_to_date(value)
            
            if len(dates) > 0:
                if len(dates) > 1:
                    close_temporal_expression = f"{dates[0]},{dates[1]}"
                else:
                    close_temporal_expression = dates[0]
            else:
                print(f"Cannot derive dates from expression: {value}")
                continue
            
            try:
                close_temporal_expression_text = expression_to_text(close_temporal_expression)
            
            except ValueError as e:
                print(f"Error converting expression to text: {e}")
            
                continue
            
            similarity = compute_similarity_expressions(value, first_ref_date, close_temporal_expression, first_ref_date)
            
            output_data.append((first_text, first_ref_date, close_temporal_expression_text, first_ref_date, similarity))
    
    return output_data

def create_real_world_dataset(input_path: Path, output_path: Path, fraction: List[float]=[0.0, 1.0]) -> None:
    print(f"Loading data from {input_path}...")
    try:
        with input_path.open('r', encoding='utf-8') as f:
            content = f.read().strip()
            
            print("Parsing JSON data...")
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines()]
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total entries loaded: {len(data)}")
    start_index: int = int(len(data) * fraction[0])
    end_index: int = int(len(data) * fraction[1])
    data = data[start_index:end_index]
    print(f"Using fraction [{fraction[0]}, {fraction[1]}]: {len(data)} entries")

    processed_sentences: List[Dict] = []
    
    print("Extracting temporal expressions from entries...")
    for entry in tqdm(data, desc="Processing entries", unit="entry"):
        try:
            if 'timex3' not in entry or not entry['timex3']:
                continue
            
            ref_date = entry.get('meta_data', {}).get('dct', None)
            
            if not ref_date:
                continue
                
            temporal_values = [t['value'] for t in entry['timex3'] if ('value' in t and is_valid_temporal_expression(t['value']) and 'explicit' in t['rulename'])]
            
            if temporal_values:
                processed_sentences.append({
                    'text': entry['text'],
                    'ref_date': ref_date,
                    'values': temporal_values
                })

        except Exception as e:
            print(f"Error processing entry: {e}")
            continue

    print(f"Found {len(processed_sentences)} sentences with valid temporal annotations.")
    
    if len(processed_sentences) == 0:
        print("No valid entries found. Check dataset content.")
        return
    
    print(f"Preparing to generate pairs (4 synthetic + 1 close per entry = 5 pairs × {len(processed_sentences)} entries)...")
    
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel generation")
    
    print("Starting parallel pair generation...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(generate_sample_from_entry, processed_sentences),
            total=len(processed_sentences),
            desc="Generating training pairs",
            unit="entry"
        ))
    
    print("Collecting results...")
    output_data: List[Tuple[str, str, str, str, float]] = []
    for result in results:
        output_data.extend(result)
    
    print(f"Creating DataFrame with {len(output_data)} pairs...")
    df = pd.DataFrame(output_data, columns=[
        "sent0", "sent0_date", "sent1", "sent1_date", "score"
    ])
    
    print(f"Writing to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"✓ Successfully wrote {len(df)} pairs to {output_path}")
    count = (df["score"] > 0.9).sum()
    print(f"✓ Number of pairs with similarity > 0.9: {count} ({count/len(df)*100:.2f}%)")
    print(f"✓ Number of pairs with similarity ≤ 0.9: {len(df) - count} ({(len(df)-count)/len(df)*100:.2f}%)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create a real-world temporal dataset from annotated data.')
    parser.add_argument('--input_path', type=Path, required=True, help='Path to the input annotated dataset (JSON format).')
    parser.add_argument('--output_path', type=Path, required=True, help='Path to save the output dataset (CSV format).')
    parser.add_argument('--fraction', type=float, nargs=2, default=[0.0, 1.0], help='Fraction of the dataset to use (start and end).')
    
    args = parser.parse_args()
    
    create_real_world_dataset(args.input_path, args.output_path, args.fraction)