import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import argparse
from typing import Tuple, List, Dict

import pandas as pd
from tqdm import tqdm

from temporal_embeddings.data_utils.utils.dates.dates_settings import START_DATE, END_DATE
from temporal_embeddings.data_utils.utils.dates.compute_distance_dates import compute_distance_dates_same_type
from temporal_embeddings.data_utils.utils.dates.compute_similarity_dates import TSFVersion, compute_similarity_dates

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def random_month_year(start_year: int=int(START_DATE.split("-")[0]), end_year: int=int(END_DATE.split("-")[0])) -> Tuple[str, str]:
    month: str = random.choice(MONTHS)
    year: int = random.randint(start_year, end_year)
    
    return month, str(year)

def month_year_to_yyyy_mm(date_str: str) -> str:
    try:
        month_index: int = MONTHS.index(date_str.split(",")[0].strip()) + 1
        year_str: str = date_str.split(",")[1].strip()
        dt = datetime.strptime(f"{month_index} {year_str}", "%m %Y")
        
        return dt.strftime("%Y-%m")
    
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")

def is_date_in_range(date_str: str, start_year: int=int(START_DATE.split("-")[0]), end_year: int=int(END_DATE.split("-")[0])) -> bool:
    try:
        month_str, year_str = date_str.split(",")
        month_index = MONTHS.index(month_str.strip()) + 1
        year = int(year_str.strip())
        
        date = datetime(year, month_index, 1)
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        return start_date <= date <= end_date
    except (ValueError, IndexError):
        return False

def phrase_and_answer() -> Tuple[str, str]:
    ref_month, ref_year = random_month_year()

    years_delta: int = random.randint(1, 10)
    months_delta: int = random.randint(0, 11)

    direction: str = random.choice(["before", "after"])

    phrase: str = f"{years_delta} year{'s' if years_delta > 1 else ''} and {months_delta} month{'s' if months_delta != 1 else ''} {direction} {ref_month}, {ref_year}"

    ref_date: datetime = datetime(int(ref_year), MONTHS.index(ref_month) + 1, 1)
    
    if direction == "before":
        answer_date = ref_date - relativedelta(years=years_delta, months=months_delta)
    
    else:
        answer_date = ref_date + relativedelta(years=years_delta, months=months_delta)
    
    answer: str = f"{MONTHS[answer_date.month - 1]}, {answer_date.year}"
    
    return phrase, answer

def random_date_str() -> str:
    month, year = random_month_year()
    
    return f"{month}, {year}"

def generate_random_date(start_year: int=int(START_DATE.split("-")[0]), end_year: int=int(END_DATE.split("-")[0])) -> str:
    start_date: datetime = datetime(start_year, 1, 1)
    end_date: datetime = datetime(end_year, 12, 31)
    delta: timedelta = end_date - start_date

    random_days: int = random.randint(0, delta.days)
    random_date: datetime = start_date + timedelta(days=random_days)
    
    if random_date < start_date:
        random_date = start_date
    
    if random_date > end_date:
        random_date = end_date

    return random_date.strftime("%Y-%m-%d")

def generate_dataset(n_phrases: int, tsf_version: TSFVersion = "v2", tsf_epsilon: float = 1e-3) -> pd.DataFrame:
    data: List[Dict] = []
    
    for _ in tqdm(range(n_phrases)):
        while True:
            phrase, answer = phrase_and_answer()
            if is_date_in_range(answer):
                break

        data.append({"sent0": phrase, "sent0_date": generate_random_date(), "sent1": answer, "sent1_date": generate_random_date(), "score": 1.0})

        incorrect_dates: set[str] = set()
        
        while len(incorrect_dates) < 4:
            d = random_date_str()
            
            if d != answer and is_date_in_range(d):
                incorrect_dates.add(d)
        
        for d in incorrect_dates:
            answer_date = month_year_to_yyyy_mm(answer)
            candidate_date = month_year_to_yyyy_mm(d)
            score = (
                1 / compute_distance_dates_same_type(answer_date, candidate_date, "yyyy-mm")
                if tsf_version == "v1"
                else compute_similarity_dates(
                    answer_date,
                    candidate_date,
                    mode="train",
                    version=tsf_version,
                    epsilon=tsf_epsilon,
                )
            )
            data.append({
                "sent0": phrase,
                "sent0_date": generate_random_date(),
                "sent1": d,
                "sent1_date": generate_random_date(),
                "score": score,
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a synthetic temporal dataset.')
    parser.add_argument('--n_phrases', type=int, default=1000000, help='Number of phrases to generate.')
    parser.add_argument('--output_file_path', type=str, default="synthetic_temporal_dataset.csv", help='Output file path for the dataset.')
    parser.add_argument('--tsf_version', type=str, choices=["v1", "v2"], default="v2", help='Temporal similarity function version to use for labels.')
    parser.add_argument('--tsf_epsilon', type=float, default=1e-3, help='Numerical floor for the TSF V2 disjoint training tail.')
    args = parser.parse_args()

    df = generate_dataset(args.n_phrases, args.tsf_version, args.tsf_epsilon)
    df.to_csv(args.output_file_path, index=False)
