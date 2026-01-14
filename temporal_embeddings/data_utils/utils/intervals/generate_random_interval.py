import random
from datetime import datetime, timedelta
from typing import List
import re

from temporal_embeddings.data_utils.utils.dates.generate_random_date import generate_random_date

def generate_random_interval(start_date: str, end_date: str, granularity_probs: List[float]) -> str:
    first_date: str = generate_random_date(start_date, end_date, granularity_probs=granularity_probs)
    first_date_granularity: str = "day" if re.match(r"\d{4}-\d{2}-\d{2}", first_date) else "month" if re.match(r"\d{4}-\d{2}", first_date) else "year"
    first_dt: datetime = datetime.strptime(first_date if not re.match(r"\d{4}s", first_date) else first_date[:4], "%Y-%m-%d" if first_date_granularity == "day" else "%Y-%m" if first_date_granularity == "month" else "%Y")
    
    if first_date_granularity == "year":
        max_span_days = random.randint(365, 365*20)
    
    elif first_date_granularity == "month":
        max_span_days = random.randint(30, 730)
    
    else:
        max_span_days = random.randint(1, 365)
    
    constrained_end = first_dt + timedelta(days=max_span_days)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    actual_end = min(constrained_end, end_dt).strftime("%Y-%m-%d")

    first_date = first_dt.strftime("%Y-%m-%d") if first_date_granularity == "day" else f"{first_dt.year}-{first_dt.month:02d}-01" if first_date_granularity == "month" else f"{first_dt.year}-01-01"
    
    second_date = generate_random_date(first_date, actual_end, granularity_probs=granularity_probs)
    
    random_dates = sorted([first_date, second_date])

    return f"{random_dates[0]},{random_dates[1]}"