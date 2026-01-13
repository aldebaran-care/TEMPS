import random
from datetime import datetime, timedelta

from temporal_embeddings.data_utils.utils.dates.generate_random_date import generate_random_date

def generate_random_interval(start_date: str, end_date: str, granularity: str) -> str:
    if granularity not in ["year", "month", "day"]:
        random_granularity: int = random.randint(0, 2)

        if random_granularity == 0:
            granularity = "year"
        
        elif random_granularity == 1:
            granularity = "month"
        
        else:
            granularity = "day"
 
    first_date: str = generate_random_date(start_date, end_date, granularity=granularity)
    first_dt: datetime = datetime.strptime(first_date, "%Y-%m-%d" if granularity == "day" else "%Y-%m" if granularity == "month" else "%Y")
    
    if granularity == "year":
        max_span_days = random.randint(365, 365*20)
    
    elif granularity == "month":
        max_span_days = random.randint(30, 730)
    
    else:
        max_span_days = random.randint(1, 365)
    
    constrained_end = first_dt + timedelta(days=max_span_days)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    actual_end = min(constrained_end, end_dt).strftime("%Y-%m-%d")

    first_date = first_dt.strftime("%Y-%m-%d") if granularity == "day" else f"{first_dt.year}-{first_dt.month:02d}-01" if granularity == "month" else f"{first_dt.year}-01-01"
    
    second_date = generate_random_date(first_date, actual_end, granularity=granularity)
    
    random_dates = sorted([first_date, second_date])

    return f"{random_dates[0]},{random_dates[1]}"