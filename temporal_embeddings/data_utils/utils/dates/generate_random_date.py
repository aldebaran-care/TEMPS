import random
import datetime

import numpy as np

def generate_random_date(start_date: str, end_date: str, granularity: str) -> str:
    random_granularity: int = random.randint(0, 3)

    if granularity == "year":
        random_granularity = 0
    
    elif granularity == "season":
        random_granularity = 1
    
    elif granularity == "month":
        random_granularity = 2
    
    elif granularity == "day":
        random_granularity = 3

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    delta = end - start
    
    # Beta distribution: alpha=1, beta=recency_bias
    # Higher beta = stronger recency bias
    recency_bias: int = 3
    bias_factor = np.random.beta(1, recency_bias)
    random_days = int(delta.days * bias_factor)
    random_date = end - datetime.timedelta(days=random_days)
    
    if random_granularity == 0:
        return random_date.strftime("%Y")
    
    if random_granularity == 1:
        seasons = ["SU", "WI", "FA", "SP"]
        return str(random_date.strftime("%Y") + f"-{seasons[random.randint(0, 3)]}")
    
    if random_granularity == 2:
        return random_date.strftime("%Y-%m")
    
    return random_date.strftime("%Y-%m-%d")