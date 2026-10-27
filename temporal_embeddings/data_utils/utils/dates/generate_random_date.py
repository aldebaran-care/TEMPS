import random
import datetime
from typing import List

import numpy as np

def generate_random_date(start_date: str, end_date: str, granularity_probs: List[float]) -> str:
    """
    Generate a random date with a specified granularity based on probabilities.
    
    Args:
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        granularity: List of probabilities for [decade, year, month, day]
    """
    granularity_types = ["decade", "year", "month", "day"]
    
    selected_granularity: str = random.choices(granularity_types, weights=granularity_probs, k=1)[0]

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    delta = end - start
    
    # Beta distribution: alpha=1, beta=recency_bias
    # Higher beta = stronger recency bias
    recency_bias: int = 3
    bias_factor: float = np.random.beta(1, recency_bias)
    random_days: int = int(delta.days * bias_factor)
    random_date: datetime.datetime = end - datetime.timedelta(days=random_days)
    
    if selected_granularity == "decade":
        year: str = str(random_date.year)
        
        # Randomly choose between "1990s" or "1990s-early/mid/late"
        if random.random() < 0.5:
            return f"{year[:3]}0s"
        
        else:
            part = random.choice(["early", "mid", "late"])
        
            return f"{year[:3]}0s-{part}"
    
    if selected_granularity == "year":
        return random_date.strftime("%Y")
    
    if selected_granularity == "month":
        return random_date.strftime("%Y-%m")
    
    if selected_granularity == "day":
        return random_date.strftime("%Y-%m-%d")
    
    raise ValueError(f"Invalid granularity selected: {selected_granularity}")