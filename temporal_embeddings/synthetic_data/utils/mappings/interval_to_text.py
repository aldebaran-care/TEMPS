import random
from typing import List

from temporal_embeddings.synthetic_data.utils.mappings.date_to_text import date_to_text

DATE_INTERVAL_PATTERNS: List[str] = [
    "from {} to {}",
    "{} – {}",
    "between {} and {}",
]

def interval_to_text(annotation: str) -> str:
    first_date, second_date = annotation.split(",")
    
    first_text: str = date_to_text(first_date)
    second_text: str = date_to_text(second_date)

    random_pattern: str = random.choice(DATE_INTERVAL_PATTERNS)
    
    return random_pattern.format(first_text, second_text)