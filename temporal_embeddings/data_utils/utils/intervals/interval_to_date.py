from typing import List

from temporal_embeddings.data_utils.utils.intervals.is_interval import is_interval
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date

def interval_to_date(annotation: str) -> List[str]:
    if is_interval(annotation)[0]:
        dates = annotation.split(",")
        
        return [to_explicit_date(dates[0])[0], to_explicit_date(dates[1])[-1]]
    
    raise ValueError(f"Cannot convert interval to date: {annotation}")
