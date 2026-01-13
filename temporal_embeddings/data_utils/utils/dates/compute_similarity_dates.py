from typing import List

from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.dates.is_in import is_in
from temporal_embeddings.data_utils.utils.dates.compute_distance_dates import compute_distance_dates_same_type, compute_distance_dates
from temporal_embeddings.data_utils.utils.compute_interval_distance import compute_interval_distance_date

def compute_similarity_dates(first_date: str, second_date: str) -> float:
    first_is_date, first_date_type = is_valid_date(first_date)
    second_is_date, second_date_type = is_valid_date(second_date)

    if first_is_date and second_is_date:
        if first_date == second_date:
            return 1.0
        
        elif first_date_type == second_date_type:
            distance = compute_distance_dates_same_type(first_date=first_date, second_date=second_date, date_type=first_date_type)
            distance = distance**2 if distance < 100 else distance
            
            return 0.4 / distance
        
        elif first_date_type != second_date_type and is_in(first_date=first_date, first_date_type=first_date_type, second_date=second_date, second_date_type=second_date_type):
            return 0.5
        
        else:
            distance = compute_distance_dates(first_date, first_date_type, second_date, second_date_type)
            distance = distance**2 if distance < 100 else distance
        
            return 0.1 / distance
    
    else:
        raise ValueError(f"Both inputs must be valid dates for similarity computation: {first_date}, {second_date}")

def compute_similarity_dates_intervals(first_date: List[str], second_date: List[str]) -> float:
    if len(first_date) == 1 and len(second_date) == 1:
        return compute_similarity_dates(first_date[0], second_date[0])
    
    if len(first_date) == 1:
        first_date = first_date * 2
    
    if len(second_date) == 1:
        second_date = second_date * 2
    
    distance, overlap = compute_interval_distance_date(first_date, second_date)
    
    if overlap:
        similarity = distance
    
        return similarity
    
    else:
        similarity = 0.5 / distance
    
        return similarity