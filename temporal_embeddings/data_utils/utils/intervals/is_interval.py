from typing import Tuple, List

from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date

def is_interval(annotation: str) -> Tuple[bool, str]:
    if "," in annotation:
        annotation_list: List[str] = annotation.split(",")
    
        return is_valid_date(annotation_list[0])[0] and is_valid_date(annotation_list[1])[0], "interval"
    
    return False, "interval"