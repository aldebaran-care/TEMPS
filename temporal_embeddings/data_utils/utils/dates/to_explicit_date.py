from calendar import monthrange
from typing import List

from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date

def to_explicit_date(annotation: str) -> List[str]:
    if not is_valid_date(annotation)[0]:
        raise ValueError(f"Invalid date format: {annotation}")

    date_format = is_valid_date(annotation)[1]

    if date_format == "yyyy":
        return [f"{annotation}-01-01", f"{annotation}-12-31"]

    elif date_format == "yyyy-mm":
        year  = annotation.split("-")[0]

        month = annotation.split("-")[-1]
        last_day = monthrange(int(year), int(month))[1]

        return [f"{year}-{month}-01", f"{year}-{month}-{last_day}"]
    
    elif date_format == "yyyys":
        decade_prefix = annotation[:4]
        
        start_year = int(decade_prefix)
        end_year   = start_year + 9
        
        if "-" in annotation:
            part = annotation.split("-")[-1]
            
            if part == "early":
                return [f"{start_year}-01-01", f"{start_year+3}-12-31"]
            elif part == "mid":
                return [f"{start_year+4}-01-01", f"{start_year+6}-12-31"]
            elif part == "late":
                return [f"{start_year+7}-01-01", f"{end_year}-12-31"]
            else:
                raise ValueError(f"Invalid decade part: {part}")
        
        else:
            return [f"{start_year}-01-01", f"{end_year}-12-31"]

    else:
        return [annotation]
