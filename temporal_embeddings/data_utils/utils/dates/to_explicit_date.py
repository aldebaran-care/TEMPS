from typing import List
import calendar

from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date

# Astronomical season dates (month, day)
season_dates = {
    "SP": {"start": (3, 20), "end": (6, 20)},   # Spring: March 20 to June 20
    "SU": {"start": (6, 21), "end": (9, 21)},   # Summer: June 21 to September 21
    "FA": {"start": (9, 22), "end": (12, 20)},  # Fall: September 22 to December 20
    "WI": {"start": (12, 21), "end": (3, 19)},  # Winter: December 21 to March 19 (next year)
}

def to_explicit_date(annotation: str) -> List[str]:
    if not is_valid_date(annotation)[0]:
        raise ValueError(f"Invalid date format: {annotation}")
    
    date_format = is_valid_date(annotation)[1]
    
    if date_format == "yyyy":
        return [f"{annotation}-01-01", f"{annotation}-12-31"]
    
    elif date_format == "yyyy-mm":
        year  = annotation.split("-")[0]
        
        month = annotation.split("-")[-1]
        if int(month) == 2:
            last_day = 28
        else:
            last_day = 30
        
        return [f"{year}-{month}-01", f"{year}-{month}-{last_day}"]
    
    elif date_format == "yyyy-s":
        year = annotation.split("-")[0]
        season = annotation.split("-")[1]
        
        start_month, start_day = season_dates[season]["start"]
        end_month, end_day = season_dates[season]["end"]
        
        start_year = int(year)
        end_year = int(year)
        
        # Winter spans two calendar years
        if season == "WI":
            end_year = start_year + 1
        
        start_date = f"{start_year}-{start_month:02d}-{start_day:02d}"
        end_date = f"{end_year}-{end_month:02d}-{end_day:02d}"
        
        return [start_date, end_date]

    else:
        return [annotation]
