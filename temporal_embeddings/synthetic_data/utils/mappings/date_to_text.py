import random
from typing import Dict, List

from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date

integer_to_month: Dict[int, str] = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

integer_to_month_short: Dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

DATE_PATTERNS: Dict[str, List[str]] = {
    "yyyy": [
        "{year}",
    ],
    "yyyy-mm": [
        "{month_short}, {year}",
        "yyyy-mm",
        "{month} {year}",
    ],
    "yyyy-mm-dd": [
        "{day} {month} {year}",
        "{day}, {month_short}, {year}",
    ]
}

def date_to_text(annotation: str) -> str:
    is_date_bool, date_format = is_valid_date(annotation)
    
    if not is_date_bool:
        raise ValueError(f"Cannot convert date to text: {annotation}")
    
    patterns = DATE_PATTERNS.get(date_format)
    if not patterns:
        raise ValueError(f"Cannot convert date to text: {annotation}")
    
    if date_format == "yyyy":
        year = annotation
        
        template = random.choice(patterns)
        
        return template.format(year=year)
    
    elif date_format == "yyyy-mm":
        year = annotation.split("-")[0]
        
        month_num = int(annotation.split("-")[-1])
        month = integer_to_month[month_num]
        month_short = integer_to_month_short[month_num]
        
        template = random.choice(patterns)
        
        return template.format(
            month=month.capitalize(),
            month_short=month_short.capitalize(),
            year=year
        )
    
    elif date_format == "yyyy-mm-dd":
        parts = annotation.split("-")
        
        year = parts[0]
        
        month_num = int(parts[1])
        month = integer_to_month[month_num]
        month_short = integer_to_month_short[month_num]
        
        day = int(parts[2])
        
        template = random.choice(patterns)
        
        return template.format(
            day=day,
            month=month.capitalize(),
            month_short=month_short.capitalize(),
            month_num=f"{month_num:02d}",
            year=year,
        )
    
    raise ValueError(f"Cannot convert date to text: {annotation}")