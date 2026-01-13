import random
from typing import Dict, List

from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date

integer_to_month: Dict[int, str] = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}

integer_to_month_short: Dict[int, str] = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}

symbol_to_season: Dict[str, str] = {
    "FA": "fall",
    "SP": "spring",
    "WI": "winter",
    "SU": "summer",
}

symbol_to_season_alt: Dict[str, str] = {
    "FA": "autumn",
    "SP": "spring",
    "WI": "winter",
    "SU": "summer",
}

DATE_PATTERNS: Dict[str, List[str]] = {
    "yyyy": [
        "{year}",
        "the year {year}",
        "in {year}",
        "during {year}",
        "year {year}",
        "in the year {year}",
        "back in {year}",
        "the year of {year}",
    ],
    "yyyy-s": [
        "{season} of {year}",
        "{season} {year}",
        "the {season} of {year}",
        "in {season} {year}",
        "during {season} {year}",
        "{season} season of {year}",
        "the {season} season in {year}",
        "{year}'s {season}",
        "{year} {season}",
        "in the {season} of {year}",
    ],
    "yyyy-mm": [
        "{month} {year}",
        "{month} of {year}",
        "in {month} {year}",
        "during {month} {year}",
        "the month of {month} {year}",
        "{month_short} {year}",
        "in {month_short} {year}",
        "{year}'s {month}",
        "the {month} of {year}",
        "{month}, {year}",
    ],
    "yyyy-mm-dd": [
        "{day} {month} {year}",
        "{month} {day}, {year}",
        "{day} of {month} {year}",
        "{day} {month_short} {year}",
        "{month_short} {day}, {year}",
        "the {day} of {month} {year}",
        "{month} {day}th, {year}",
        "{day}/{month_num}/{year}",
        "{month_num}/{day}/{year}",
        "{year}-{month_num}-{day}",
        "on {day} {month} {year}",
        "on {month} {day}, {year}",
        "{day}.{month_num}.{year}",
        "{ordinal_day} of {month}, {year}",
        "{month} {ordinal_day}, {year}",
    ]
}

def _get_ordinal(day: int) -> str:
    """Convert day number to ordinal (1st, 2nd, 3rd, etc.)"""
    if 10 <= day % 100 <= 20:
        suffix = "th"
    
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    
    return f"{day}{suffix}"

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
    
    elif date_format == "yyyy-s":
        year = annotation.split("-")[0]
        
        season_symbol = annotation.split("-")[-1]
        season = symbol_to_season[season_symbol]
        season_alt = symbol_to_season_alt[season_symbol]
        chosen_season = random.choice([season, season_alt])
        
        template = random.choice(patterns)
        
        return template.format(season=chosen_season, year=year)
    
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
        ordinal_day = _get_ordinal(day)
        
        template = random.choice(patterns)
        
        return template.format(
            day=day,
            month=month.capitalize(),
            month_short=month_short.capitalize(),
            month_num=f"{month_num:02d}",
            year=year,
            ordinal_day=ordinal_day
        )
    
    raise ValueError(f"Cannot convert date to text: {annotation}")