import random
from typing import Dict

from temporal_embeddings.synthetic_data.utils.periods.is_period import is_period
from temporal_embeddings.data_utils.utils.extract_integers import extract_integers

PERIOD_PATTERNS: Dict[str, Dict[str, list[str]]] = {
    "pd": {
        "singular": [
            "a day", "one day", "just a day", "a single day", "1 day",
            "a full day", "a single 24-hour period", "one full day", 
            "a single calendar day", "a day-long period"
        ],
        "plural": [
            "{0} days", "{0} full days", "{0} calendar days",
            "{0} day-long periods", "about {0} days", "nearly {0} days",
            "approximately {0} days", "{0} consecutive days", 
            "{0} 24-hour periods", "{0} days in total"
        ]
    },
    "pw": {
        "singular": [
            "a week", "one week", "just a week", "a single week", "1 week",
            "a full week", "a seven-day period", "one full week", 
            "a single calendar week", "a week-long period"
        ],
        "plural": [
            "{0} weeks", "{0} full weeks", "{0} calendar weeks",
            "{0} week-long periods", "about {0} weeks", "nearly {0} weeks",
            "approximately {0} weeks", "{0} consecutive weeks", 
            "{0} seven-day periods", "{0} weeks in total"
        ]
    },
    "pm": {
        "singular": [
            "a month", "one month", "just a month", "a single month", "1 month",
            "a full month", "a 30-day period", "one full month", 
            "a single calendar month", "a month-long period"
        ],
        "plural": [
            "{0} months", "{0} full months", "{0} calendar months",
            "{0} month-long periods", "about {0} months", "nearly {0} months",
            "approximately {0} months", "{0} consecutive months", 
            "{0} 30-day periods", "{0} months in total"
        ]
    },
    "py": {
        "singular": [
            "a year", "one year", "just a year", "a single year", "1 year",
            "a full year", "a 12-month period", "one full year", 
            "a single calendar year", "a year-long period"
        ],
        "plural": [
            "{0} years", "{0} full years", "{0} calendar years",
            "{0} year-long periods", "about {0} years", "nearly {0} years",
            "approximately {0} years", "{0} consecutive years", 
            "{0} 12-month periods", "{0} years in total"
        ]
    },
    "pdi": {
        "range": [
            "{0} to {1} days", "between {0} and {1} days",
            "from {0} to {1} days", "around {0}-{1} days",
            "approximately {0} to {1} days", "nearly {0}-{1} days",
            "{0}-{1} days range", "about {0} to {1} days",
            "{0}-{1} days in total", "roughly {0} to {1} days"
        ]
    },
    "pwi": {
        "range": [
            "{0} to {1} weeks", "between {0} and {1} weeks",
            "from {0} to {1} weeks", "around {0}-{1} weeks",
            "approximately {0} to {1} weeks", "nearly {0}-{1} weeks",
            "{0}-{1} weeks range", "about {0} to {1} weeks",
            "{0}-{1} weeks in total", "roughly {0} to {1} weeks"
        ]
    },
    "pmi": {
        "range": [
            "{0} to {1} months", "between {0} and {1} months",
            "from {0} to {1} months", "around {0}-{1} months",
            "approximately {0} to {1} months", "nearly {0}-{1} months",
            "{0}-{1} months range", "about {0} to {1} months",
            "{0}-{1} months in total", "roughly {0} to {1} months"
        ]
    },
    "pyi": {
        "range": [
            "{0} to {1} years", "between {0} and {1} years",
            "from {0} to {1} years", "around {0}-{1} years",
            "approximately {0} to {1} years", "nearly {0}-{1} years",
            "{0}-{1} years range", "about {0} to {1} years",
            "{0}-{1} years in total", "roughly {0} to {1} years"
        ]
    },
    "pdn": {
        "first": [
            "first day", "day one", "the initial day", "the first 24 hours", 
            "the starting day", "day 1", "the first calendar day", 
            "the first day of the period", "the first day in sequence", 
            "the first day overall"
        ],
        "nth": [
            "{1}th day", "day {1}", "the {1}th calendar day",
            "the {1}th day of the period", "the {1}th day in sequence", 
            "the {1}th day overall", "day {1} in total", 
            "the {1}th day of the timeline", "the {1}th day in order", 
            "the {1}th day of the range"
        ]
    },
    "pwn": {
        "first": [
            "first week", "week one", "the initial week", "the first 7 days", 
            "the starting week", "week 1", "the first calendar week", 
            "the first week of the period", "the first week in sequence", 
            "the first week overall"
        ],
        "nth": [
            "{1}th week", "week {1}", "the {1}th calendar week",
            "the {1}th week of the period", "the {1}th week in sequence", 
            "the {1}th week overall", "week {1} in total", 
            "the {1}th week of the timeline", "the {1}th week in order", 
            "the {1}th week of the range"
        ]
    },
    "pmn": {
        "first": [
            "first month", "month one", "the initial month", "the first 30 days", 
            "the starting month", "month 1", "the first calendar month", 
            "the first month of the period", "the first month in sequence", 
            "the first month overall"
        ],
        "nth": [
            "{1}th month", "month {1}", "the {1}th calendar month",
            "the {1}th month of the period", "the {1}th month in sequence", 
            "the {1}th month overall", "month {1} in total", 
            "the {1}th month of the timeline", "the {1}th month in order", 
            "the {1}th month of the range"
        ]
    },
    "pyn": {
        "first": [
            "first year", "year one", "the initial year", "the first 12 months", 
            "the starting year", "year 1", "the first calendar year", 
            "the first year of the period", "the first year in sequence", 
            "the first year overall"
        ],
        "nth": [
            "{1}th year", "year {1}", "the {1}th calendar year",
            "the {1}th year of the period", "the {1}th year in sequence", 
            "the {1}th year overall", "year {1} in total", 
            "the {1}th year of the timeline", "the {1}th year in order", 
            "the {1}th year of the range"
        ]
    }
}

def period_to_text(annotation: str) -> str:
    is_period_bool, period_format = is_period(annotation)
    
    if not is_period_bool:
        raise ValueError(f"Cannot convert period to text: {annotation}")
    
    extracted_integers = extract_integers(annotation)
    patterns = PERIOD_PATTERNS.get(period_format)
    
    if not patterns:
        raise ValueError(f"Cannot convert period to text: {annotation}")
    
    # Handle range formats (pdi, pwi, pmi, pyi)
    if "range" in patterns:
        template = random.choice(patterns["range"])
        return template.format(extracted_integers[0], extracted_integers[1])
    
    # Handle nth formats (pdn, pwn, pmn, pyn)
    if "first" in patterns and "nth" in patterns:
        if extracted_integers[1] == 1:
            return random.choice(patterns["first"])
        else:
            template = random.choice(patterns["nth"])
            return template.format(extracted_integers[0], extracted_integers[1])
    
    # Handle singular/plural formats (pd, pw, pm, py)
    if "singular" in patterns and "plural" in patterns:
        if extracted_integers[0] == 1:
            return random.choice(patterns["singular"])
        else:
            template = random.choice(patterns["plural"])
            return template.format(extracted_integers[0])
    
    raise ValueError(f"Cannot convert period to text: {annotation}")