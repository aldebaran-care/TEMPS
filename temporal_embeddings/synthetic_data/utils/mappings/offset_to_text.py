import re
import random
from typing import Dict, List

OFFSET_PATTERNS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "day": {
        "this_offset": {
            "past_singular": ["last day", "yesterday", "the previous day", "one day ago", "the day before"],
            "past_plural": [
                "last {0} days", "the previous {0} days", "{0} days ago", 
                "the past {0} days", "the preceding {0} days"
            ],
            "future_singular": ["next day", "tomorrow", "the following day", "one day later", "the day after"],
            "future_plural": [
                "next {0} days", "the following {0} days", "{0} days later", 
                "the upcoming {0} days", "the succeeding {0} days"
            ]
        },
        "offset": {
            "past_singular": ["yesterday", "the previous day", "one day ago", "the day before", "a day earlier"],
            "past_plural": [
                "{0} days ago", "the past {0} days", "the preceding {0} days", 
                "{0} days earlier", "{0} days in the past"
            ],
            "future_singular": ["tomorrow", "the next day", "one day later", "the day after", "a day ahead"],
            "future_plural": [
                "{0} days later", "the next {0} days", "the following {0} days", 
                "{0} days in the future", "{0} days ahead"
            ]
        },
        "this": {
            "singular": ["today", "this day", "the current day", "the present day", "this very day"],
            "plural": [
                "these {0} days", "the current {0} days", "the present {0} days", 
                "these very {0} days", "this span of {0} days"
            ]
        },
        "immediate": {
            "past_singular": ["the last day", "the previous day", "one day ago", "the day before", "a day earlier"],
            "past_plural": [
                "these past {0} days", "the preceding {0} days", "{0} days ago", 
                "{0} days earlier", "{0} days in the past"
            ],
            "future_singular": ["the next day", "the following day", "one day later", "the day after", "a day ahead"],
            "future_plural": [
                "these next {0} days", "the succeeding {0} days", "{0} days later", 
                "{0} days in the future", "{0} days ahead"
            ]
        }
    },
    "week": {
        "this_offset": {
            "past_singular": ["last week", "the previous week", "a week ago", "the past week", "the preceding week"],
            "past_plural": [
                "last {0} weeks", "the previous {0} weeks", "{0} weeks ago", 
                "the past {0} weeks", "the preceding {0} weeks"
            ],
            "future_singular": ["next week", "the following week", "a week later", "the upcoming week", "the succeeding week"],
            "future_plural": [
                "next {0} weeks", "the following {0} weeks", "{0} weeks later", 
                "the upcoming {0} weeks", "the succeeding {0} weeks"
            ]
        },
        "offset": {
            "past_singular": ["a week ago", "the previous week", "one week ago", "the past week", "the preceding week"],
            "past_plural": [
                "{0} weeks ago", "the past {0} weeks", "the preceding {0} weeks", 
                "{0} weeks earlier", "{0} weeks in the past"
            ],
            "future_singular": ["a week later", "the next week", "one week later", "the following week", "a week ahead"],
            "future_plural": [
                "{0} weeks later", "the next {0} weeks", "the following {0} weeks", 
                "{0} weeks in the future", "{0} weeks ahead"
            ]
        },
        "this": {
            "singular": ["this week", "the current week", "the present week", "this very week", "this span of a week"],
            "plural": [
                "these {0} weeks", "the current {0} weeks", "the present {0} weeks", 
                "these very {0} weeks", "this span of {0} weeks"
            ]
        },
        "immediate": {
            "past_singular": ["the last week", "the previous week", "one week ago", "the week before", "a week earlier"],
            "past_plural": [
                "these past {0} weeks", "the preceding {0} weeks", "{0} weeks ago", 
                "{0} weeks earlier", "{0} weeks in the past"
            ],
            "future_singular": ["the next week", "the following week", "one week later", "the week after", "a week ahead"],
            "future_plural": [
                "these next {0} weeks", "the succeeding {0} weeks", "{0} weeks later", 
                "{0} weeks in the future", "{0} weeks ahead"
            ]
        }
    },
    "month": {
        "this_offset": {
            "past_singular": ["last month", "the previous month", "a month ago", "the past month", "the preceding month"],
            "past_plural": [
                "last {0} months", "the previous {0} months", "{0} months ago", 
                "the past {0} months", "the preceding {0} months"
            ],
            "future_singular": ["next month", "the following month", "a month later", "the upcoming month", "the succeeding month"],
            "future_plural": [
                "next {0} months", "the following {0} months", "{0} months later", 
                "the upcoming {0} months", "the succeeding {0} months"
            ]
        },
        "offset": {
            "past_singular": ["a month ago", "the previous month", "one month ago", "the past month", "the preceding month"],
            "past_plural": [
                "{0} months ago", "the past {0} months", "the preceding {0} months", 
                "{0} months earlier", "{0} months in the past"
            ],
            "future_singular": ["a month later", "the next month", "one month later", "the following month", "a month ahead"],
            "future_plural": [
                "{0} months later", "the next {0} months", "the following {0} months", 
                "{0} months in the future", "{0} months ahead"
            ]
        },
        "this": {
            "singular": ["this month", "the current month", "the present month", "this very month", "this span of a month"],
            "plural": [
                "these {0} months", "the current {0} months", "the present {0} months", 
                "these very {0} months", "this span of {0} months"
            ]
        },
        "immediate": {
            "past_singular": ["the last month", "the previous month", "one month ago", "the month before", "a month earlier"],
            "past_plural": [
                "these past {0} months", "the preceding {0} months", "{0} months ago", 
                "{0} months earlier", "{0} months in the past"
            ],
            "future_singular": ["the next month", "the following month", "one month later", "the month after", "a month ahead"],
            "future_plural": [
                "these next {0} months", "the succeeding {0} months", "{0} months later", 
                "{0} months in the future", "{0} months ahead"
            ]
        }
    },
    "year": {
        "this_offset": {
            "past_singular": ["last year", "the previous year", "a year ago", "the past year", "the preceding year"],
            "past_plural": [
                "last {0} years", "the previous {0} years", "{0} years ago", 
                "the past {0} years", "the preceding {0} years"
            ],
            "future_singular": ["next year", "the following year", "a year later", "the upcoming year", "the succeeding year"],
            "future_plural": [
                "next {0} years", "the following {0} years", "{0} years later", 
                "the upcoming {0} years", "the succeeding {0} years"
            ]
        },
        "offset": {
            "past_singular": ["a year ago", "the previous year", "one year ago", "the past year", "the preceding year"],
            "past_plural": [
                "{0} years ago", "the past {0} years", "the preceding {0} years", 
                "{0} years earlier", "{0} years in the past"
            ],
            "future_singular": ["a year later", "the next year", "one year later", "the following year", "a year ahead"],
            "future_plural": [
                "{0} years later", "the next {0} years", "the following {0} years", 
                "{0} years in the future", "{0} years ahead"
            ]
        },
        "this": {
            "singular": ["this year", "the current year", "the present year", "this very year", "this span of a year"],
            "plural": [
                "these {0} years", "the current {0} years", "the present {0} years", 
                "these very {0} years", "this span of {0} years"
            ]
        },
        "immediate": {
            "past_singular": ["the last year", "the previous year", "one year ago", "the year before", "a year earlier"],
            "past_plural": [
                "these past {0} years", "the preceding {0} years", "{0} years ago", 
                "{0} years earlier", "{0} years in the past"
            ],
            "future_singular": ["the next year", "the following year", "one year later", "the year after", "a year ahead"],
            "future_plural": [
                "these next {0} years", "the succeeding {0} years", "{0} years later", 
                "{0} years in the future", "{0} years ahead"
            ]
        }
    }
}

PATTERN_CONFIGS: Dict[str, Dict[str, str]] = {
    "this_offset": {
        "day": r"^THIS P(-?\d+)D OFFSET P(-?\d+)D$",
        "week": r"^THIS P(-?\d+)W OFFSET P(-?\d+)W$",
        "month": r"^THIS P(-?\d+)M OFFSET P(-?\d+)M$",
        "year": r"^THIS P(-?\d+)Y OFFSET P(-?\d+)Y$"
    },
    "offset": {
        "day": r'^OFFSET P(-?\d+)D$',
        "week": r'^OFFSET P(-?\d+)W$',
        "month": r'^OFFSET P(-?\d+)M$',
        "year": r'^OFFSET P(-?\d+)Y$'
    },
    "this": {
        "day": r'^THIS P(-?\d+)D$',
        "week": r'^THIS P(-?\d+)W$',
        "month": r'^THIS P(-?\d+)M$',
        "year": r'^THIS P(-?\d+)Y$'
    },
    "immediate": {
        "day": r'^(?:NEXT_IMMEDIATE|PREV_IMMEDIATE) P(-?\d+)D$',
        "week": r'^(?:NEXT_IMMEDIATE|PREV_IMMEDIATE) P(-?\d+)W$',
        "month": r'^(?:NEXT_IMMEDIATE|PREV_IMMEDIATE) P(-?\d+)M$',
        "year": r'^(?:NEXT_IMMEDIATE|PREV_IMMEDIATE) P(-?\d+)Y$'
    }
}

def _get_text_from_patterns(unit: str, pattern_type: str, value: int, is_past: bool) -> str:
    """Helper function to get text from pattern dictionary."""
    patterns = OFFSET_PATTERNS[unit][pattern_type]
    value_abs = abs(value)
    
    if pattern_type == "this":
        if value_abs == 1:
            return random.choice(patterns["singular"])
        else:
            template = random.choice(patterns["plural"])
            return template.format(value_abs)
    
    if is_past:
        if value_abs == 1:
            return random.choice(patterns["past_singular"])
        else:
            template = random.choice(patterns["past_plural"])
            return template.format(value_abs)
    else:
        if value_abs == 1:
            return random.choice(patterns["future_singular"])
        else:
            template = random.choice(patterns["future_plural"])
            return template.format(value_abs)

def offset_to_text(annotation: str) -> str:
    for unit in ["day", "week", "month", "year"]:
        if match := re.search(PATTERN_CONFIGS["this_offset"][unit], annotation):
            first_integer = int(match.group(1))
            second_integer = int(match.group(2))
            
            if abs(first_integer) == abs(second_integer):
                is_past = second_integer < 0
                return _get_text_from_patterns(unit, "this_offset", second_integer, is_past)
    
    for unit in ["day", "week", "month", "year"]:
        if match := re.search(PATTERN_CONFIGS["offset"][unit], annotation):
            first_integer = int(match.group(1))
            is_past = first_integer < 0
            return _get_text_from_patterns(unit, "offset", first_integer, is_past)
    
    for unit in ["day", "week", "month", "year"]:
        if match := re.search(PATTERN_CONFIGS["this"][unit], annotation):
            first_integer = int(match.group(1))
            if first_integer > 0:
                return _get_text_from_patterns(unit, "this", first_integer, is_past=False)
    
    for unit in ["day", "week", "month", "year"]:
        if match := re.search(PATTERN_CONFIGS["immediate"][unit], annotation):
            first_integer = int(match.group(1))
            
            if "PREV_IMMEDIATE" in annotation:
                return _get_text_from_patterns(unit, "immediate", first_integer, is_past=True)
            
            if "NEXT_IMMEDIATE" in annotation:
                return _get_text_from_patterns(unit, "immediate", first_integer, is_past=False)
    
    raise ValueError(f"Cannot convert offset to text: {annotation}")