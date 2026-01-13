import random
from typing import List

DATE_INTERVAL_PATTERNS: List[str] = [
    "from {} to {}",
    "between {} and {}",
    "{} to {}",
    "{} through {}",
    "{} until {}",
    "{} till {}",
    
    "ranging from {} to {}",
    "within the range of {} to {}",
    "in the range of {} to {}",
    "across the range from {} to {}",
    
    "in the period from {} to {}",
    "during the period from {} to {}",
    "over the period from {} to {}",
    "throughout the period of {} to {}",
    "for the period {} to {}",
    "across the period from {} to {}",
    "within the period of {} to {}",
    
    "spanning from {} to {}",
    "spanning {} to {}",
    "spanning the dates {} to {}",
    "spanning between {} and {}",
    "spanning from {} through {}",
    
    "over the time frame of {} to {}",
    "within the time frame of {} to {}",
    "during the time frame from {} to {}",
    "across the time frame of {} to {}",
    "in the time frame {} to {}",
    
    "over the span of {} to {}",
    "throughout the duration of {} to {}",
    "during the duration from {} to {}",
    "across the duration of {} to {}",
    "for the duration {} to {}",
    
    "starting on {} and ending on {}",
    "beginning on {} and ending on {}",
    "commencing on {} and concluding on {}",
    "beginning on {} and concluding on {}",
    "starting {} and ending {}",
    "starting from {} and ending at {}",
    "beginning {} and finishing {}",
    "commencing {} and finishing {}",
    "starting at {} and finishing at {}",
    
    "extending from {} to {}",
    "extending from {} through {}",
    "extending between {} and {}",
    "stretching from {} to {}",
    "stretching from {} through {}",
    
    "covering the period from {} to {}",
    "covering {} to {}",
    "covering the dates {} to {}",
    "covering from {} to {}",
    "covering the time from {} to {}",
    "covering the span of {} to {}",
    
    "during the interval from {} to {}",
    "within the interval of {} to {}",
    "over the interval {} to {}",
    "across the interval from {} to {}",
    "in the interval {} to {}",
    
    "from the {} until the {}",
    "from {} up to {}",
    "from {} up until {}",
    "since {} until {}",
    "as of {} through {}",
    "as of {} until {}",
    
    "for the dates {} to {}",
    "across the dates from {} to {}",
    "between the dates of {} and {}",
    "from the date of {} to {}",
    "over the dates {} to {}",
    "during the dates {} to {}",
    
    "including {} to {}",
    "from {} inclusive to {}",
    "from {} to {} inclusive",
    "from {} through {} inclusive",
    
    "continuously from {} to {}",
    "running from {} to {}",
    "running from {} through {}",
    "ongoing from {} to {}",
    
    "throughout {} to {}",
    "throughout the time from {} to {}",
    "all through {} to {}",
    "all the way from {} to {}",
    
    "bounded by {} and {}",
    "delimited by {} and {}",
    "confined to {} to {}",
    "limited to {} through {}",
    
    "sequentially from {} to {}",
    "in sequence from {} to {}",
    
    "in the timeframe {} to {}",
    "within the timespan of {} to {}",
    "during the timespan from {} to {}",
    "over the course of {} to {}",
    "throughout the course of {} to {}",
]

def _date_to_interval_text(annotation: str) -> str:
    """Convert date annotation to text format optimized for interval patterns."""
    parts = annotation.split("-")
    
    months = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December"
    }
    
    months_short = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
    }
    
    seasons = {
        "SP": "Spring", "SU": "Summer", "FA": "Fall", "WI": "Winter"
    }
    
    if len(parts) == 1:
        return parts[0]
    
    if len(parts) == 2 and parts[1] in seasons:
        season = seasons[parts[1]]
        return random.choice([
            f"{season} {parts[0]}",
            f"{parts[0]} {season}"
        ])
    
    if len(parts) == 2 and parts[1] in months:
        month = random.choice([months[parts[1]], months_short[parts[1]]])
        return random.choice([
            f"{month} {parts[0]}",
            f"{parts[0]}-{parts[1]}"
        ])
    
    if len(parts) == 3:
        year, month, day = parts
        month_name = random.choice([months[month], months_short[month]])
        day_num = str(int(day))
        
        return random.choice([
            f"{month_name} {day_num}, {year}",
            f"{day_num} {month_name} {year}",
            f"{year}-{month}-{day}",
            f"{month}/{day}/{year}"
        ])
    
    return annotation

def interval_to_text(annotation: str) -> str:
    first_date, second_date = annotation.split(",")
    
    first_text: str = _date_to_interval_text(first_date)
    second_text: str = _date_to_interval_text(second_date)

    random_pattern: str = random.choice(DATE_INTERVAL_PATTERNS)
    
    return random_pattern.format(first_text, second_text)