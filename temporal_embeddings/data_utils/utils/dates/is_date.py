import re
from typing import Tuple
from datetime import datetime

from temporal_embeddings.data_utils.utils.dates.dates_settings import START_DATE, END_DATE

def is_valid_date(text: str) -> Tuple[bool, str]:
    """
    Check if the given text is a valid date in one of the accepted formats.
    Returns a tuple (is_valid: bool, date_type: str).
    """
    start_date_obj: datetime = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date_obj: datetime = datetime.strptime(END_DATE, "%Y-%m-%d")

    if re.match(r'^\d{4}-\d{2}-\d{2}$', text):
        date_obj: datetime = datetime.strptime(text, "%Y-%m-%d")
        
        if date_obj < start_date_obj or date_obj > end_date_obj:
            return False, f"Date \"{text}\" out of range ({START_DATE} to {END_DATE})"
        
        return True, "yyyy-mm-dd"
    
    elif re.match(r'^\d{4}-\d{2}$', text):
        date_obj: datetime = datetime.strptime(text, "%Y-%m")

        if date_obj < start_date_obj or date_obj > end_date_obj:
            return False, f"Date \"{text}\" out of range ({START_DATE} to {END_DATE})"

        return True, "yyyy-mm"
    
    elif re.match(r'^\d{4}$', text):
        date_obj: datetime = datetime.strptime(text, "%Y")

        if date_obj < start_date_obj or date_obj > end_date_obj:
            return False, f"Date \"{text}\" out of range ({START_DATE} to {END_DATE})"

        return True, "yyyy"
    
    elif re.match(r'^\d{4}-(?:SU|WI|FA|SP)$', text):
        return True, "yyyy-s"
    
    else:
        return False, "Invalid format"