import random
from typing import List

import numpy as np

from temporal_embeddings.data_utils.utils.dates.dates_settings import START_DATE, END_DATE
from temporal_embeddings.data_utils.utils.dates.generate_random_date import generate_random_date
from temporal_embeddings.data_utils.utils.offsets.generate_random_offset import generate_random_offset
from temporal_embeddings.data_utils.utils.refs.generate_random_ref import generate_random_ref
from temporal_embeddings.data_utils.utils.intervals.generate_random_interval import generate_random_interval
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.offsets.is_offset import is_offset
from temporal_embeddings.data_utils.utils.refs.is_ref import is_ref
from temporal_embeddings.data_utils.utils.intervals.is_interval import is_interval
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date

def generate_random_temporal_expression(probabilities: List[float], close: bool, expression: str, current_date: str) -> str:
    """
    Generate a random TimeML expression based on specified probabilities for each type. The order of probabilities is: [date, offset, reference, interval].
    """
    if not close:
        random_temporal_expression_type: int = np.random.choice(np.arange(len(probabilities)), p=np.array(probabilities))
        
        if random_temporal_expression_type == 0:
            return generate_random_date(START_DATE, END_DATE, granularity="")
        
        if random_temporal_expression_type == 1:
            return generate_random_offset(close=False, value="", type="")
        
        if random_temporal_expression_type == 2:
            return generate_random_ref()
        
        return generate_random_interval(START_DATE, END_DATE, granularity="")
    
    else:
        random_generate_date: bool = bool(random.getrandbits(1))

        if is_valid_date(expression)[0]:
            year: int = int(expression.split("-")[0])

            return generate_random_date(f"{year}-01-01", f"{year}-12-31", granularity="day" if random_generate_date else "")
        
        if is_interval(expression)[0]:
            start_date, end_date = expression.split(",")
            start_date = to_explicit_date(start_date)[0]
            end_date = to_explicit_date(end_date)[-1]
            
            return generate_random_interval(start_date, end_date, granularity="day" if random_generate_date else "")
        
        if is_offset(expression)[0]:
            if not random_generate_date:
                return generate_random_offset(close=True, value=expression, type=is_offset(expression)[1])
            else:
                return generate_random_date(current_date, current_date, granularity="day")
        
        if is_ref(expression)[0]:
            if random_generate_date:
                return generate_random_ref()
            else:
                return generate_random_date(current_date, current_date, granularity="day")
        
        raise ValueError(f"Cannot generate close temporal expression for: {expression}")