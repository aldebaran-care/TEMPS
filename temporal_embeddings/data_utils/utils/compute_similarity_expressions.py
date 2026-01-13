from typing import List

from temporal_embeddings.data_utils.utils.dates.compute_similarity_dates import compute_similarity_dates_intervals
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.dates.compute_similarity_dates import compute_similarity_dates_intervals
from temporal_embeddings.data_utils.utils.offsets.offset_to_date import offset_to_date
from temporal_embeddings.data_utils.utils.refs.ref_to_date import ref_to_date
from temporal_embeddings.data_utils.utils.intervals.interval_to_date import interval_to_date
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date
from temporal_embeddings.data_utils.utils.refs.compute_similarity_refs import compute_similarity_refs
from temporal_embeddings.data_utils.utils.offsets.is_offset import is_offset
from temporal_embeddings.data_utils.utils.refs.is_ref import is_ref
from temporal_embeddings.data_utils.utils.intervals.is_interval import is_interval

def compute_similarity_expressions(first_expression: str, first_reference_date: str, second_expression: str, second_reference_date: str) -> float:
    first_is_date: bool = False
    second_is_date: bool = False

    first_expression_explicit_date: List[str]
    second_expression_explicit_date: List[str]

    if is_valid_date(first_expression)[0]:
        first_is_date = True
        first_expression_explicit_date = to_explicit_date(first_expression)
    
    elif is_offset(first_expression)[0]:
        first_is_date = True
        first_expression_explicit_date = offset_to_date(first_expression, first_reference_date)
    
    elif is_ref(first_expression)[0]:
        first_is_date = True
        first_expression_explicit_date = ref_to_date(first_expression, first_reference_date)
    
    elif is_interval(first_expression)[0]:
        first_is_date = True
        first_expression_explicit_date = interval_to_date(first_expression)

    else:
        raise ValueError(f"Cannot compute similarity for expressions: {first_expression}, {second_expression}")
    
    if is_valid_date(second_expression)[0]:
        second_is_date = True
        second_expression_explicit_date = to_explicit_date(second_expression)
    
    elif is_offset(second_expression)[0]:
        second_is_date = True
        second_expression_explicit_date = offset_to_date(second_expression, second_reference_date)
    
    elif is_ref(second_expression)[0]:
        second_is_date = True
        second_expression_explicit_date = ref_to_date(second_expression, second_reference_date)
    
    elif is_interval(second_expression)[0]:
        second_is_date = True
        second_expression_explicit_date = interval_to_date(second_expression)

    else:
        raise ValueError(f"Cannot compute similarity for expressions: {first_expression}, {second_expression}")
   
    if first_is_date and second_is_date:
        return compute_similarity_dates_intervals(first_expression_explicit_date, second_expression_explicit_date)
    
    else:
        return 0.0