from typing import List

from temporal_embeddings.data_utils.utils.dates.compute_similarity_dates import compute_similarity_dates_intervals
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.offsets.offset_to_date import offset_to_date
from temporal_embeddings.data_utils.utils.refs.ref_to_date import ref_to_date
from temporal_embeddings.data_utils.utils.intervals.interval_to_date import interval_to_date
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date
from temporal_embeddings.data_utils.utils.offsets.is_offset import is_offset
from temporal_embeddings.data_utils.utils.refs.is_ref import is_ref
from temporal_embeddings.data_utils.utils.intervals.is_interval import is_interval


def _to_explicit(expression: str, reference_date: str) -> List[str]:
    if is_valid_date(expression)[0]:
        return to_explicit_date(expression)

    if is_offset(expression)[0]:
        return offset_to_date(expression, reference_date)

    if is_ref(expression)[0]:
        return ref_to_date(expression, reference_date)

    if is_interval(expression)[0]:
        return interval_to_date(expression)

    raise ValueError(f"Cannot resolve temporal expression: {expression}")


def compute_similarity_expressions(
    first_expression: str,
    first_reference_date: str,
    second_expression: str,
    second_reference_date: str,
) -> float:
    first_explicit = _to_explicit(first_expression, first_reference_date)
    second_explicit = _to_explicit(second_expression, second_reference_date)

    return compute_similarity_dates_intervals(first_explicit, second_explicit)
