"""Temporal similarity functions over inclusive date intervals.

Dates are grounded as inclusive integer day intervals, e.g. a single day is
``[d, d]`` and a year is ``[YYYY-01-01, YYYY-12-31]``. TSF V2 therefore uses
``+ 1`` when computing interval lengths, intersections, unions, and gaps.
"""

import math
from typing import List, Literal, Tuple

from temporal_embeddings.data_utils.utils.compute_interval_distance import (
    compute_interval_distance_date,
    days_since_base,
)
from temporal_embeddings.data_utils.utils.dates.compute_distance_dates import (
    compute_distance_dates,
    compute_distance_dates_same_type,
)
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.dates.is_in import is_in
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date

Interval = Tuple[int, int]
TSFMode = Literal["inference", "train"]
TSFVersion = Literal["v1", "v2"]


def _date_list_to_interval(dates: List[str]) -> Interval:
    if len(dates) == 1:
        dates = dates * 2

    if len(dates) != 2:
        raise ValueError(f"Expected one date or an interval boundary pair: {dates}")

    start, end = sorted((days_since_base(dates[0]), days_since_base(dates[1])))
    return start, end


def _tsf_v1_dates(first_date: str, second_date: str) -> float:
    first_is_date, first_date_type = is_valid_date(first_date)
    second_is_date, second_date_type = is_valid_date(second_date)

    if first_is_date and second_is_date:
        if first_date == second_date:
            return 1.0

        elif first_date_type == second_date_type:
            distance = compute_distance_dates_same_type(first_date=first_date, second_date=second_date, date_type=first_date_type)
            distance = distance**2 if distance < 100 else distance

            return 0.4 / distance

        elif first_date_type != second_date_type and is_in(first_date=first_date, first_date_type=first_date_type, second_date=second_date, second_date_type=second_date_type):
            return 0.5

        else:
            distance = compute_distance_dates(first_date, first_date_type, second_date, second_date_type)
            distance = distance**2 if distance < 100 else distance

            return 0.1 / distance

    else:
        raise ValueError(f"Both inputs must be valid dates for similarity computation: {first_date}, {second_date}")


def _tsf_v1(first_interval: List[str], second_interval: List[str]) -> float:
    if len(first_interval) == 1 and len(second_interval) == 1:
        return _tsf_v1_dates(first_interval[0], second_interval[0])

    if len(first_interval) == 1:
        first_interval = first_interval * 2

    if len(second_interval) == 1:
        second_interval = second_interval * 2

    distance, overlap = compute_interval_distance_date(first_interval, second_interval)

    if overlap:
        return distance

    return 0.5 / distance


def _tsf_v2_inference(first_interval: Interval, second_interval: Interval) -> float:
    a_q, b_q = first_interval
    a_d, b_d = second_interval

    if a_q == a_d and b_q == b_d:
        return 1.0

    inter = max(0, min(b_q, b_d) - max(a_q, a_d) + 1)
    if inter == 0:
        return 0.0

    union = max(b_q, b_d) - min(a_q, a_d) + 1
    iou = inter / union

    q_in_d = a_d <= a_q and b_q <= b_d
    d_in_q = a_q <= a_d and b_d <= b_q
    if q_in_d or d_in_q:
        return (1.0 + iou) / 2.0

    return iou


def _tsf_v2_train(first_interval: Interval, second_interval: Interval, epsilon: float = 1e-3) -> float:
    score = _tsf_v2_inference(first_interval, second_interval)
    if score > 0.0:
        return score

    a_q, b_q = first_interval
    a_d, b_d = second_interval
    gap = max(a_q, a_d) - min(b_q, b_d) - 1
    length = (b_q - a_q + 1) + (b_d - a_d + 1) + 1

    return epsilon * math.exp(-((gap / length) ** 2))


def tsf(
    first_interval: List[str],
    second_interval: List[str],
    *,
    mode: TSFMode = "train",
    version: TSFVersion = "v2",
    epsilon: float = 1e-3,
) -> float:
    if version == "v1":
        return _tsf_v1(first_interval, second_interval)

    first_grounded_interval = _date_list_to_interval(first_interval)
    second_grounded_interval = _date_list_to_interval(second_interval)

    if mode == "inference":
        return _tsf_v2_inference(first_grounded_interval, second_grounded_interval)

    return _tsf_v2_train(first_grounded_interval, second_grounded_interval, epsilon=epsilon)


def compute_similarity_dates(
    first_date: str,
    second_date: str,
    *,
    mode: TSFMode = "train",
    version: TSFVersion = "v2",
    epsilon: float = 1e-3,
) -> float:
    if version == "v1":
        return _tsf_v1_dates(first_date, second_date)

    first_is_date = is_valid_date(first_date)[0]
    second_is_date = is_valid_date(second_date)[0]

    if not first_is_date or not second_is_date:
        raise ValueError(f"Both inputs must be valid dates for similarity computation: {first_date}, {second_date}")

    return tsf(
        to_explicit_date(first_date),
        to_explicit_date(second_date),
        mode=mode,
        version=version,
        epsilon=epsilon,
    )


def compute_similarity_dates_intervals(
    first_date: List[str],
    second_date: List[str],
    *,
    mode: TSFMode = "train",
    version: TSFVersion = "v2",
    epsilon: float = 1e-3,
) -> float:
    return tsf(first_date, second_date, mode=mode, version=version, epsilon=epsilon)
