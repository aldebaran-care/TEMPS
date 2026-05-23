"""Asymmetric temporal similarity over inclusive date intervals.

Each grounded TimeML annotation `I = [a, b]` (inclusive integer day indices)
is mapped to the moment-matched Gaussian of the continuous uniform
`U[a, b + 1)`:

    mu_I    = (a + b + 1) / 2
    sigma_I = (b - a + 1) / sqrt(12)

The temporal distance from query interval `I_q` to document interval `I_d`
is the asymmetric KL divergence between the two Gaussians:

    KL( N(mu_q, sigma_q^2) || N(mu_d, sigma_d^2) )

The public similarity API returns `1 / (1 + KL)`, matching the convention of
`asymmetrical_kl_sim` in `temporal_embeddings/utils/similarity.py` so that
the label geometry mirrors the model head's output geometry.
"""

import math
from typing import List, Tuple

from temporal_embeddings.data_utils.utils.compute_interval_distance import days_since_base
from temporal_embeddings.data_utils.utils.dates.is_date import is_valid_date
from temporal_embeddings.data_utils.utils.dates.to_explicit_date import to_explicit_date

Interval = Tuple[int, int]


def _date_list_to_interval(dates: List[str]) -> Interval:
    if len(dates) == 1:
        dates = dates * 2

    if len(dates) != 2:
        raise ValueError(f"Expected one date or an interval boundary pair: {dates}")

    start = days_since_base(dates[0])
    end = days_since_base(dates[1])

    if start > end:
        raise ValueError(f"Interval has start after end: {dates}")

    return start, end


def _gaussian_of_interval(interval: Interval) -> Tuple[float, float]:
    a, b = interval
    length = b - a + 1
    mean = (a + b + 1) / 2.0
    variance = (length ** 2) / 12.0
    return mean, variance


def temporal_kl(first_interval: Interval, second_interval: Interval) -> float:
    """Asymmetric KL divergence from `first_interval` to `second_interval`."""
    mu_q, var_q = _gaussian_of_interval(first_interval)
    mu_d, var_d = _gaussian_of_interval(second_interval)

    return (
        0.5 * math.log(var_d / var_q)
        + (var_q + (mu_q - mu_d) ** 2) / (2.0 * var_d)
        - 0.5
    )


def tsf(first_interval: List[str], second_interval: List[str]) -> float:
    """Asymmetric similarity in (0, 1]: 1 / (1 + KL(q -> d))."""
    first_grounded = _date_list_to_interval(first_interval)
    second_grounded = _date_list_to_interval(second_interval)

    return 1.0 / (1.0 + temporal_kl(first_grounded, second_grounded))


def compute_similarity_dates(first_date: str, second_date: str) -> float:
    if not is_valid_date(first_date)[0] or not is_valid_date(second_date)[0]:
        raise ValueError(
            f"Both inputs must be valid dates for similarity computation: {first_date}, {second_date}"
        )

    return tsf(to_explicit_date(first_date), to_explicit_date(second_date))


def compute_similarity_dates_intervals(
    first_date: List[str],
    second_date: List[str],
) -> float:
    return tsf(first_date, second_date)
