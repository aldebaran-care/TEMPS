import random
import re
from typing import Dict, List, Optional, Tuple


# Month name -> ordinal (1-12), used to turn a month-year mention into a single
# absolute month index so that two dates can be compared / subtracted.
MONTH_TO_NUM = {
    name: i
    for i, names in enumerate(
        [
            ("Jan", "January"),
            ("Feb", "February"),
            ("Mar", "March"),
            ("Apr", "April"),
            ("May",),
            ("Jun", "June"),
            ("Jul", "July"),
            ("Aug", "August"),
            ("Sep", "September"),
            ("Oct", "October"),
            ("Nov", "November"),
            ("Dec", "December"),
        ],
        start=1,
    )
    for name in names
}


def parse_month_year(text: str) -> Optional[int]:
    """Parse a month-year mention into an absolute month index."""
    for match in re.finditer(r"\b([A-Za-z]{3,9}),?\s+(\d{1,4})\b", text.strip()):
        month = MONTH_TO_NUM.get(match.group(1).capitalize())
        if month is not None:
            return int(match.group(2)) * 12 + (month - 1)
    return None


def _positive_paragraphs(item: Dict) -> List[str]:
    paragraphs = item.get("paragraphs", [])
    answers = item.get("answer", [])
    return [paragraph for idx, paragraph in enumerate(paragraphs) if idx in answers]


def add_query_anchored_l1_negatives(
    l1_data: List[Dict],
    num_negative_samples: int = 15,
    hard_negative_pool: int = 20,
    seed: int = 42,
) -> Dict[str, int]:
    """Add TempReason L1 negatives sampled near the reference date in the query.

    The function first resets each item to its positive paragraph(s), so it is
    safe to call on either freshly processed L1 data or an already-augmented
    L1 file without appending duplicate negative batches.
    """
    random.seed(seed)
    hard_negative_pool = max(hard_negative_pool, num_negative_samples)

    for item in l1_data:
        positives = _positive_paragraphs(item)
        if not positives and item.get("paragraphs"):
            positives = [item["paragraphs"][0]]
        item["paragraphs"] = positives
        item["answer"] = list(range(len(positives)))

    all_l1_paragraphs = []
    for item in l1_data:
        if "paragraphs" in item and item["paragraphs"]:
            all_l1_paragraphs.extend(item["paragraphs"])

    seen_texts = set()
    dated_candidates: List[Tuple[int, str]] = []
    for paragraph in all_l1_paragraphs:
        if paragraph in seen_texts:
            continue
        value = parse_month_year(paragraph)
        if value is not None:
            dated_candidates.append((value, paragraph))
            seen_texts.add(paragraph)

    query_anchor_count = 0
    fallback_count = 0

    for item in l1_data:
        current_paragraphs = set(item.get("paragraphs", []))
        gold_text = item["paragraphs"][0] if item.get("paragraphs") else None
        gold_value = parse_month_year(gold_text) if gold_text else None
        query_value = parse_month_year(item.get("question", ""))

        if query_value is None or not dated_candidates:
            available_negatives = [
                p for p in all_l1_paragraphs if p not in current_paragraphs
            ]
            num_samples = min(num_negative_samples, len(available_negatives))
            negative_paragraphs = random.sample(available_negatives, num_samples)
            fallback_count += 1
        else:
            ranked = sorted(
                dated_candidates, key=lambda candidate: (abs(candidate[0] - query_value), candidate[0])
            )
            nearest = []
            for value, text in ranked:
                if value == gold_value or text in current_paragraphs:
                    continue
                nearest.append(text)
                if len(nearest) >= hard_negative_pool:
                    break

            num_samples = min(num_negative_samples, len(nearest))
            negative_paragraphs = random.sample(nearest, num_samples)
            query_anchor_count += 1

        item["paragraphs"].extend(negative_paragraphs)

    return {
        "all_paragraphs": len(all_l1_paragraphs),
        "dated_candidates": len(dated_candidates),
        "query_anchor_count": query_anchor_count,
        "fallback_count": fallback_count,
    }
