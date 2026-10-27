"""Val/test split selection for benchmark evaluation.

Datapoints are tagged with a per-item "split" field ("val" or "test") by
split_and_tune_alpha.py. Similarity caches are split-agnostic (built over the
full benchmark), so only metric computation is restricted to a split. This
helper is deliberately dependency-free (no torch / pandas) so lightweight
evaluators (BM25, SUTime) can filter to a split without importing the neural
stack.
"""

from typing import Any, Dict, List

SPLIT_KEY: str = "split"
VALID_SPLITS = ("val", "test")


def select_split_items(
    benchmark_data: List[Dict[str, Any]],
    split: str,
) -> List[Dict[str, Any]]:
    """Return only the benchmark items belonging to `split` ("val"/"test").

    If the benchmark carries no split tags (e.g. it predates the split step),
    fall back to evaluating on every item and warn, so existing caches keep
    working without an error.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'.")

    tagged = [item for item in benchmark_data if item.get(SPLIT_KEY) in VALID_SPLITS]
    if not tagged:
        print(
            f"  [warn] No val/test tags on this benchmark; evaluating on all "
            f"{len(benchmark_data)} items. Run `python split_and_tune_alpha.py tag` "
            f"to create splits."
        )
        return benchmark_data

    selected = [item for item in benchmark_data if item.get(SPLIT_KEY) == split]
    if not selected:
        present = sorted({item[SPLIT_KEY] for item in tagged})
        raise ValueError(
            f"No items tagged split='{split}' for this benchmark "
            f"({len(tagged)} tagged items; splits present: {present})."
        )
    print(f"  Evaluating on split='{split}': {len(selected)}/{len(benchmark_data)} items")
    return selected
