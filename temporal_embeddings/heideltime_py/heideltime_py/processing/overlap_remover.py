"""
Overlap Remover: removes duplicate and overlapping timex expressions.
Port of HeidelTime.deleteOverlappedTimexesPreprocessing/Postprocessing.
"""

from ..models import Timex3


def remove_overlaps_preprocessing(timexes: list[Timex3]) -> list[Timex3]:
    """
    Pre-processing overlap removal:
    - Remove shorter timex if fully contained in a longer one
    - For same-span timexes: prefer non-UNDEF, prefer explicit, prefer with emptyValue
    """
    to_remove = set()

    for idx1, t1 in enumerate(timexes):
        for idx2, t2 in enumerate(timexes):
            if idx1 == idx2:
                continue

            # t1 is strictly contained in t2
            if (t1.begin >= t2.begin and t1.end < t2.end) or (
                t1.begin > t2.begin and t1.end <= t2.end
            ):
                to_remove.add(idx1)

            # t2 is strictly contained in t1
            elif (t2.begin >= t1.begin and t2.end < t1.end) or (
                t2.begin > t1.begin and t2.end <= t1.end
            ):
                to_remove.add(idx2)

            # Same span
            if idx1 != idx2 and t1.begin == t2.begin and t1.end == t2.end:
                # Prefer non-UNDEF
                if t1.timex_value.startswith("UNDEF") and not t2.timex_value.startswith("UNDEF"):
                    to_remove.add(idx1)
                elif (
                    not t1.timex_value.startswith("UNDEF")
                    and t2.timex_value.startswith("UNDEF")
                    or t1.found_by_rule.endswith("explicit")
                    and not t2.found_by_rule.endswith("explicit")
                    or t2.empty_value == ""
                    and t1.empty_value != ""
                ):
                    to_remove.add(idx2)
                # Remove lower ID (earlier rule)
                elif idx1 not in to_remove and idx2 not in to_remove:
                    id1 = int(t1.timex_id[1:]) if t1.timex_id.startswith("t") else 0
                    id2 = int(t2.timex_id[1:]) if t2.timex_id.startswith("t") else 0
                    if id1 < id2:
                        to_remove.add(idx1)

    return [t for i, t in enumerate(timexes) if i not in to_remove]


def remove_overlaps_postprocessing(timexes: list[Timex3]) -> list[Timex3]:
    """
    Post-processing overlap removal:
    - Find all groups of overlapping timexes
    - Keep the one with the longest/most specific value from each group
    """
    if not timexes:
        return timexes

    # Filter out REMOVE values
    timexes = [t for t in timexes if t.timex_value != "REMOVE"]

    # Build overlap groups
    n = len(timexes)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i] or timexes[i].timex_type == "TEMPONYM":
            continue

        group = [i]
        visited[i] = True

        # Find all timexes overlapping with this group
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if visited[j] or timexes[j].timex_type == "TEMPONYM":
                    continue
                # Check overlap with any member of the group
                for gi in group:
                    t1 = timexes[gi]
                    t2 = timexes[j]
                    if _overlaps(t1, t2):
                        group.append(j)
                        visited[j] = True
                        changed = True
                        break

        if len(group) > 1:
            groups.append(group)

    # For each group, select the best timex
    indices_to_remove = set()
    for group in groups:
        group_timexes = [(idx, timexes[idx]) for idx in group]

        # Filter out REMOVE values
        group_timexes = [(idx, t) for idx, t in group_timexes if t.timex_value != "REMOVE"]
        if not group_timexes:
            continue

        # Find the best timex
        best_idx, best = group_timexes[0]
        all_same_type = all(t.timex_type == best.timex_type for _, t in group_timexes)
        all_date_or_time = all(t.timex_type in ("DATE", "TIME") for _, t in group_timexes)

        for idx, t in group_timexes[1:]:
            if all_same_type and all_date_or_time:
                # Prefer BCADhint
                if (
                    "-BCADhint" in t.found_by_rule
                    or "relative" not in t.found_by_rule
                    and "relative" in best.found_by_rule
                ):
                    best_idx, best = idx, t
                elif len(t.timex_value) == len(best.timex_value):
                    if t.begin < best.begin:
                        best_idx, best = idx, t
                elif len(t.timex_value) > len(best.timex_value):
                    best_idx, best = idx, t
            else:
                if len(t.timex_value) > len(best.timex_value):
                    best_idx, best = idx, t

        # Mark all except best for removal
        for idx, _ in group_timexes:
            if idx != best_idx:
                indices_to_remove.add(idx)

        # If same type, extend best to cover the whole group
        if all_same_type and all_date_or_time:
            combined_begin = min(t.begin for _, t in group_timexes)
            combined_end = max(t.end for _, t in group_timexes)
            best.begin = combined_begin
            best.end = combined_end

    return [t for i, t in enumerate(timexes) if i not in indices_to_remove]


def _overlaps(t1: Timex3, t2: Timex3) -> bool:
    """Check if two timexes overlap."""
    return (
        (t1.begin <= t2.begin < t1.end)
        or (t2.begin <= t1.begin < t2.end)
        or (t2.begin <= t1.begin and t1.end <= t2.end)
        or (t1.begin <= t2.begin and t2.end <= t1.end)
    )
