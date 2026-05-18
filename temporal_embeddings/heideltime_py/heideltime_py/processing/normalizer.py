"""
Normalizer: applies rule functions to produce TIMEX3 values.
Port of HeidelTime.applyRuleFunctions().
"""

import re

from ..resources.normalization_manager import NormalizationManager


def apply_rule_functions(
    tonormalize: str, match: re.Match, norm_manager: NormalizationManager, language: str = "english"
) -> str | None:
    """
    Apply normalization functions to produce a TIMEX3 value string.

    Handles:
    - %funcName(group(N))  -> lookup match.group(N) in normalization map
    - group(N)             -> replace with match.group(N)
    - %SUBSTRING%(str,start,end)
    - %LOWERCASE%(str)
    - %UPPERCASE%(str)
    - %SUM%(a,b)
    - %funcName(literal)   -> lookup literal in normalization map
    - %CHINESENUMBERS%(str)
    """
    pa_norm = re.compile(r"%([A-Za-z0-9]+?)\(group\(([0-9]+)\)\)")
    pa_group = re.compile(r"group\(([0-9]+)\)")
    pa_substring = re.compile(r"%SUBSTRING%\((.*?),([0-9]+),([0-9]+)\)")
    pa_lowercase = re.compile(r"%LOWERCASE%\((.*?)\)")
    pa_uppercase = re.compile(r"%UPPERCASE%\((.*?)\)")
    pa_sum = re.compile(r"%SUM%\((.*?),(.*?)\)")
    pa_norm_no_group = re.compile(r"%([A-Za-z0-9]+?)\((.*?)\)")
    pa_chinese = re.compile(r"%CHINESENUMBERS%\((.*?)\)")

    max_iterations = 50
    iteration = 0

    while ("%" in tonormalize or "group" in tonormalize) and iteration < max_iterations:
        iteration += 1
        changed = False

        # Replace %funcName(group(N))
        for mr in pa_norm.finditer(tonormalize):
            func_name = mr.group(1)
            group_num = int(mr.group(2))
            try:
                group_val = match.group(group_num)
            except IndexError:
                group_val = None

            if group_val is not None:
                # Normalize whitespace in the matched text
                part = re.sub(r"[\n\s]+", " ", group_val)
                norm_map = norm_manager.get_normalization(func_name)
                if norm_map is not None and norm_map.contains_key(part):
                    replacement = norm_map.get(part, "")
                    tonormalize = tonormalize.replace(mr.group(0), replacement, 1)
                    changed = True
                elif func_name.startswith("Temponym") or (func_name and "Temponym" in func_name):
                    return None
            else:
                tonormalize = tonormalize.replace(mr.group(0), "", 1)
                changed = True
            break  # re-scan after replacement

        if changed:
            continue

        # Replace group(N)
        for mr in pa_group.finditer(tonormalize):
            group_num = int(mr.group(1))
            try:
                group_val = match.group(group_num) or ""
            except IndexError:
                group_val = ""
            tonormalize = tonormalize.replace(mr.group(0), group_val, 1)
            changed = True
            break

        if changed:
            continue

        # Replace %SUBSTRING%(str,start,end)
        for mr in pa_substring.finditer(tonormalize):
            s = mr.group(1)
            start = int(mr.group(2))
            end = int(mr.group(3))
            tonormalize = tonormalize.replace(mr.group(0), s[start:end], 1)
            changed = True
            break

        if changed:
            continue

        # Replace %LOWERCASE%
        if language != "arabic":
            for mr in pa_lowercase.finditer(tonormalize):
                tonormalize = tonormalize.replace(mr.group(0), mr.group(1).lower(), 1)
                changed = True
                break

            if changed:
                continue

            for mr in pa_uppercase.finditer(tonormalize):
                tonormalize = tonormalize.replace(mr.group(0), mr.group(1).upper(), 1)
                changed = True
                break

            if changed:
                continue

        # Replace %SUM%(a,b)
        for mr in pa_sum.finditer(tonormalize):
            try:
                result = int(mr.group(1)) + int(mr.group(2))
                tonormalize = tonormalize.replace(mr.group(0), str(result), 1)
                changed = True
            except ValueError:
                pass
            break

        if changed:
            continue

        # Replace %CHINESENUMBERS%
        for mr in pa_chinese.finditer(tonormalize):
            chinese_map = {
                "零": "0",
                "０": "0",
                "0": "0",
                "一": "1",
                "１": "1",
                "1": "1",
                "二": "2",
                "２": "2",
                "2": "2",
                "三": "3",
                "３": "3",
                "3": "3",
                "四": "4",
                "４": "4",
                "4": "4",
                "五": "5",
                "５": "5",
                "5": "5",
                "六": "6",
                "６": "6",
                "6": "6",
                "七": "7",
                "７": "7",
                "7": "7",
                "八": "8",
                "８": "8",
                "8": "8",
                "九": "9",
                "９": "9",
                "9": "9",
            }
            out = ""
            for ch in mr.group(1):
                out += chinese_map.get(ch, ch)
            tonormalize = tonormalize.replace(mr.group(0), out, 1)
            changed = True
            break

        if changed:
            continue

        # Replace %funcName(literal) -- no group reference
        for mr in pa_norm_no_group.finditer(tonormalize):
            func_name = mr.group(1)
            literal = mr.group(2)
            # Skip if this looks like a special function we already handle
            if func_name in ("SUBSTRING", "LOWERCASE", "UPPERCASE", "SUM", "CHINESENUMBERS"):
                continue
            norm_map = norm_manager.get_normalization(func_name)
            if norm_map is not None:
                result = norm_map.get(literal, "")
                if result is not None:
                    tonormalize = tonormalize.replace(mr.group(0), result, 1)
                    changed = True
            break

        if not changed:
            break

    return tonormalize


def correct_duration_value(value: str) -> str:
    """Convert finer granularity durations to coarser ones, e.g. PT24H -> P1D."""
    m = re.fullmatch(r"PT(\d+)H", value)
    if m:
        hours = int(m.group(1))
        if hours % 24 == 0:
            return f"P{hours // 24}D"
        return value

    m = re.fullmatch(r"PT(\d+)M", value)
    if m:
        minutes = int(m.group(1))
        if minutes % 60 == 0:
            return f"PT{minutes // 60}H"
        return value

    m = re.fullmatch(r"P(\d+)M", value)
    if m:
        months = int(m.group(1))
        if months % 12 == 0:
            return f"P{months // 12}Y"
        return value

    return value
