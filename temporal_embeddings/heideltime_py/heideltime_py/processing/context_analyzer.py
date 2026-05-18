"""
Context Analyzer: tense detection, last-mentioned-X lookups, boundary checks.
Port of ContextAnalyzer.java.
"""

import re

from ..models import Sentence, Timex3


def check_infront_behind(match_start: int, match_end: int, sentence_text: str) -> bool:
    """Check token boundaries - reject if surrounded by invalid chars."""
    # reject if preceded by digit+dot (e.g., "53453.1999")
    if match_start > 1:
        if re.match(r"\d\.", sentence_text[match_start - 2 : match_start]):
            return False

    # reject if preceded by word char, $, or +
    if match_start > 0:
        ch = sentence_text[match_start - 1 : match_start]
        if re.match(r"[\w$+]", ch) and ch != "(":
            return False

    # reject if followed by degree sign or word char
    if match_end < len(sentence_text):
        ch = sentence_text[match_end : match_end + 1]
        if re.match(r"[°\w]", ch) and ch != ")":
            return False
        # reject if followed by ,digit or .digit
        if match_end + 1 < len(sentence_text):
            two = sentence_text[match_end : match_end + 2]
            if re.match(r"[.,]\d", two):
                return False

    return True


def check_token_boundaries(match_start: int, match_end: int, sentence: Sentence) -> bool:
    """Check that match boundaries align with token boundaries."""
    sent_begin = sentence.begin
    sent_text = sentence.text

    # whole expression is the whole sentence
    if (match_end - match_start) == len(sent_text):
        return True

    # whitespace boundaries
    if match_start > 0 and match_end < len(sent_text):
        if (
            sent_text[match_start - 1 : match_start] == " "
            and sent_text[match_end : match_end + 1] == " "
        ):
            return True

    begin_ok = False
    end_ok = False
    boundary_chars = {".", "/", "-", "\u2013", ","}  # ndash, comma

    for tok in sentence.tokens:
        # Check begin
        if (
            (match_start + sent_begin) == tok.begin
            or match_start > 0
            and sent_text[match_start - 1 : match_start] in boundary_chars
        ):
            begin_ok = True

        # Check end
        if (
            (match_end + sent_begin) == tok.end
            or match_end < len(sent_text)
            and sent_text[match_end : match_end + 1] in boundary_chars
        ):
            end_ok = True

        if begin_ok and end_ok:
            return True

    return False


def get_last_mentioned_x(linear_dates: list[Timex3], i: int, x: str, norm_manager=None) -> str:
    """Walk backward through prior timexes to find century/decade/year/month/day/week/quarter/season."""
    if i <= 0:
        return ""

    t_i = linear_dates[i]
    j = i - 1

    while j >= 0:
        timex = linear_dates[j]
        # skip same-offset timexes
        if t_i.begin == timex.begin:
            j -= 1
            continue

        value = timex.timex_value
        if "funcDate" in value:
            j -= 1
            continue

        if x == "century":
            if re.match(r"^\d\d", value):
                return value[:2]
            elif re.match(r"^BC\d\d", value):
                return value[:4]
        elif x == "decade":
            if re.match(r"^\d{3}", value):
                return value[:3]
            elif re.match(r"^BC\d{3}", value):
                return value[:5]
        elif x == "year":
            if re.match(r"^\d{4}", value):
                return value[:4]
            elif re.match(r"^BC\d{4}", value):
                return value[:6]
        elif x == "dateYear":
            if re.match(r"^\d{4}", value) or re.match(r"^BC\d{4}", value):
                return value
        elif x == "month":
            if re.match(r"^\d{4}-\d{2}", value):
                return value[:7]
            elif re.match(r"^BC\d{4}-\d{2}", value):
                return value[:9]
        elif x == "month-with-details":
            if re.match(r"^\d{4}-\d{2}", value):
                return value
        elif x == "day":
            if re.match(r"^\d{4}-\d{2}-\d{2}", value):
                return value[:10]
        elif x == "week":
            if re.match(r"^\d{4}-\d{2}-\d{2}", value):
                m = re.match(r"^(\d{4})-\d{2}-\d{2}", value)
                if m:
                    from .date_calculator import get_week_of_date

                    week = get_week_of_date(value[:10])
                    return f"{m.group(1)}-W{week}"
            elif re.match(r"^\d{4}-W\d{2}", value):
                m = re.match(r"^(\d{4}-W\d{2})", value)
                if m:
                    return m.group(1)
        elif x == "quarter":
            if re.match(r"^\d{4}-\d{2}", value):
                month = value[5:7]
                quarter = (
                    norm_manager.get_from_norm_month_in_quarter(month) if norm_manager else "1"
                )
                return f"{value[:4]}-Q{quarter}"
            elif re.match(r"^\d{4}-Q[1234]", value):
                return value[:7]
        elif x == "dateQuarter":
            if re.match(r"^\d{4}-Q[1234]", value):
                return value[:7]
        elif x == "season":
            if re.match(r"^\d{4}-\d{2}", value):
                month = value[5:7]
                season = norm_manager.get_from_norm_month_in_season(month) if norm_manager else ""
                return f"{value[:4]}-{season}"
            elif re.match(r"^\d{4}-(SP|SU|FA|WI)", value):
                return value[:7]
        else:
            j -= 1
            continue

        j -= 1

    return ""


def get_last_tense(timex: Timex3, sentence: Sentence, repatterns: dict[str, str]) -> str:
    """
    Get the tense from POS tags of tokens before/after the timex.
    Port of ContextAnalyzer.getLastTense().
    """
    tense_past_pattern = repatterns.get("tensePos4Past", "")
    tense_pf_pattern = repatterns.get("tensePos4PresentFuture", "")
    tense_future_pattern = repatterns.get("tensePos4Future", "")
    tense_future_word = repatterns.get("tenseWord4Future", "")

    last_tense = ""

    # Look at tokens before the timex
    for tok in sentence.tokens:
        if tok.end <= timex.begin:
            if not tok.pos:
                continue
            if tense_pf_pattern and re.fullmatch(tense_pf_pattern, tok.pos):
                last_tense = "PRESENTFUTURE"
            elif tense_past_pattern and re.fullmatch(tense_past_pattern, tok.pos):
                last_tense = "PAST"
            elif tense_future_pattern and re.fullmatch(tense_future_pattern, tok.pos):
                if tense_future_word and re.fullmatch(tense_future_word, tok.text):
                    last_tense = "FUTURE"
            if tok.text == "since" or tok.text == "depuis":
                last_tense = "PAST"

    # If no tense found before, look after
    if last_tense == "":
        for tok in sentence.tokens:
            if tok.begin >= timex.end:
                if not tok.pos:
                    continue
                if tense_pf_pattern and re.fullmatch(tense_pf_pattern, tok.pos):
                    last_tense = "PRESENTFUTURE"
                    break
                elif tense_past_pattern and re.fullmatch(tense_past_pattern, tok.pos):
                    last_tense = "PAST"
                    break
                elif tense_future_pattern and re.fullmatch(tense_future_pattern, tok.pos):
                    if tense_future_word and re.fullmatch(tense_future_word, tok.text):
                        last_tense = "FUTURE"
                        break

    # Check for compound past tenses (VHZ/VBZ + VVN)
    if last_tense == "PRESENTFUTURE":
        prev_pos = ""
        for tok in sentence.tokens:
            if tok.end <= timex.begin:
                if prev_pos in ("VHZ", "VBZ", "VHP", "VBP", "VER:pres"):
                    if tok.pos in ("VVN", "VER:pper"):
                        if tok.text not in ("expected", "scheduled"):
                            last_tense = "PAST"
                            break
                prev_pos = tok.pos

    return last_tense


def get_closest_tense(timex: Timex3, sentence: Sentence, repatterns: dict[str, str]) -> str:
    """
    Get the closest tense (before or after) the timex.
    Port of ContextAnalyzer.getClosestTense().
    """
    tense_past_pattern = repatterns.get("tensePos4Past", "")
    tense_pf_pattern = repatterns.get("tensePos4PresentFuture", "")
    tense_future_pattern = repatterns.get("tensePos4Future", "")
    tense_future_word = repatterns.get("tenseWord4Future", "")

    last_tense = ""
    next_tense = ""
    last_dist = 0
    next_dist = 0
    timex_pos = 0
    token_counter = 0

    for tok in sentence.tokens:
        token_counter += 1
        if tok.end <= timex.begin:
            if not tok.pos:
                continue
            if tense_pf_pattern and re.fullmatch(tense_pf_pattern, tok.pos):
                last_tense = "PRESENTFUTURE"
                last_dist = token_counter
            elif tense_past_pattern and re.fullmatch(tense_past_pattern, tok.pos):
                last_tense = "PAST"
                last_dist = token_counter
            elif tense_future_pattern and re.fullmatch(tense_future_pattern, tok.pos):
                if tense_future_word and re.fullmatch(tense_future_word, tok.text):
                    last_tense = "FUTURE"
                    last_dist = token_counter
        else:
            if timex_pos == 0:
                timex_pos = token_counter

    token_counter = 0
    for tok in sentence.tokens:
        token_counter += 1
        if next_tense == "" and tok.begin >= timex.end:
            if not tok.pos:
                continue
            if tense_pf_pattern and re.fullmatch(tense_pf_pattern, tok.pos):
                next_tense = "PRESENTFUTURE"
                next_dist = token_counter
            elif tense_past_pattern and re.fullmatch(tense_past_pattern, tok.pos):
                next_tense = "PAST"
                next_dist = token_counter
            elif tense_future_pattern and re.fullmatch(tense_future_pattern, tok.pos):
                if tense_future_word and re.fullmatch(tense_future_word, tok.text):
                    next_tense = "FUTURE"
                    next_dist = token_counter

    if last_tense == "":
        return next_tense
    if next_tense == "":
        return last_tense

    # Return the closer one
    if (timex_pos - last_dist) > (next_dist - timex_pos):
        return next_tense
    return last_tense
