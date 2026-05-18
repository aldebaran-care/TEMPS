"""
Disambiguator: resolves UNDEF values in timex expressions.
Port of HeidelTime.specifyAmbiguousValues() / specifyAmbiguousValuesString().
"""

import re

from ..models import Sentence, Timex3
from ..resources.normalization_manager import NormalizationManager
from . import date_calculator as dc
from .context_analyzer import get_last_mentioned_x, get_last_tense

# Caps for UNDEF-this-unit-PLUS/MINUS-N: pathological diffs avoid slow or fragile date math.
_MAX_SANE_DAY_ARITHMETIC = 500_000
_MAX_SANE_WEEK_ARITHMETIC = 100_000


def _is_news_type(doc_type: str) -> bool:
    return doc_type in ("news", "colloquial", "scientific")


def specify_ambiguous_value(
    ambig: str,
    t_i: Timex3,
    i: int,
    linear_dates: list[Timex3],
    sentence: Sentence,
    norm: NormalizationManager,
    repatterns: dict[str, str],
    doc_type: str = "news",
    dct_value: str | None = None,
    language: str = "english",
) -> str:
    """Resolve a single UNDEF value string."""

    dct_available = dct_value is not None and len(dct_value) >= 8
    is_news = _is_news_type(doc_type)

    # Parse DCT
    dct_century = dct_year = dct_decade = dct_month = dct_day = 0
    dct_season = dct_quarter = dct_half = ""
    dct_weekday = dct_week = 0

    if dct_available:
        dv = dct_value.replace("-", "") if "-" in dct_value else dct_value
        if len(dv) >= 8:
            dct_century = int(dv[0:2])
            dct_year = int(dv[0:4])
            dct_decade = int(dv[2:3])
            dct_month = int(dv[4:6])
            dct_day = int(dv[6:8])

        dct_quarter = "Q" + norm.get_from_norm_month_in_quarter(
            norm.get_from_norm_number(str(dct_month))
        )
        dct_half = "H1" if dct_month <= 6 else "H2"
        dct_season = norm.get_from_norm_month_in_season(norm.get_from_norm_number(str(dct_month)))
        dct_date_str = f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}-{norm.get_from_norm_number(str(dct_day))}"
        dct_weekday = dc.get_weekday_of_date(dct_date_str)
        dct_week = dc.get_week_of_date(dct_date_str)

    # Parse value parts for UNDEF-year / UNDEF-century
    value_parts = ambig.split("-")
    vi_has_month = vi_has_day = vi_has_season = vi_has_week = vi_has_quarter = vi_has_half = False
    vi_month = vi_day = 0
    vi_season = vi_quarter = vi_half = ""

    if ambig.startswith("UNDEF-year") or ambig.startswith("UNDEF-century"):
        idx = 2  # parts after "UNDEF-year" or "UNDEF-century"
        if len(value_parts) > idx:
            p = value_parts[idx]
            if re.fullmatch(r"\d\d", p):
                vi_has_month = True
                vi_month = int(p)
            elif p in ("SP", "SU", "FA", "WI"):
                vi_has_season = True
                vi_season = p
            elif p in ("Q1", "Q2", "Q3", "Q4"):
                vi_has_quarter = True
                vi_quarter = p
            elif p in ("H1", "H2"):
                vi_has_half = True
                vi_half = p
            if len(value_parts) > idx + 1 and re.fullmatch(r"\d\d", value_parts[idx + 1]):
                vi_has_day = True
                vi_day = int(value_parts[idx + 1])

    last_tense = get_last_tense(t_i, sentence, repatterns)
    value_new = ambig

    ############################
    # UNDEF-year resolution
    ############################
    if ambig.startswith("UNDEF-year"):
        new_year = str(dct_year) if dct_available else ""

        if vi_has_month and not vi_has_season:
            if is_news and dct_available:
                if last_tense in ("FUTURE", "PRESENTFUTURE"):
                    if dct_month > vi_month:
                        new_year = str(dct_year + 1)
                if last_tense == "PAST":
                    if dct_month < vi_month:
                        new_year = str(dct_year - 1)
            else:
                new_year = get_last_mentioned_x(linear_dates, i, "year", norm)

        if vi_has_quarter:
            if is_news and dct_available:
                dq = int(dct_quarter[1:])
                vq = int(vi_quarter[1:])
                if last_tense in ("FUTURE", "PRESENTFUTURE") and dq < vq:
                    new_year = str(dct_year + 1)
                if last_tense == "PAST" and dq < vq:
                    new_year = str(dct_year - 1)
                if last_tense == "":
                    if doc_type == "colloquial" and dq < vq:
                        new_year = str(dct_year + 1)
                    elif dq < vq:
                        new_year = str(dct_year - 1)
            else:
                new_year = get_last_mentioned_x(linear_dates, i, "year", norm)

        if vi_has_half:
            if is_news and dct_available:
                dh = int(dct_half[1:])
                vh = int(vi_half[1:])
                if last_tense in ("FUTURE", "PRESENTFUTURE") and dh < vh:
                    new_year = str(dct_year + 1)
                if last_tense == "PAST" and dh < vh:
                    new_year = str(dct_year - 1)
                if last_tense == "":
                    if doc_type == "colloquial" and dh < vh:
                        new_year = str(dct_year + 1)
                    elif dh < vh:
                        new_year = str(dct_year - 1)
            else:
                new_year = get_last_mentioned_x(linear_dates, i, "year", norm)

        if vi_has_season and not vi_has_month and not vi_has_day:
            if is_news and dct_available:
                new_year = str(dct_year)
            else:
                new_year = get_last_mentioned_x(linear_dates, i, "year", norm)

        if not new_year:
            value_new = ambig.replace("UNDEF-year", "XXXX", 1)
        else:
            value_new = ambig.replace("UNDEF-year", new_year, 1)

    ############################
    # UNDEF-century resolution
    ############################
    elif ambig.startswith("UNDEF-century"):
        new_century = str(dct_century)
        if is_news and dct_available and ambig != "UNDEF-century":
            # ambig[13] is meant to be the decade digit, but noisy inputs (e.g.
            # "UNDEF-centurydecade") put a non-digit there. Skip the news/DCT
            # tense adjustment when we can't extract a numeric decade.
            decade_char = ambig[13:14]
            vi_decade = int(decade_char) if decade_char.isdigit() else None
            if vi_decade is not None:
                if last_tense in ("FUTURE", "PRESENTFUTURE"):
                    new_century = (
                        str(dct_century + 1) if vi_decade < dct_decade else str(dct_century)
                    )
                if last_tense == "PAST":
                    new_century = (
                        str(dct_century - 1) if dct_decade < vi_decade else str(dct_century)
                    )
        elif doc_type in ("narrative", "narratives"):
            new_century = get_last_mentioned_x(linear_dates, i, "century", norm)
            if new_century and not new_century.startswith("BC"):
                if re.match(r"^\d\d", new_century) and int(new_century[:2]) < 10:
                    new_century = "00"
            else:
                new_century = "00"
        else:
            new_century = get_last_mentioned_x(linear_dates, i, "century", norm)

        if not new_century:
            if doc_type not in ("narrative", "narratives"):
                value_new = ambig.replace("UNDEF-century", "19", 1)
            else:
                value_new = ambig.replace("UNDEF-century", "00", 1)
        else:
            value_new = ambig.replace("UNDEF-century", new_century, 1)

        # Fix 3-digit years in non-narrative
        if re.fullmatch(r"\d{3}", value_new) and doc_type not in ("narrative", "narratives"):
            value_new = "19" + value_new[2:]

    ############################
    # Other UNDEF patterns
    ############################
    elif ambig.startswith("UNDEF"):
        value_new = ambig

        # UNDEF-REFDATE
        if ambig == "UNDEF-REFDATE":
            if i > 0:
                value_new = linear_dates[i - 1].timex_value
            else:
                value_new = "XXXX-XX-XX"

        # Arithmetic: UNDEF-(this|REFUNIT|REF)-unit-(MINUS|PLUS)-N
        elif re.match(r"^UNDEF-(this|REFUNIT|REF)-(.*?)-(MINUS|PLUS)-(\d+)", ambig):
            m = re.match(r"^(UNDEF-(this|REFUNIT|REF)-(.*?)-(MINUS|PLUS)-(\d+))(.*)", ambig)
            if m:
                check_undef = m.group(1)
                ltn = m.group(2)
                unit = m.group(3)
                op = m.group(4)
                try:
                    diff = int(m.group(5))
                except ValueError:
                    value_new = "FUTURE_REF" if op == "PLUS" else "PAST_REF"
                    return value_new

                if doc_type == "scientific":
                    value_new = _resolve_scientific_arithmetic(ambig, check_undef, unit, op, diff)
                else:
                    value_new = _resolve_arithmetic(
                        value_new,
                        check_undef,
                        ltn,
                        unit,
                        op,
                        diff,
                        i,
                        linear_dates,
                        norm,
                        dct_available,
                        is_news,
                        dct_century,
                        dct_decade,
                        dct_year,
                        dct_month,
                        dct_day,
                        dct_week,
                        dct_quarter,
                        language,
                    )

        # century: last/this/next
        elif ambig.startswith("UNDEF-last-century"):
            value_new = _resolve_last_this_next(
                "UNDEF-last-century",
                -1,
                "century",
                value_new,
                i,
                linear_dates,
                norm,
                is_news,
                dct_available,
                dct_century,
            )
        elif ambig.startswith("UNDEF-this-century"):
            value_new = _resolve_last_this_next(
                "UNDEF-this-century",
                0,
                "century",
                value_new,
                i,
                linear_dates,
                norm,
                is_news,
                dct_available,
                dct_century,
            )
        elif ambig.startswith("UNDEF-next-century"):
            value_new = _resolve_last_this_next(
                "UNDEF-next-century",
                1,
                "century",
                value_new,
                i,
                linear_dates,
                norm,
                is_news,
                dct_available,
                dct_century,
            )

        # decade
        elif ambig.startswith("UNDEF-last-decade"):
            if is_news and dct_available:
                value_new = value_new.replace("UNDEF-last-decade", str(dct_year - 10)[:3])
            else:
                lm = get_last_mentioned_x(linear_dates, i, "decade", norm)
                value_new = value_new.replace(
                    "UNDEF-last-decade", dc.get_x_next_decade(lm, -1) if lm else "XXXX"
                )
        elif ambig.startswith("UNDEF-this-decade"):
            if is_news and dct_available:
                value_new = value_new.replace("UNDEF-this-decade", str(dct_year)[:3])
            else:
                lm = get_last_mentioned_x(linear_dates, i, "decade", norm)
                value_new = value_new.replace("UNDEF-this-decade", lm if lm else "XXXX")
        elif ambig.startswith("UNDEF-next-decade"):
            if is_news and dct_available:
                value_new = value_new.replace("UNDEF-next-decade", str(dct_year + 10)[:3])
            else:
                lm = get_last_mentioned_x(linear_dates, i, "decade", norm)
                value_new = value_new.replace(
                    "UNDEF-next-decade", dc.get_x_next_decade(lm, 1) if lm else "XXXX"
                )

        # year
        elif ambig.startswith("UNDEF-last-year"):
            cu = "UNDEF-last-year"
            if is_news and dct_available:
                value_new = value_new.replace(cu, str(dct_year - 1))
            else:
                lm = get_last_mentioned_x(linear_dates, i, "year", norm)
                value_new = value_new.replace(cu, dc.get_x_next_year(lm, -1) if lm else "XXXX")
            if value_new.endswith("-FY"):
                value_new = "FY" + value_new[: min(len(value_new), 4)]
        elif ambig.startswith("UNDEF-this-year"):
            cu = "UNDEF-this-year"
            if is_news and dct_available:
                value_new = value_new.replace(cu, str(dct_year))
            else:
                lm = get_last_mentioned_x(linear_dates, i, "year", norm)
                value_new = value_new.replace(cu, lm if lm else "XXXX")
            if value_new.endswith("-FY"):
                value_new = "FY" + value_new[: min(len(value_new), 4)]
        elif ambig.startswith("UNDEF-next-year"):
            cu = "UNDEF-next-year"
            if is_news and dct_available:
                value_new = value_new.replace(cu, str(dct_year + 1))
            else:
                lm = get_last_mentioned_x(linear_dates, i, "year", norm)
                value_new = value_new.replace(cu, dc.get_x_next_year(lm, 1) if lm else "XXXX")
            if value_new.endswith("-FY"):
                value_new = "FY" + value_new[: min(len(value_new), 4)]

        # month
        elif ambig.startswith("UNDEF-last-month"):
            cu = "UNDEF-last-month"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    dc.get_x_next_month(
                        f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}", -1
                    ),
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "month", norm)
                value_new = value_new.replace(cu, dc.get_x_next_month(lm, -1) if lm else "XXXX-XX")
        elif ambig.startswith("UNDEF-this-month"):
            cu = "UNDEF-this-month"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu, f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}"
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "month", norm)
                value_new = value_new.replace(cu, lm if lm else "XXXX-XX")
        elif ambig.startswith("UNDEF-next-month"):
            cu = "UNDEF-next-month"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    dc.get_x_next_month(
                        f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}", 1
                    ),
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "month", norm)
                value_new = value_new.replace(cu, dc.get_x_next_month(lm, 1) if lm else "XXXX-XX")

        # day
        elif ambig.startswith("UNDEF-last-day"):
            cu = "UNDEF-last-day"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    dc.get_x_next_day(
                        f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}-{dct_day}", -1
                    ),
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                value_new = value_new.replace(cu, dc.get_x_next_day(lm, -1) if lm else "XXXX-XX-XX")
        elif ambig.startswith("UNDEF-this-day"):
            cu = "UNDEF-this-day"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}-{norm.get_from_norm_number(str(dct_day))}",
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                if lm:
                    value_new = value_new.replace(cu, lm)
                else:
                    value_new = value_new.replace(cu, "XXXX-XX-XX")
                if ambig == "UNDEF-this-day":
                    value_new = "PRESENT_REF"
        elif ambig.startswith("UNDEF-next-day"):
            cu = "UNDEF-next-day"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    dc.get_x_next_day(
                        f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}-{dct_day}", 1
                    ),
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                value_new = value_new.replace(cu, dc.get_x_next_day(lm, 1) if lm else "XXXX-XX-XX")

        # week
        elif ambig.startswith("UNDEF-last-week"):
            cu = "UNDEF-last-week"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    dc.get_x_next_week(
                        f"{dct_year}-W{norm.get_from_norm_number(str(dct_week))}", -1
                    ),
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "week", norm)
                value_new = value_new.replace(cu, dc.get_x_next_week(lm, -1) if lm else "XXXX-WXX")
        elif ambig.startswith("UNDEF-this-week"):
            cu = "UNDEF-this-week"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu, f"{dct_year}-W{norm.get_from_norm_number(str(dct_week))}"
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "week", norm)
                value_new = value_new.replace(cu, lm if lm else "XXXX-WXX")
        elif ambig.startswith("UNDEF-next-week"):
            cu = "UNDEF-next-week"
            if is_news and dct_available:
                value_new = value_new.replace(
                    cu,
                    dc.get_x_next_week(
                        f"{dct_year}-W{norm.get_from_norm_number(str(dct_week))}", 1
                    ),
                )
            else:
                lm = get_last_mentioned_x(linear_dates, i, "week", norm)
                value_new = value_new.replace(cu, dc.get_x_next_week(lm, 1) if lm else "XXXX-WXX")

        # quarter
        elif ambig.startswith("UNDEF-last-quarter"):
            cu = "UNDEF-last-quarter"
            if is_news and dct_available:
                if dct_quarter == "Q1":
                    value_new = value_new.replace(cu, f"{dct_year - 1}-Q4")
                else:
                    value_new = value_new.replace(cu, f"{dct_year}-Q{int(dct_quarter[1:]) - 1}")
            else:
                lm = get_last_mentioned_x(linear_dates, i, "quarter", norm)
                if lm:
                    lq = int(lm[6:7])
                    ly = int(lm[:4])
                    if lq == 1:
                        value_new = value_new.replace(cu, f"{ly - 1}-Q4")
                    else:
                        value_new = value_new.replace(cu, f"{ly}-Q{lq - 1}")
                else:
                    value_new = value_new.replace(cu, "XXXX-QX")
        elif ambig.startswith("UNDEF-this-quarter"):
            cu = "UNDEF-this-quarter"
            if is_news and dct_available:
                value_new = value_new.replace(cu, f"{dct_year}-{dct_quarter}")
            else:
                lm = get_last_mentioned_x(linear_dates, i, "quarter", norm)
                value_new = value_new.replace(cu, lm if lm else "XXXX-QX")
        elif ambig.startswith("UNDEF-next-quarter"):
            cu = "UNDEF-next-quarter"
            if is_news and dct_available:
                if dct_quarter == "Q4":
                    value_new = value_new.replace(cu, f"{dct_year + 1}-Q1")
                else:
                    value_new = value_new.replace(cu, f"{dct_year}-Q{int(dct_quarter[1:]) + 1}")
            else:
                lm = get_last_mentioned_x(linear_dates, i, "quarter", norm)
                if lm:
                    lq = int(lm[6:7])
                    ly = int(lm[:4])
                    if lq == 4:
                        value_new = value_new.replace(cu, f"{ly + 1}-Q1")
                    else:
                        value_new = value_new.replace(cu, f"{ly}-Q{lq + 1}")
                else:
                    value_new = value_new.replace(cu, "XXXX-QX")

        # MONTH NAMES: UNDEF-(last|this|next)-january etc.
        elif re.match(
            r"UNDEF-(last|this|next)-(january|february|march|april|may|june|july|august|september|october|november|december)",
            ambig,
        ):
            m = re.match(
                r"(UNDEF-(last|this|next)-(january|february|march|april|may|june|july|august|september|october|november|december))(.*)",
                ambig,
            )
            if m:
                check_undef = m.group(1)
                ltn = m.group(2)
                new_month = norm.get_from_norm_month_name(m.group(3))
                new_month_int = int(new_month)
                rest = m.group(4)
                day = 0
                day_m = re.search(r"-(\d\d)", rest)
                if day_m:
                    day = int(day_m.group(1))

                if ltn == "last":
                    if is_news and dct_available:
                        if dct_month == new_month_int and day != 0:
                            if dct_day > day:
                                value_new = value_new.replace(
                                    check_undef, f"{dct_year}-{new_month}"
                                )
                            else:
                                value_new = value_new.replace(
                                    check_undef, f"{dct_year - 1}-{new_month}"
                                )
                        elif dct_month <= new_month_int:
                            value_new = value_new.replace(
                                check_undef, f"{dct_year - 1}-{new_month}"
                            )
                        else:
                            value_new = value_new.replace(check_undef, f"{dct_year}-{new_month}")
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "month-with-details", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX")
                        else:
                            lm_month_int = int(lm[5:7])
                            if lm_month_int <= new_month_int:
                                value_new = value_new.replace(
                                    check_undef, f"{int(lm[:4]) - 1}-{new_month}"
                                )
                            else:
                                value_new = value_new.replace(check_undef, f"{lm[:4]}-{new_month}")
                elif ltn == "this":
                    if is_news and dct_available:
                        value_new = value_new.replace(check_undef, f"{dct_year}-{new_month}")
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "month-with-details", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX")
                        else:
                            value_new = value_new.replace(check_undef, f"{lm[:4]}-{new_month}")
                elif ltn == "next":
                    if is_news and dct_available:
                        if dct_month == new_month_int and day != 0:
                            if dct_day < day:
                                value_new = value_new.replace(
                                    check_undef, f"{dct_year}-{new_month}"
                                )
                            else:
                                value_new = value_new.replace(
                                    check_undef, f"{dct_year + 1}-{new_month}"
                                )
                        elif dct_month >= new_month_int:
                            value_new = value_new.replace(
                                check_undef, f"{dct_year + 1}-{new_month}"
                            )
                        else:
                            value_new = value_new.replace(check_undef, f"{dct_year}-{new_month}")
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "month-with-details", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX")
                        else:
                            lm_month_int = int(lm[5:7])
                            if lm_month_int >= new_month_int:
                                value_new = value_new.replace(
                                    check_undef, f"{int(lm[:4]) + 1}-{new_month}"
                                )
                            else:
                                value_new = value_new.replace(check_undef, f"{lm[:4]}-{new_month}")

        # SEASON NAMES: UNDEF-(last|this|next)-(SP|SU|FA|WI)
        elif re.match(r"^UNDEF-(last|this|next)-(SP|SU|FA|WI)", ambig):
            m = re.match(r"(UNDEF-(last|this|next)-(SP|SU|FA|WI))", ambig)
            if m:
                check_undef = m.group(1)
                ltn = m.group(2)
                new_season = m.group(3)
                value_new = _resolve_season(
                    value_new,
                    check_undef,
                    ltn,
                    new_season,
                    is_news,
                    dct_available,
                    dct_year,
                    dct_month,
                    dct_season,
                    i,
                    linear_dates,
                    norm,
                    doc_type,
                )

        # WEEKDAY NAMES: UNDEF-(last|this|next|day)-weekday
        elif re.match(
            r"^UNDEF-(last|this|next|day)-(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            ambig,
        ):
            m = re.match(
                r"(UNDEF-(last|this|next|day)-(monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
                ambig,
            )
            if m:
                check_undef = m.group(1)
                ltnd = m.group(2)
                new_weekday = m.group(3)
                new_weekday_int = int(norm.get_from_norm_day_in_week(new_weekday))
                dct_date = f"{dct_year}-{dct_month}-{dct_day}"

                if ltnd == "last":
                    if is_news and dct_available:
                        diff = -(dct_weekday - new_weekday_int)
                        if diff >= 0:
                            diff -= 7
                        value_new = value_new.replace(
                            check_undef, dc.get_x_next_day(dct_date, diff)
                        )
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX-XX")
                        else:
                            lm_wd = dc.get_weekday_of_date(lm)
                            diff = -(lm_wd - new_weekday_int)
                            if diff >= 0:
                                diff -= 7
                            value_new = value_new.replace(check_undef, dc.get_x_next_day(lm, diff))

                elif ltnd == "this":
                    if is_news and dct_available:
                        diff = -(dct_weekday - new_weekday_int)
                        if diff >= 0:
                            diff -= 7
                        if diff == -7:
                            diff = 0
                        value_new = value_new.replace(
                            check_undef, dc.get_x_next_day(dct_date, diff)
                        )
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX-XX")
                        else:
                            lm_wd = dc.get_weekday_of_date(lm)
                            diff = -(lm_wd - new_weekday_int)
                            if diff >= 0:
                                diff -= 7
                            if diff == -7:
                                diff = 0
                            value_new = value_new.replace(check_undef, dc.get_x_next_day(lm, diff))

                elif ltnd == "next":
                    if is_news and dct_available:
                        diff = new_weekday_int - dct_weekday
                        if diff <= 0:
                            diff += 7
                        value_new = value_new.replace(
                            check_undef, dc.get_x_next_day(dct_date, diff)
                        )
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX-XX")
                        else:
                            lm_wd = dc.get_weekday_of_date(lm)
                            diff = new_weekday_int - lm_wd
                            if diff <= 0:
                                diff += 7
                            value_new = value_new.replace(check_undef, dc.get_x_next_day(lm, diff))

                elif ltnd == "day":
                    if is_news and dct_available:
                        diff = -(dct_weekday - new_weekday_int)
                        if diff >= 0:
                            diff -= 7
                        if diff == -7:
                            diff = 0
                        if last_tense == "FUTURE" and diff != 0:
                            diff += 7
                        value_new = value_new.replace(
                            check_undef, dc.get_x_next_day(dct_date, diff)
                        )
                    else:
                        lm = get_last_mentioned_x(linear_dates, i, "day", norm)
                        if not lm:
                            value_new = value_new.replace(check_undef, "XXXX-XX-XX")
                        else:
                            lm_wd = dc.get_weekday_of_date(lm)
                            diff = -(lm_wd - new_weekday_int)
                            if diff >= 0:
                                diff -= 7
                            if diff == -7:
                                diff = 0
                            value_new = value_new.replace(check_undef, dc.get_x_next_day(lm, diff))

    return value_new


def _resolve_last_this_next(
    check_undef, offset, unit, value_new, i, linear_dates, norm, is_news, dct_available, dct_val
):
    """Resolve UNDEF-last/this/next-century."""
    if is_news and dct_available:
        value_new = value_new.replace(check_undef, norm.get_from_norm_number(str(dct_val + offset)))
    else:
        lm = get_last_mentioned_x(linear_dates, i, unit, norm)
        if not lm:
            value_new = value_new.replace(check_undef, "XX")
        elif offset == 0:
            value_new = value_new.replace(check_undef, lm)
        else:
            lm = dc.get_x_next_century(lm, offset)
            value_new = value_new.replace(check_undef, lm)
    return value_new


def _resolve_season(
    value_new,
    check_undef,
    ltn,
    new_season,
    is_news,
    dct_available,
    dct_year,
    dct_month,
    dct_season,
    i,
    linear_dates,
    norm,
    doc_type,
):
    """Resolve season references."""
    season_order = {"SP": 0, "SU": 1, "FA": 2, "WI": 3}

    if ltn == "this":
        if is_news and dct_available:
            value_new = value_new.replace(check_undef, f"{dct_year}-{new_season}")
        else:
            lm = get_last_mentioned_x(linear_dates, i, "season", norm)
            if not lm:
                value_new = value_new.replace(check_undef, "XXXX-XX")
            else:
                value_new = value_new.replace(check_undef, f"{lm[:4]}-{new_season}")
    elif ltn == "last":
        if is_news and dct_available:
            # If current season is before or equal to target, go back a year
            if dct_season == "SP":
                value_new = value_new.replace(check_undef, f"{dct_year - 1}-{new_season}")
            elif dct_season == "SU":
                if new_season == "SP":
                    value_new = value_new.replace(check_undef, f"{dct_year}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{dct_year - 1}-{new_season}")
            elif dct_season == "FA":
                if new_season in ("SP", "SU"):
                    value_new = value_new.replace(check_undef, f"{dct_year}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{dct_year - 1}-{new_season}")
            elif dct_season == "WI":
                if new_season == "WI":
                    value_new = value_new.replace(check_undef, f"{dct_year - 1}-{new_season}")
                else:
                    yr = dct_year - 1 if dct_month < 12 else dct_year
                    value_new = value_new.replace(check_undef, f"{yr}-{new_season}")
        else:
            lm = get_last_mentioned_x(linear_dates, i, "season", norm)
            if not lm:
                value_new = value_new.replace(check_undef, "XXXX-XX")
            else:
                lm_season = lm[5:7]
                lm_year = int(lm[:4])
                if season_order.get(new_season, 0) >= season_order.get(lm_season, 0):
                    value_new = value_new.replace(check_undef, f"{lm_year - 1}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{lm_year}-{new_season}")
    elif ltn == "next":
        if is_news and dct_available:
            if dct_season == "SP":
                if new_season == "SP":
                    value_new = value_new.replace(check_undef, f"{dct_year + 1}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{dct_year}-{new_season}")
            elif dct_season == "SU":
                if new_season in ("SP", "SU"):
                    value_new = value_new.replace(check_undef, f"{dct_year + 1}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{dct_year}-{new_season}")
            elif dct_season == "FA":
                if new_season == "WI":
                    value_new = value_new.replace(check_undef, f"{dct_year}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{dct_year + 1}-{new_season}")
            elif dct_season == "WI":
                value_new = value_new.replace(check_undef, f"{dct_year + 1}-{new_season}")
        else:
            lm = get_last_mentioned_x(linear_dates, i, "season", norm)
            if not lm:
                value_new = value_new.replace(check_undef, "XXXX-XX")
            else:
                lm_season = lm[5:7]
                lm_year = int(lm[:4])
                if season_order.get(new_season, 0) <= season_order.get(lm_season, 0):
                    value_new = value_new.replace(check_undef, f"{lm_year + 1}-{new_season}")
                else:
                    value_new = value_new.replace(check_undef, f"{lm_year}-{new_season}")

    return value_new


def _resolve_scientific_arithmetic(ambig, check_undef, unit, op, diff):
    """Resolve arithmetic for scientific documents using TPZ format."""
    sym = "+" if op == "PLUS" else "-"
    unit_formats = {
        "year": f"{diff:04d}",
        "month": f"0000-{diff:02d}",
        "week": f"0000-W{diff:02d}",
        "day": f"0000-00-{diff:02d}",
        "hour": f"0000-00-00T{diff:02d}",
        "minute": f"0000-00-00T00:{diff:02d}",
        "second": f"0000-00-00T00:00:{diff:02d}",
    }
    fmt = unit_formats.get(unit, str(diff))
    return f"TPZ{sym}{fmt}"


def _resolve_arithmetic(
    value_new,
    check_undef,
    ltn,
    unit,
    op,
    diff,
    i,
    linear_dates,
    norm,
    dct_available,
    is_news,
    dct_century,
    dct_decade,
    dct_year,
    dct_month,
    dct_day,
    dct_week,
    dct_quarter,
    language,
):
    """Resolve UNDEF-(this|REFUNIT|REF)-unit-(MINUS|PLUS)-N."""
    if op == "MINUS":
        signed_diff = -diff
    else:
        signed_diff = diff

    if unit == "day" and abs(signed_diff) > _MAX_SANE_DAY_ARITHMETIC:
        return value_new.replace(check_undef, "XXXX-XX-XX")
    if unit == "week" and abs(signed_diff) > _MAX_SANE_WEEK_ARITHMETIC:
        return value_new.replace(check_undef, "XXXX-XX-XX")

    # REFUNIT special case for year
    if ltn == "REFUNIT" and unit == "year":
        date_with_year = get_last_mentioned_x(linear_dates, i, "dateYear", norm)
        if not date_with_year:
            return value_new.replace(check_undef, "XXXX")
        year_new = dc.get_x_next_year(date_with_year, signed_diff)
        rest = date_with_year[4:] if not date_with_year.startswith("BC") else date_with_year[6:]
        return value_new.replace(check_undef, year_new + rest)

    use_dct = is_news and dct_available and ltn == "this"

    if unit == "century":
        if use_dct:
            return value_new.replace(check_undef, str(dct_century + signed_diff))
        lm = get_last_mentioned_x(linear_dates, i, "century", norm)
        if not lm:
            return value_new.replace(check_undef, "XX")
        return value_new.replace(check_undef, dc.get_x_next_century(lm, signed_diff))

    elif unit == "decade":
        if use_dct:
            dct_decade_long = int(f"{dct_century}{dct_decade}")
            return value_new.replace(check_undef, f"{dct_decade_long + signed_diff}X")
        lm = get_last_mentioned_x(linear_dates, i, "decade", norm)
        if not lm:
            return value_new.replace(check_undef, "XXX")
        return value_new.replace(check_undef, dc.get_x_next_decade(lm, signed_diff))

    elif unit == "year":
        if use_dct:
            return value_new.replace(check_undef, str(dct_year + signed_diff))
        lm = get_last_mentioned_x(linear_dates, i, "year", norm)
        if not lm:
            return value_new.replace(check_undef, "XXXX")
        return value_new.replace(check_undef, dc.get_x_next_year(lm, signed_diff))

    elif unit == "quarter":
        if use_dct:
            int_quarter = int(dct_quarter[1:])
            diff_q = diff % 4
            diff_y = (diff - diff_q) // 4
            if op == "MINUS":
                diff_q, diff_y = -diff_q, -diff_y
            return value_new.replace(check_undef, f"{dct_year + diff_y}-Q{int_quarter + diff_q}")
        lm = get_last_mentioned_x(linear_dates, i, "quarter", norm)
        if not lm:
            return value_new.replace(check_undef, "XXXX-XX")
        int_year = int(lm[:4])
        int_quarter = int(lm[6:])
        diff_q = diff % 4
        diff_y = (diff - diff_q) // 4
        if op == "MINUS":
            diff_q, diff_y = -diff_q, -diff_y
        return value_new.replace(check_undef, f"{int_year + diff_y}-Q{int_quarter + diff_q}")

    elif unit == "month":
        if use_dct:
            return value_new.replace(
                check_undef,
                dc.get_x_next_month(
                    f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}", signed_diff
                ),
            )
        lm = get_last_mentioned_x(linear_dates, i, "month", norm)
        if not lm:
            return value_new.replace(check_undef, "XXXX-XX")
        return value_new.replace(check_undef, dc.get_x_next_month(lm, signed_diff))

    elif unit == "week":
        if use_dct:
            return value_new.replace(
                check_undef,
                dc.get_x_next_week(
                    f"{dct_year}-W{norm.get_from_norm_number(str(dct_week))}", signed_diff
                ),
            )
        lm = get_last_mentioned_x(linear_dates, i, "day", norm)
        if not lm:
            return value_new.replace(check_undef, "XXXX-XX-XX")
        return value_new.replace(check_undef, dc.get_x_next_day(lm, signed_diff * 7))

    elif unit == "day":
        if use_dct:
            return value_new.replace(
                check_undef,
                dc.get_x_next_day(
                    f"{dct_year}-{norm.get_from_norm_number(str(dct_month))}-{dct_day}", signed_diff
                ),
            )
        lm = get_last_mentioned_x(linear_dates, i, "day", norm)
        if not lm:
            return value_new.replace(check_undef, "XXXX-XX-XX")
        return value_new.replace(check_undef, dc.get_x_next_day(lm, signed_diff))

    return value_new
