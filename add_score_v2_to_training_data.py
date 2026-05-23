import argparse
import calendar
import csv
import math
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Optional, Tuple

from temporal_embeddings.data_utils.utils.dates.compute_similarity_dates import compute_similarity_dates_intervals

HEIDELTIME_PACKAGE_ROOT = Path(__file__).resolve().parent / "temporal_embeddings" / "heideltime_py"
if str(HEIDELTIME_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(HEIDELTIME_PACKAGE_ROOT))

from heideltime_py import HeidelTime, Timex3  # noqa: E402

DateInterval = List[str]

DEFAULT_INPUTS: Dict[str, Path] = {
    "real": Path("data/new_training_dataset/real_world_dataset/dataset.csv"),
    "synthetic": Path("data/new_training_dataset/synthetic_dataset/synthetic_dataset.csv"),
    "temporal_relationships": Path("data/new_training_dataset/synthetic_dataset/temporal_relationships.csv"),
}

MONTHS: Dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_PATTERN = (
    r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
    r"Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)"
)

SEASON_MONTHS: Dict[str, Tuple[int, int]] = {
    "SP": (3, 5),
    "SU": (6, 8),
    "FA": (9, 11),
    "WI": (12, 2),
}

_HEIDELTIME: Optional[HeidelTime] = None


def configure_heideltime(language: str = "english", document_type: str = "news", use_spacy: bool = False) -> None:
    global _HEIDELTIME
    _HEIDELTIME = HeidelTime(
        language=language,
        document_type=document_type,
        find_dates=True,
        find_times=True,
        find_durations=True,
        find_sets=False,
        find_temponyms=False,
        use_spacy=use_spacy,
    )
    extract_intervals.cache_clear()


def heideltime() -> HeidelTime:
    global _HEIDELTIME
    if _HEIDELTIME is None:
        configure_heideltime()
    return _HEIDELTIME


def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _month_number(month: str) -> int:
    return MONTHS[month.lower().rstrip(".")]


def _year_interval(year: int) -> DateInterval:
    return [f"{year:04d}-01-01", f"{year:04d}-12-31"]


def _month_interval(year: int, month: int) -> DateInterval:
    return [f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{_last_day(year, month):02d}"]


def _day_interval(year: int, month: int, day: int) -> DateInterval:
    return [f"{year:04d}-{month:02d}-{day:02d}"]


def _date_range(start: datetime, end: datetime) -> DateInterval:
    start, end = sorted((start, end))
    return [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]


def _add_months(date: datetime, months: int) -> datetime:
    total_months = date.year * 12 + date.month - 1 + months
    year = total_months // 12
    month = total_months % 12 + 1
    day = min(date.day, _last_day(year, month))
    return date.replace(year=year, month=month, day=day)


def _normalize_for_heideltime(text: str) -> str:
    # HeidelTime handles "Jun 2001" but this port misses "Jun, 2001".
    text = re.sub(rf"\b(\d{{1,2}}),\s+({MONTH_PATTERN}),\s+(\d{{4}})\b", r"\1 \2 \3", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b({MONTH_PATTERN}),\s+(\d{{4}})\b", r"\1 \2", text, flags=re.IGNORECASE)
    return text


def _normalize_dct(reference_date: str) -> str:
    try:
        return datetime.strptime(reference_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return "2021-11-09"


def _parse_duration_months(value: str) -> Optional[int]:
    match = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?", value)
    if not match:
        return None

    years = int(match.group(1) or 0)
    months = int(match.group(2) or 0)
    weeks = int(match.group(3) or 0)
    days = int(match.group(4) or 0)
    if weeks or days:
        return None
    return years * 12 + months


def _interval_granularity(interval: DateInterval) -> str:
    if len(interval) == 1:
        return "day"

    start = datetime.strptime(interval[0], "%Y-%m-%d")
    end = datetime.strptime(interval[-1], "%Y-%m-%d")

    if start.day == 1 and end.day == _last_day(end.year, end.month) and start.year == end.year and start.month == end.month:
        return "month"
    if start.month == 1 and start.day == 1 and end.month == 12 and end.day == 31 and start.year == end.year:
        return "year"
    return "interval"


def _shift_interval_by_months(interval: DateInterval, months: int) -> DateInterval:
    start = datetime.strptime(interval[0], "%Y-%m-%d")
    end = datetime.strptime(interval[-1], "%Y-%m-%d") if len(interval) > 1 else start
    granularity = _interval_granularity(interval)

    shifted_start = _add_months(start, months)
    shifted_end = _add_months(end, months)

    if granularity == "day":
        return [shifted_start.strftime("%Y-%m-%d")]
    if granularity == "month":
        return _month_interval(shifted_start.year, shifted_start.month)
    if granularity == "year":
        return _year_interval(shifted_start.year)
    return _date_range(shifted_start, shifted_end)


def _timeml_value_to_interval(value: str, reference_date: str) -> Optional[DateInterval]:
    value = (value or "").strip()
    if not value or value in {"REMOVE", "PAST_REF", "FUTURE_REF"} or value.startswith("UNDEF"):
        return None

    if value == "PRESENT_REF":
        return [_normalize_dct(reference_date)]

    if "/" in value:
        boundaries = [_timeml_value_to_interval(part, reference_date) for part in value.split("/", 1)]
        if boundaries[0] and boundaries[1]:
            return [boundaries[0][0], boundaries[1][-1]]
        return None

    value = value.split("T", 1)[0]

    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if 1 <= month <= 12 and 1 <= day <= _last_day(year, month):
            return _day_interval(year, month, day)
        return None

    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return _month_interval(year, month)
        return None

    match = re.fullmatch(r"(\d{4})", value)
    if match:
        return _year_interval(int(match.group(1)))

    match = re.fullmatch(r"(\d{4})-W(\d{2})", value)
    if match:
        year, week = int(match.group(1)), int(match.group(2))
        if 1 <= week <= 53:
            start = datetime.fromisocalendar(year, week, 1)
            return _date_range(start, start + timedelta(days=6))
        return None

    match = re.fullmatch(r"(\d{4})-Q([1-4])", value)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        start_month = ((quarter - 1) * 3) + 1
        end_month = start_month + 2
        return [f"{year:04d}-{start_month:02d}-01", f"{year:04d}-{end_month:02d}-{_last_day(year, end_month):02d}"]

    match = re.fullmatch(r"(\d{4})-H([12])", value)
    if match:
        year, half = int(match.group(1)), int(match.group(2))
        return [f"{year:04d}-01-01", f"{year:04d}-06-30"] if half == 1 else [f"{year:04d}-07-01", f"{year:04d}-12-31"]

    match = re.fullmatch(r"(\d{4})-(SP|SU|FA|WI)", value)
    if match:
        year, season = int(match.group(1)), match.group(2)
        start_month, end_month = SEASON_MONTHS[season]
        end_year = year + 1 if season == "WI" else year
        return [f"{year:04d}-{start_month:02d}-01", f"{end_year:04d}-{end_month:02d}-{_last_day(end_year, end_month):02d}"]

    return None


def _deduplicate(intervals: List[DateInterval]) -> List[DateInterval]:
    output: List[DateInterval] = []
    seen = set()
    for interval in intervals:
        key = tuple(interval)
        if key not in seen:
            seen.add(key)
            output.append(interval)
    return output


def _extract_arithmetic_target(text: str, timexes: List[Timex3], reference_date: str) -> List[DateInterval]:
    durations = [t for t in timexes if t.timex_type == "DURATION"]
    dates = [t for t in timexes if t.timex_type in {"DATE", "TIME"}]
    if not durations or not dates:
        return []

    targets: List[DateInterval] = []
    lowered = text.lower()

    for anchor in dates:
        anchor_interval = _timeml_value_to_interval(anchor.timex_value, reference_date)
        if not anchor_interval:
            continue

        before_anchor = [d for d in durations if d.end <= anchor.begin]
        if not before_anchor:
            continue

        context_start = min(d.begin for d in before_anchor)
        context = lowered[context_start:anchor.begin]
        if "after" not in context and "before" not in context:
            continue

        months = 0
        for duration in before_anchor:
            duration_months = _parse_duration_months(duration.timex_value)
            if duration_months is None:
                months = 0
                break
            months += duration_months
        if months == 0:
            continue

        direction = 1 if "after" in context else -1
        targets.append(_shift_interval_by_months(anchor_interval, direction * months))

    return targets


def _extract_heideltime_ranges(text: str, timexes: List[Timex3], reference_date: str) -> Tuple[List[DateInterval], set[int]]:
    date_timexes = [
        (index, timex)
        for index, timex in sorted(enumerate(timexes), key=lambda item: item[1].begin)
        if timex.timex_type in {"DATE", "TIME"}
    ]
    ranges: List[DateInterval] = []
    consumed: set[int] = set()

    for (first_index, first), (second_index, second) in zip(date_timexes, date_timexes[1:]):
        first_interval = _timeml_value_to_interval(first.timex_value, reference_date)
        second_interval = _timeml_value_to_interval(second.timex_value, reference_date)
        if not first_interval or not second_interval:
            continue

        prefix = text[max(0, first.begin - 16):first.begin].lower()
        connector = text[first.end:second.begin].lower()
        interval_context = (
            re.search(r"\b(from|between)\b", prefix) is not None
            or re.search(r"\b(to|and|until|through|thru)\b|[-–]", connector) is not None
        )
        if not interval_context:
            continue

        ranges.append([first_interval[0], second_interval[-1]])
        consumed.update({first_index, second_index})

    return ranges, consumed


@lru_cache(maxsize=200_000)
def extract_intervals(text: str, reference_date: str) -> Tuple[Tuple[str, ...], ...]:
    normalized_text = _normalize_for_heideltime(text or "")
    dct = _normalize_dct(reference_date)
    timexes = heideltime().extract(normalized_text, dct=dct)

    arithmetic_intervals = _extract_arithmetic_target(normalized_text, timexes, dct)
    if arithmetic_intervals:
        return tuple(tuple(interval) for interval in _deduplicate(arithmetic_intervals))

    intervals: List[DateInterval] = []
    range_intervals, consumed = _extract_heideltime_ranges(normalized_text, timexes, dct)
    intervals.extend(range_intervals)

    for index, timex in enumerate(timexes):
        if index in consumed:
            continue
        if timex.timex_type not in {"DATE", "TIME"}:
            continue
        interval = _timeml_value_to_interval(timex.timex_value, dct)
        if interval:
            intervals.append(interval)

    return tuple(tuple(interval) for interval in _deduplicate(intervals))


def score_v2(sent0: str, sent0_date: str, sent1: str, sent1_date: str) -> Optional[float]:
    if "yyyy-mm" in sent0 or "yyyy-mm" in sent1:
        return None

    sent0_intervals = [list(interval) for interval in extract_intervals(sent0, sent0_date)]
    sent1_intervals = [list(interval) for interval in extract_intervals(sent1, sent1_date)]

    if not sent0_intervals or not sent1_intervals:
        return None

    best = 0.0
    for first_interval in sent0_intervals:
        for second_interval in sent1_intervals:
            best = max(
                best,
                compute_similarity_dates_intervals(
                    first_interval,
                    second_interval,
                ),
            )
    return best


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_with_score_v2{input_path.suffix}")


def _chunked(reader: Iterable[Dict[str, str]], chunk_size: int) -> Iterable[List[Dict[str, str]]]:
    chunk: List[Dict[str, str]] = []
    for row in reader:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _chunked_with_limits(
    reader: Iterable[Dict[str, str]],
    chunk_size: int,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> Iterable[List[Dict[str, str]]]:
    chunk: List[Dict[str, str]] = []
    bounded_end = float("inf") if end_index is None else end_index

    for row_index, row in enumerate(reader):
        if row_index < start_index:
            continue
        if row_index > bounded_end:
            break

        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def _worker_init(language: str, document_type: str, use_spacy: bool) -> None:
    configure_heideltime(language=language, document_type=document_type, use_spacy=use_spacy)


def _score_chunk(args: Tuple[List[Dict[str, str]], List[str], str, bool]) -> Tuple[List[Dict[str, str]], Dict[str, int], List[str]]:
    chunk, fieldnames, empty_value, skip_unscored = args
    stats = {
        "rows": 0,
        "scored": 0,
        "unscored": 0,
        "errors": 0,
    }
    output_rows: List[Dict[str, str]] = []
    warnings: List[str] = []

    for row in chunk:
        stats["rows"] += 1
        try:
            score = score_v2(row["sent0"], row["sent0_date"], row["sent1"], row["sent1_date"])
        except Exception as exc:
            score = None
            stats["errors"] += 1
            if len(warnings) < 3:
                warnings.append(f"{type(exc).__name__}: {exc}")

        if score is None or math.isnan(score):
            stats["unscored"] += 1
            if skip_unscored:
                continue
            row["score_v2"] = empty_value
        else:
            row["score_v2"] = f"{score:.17g}"
            stats["scored"] += 1

        output_rows.append({name: row.get(name, "") for name in fieldnames})

    return output_rows, stats, warnings


def _merge_stats(total: Dict[str, int], update: Dict[str, int]) -> None:
    for key, value in update.items():
        total[key] += value


def _log_progress(input_path: Path, stats: Dict[str, int], start_time: float, workers: int, force: bool = False) -> None:
    elapsed = max(time.time() - start_time, 1e-9)
    rows_per_second = stats["rows"] / elapsed
    if force or stats["rows"] > 0:
        print(
            f"{input_path}: processed {stats['rows']:,} rows; "
            f"scored {stats['scored']:,}; skipped {stats['unscored']:,}; "
            f"errors {stats['errors']:,}; written {stats['scored'] if stats['unscored'] else stats['scored']:,}; "
            f"{rows_per_second:,.1f} rows/s; elapsed {elapsed / 60:.1f} min; workers {workers}",
            flush=True,
        )


def add_score_v2_column(
    input_path: Path,
    output_path: Path,
    empty_value: str = "",
    skip_unscored: bool = True,
    workers: int = 1,
    chunk_size: int = 1_000,
    log_interval_seconds: float = 10.0,
    heideltime_language: str = "english",
    heideltime_document_type: str = "news",
    use_spacy: bool = False,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> Dict[str, int]:
    stats = {
        "rows": 0,
        "scored": 0,
        "unscored": 0,
        "errors": 0,
    }
    written = 0
    warning_count = 0
    workers = max(1, workers)
    chunk_size = max(1, chunk_size)

    with input_path.open("r", encoding="utf-8", newline="") as infile, output_path.open("w", encoding="utf-8", newline="") as outfile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_path}")

        fieldnames = [name for name in reader.fieldnames if name != "score_v2"] + ["score_v2"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        start_time = time.time()
        last_log_time = start_time

        print(
            f"Starting {input_path} -> {output_path} with {workers} worker(s), "
            f"chunk_size={chunk_size}, skip_unscored={skip_unscored}",
            flush=True,
        )

        chunk_args = (
            (chunk, fieldnames, empty_value, skip_unscored)
            for chunk in _chunked_with_limits(
                reader,
                chunk_size,
                start_index=start_index,
                end_index=end_index,
            )
        )

        if workers == 1:
            configure_heideltime(heideltime_language, heideltime_document_type, use_spacy)
            results = map(_score_chunk, chunk_args)
        else:
            pool = mp.Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(heideltime_language, heideltime_document_type, use_spacy),
            )
            results = pool.imap(_score_chunk, chunk_args, chunksize=1)

        try:
            for output_rows, chunk_stats, warnings in results:
                _merge_stats(stats, chunk_stats)
                writer.writerows(output_rows)
                written += len(output_rows)

                for warning in warnings:
                    warning_count += 1
                    if warning_count <= 10:
                        print(f"Warning: could not compute score_v2 in {input_path}: {warning}", flush=True)

                current_time = time.time()
                if current_time - last_log_time >= log_interval_seconds:
                    elapsed = max(current_time - start_time, 1e-9)
                    rows_per_second = stats["rows"] / elapsed
                    print(
                        f"{input_path}: processed {stats['rows']:,} rows; "
                        f"scored {stats['scored']:,}; skipped {stats['unscored']:,}; "
                        f"errors {stats['errors']:,}; written {written:,}; "
                        f"{rows_per_second:,.1f} rows/s; elapsed {elapsed / 60:.1f} min; "
                        f"workers {workers}",
                        flush=True,
                    )
                    last_log_time = current_time
        finally:
            if workers > 1:
                pool.close()
                pool.join()

        elapsed = max(time.time() - start_time, 1e-9)
        print(
            f"{input_path}: final throughput {stats['rows'] / elapsed:,.1f} rows/s over {elapsed / 60:.1f} min; "
            f"written {written:,}",
            flush=True,
        )

    return stats


def update_file(
    input_path: Path,
    output_path: Optional[Path],
    in_place: bool,
    empty_value: str,
    skip_unscored: bool,
    workers: int,
    chunk_size: int,
    log_interval_seconds: float,
    heideltime_language: str,
    heideltime_document_type: str,
    use_spacy: bool,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> None:
    input_path = input_path.resolve()
    final_output_path = input_path if in_place else (output_path or default_output_path(input_path)).resolve()

    if in_place:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(input_path.parent), suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
        try:
            stats = add_score_v2_column(
                input_path,
                tmp_path,
                empty_value=empty_value,
                skip_unscored=skip_unscored,
                workers=workers,
                chunk_size=chunk_size,
                log_interval_seconds=log_interval_seconds,
                heideltime_language=heideltime_language,
                heideltime_document_type=heideltime_document_type,
                use_spacy=use_spacy,
                start_index=start_index,
                end_index=end_index,
            )
            os.replace(tmp_path, input_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = add_score_v2_column(
            input_path,
            final_output_path,
            empty_value=empty_value,
            skip_unscored=skip_unscored,
            workers=workers,
            chunk_size=chunk_size,
            log_interval_seconds=log_interval_seconds,
            heideltime_language=heideltime_language,
            heideltime_document_type=heideltime_document_type,
            use_spacy=use_spacy,
            start_index=start_index,
            end_index=end_index,
        )

    print(f"Finished {input_path}")
    print(f"Output: {final_output_path}")
    print(
        f"Rows: {stats['rows']:,}; scored: {stats['scored']:,}; "
        f"unscored: {stats['unscored']:,}; errors: {stats['errors']:,}"
    )


def _read_index(prompt: str, default: Optional[int]) -> Optional[int]:
    while True:
        suffix = "" if default is None else f" [{default}]"
        value = input(f"{prompt}{suffix}: ").strip()
        if not value:
            return default
        if value.lower() in {"none", "all"}:
            return None
        try:
            parsed = int(value)
            if parsed < 0:
                print("Please enter a non-negative integer.")
                continue
            return parsed
        except ValueError:
            print("Invalid input. Enter a non-negative integer, 'none', or leave empty.")


def _prompt_slice_for_dataset(dataset_name: str) -> Tuple[int, Optional[int]]:
    print(f"\nSelect row range for dataset '{dataset_name}' (0-based, inclusive end).")
    start_index = _read_index("Start index", 0)
    end_index = _read_index("End index (None for full tail)", None)

    if start_index is None:
        start_index = 0
    if end_index is not None and end_index < start_index:
        print("End index cannot be smaller than start index. Re-enter values.")
        return _prompt_slice_for_dataset(dataset_name)

    return start_index, end_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a score_v2 column to temporal training CSV files.")
    parser.add_argument("--dataset", choices=["real", "synthetic", "temporal_relationships", "all"], default="all", help="Default dataset path to process.")
    parser.add_argument("--input_path", type=Path, help="Specific CSV to process. Overrides --dataset.")
    parser.add_argument("--output_path", type=Path, help="Output CSV path. Not allowed with --dataset all unless --input_path is set.")
    parser.add_argument("--in-place", action="store_true", help="Replace the input CSV atomically after writing a temporary file.")
    parser.add_argument("--keep-unscored", action="store_true", help="Keep rows whose score_v2 cannot be recovered and write --empty_value.")
    parser.add_argument("--empty_value", default="", help="Value to write for unscored rows when --keep-unscored is used.")
    parser.add_argument("--heideltime-language", default="english", help="HeidelTime language resources to use.")
    parser.add_argument("--heideltime-document-type", default="news", choices=["news", "narrative", "narratives", "colloquial", "scientific"], help="HeidelTime document type.")
    parser.add_argument("--use-spacy", action="store_true", help="Use spaCy tokenization inside heideltime_py.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Number of worker processes. Defaults to all available CPUs.")
    parser.add_argument("--chunk-size", type=int, default=1_000, help="Number of CSV rows scored per worker task.")
    parser.add_argument("--log-interval-seconds", type=float, default=10.0, help="How often to print progress logs.")
    parser.add_argument(
        "--interactive-index-range",
        action="store_true",
        help="Prompt in CLI for start/end row indices (0-based, inclusive end) for each processed dataset.",
    )
    args = parser.parse_args()

    workers = max(1, args.workers)

    if args.input_path:
        start_index = 0
        end_index = None
        if args.interactive_index_range:
            start_index, end_index = _prompt_slice_for_dataset(args.input_path.name)
        update_file(
            args.input_path,
            args.output_path,
            args.in_place,
            args.empty_value,
            not args.keep_unscored,
            workers,
            args.chunk_size,
            args.log_interval_seconds,
            args.heideltime_language,
            args.heideltime_document_type,
            args.use_spacy,
            start_index=start_index,
            end_index=end_index,
        )
        return

    if args.output_path and args.dataset == "all":
        raise ValueError("--output_path can only be used with --input_path or a single --dataset")

    datasets = DEFAULT_INPUTS if args.dataset == "all" else {args.dataset: DEFAULT_INPUTS[args.dataset]}
    for dataset_name, dataset_path in datasets.items():
        start_index = 0
        end_index = None
        if args.interactive_index_range:
            start_index, end_index = _prompt_slice_for_dataset(dataset_name)
        update_file(
            dataset_path,
            args.output_path,
            args.in_place,
            args.empty_value,
            not args.keep_unscored,
            workers,
            args.chunk_size,
            args.log_interval_seconds,
            args.heideltime_language,
            args.heideltime_document_type,
            args.use_spacy,
            start_index=start_index,
            end_index=end_index,
        )


if __name__ == "__main__":
    main()
