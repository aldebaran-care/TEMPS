"""
Date Calculator: calendar arithmetic for temporal disambiguation.
Port of DateCalculator.java using Python datetime.
"""

from datetime import datetime, timedelta


def get_x_next_day(date_str: str, x: int) -> str:
    """Add x days to a YYYY-MM-DD date string."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        dt += timedelta(days=x)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        # OverflowError: disambiguation can pass absurd deltas; timedelta uses a C
        # int range for days on some platforms. Fall back to the input unchanged.
        return date_str


def get_x_next_month(date_str: str, x: int) -> str:
    """Add x months to a YYYY-MM date string."""
    try:
        # Handle BC dates
        bc = False
        if date_str.startswith("BC"):
            bc = True
            date_str = date_str[2:]

        parts = date_str.split("-")
        year = int(parts[0])
        month = int(parts[1])

        # Calculate new month/year
        total_months = year * 12 + (month - 1) + x
        new_year = total_months // 12
        new_month = (total_months % 12) + 1

        if new_year <= 0 and not bc:
            # Transition to BC
            bc = True
            new_year = abs(new_year) + 1

        prefix = "BC" if bc else ""
        return f"{prefix}{new_year:04d}-{new_month:02d}"
    except (ValueError, IndexError):
        return date_str


def get_x_next_year(date_str: str, x: int) -> str:
    """Add x years to a year string (handles BC)."""
    try:
        bc = False
        if date_str.startswith("BC"):
            bc = True
            year = -int(date_str[2:6]) + 1  # BC year to astronomical
        else:
            year = int(date_str[:4])

        new_year = year + x

        if new_year <= 0:
            return f"BC{abs(new_year - 1):04d}"
        else:
            return f"{new_year:04d}"
    except (ValueError, IndexError):
        return date_str


def get_x_next_week(date_str: str, x: int) -> str:
    """Add x weeks to a YYYY-Www date string."""
    try:
        # Parse YYYY-Www
        date_no_w = date_str.replace("W", "")
        parts = date_no_w.split("-")
        year = int(parts[0])
        week = int(parts[1])

        # Use ISO calendar: create a date from year+week, then add weeks
        # Monday of the given ISO week
        dt = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")
        dt += timedelta(weeks=x)

        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except (ValueError, IndexError, OverflowError):
        return date_str


def get_x_next_decade(date_str: str, x: int) -> str:
    """Add x decades to a decade string (3 digits like '199')."""
    try:
        bc = False
        if date_str.startswith("BC"):
            bc = True
            decade = -int(date_str[2:]) + 1
        else:
            decade = int(date_str)

        new_decade = decade + x

        if new_decade <= 0:
            return f"BC{abs(new_decade - 1):03d}"
        else:
            return f"{new_decade:03d}"
    except (ValueError, IndexError):
        return date_str


def get_x_next_century(date_str: str, x: int) -> str:
    """Add x centuries to a century string (2 digits like '19')."""
    try:
        bc = False
        if date_str.startswith("BC"):
            bc = True
            century = -int(date_str[2:])
        else:
            century = int(date_str)

        new_century = century + x

        if new_century < 0:
            return f"BC{abs(new_century):02d}"
        else:
            return f"{new_century:02d}"
    except (ValueError, IndexError):
        return date_str


def get_weekday_of_date(date_str: str) -> int:
    """
    Get day of week for a date. Returns Java Calendar convention:
    Sunday=1, Monday=2, ..., Saturday=7.
    """
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        # Python: Monday=0..Sunday=6 -> Java: Sunday=1..Saturday=7
        py_weekday = dt.weekday()  # Mon=0, Tue=1, ..., Sun=6
        return (py_weekday + 2) % 7 or 7  # Mon=2, Tue=3, ..., Sat=7, Sun=1
    except ValueError:
        return 0


def get_week_of_date(date_str: str) -> int:
    """Get ISO week number of a date."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.isocalendar()[1]
    except ValueError:
        return 0
