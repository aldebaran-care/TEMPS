import random
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from temporal_embeddings.data_utils.utils.dates.compute_distance_dates import compute_distance_dates_same_type

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def random_month_year(start_year=1100, end_year=2100):
    month = random.choice(MONTHS)
    year = random.randint(start_year, end_year)
    return month, year

def month_year_to_yyyy_mm(date_str):
    try:
        dt = datetime.strptime(date_str, "%B %Y")
        return dt.strftime("%Y-%m")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")

def phrase_and_answer():
    # Generate a random reference date
    ref_month, ref_year = random_month_year()
    # Generate random years and months to shift
    years_delta = random.randint(1, 10)
    months_delta = random.randint(0, 11)
    # Randomly choose "before" or "after"
    direction = random.choice(["before", "after"])
    # Compose the phrase
    phrase = f"{years_delta} year{'s' if years_delta > 1 else ''} and {months_delta} month{'s' if months_delta != 1 else ''} {direction} {ref_month} {ref_year}"
    # Compute the referenced date
    ref_date = datetime(ref_year, MONTHS.index(ref_month) + 1, 1)
    if direction == "before":
        answer_date = ref_date - relativedelta(years=years_delta, months=months_delta)
    else:
        answer_date = ref_date + relativedelta(years=years_delta, months=months_delta)
    answer = f"{MONTHS[answer_date.month - 1]} {answer_date.year}"
    return phrase, answer

def random_date_str():
    month, year = random_month_year()
    return f"{month} {year}"

def generate_random_date(start_year=1100, end_year=2100):
    # Generate a random date between the start and end years
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date

    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)

    # Format: day month year (e.g., 24 July 2025)
    return random_date.strftime("%d %B %Y")

def generate_dataset(n_phrases=100):
    data = []
    for _ in range(n_phrases):
        phrase, answer = phrase_and_answer()

        # Add the correct answer
        data.append({"sent0": phrase, "sent0_date": generate_random_date(), "sent1": answer, "sent1_date": generate_random_date(), "score": 1.0})
        # Add 4 random incorrect answers
        incorrect_dates = set()
        while len(incorrect_dates) < 4:
            d = random_date_str()
            if d != answer:
                incorrect_dates.add(d)
        for d in incorrect_dates:
            data.append({"sent0": phrase, "sent0_date": generate_random_date(), "sent1": d, "sent1_date": generate_random_date(), "ref_date": answer, "score": (1 / compute_distance_dates_same_type(month_year_to_yyyy_mm(answer), month_year_to_yyyy_mm(d), "yyyy-mm"))})
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(1000000)
    df.to_csv("synthetic_temporal_dataset.csv", index=False)