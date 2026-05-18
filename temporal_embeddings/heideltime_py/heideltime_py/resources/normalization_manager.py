"""Normalization Manager: loads normalization resources into RegexHashMaps."""

import re

from .regex_hashmap import RegexHashMap
from .scanner import scan_resources

# Unicode space replacement
SPACE_REGEX = r"[\u2000-\u200a \u202f\u205f\u3000\u00a0\u1680\u180e]+"

# Pattern to parse normalization lines: "key","value"
PA_READ_NORM = re.compile(r'"(.*?)","(.*?)"')


def _replace_spaces(text: str) -> str:
    return text.replace(" ", SPACE_REGEX)


class NormalizationManager:
    """Manages all normalization lookup tables for a language."""

    def __init__(
        self, language: str, load_temponyms: bool = False, resources_path: str | None = None
    ):
        self.all_normalizations: dict[str, RegexHashMap] = {}

        # Global normalization mappings
        self.norm_day_in_week = {
            "sunday": "1",
            "monday": "2",
            "tuesday": "3",
            "wednesday": "4",
            "thursday": "5",
            "friday": "6",
            "saturday": "7",
            "Sunday": "1",
            "Monday": "2",
            "Tuesday": "3",
            "Wednesday": "4",
            "Thursday": "5",
            "Friday": "6",
            "Saturday": "7",
        }

        self.norm_number = {}
        for i in range(61):
            self.norm_number[str(i)] = f"{i:02d}"
            self.norm_number[f"{i:02d}"] = f"{i:02d}"

        self.norm_month_name = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }

        self.norm_month_in_season = {
            "": "",
            "01": "WI",
            "02": "WI",
            "03": "SP",
            "04": "SP",
            "05": "SP",
            "06": "SU",
            "07": "SU",
            "08": "SU",
            "09": "FA",
            "10": "FA",
            "11": "FA",
            "12": "WI",
        }

        self.norm_month_in_quarter = {
            "01": "1",
            "02": "1",
            "03": "1",
            "04": "2",
            "05": "2",
            "06": "2",
            "07": "3",
            "08": "3",
            "09": "3",
            "10": "4",
            "11": "4",
            "12": "4",
        }

        # Load from files
        resource_files = scan_resources(language, "normalization", resources_path)
        for name in resource_files:
            self.all_normalizations[name] = RegexHashMap()

        for name, filepath in resource_files.items():
            if "Temponym" in name and not load_temponyms:
                continue
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if line.startswith("//") or line == "":
                        continue
                    m = PA_READ_NORM.search(line)
                    if m:
                        key = _replace_spaces(m.group(1))
                        val = m.group(2)
                        self.all_normalizations[name].put(key, val)

    def get_normalization(self, name: str) -> RegexHashMap | None:
        return self.all_normalizations.get(name)

    def get_from_norm_number(self, key: str) -> str:
        return self.norm_number.get(key, key)

    def get_from_norm_day_in_week(self, key: str) -> str:
        return self.norm_day_in_week.get(key, "0")

    def get_from_norm_month_name(self, key: str) -> str:
        return self.norm_month_name.get(key, "XX")

    def get_from_norm_month_in_season(self, key: str) -> str:
        return self.norm_month_in_season.get(key, "")

    def get_from_norm_month_in_quarter(self, key: str) -> str:
        return self.norm_month_in_quarter.get(key, "1")
