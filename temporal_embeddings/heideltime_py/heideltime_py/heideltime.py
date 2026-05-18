"""
HeidelTime main class: orchestrates the temporal tagging pipeline.
"""

import re
from datetime import date

from .models import Sentence, Timex3
from .output.timeml_formatter import format_timeml
from .processing.disambiguator import specify_ambiguous_value
from .processing.overlap_remover import (
    remove_overlaps_postprocessing,
    remove_overlaps_preprocessing,
)
from .processing.timex_finder import find_timexes
from .processing.tokenizer import tokenize
from .resources.normalization_manager import NormalizationManager
from .resources.re_pattern_manager import load_repatterns
from .resources.rule_manager import load_rules


class HeidelTime:
    """
    HeidelTime temporal expression tagger and normalizer.

    Finds temporal expressions in text and normalizes them to TIMEX3 format.
    """

    def __init__(
        self,
        language: str = "english",
        document_type: str = "news",
        resources_path: str | None = None,
        find_dates: bool = True,
        find_times: bool = True,
        find_durations: bool = True,
        find_sets: bool = True,
        find_temponyms: bool = False,
        use_spacy: bool = True,
    ):
        """
        Initialize HeidelTime.

        Args:
            language: Language of documents to process (english, german, etc.)
            document_type: Type of documents (news, narrative, colloquial, scientific)
            resources_path: Custom path to resources directory
            find_dates: Whether to extract DATE expressions
            find_times: Whether to extract TIME expressions
            find_durations: Whether to extract DURATION expressions
            find_sets: Whether to extract SET expressions
            find_temponyms: Whether to extract TEMPONYM expressions
            use_spacy: Whether to use spaCy for tokenization/POS tagging
        """
        self.language = language
        self.document_type = document_type
        self.find_dates = find_dates
        self.find_times = find_times
        self.find_durations = find_durations
        self.find_sets = find_sets
        self.find_temponyms = find_temponyms
        self.use_spacy = use_spacy

        # Load resources
        self.repatterns = load_repatterns(language, find_temponyms, resources_path)
        self.norm_manager = NormalizationManager(language, find_temponyms, resources_path)
        self.rules = load_rules(language, self.repatterns, find_temponyms, resources_path)

        # Group rules by type
        self.rules_by_type = {}
        for rule in self.rules:
            self.rules_by_type.setdefault(rule.rule_type, []).append(rule)

    def process(self, text: str, dct: str | None = None) -> str:
        """
        Process text and return TimeML-annotated XML.

        Args:
            text: Input document text
            dct: Document Creation Time in YYYY-MM-DD or YYYYMMDD format

        Returns:
            TimeML XML string with TIMEX3 annotations
        """
        timexes = self.extract(text, dct)

        # Normalize DCT format
        dct_normalized = None
        if dct:
            dct_normalized = dct if "-" in dct else f"{dct[:4]}-{dct[4:6]}-{dct[6:8]}"

        return format_timeml(text, timexes, dct_normalized)

    def extract(self, text: str, dct: str | None = None) -> list[Timex3]:
        """
        Extract temporal expressions and return Timex3 objects.

        Args:
            text: Input document text
            dct: Document Creation Time in YYYY-MM-DD or YYYYMMDD format

        Returns:
            List of Timex3 objects
        """
        if dct:
            dct = self._normalize_dct_string(dct)
        if dct and not self._is_valid_dct(dct):
            raise ValueError(f"Invalid DCT format: '{dct}'. Expected YYYYMMDD or YYYY-MM-DD.")

        # Tokenize
        sentences = tokenize(text, self.language, self.use_spacy)

        # Find timexes sentence by sentence
        all_timexes = []
        timex_id = 1
        flag_historic = False

        for sentence in sentences:
            if self.find_dates:
                found, timex_id = find_timexes(
                    sentence,
                    self.rules_by_type.get("DATE", []),
                    self.norm_manager,
                    self.language,
                    timex_id,
                )
                all_timexes.extend(found)

            if self.find_times:
                found, timex_id = find_timexes(
                    sentence,
                    self.rules_by_type.get("TIME", []),
                    self.norm_manager,
                    self.language,
                    timex_id,
                )
                all_timexes.extend(found)

            # Check for BC dates (for narrative type)
            if self.document_type in ("narrative", "narratives"):
                for t in all_timexes:
                    if t.timex_value.startswith("BC"):
                        flag_historic = True
                        break

            if self.find_sets:
                found, timex_id = find_timexes(
                    sentence,
                    self.rules_by_type.get("SET", []),
                    self.norm_manager,
                    self.language,
                    timex_id,
                )
                all_timexes.extend(found)

            if self.find_durations:
                found, timex_id = find_timexes(
                    sentence,
                    self.rules_by_type.get("DURATION", []),
                    self.norm_manager,
                    self.language,
                    timex_id,
                )
                all_timexes.extend(found)

            if self.find_temponyms:
                found, timex_id = find_timexes(
                    sentence,
                    self.rules_by_type.get("TEMPONYM", []),
                    self.norm_manager,
                    self.language,
                    timex_id,
                )
                all_timexes.extend(found)

        # Pre-processing overlap removal
        all_timexes = remove_overlaps_preprocessing(all_timexes)

        # Disambiguate UNDEF values
        self._specify_ambiguous_values(all_timexes, sentences, dct)

        # Post-processing overlap removal
        all_timexes = remove_overlaps_postprocessing(all_timexes)

        # Remove invalids
        all_timexes = [t for t in all_timexes if t.timex_value != "REMOVE"]

        return all_timexes

    def _specify_ambiguous_values(
        self, timexes: list[Timex3], sentences: list[Sentence], dct: str | None
    ):
        """Resolve UNDEF values in timex expressions."""
        # Build linear list of DATE/TIME timexes (+ DURATIONs with emptyValue)
        linear_dates = []
        for t in timexes:
            if t.timex_type in ("DATE", "TIME") or t.timex_type == "DURATION" and t.empty_value:
                linear_dates.append(t)

        # Find sentence for each timex
        sent_map = {}
        for t in linear_dates:
            for s in sentences:
                if s.begin <= t.begin and s.end >= t.end:
                    sent_map[id(t)] = s
                    break

        # Process each timex
        for i, t in enumerate(linear_dates):
            sentence = sent_map.get(id(t))
            if sentence is None:
                # Fallback: use first sentence
                sentence = sentences[0] if sentences else Sentence(begin=0, end=len(""), text="")

            if t.timex_type in ("DATE", "TIME"):
                new_value = specify_ambiguous_value(
                    t.timex_value,
                    t,
                    i,
                    linear_dates,
                    sentence,
                    self.norm_manager,
                    self.repatterns,
                    self.document_type,
                    dct,
                    self.language,
                )
                t.timex_value = new_value

            # Also handle emptyValue
            if t.empty_value:
                new_empty = specify_ambiguous_value(
                    t.empty_value,
                    t,
                    i,
                    linear_dates,
                    sentence,
                    self.norm_manager,
                    self.repatterns,
                    self.document_type,
                    dct,
                    self.language,
                )
                t.empty_value = new_empty

    @staticmethod
    def _normalize_dct_string(dct: str) -> str:
        """Coerce common variants (e.g. YYYY-M-D) to YYYY-MM-DD; validate calendar date."""
        s = dct.strip()
        if re.match(r"^\d{8}$", s):
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
        if m:
            y, mo, d_ = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            try:
                date(y, mo, d_)
                return f"{y:04d}-{mo:02d}-{d_:02d}"
            except ValueError:
                pass
        return s

    @staticmethod
    def _is_valid_dct(dct: str) -> bool:
        """Check if DCT is in valid format."""
        if dct is None:
            return True
        return bool(re.match(r"^\d{8}$", dct) or re.match(r"^\d{4}-\d{2}-\d{2}", dct))
