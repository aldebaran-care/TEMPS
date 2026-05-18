"""Rule Manager: loads extraction rules and compiles regex patterns."""

import re
import warnings
from dataclasses import dataclass

from .scanner import scan_resources


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    normalization: str
    rule_type: str = ""
    offset: str = ""
    quant: str = ""
    freq: str = ""
    mod: str = ""
    pos_constraint: str = ""
    empty_value: str = ""
    fast_check: re.Pattern | None = None


# Pattern to parse rule lines
PA_READ_RULES = re.compile(r'RULENAME="(.*?)",EXTRACTION="(.*?)",NORM_VALUE="(.*?)"(.*)')

# Unicode space replacement
SPACE_REGEX = r"[\u2000-\u200a \u202f\u205f\u3000\u00a0\u1680\u180e]+"


def _replace_spaces(text: str) -> str:
    return text.replace(" ", SPACE_REGEX)


def _extract_field(line: str, field_name: str) -> str:
    """Extract a quoted field value from the remainder of a rule line."""
    m = re.search(f'{field_name}="(.*?)"', line)
    return m.group(1) if m else ""


def _substitute_repatterns(extraction: str, repatterns: dict[str, str]) -> str:
    """Replace %reXXX references with actual regex patterns."""
    pa_variable = re.compile(r"%(re[a-zA-Z0-9]*)")
    for m in pa_variable.finditer(extraction):
        pattern_name = m.group(1)
        if pattern_name not in repatterns:
            raise ValueError(f"Pattern %{pattern_name} not found in repattern resources")
        extraction = extraction.replace(f"%{pattern_name}", repatterns[pattern_name])
    return extraction


# Rule type ordering: DATE > TIME > DURATION > SET > rest
_RULE_TYPE_ORDER = {"daterules": 0, "timerules": 1, "durationrules": 2, "setrules": 3}
_RULE_TYPE_MAP = {
    "daterules": "DATE",
    "timerules": "TIME",
    "durationrules": "DURATION",
    "setrules": "SET",
    "temponymrules": "TEMPONYM",
}


def load_rules(
    language: str,
    repatterns: dict[str, str],
    load_temponyms: bool = False,
    resources_path: str | None = None,
) -> list[Rule]:
    """
    Load all rules for a language, substitute repatterns, and compile.

    Returns a list of Rule objects sorted by type then name.
    """
    resource_files = scan_resources(language, "rules", resources_path)

    # Sort resource keys: daterules > timerules > durationrules > setrules > rest
    sorted_keys = sorted(resource_files.keys(), key=lambda k: _RULE_TYPE_ORDER.get(k, 99))

    rules = []
    seen_names = set()

    for resource_key in sorted_keys:
        if resource_key == "temponymrules" and not load_temponyms:
            continue
        if resource_key == "intervalrules":
            continue  # intervals handled separately

        rule_type = _RULE_TYPE_MAP.get(resource_key, resource_key.upper())
        filepath = resource_files[resource_key]

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if line.startswith("//") or line == "":
                    continue

                m = PA_READ_RULES.match(line)
                if not m:
                    continue

                rule_name = m.group(1)
                rule_extraction = m.group(2)
                rule_normalization = m.group(3)
                remainder = m.group(4) or ""

                if rule_name in seen_names:
                    continue
                seen_names.add(rule_name)

                # Replace literal spaces with [\s]+ BEFORE substituting repatterns.
                # Repatterns already have their own internal space handling via
                # Unicode-aware character classes, so we only transform
                # EXTRACTION-level spaces here.
                rule_extraction = rule_extraction.replace(" ", r"[\s]+")

                # Substitute repattern references
                try:
                    rule_extraction = _substitute_repatterns(rule_extraction, repatterns)
                except ValueError as e:
                    print(f"Warning: {e} in rule {rule_name}")
                    continue

                # Compile pattern
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        pattern = re.compile(rule_extraction)
                except re.error as e:
                    print(f"Warning: Cannot compile pattern for rule {rule_name}: {e}")
                    continue

                # Extract optional fields
                offset = _extract_field(remainder, "OFFSET") if "OFFSET" in remainder else ""
                quant = _extract_field(remainder, "NORM_QUANT") if "NORM_QUANT" in remainder else ""
                freq = _extract_field(remainder, "NORM_FREQ") if "NORM_FREQ" in remainder else ""
                mod = _extract_field(remainder, "NORM_MOD") if "NORM_MOD" in remainder else ""
                pos_constraint = (
                    _extract_field(remainder, "POS_CONSTRAINT")
                    if "POS_CONSTRAINT" in remainder
                    else ""
                )
                empty_value = (
                    _extract_field(remainder, "EMPTY_VALUE") if "EMPTY_VALUE" in remainder else ""
                )

                # Fast check pattern
                fast_check = None
                if "FAST_CHECK" in remainder:
                    fc_str = _extract_field(remainder, "FAST_CHECK")
                    if fc_str:
                        try:
                            fc_str = _substitute_repatterns(fc_str, repatterns)
                            fc_str = fc_str.replace(" ", r"[\s]+")
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", FutureWarning)
                                fast_check = re.compile(fc_str)
                        except (re.error, ValueError):
                            pass

                rules.append(
                    Rule(
                        name=rule_name,
                        pattern=pattern,
                        normalization=rule_normalization,
                        rule_type=rule_type,
                        offset=offset,
                        quant=quant,
                        freq=freq,
                        mod=mod,
                        pos_constraint=pos_constraint,
                        empty_value=empty_value,
                        fast_check=fast_check,
                    )
                )

    return rules
