"""RePattern Manager: loads regex pattern files and builds alternation patterns."""

import re

from .scanner import scan_resources

# Unicode space replacement (matches Java's GenericResourceManager.replaceSpaces)
SPACE_REGEX = r"[\u2000-\u200a \u202f\u205f\u3000\u00a0\u1680\u180e]+"


def _replace_spaces(text: str) -> str:
    """Replace literal spaces with Unicode-aware whitespace regex."""
    return text.replace(" ", SPACE_REGEX)


def _effective_length(pattern: str) -> int:
    """Calculate effective length for sorting (longer patterns first)."""
    s = re.sub(r"\[[^\]]*\]", "X", pattern)
    s = re.sub(r"\?", "", s)
    s = re.sub(r"\\.(?:\{([^\}])+\})?", lambda m: "X" + (m.group(1) or ""), s)
    return len(s)


def _finalize_pattern(pattern: str) -> str:
    """
    Finalize a repattern:
    1. Strip leading |
    2. Convert user parentheses to non-capturing
    3. Wrap in capturing group
    """
    # strip leading |
    if pattern.startswith("|"):
        pattern = pattern[1:]
    # convert (X to (?:X  -- but not (?
    pattern = re.sub(r"\(([^?])", r"(?:\1", pattern)
    # wrap in capturing group
    pattern = "(" + pattern + ")"
    return pattern


def load_repatterns(
    language: str, load_temponyms: bool = False, resources_path: str | None = None
) -> dict[str, str]:
    """
    Load all repattern resources for a language.

    Returns dict mapping pattern name -> finalized regex string.
    """
    resource_files = scan_resources(language, "repattern", resources_path)
    patterns = {}

    for name, filepath in resource_files.items():
        # skip temponym patterns if not requested
        if "Temponym" in name and not load_temponyms:
            continue

        lines = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if line.startswith("//") or line == "":
                    continue
                lines.append(_replace_spaces(line))

        # sort by effective length descending
        lines.sort(key=_effective_length, reverse=True)

        # join with |
        raw = "|".join(lines)
        patterns[name] = _finalize_pattern(raw)

    return patterns
