"""
TimeML Output Formatter: produces TimeML XML output with TIMEX3 tags.
"""

from ..models import Timex3


def format_timeml(text: str, timexes: list[Timex3], dct_value: str = None) -> str:
    """
    Format document text with inline TIMEX3 annotations in TimeML format.

    Returns XML string.
    """
    # Sort timexes by begin position (ascending), then by length (descending)
    sorted_timexes = sorted(timexes, key=lambda t: (t.begin, -(t.end - t.begin)))

    # Remove any with REMOVE value
    sorted_timexes = [t for t in sorted_timexes if t.timex_value != "REMOVE"]

    # Build output by inserting tags
    result = []
    last_pos = 0

    for t in sorted_timexes:
        if t.begin < last_pos:
            continue  # skip overlapping

        # Add text before this timex
        result.append(_escape_xml(text[last_pos : t.begin]))

        # Build TIMEX3 tag
        attrs = (
            f'tid="{t.timex_id}" type="{t.timex_type}" value="{_escape_xml_attr(t.timex_value)}"'
        )
        if t.timex_quant:
            attrs += f' quant="{_escape_xml_attr(t.timex_quant)}"'
        if t.timex_freq:
            attrs += f' freq="{_escape_xml_attr(t.timex_freq)}"'
        if t.timex_mod:
            attrs += f' mod="{_escape_xml_attr(t.timex_mod)}"'

        covered_text = text[t.begin : t.end]
        result.append(f"<TIMEX3 {attrs}>{_escape_xml(covered_text)}</TIMEX3>")
        last_pos = t.end

    # Add remaining text
    result.append(_escape_xml(text[last_pos:]))

    body = "".join(result)

    # Wrap in TimeML document
    dct_tag = ""
    if dct_value:
        dct_tag = (
            f'\n<DCT><TIMEX3 tid="t0" type="DATE" value="{_escape_xml_attr(dct_value)}" '
            f'temporalFunction="false" functionInDocument="CREATION_TIME">'
            f"{_escape_xml(dct_value)}</TIMEX3></DCT>"
        )

    return (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE TimeML SYSTEM "TimeML.dtd">\n'
        f"<TimeML>{dct_tag}\n"
        f"<TEXT>{body}</TEXT>\n"
        "</TimeML>\n"
    )


def _escape_xml(text: str) -> str:
    """Escape XML special characters in text content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_xml_attr(text: str) -> str:
    """Escape XML special characters in attribute values."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
