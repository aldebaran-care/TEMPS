"""
Timex Finder: applies extraction rules to sentences to find temporal expressions.
Port of HeidelTime.findTimexes().
"""

import re

from ..models import Sentence, Timex3
from ..resources.normalization_manager import NormalizationManager
from ..resources.rule_manager import Rule
from .context_analyzer import check_infront_behind, check_token_boundaries
from .normalizer import apply_rule_functions, correct_duration_value


def _check_pos_constraint(sentence: Sentence, pos_constraint: str, match: re.Match) -> bool:
    """Check POS constraints defined in a rule."""
    pa_constraint = re.compile(r"group\(([0-9]+)\):(.*?):")
    for mr in pa_constraint.finditer(pos_constraint):
        group_num = int(mr.group(1))
        token_begin = sentence.begin + match.start(group_num)
        token_end = sentence.begin + match.end(group_num)
        required_pos = mr.group(2)

        # Find the token at this position
        pos_found = ""
        for tok in sentence.tokens:
            if tok.begin == token_begin:
                pos_found = tok.pos
                break

        if not re.fullmatch(required_pos, pos_found):
            return False
    return True


def find_timexes(
    sentence: Sentence,
    rules: list[Rule],
    norm_manager: NormalizationManager,
    language: str = "english",
    timex_id_counter: int = 1,
    group_gran: bool = True,
) -> tuple:
    """
    Find temporal expressions in a sentence using rules.

    Returns (list of Timex3, updated timex_id_counter).
    """
    found = []

    # Sort rules by name (important for overlap resolution later)
    sorted_rules = sorted(rules, key=lambda r: r.name)

    for rule in sorted_rules:
        # Fast check
        if rule.fast_check is not None:
            if not rule.fast_check.search(sentence.text):
                continue

        # Find all matches
        for match in rule.pattern.finditer(sentence.text):
            match_start = match.start()
            match_end = match.end()

            # Check boundaries
            if not check_token_boundaries(match_start, match_end, sentence):
                continue
            if not check_infront_behind(match_start, match_end, sentence.text):
                continue

            # Check POS constraint
            if rule.pos_constraint:
                if not _check_pos_constraint(sentence, rule.pos_constraint, match):
                    continue

            # Apply offset if any
            if rule.offset:
                offset_match = re.match(r"group\((\d+)\)-group\((\d+)\)", rule.offset)
                if offset_match:
                    start_group = int(offset_match.group(1))
                    end_group = int(offset_match.group(2))
                    try:
                        match_start = match.start(start_group)
                        match_end = match.end(end_group)
                    except IndexError:
                        pass

            # Normalize
            value = apply_rule_functions(rule.normalization, match, norm_manager, language)
            if value is None:
                continue

            quant = ""
            if rule.quant:
                quant = apply_rule_functions(rule.quant, match, norm_manager, language) or ""

            freq = ""
            if rule.freq:
                freq = apply_rule_functions(rule.freq, match, norm_manager, language) or ""

            mod = ""
            if rule.mod:
                mod = apply_rule_functions(rule.mod, match, norm_manager, language) or ""

            empty_value = ""
            if rule.empty_value:
                empty_value = (
                    apply_rule_functions(rule.empty_value, match, norm_manager, language) or ""
                )
                empty_value = correct_duration_value(empty_value)

            if group_gran:
                value = correct_duration_value(value)

            # Determine found_by_rule suffix
            found_by = rule.name
            if rule.rule_type in ("DATE", "TIME"):
                if value.startswith("X") or value.startswith("UNDEF"):
                    found_by = rule.name + "-relative"
                else:
                    found_by = rule.name + "-explicit"

            # Build token IDs
            abs_start = match_start + sentence.begin
            abs_end = match_end + sentence.begin
            first_tok_id = 0
            all_tok_ids = ""
            for tok in sentence.tokens:
                if tok.begin <= abs_start and tok.end > abs_start:
                    first_tok_id = tok.token_id
                    all_tok_ids = f"BEGIN<-->{tok.token_id}"
                if tok.begin > abs_start and tok.end <= abs_end:
                    all_tok_ids += f"<-->{tok.token_id}"

            timex = Timex3(
                begin=abs_start,
                end=abs_end,
                text=sentence.text[match_start:match_end],
                timex_type=rule.rule_type,
                timex_value=value,
                timex_id=f"t{timex_id_counter}",
                timex_quant=quant,
                timex_freq=freq,
                timex_mod=mod,
                empty_value=empty_value,
                found_by_rule=found_by,
                first_tok_id=first_tok_id,
                all_tok_ids=all_tok_ids,
                filename=sentence.filename,
                sent_id=sentence.sentence_id,
            )
            found.append(timex)
            timex_id_counter += 1

    return found, timex_id_counter
