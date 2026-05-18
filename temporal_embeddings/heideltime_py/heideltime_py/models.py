"""Data models replacing UIMA type system."""

from dataclasses import dataclass, field


@dataclass
class Token:
    begin: int
    end: int
    text: str
    pos: str = ""
    token_id: int = 0


@dataclass
class Sentence:
    begin: int
    end: int
    text: str
    tokens: list[Token] = field(default_factory=list)
    sentence_id: int = 0
    filename: str = ""


@dataclass
class Timex3:
    begin: int = 0
    end: int = 0
    text: str = ""
    timex_type: str = ""
    timex_value: str = ""
    timex_id: str = ""
    timex_quant: str = ""
    timex_freq: str = ""
    timex_mod: str = ""
    empty_value: str = ""
    found_by_rule: str = ""
    first_tok_id: int = 0
    all_tok_ids: str = ""
    filename: str = ""
    sent_id: int = 0


@dataclass
class Document:
    text: str
    sentences: list[Sentence] = field(default_factory=list)
    timexes: list[Timex3] = field(default_factory=list)
    dct_value: str | None = None
