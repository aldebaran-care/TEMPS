"""
Tokenizer: sentence splitting and POS tagging.
Supports spaCy, or a simple whitespace fallback.
"""

from ..models import Sentence, Token


def tokenize_with_spacy(text: str, language: str = "english") -> list[Sentence]:
    """Tokenize text using spaCy, creating Sentence and Token objects."""
    import spacy

    model_map = {
        "english": "en_core_web_sm",
        "englishcoll": "en_core_web_sm",
        "englishsci": "en_core_web_sm",
        "german": "de_core_news_sm",
        "dutch": "nl_core_news_sm",
        "french": "fr_core_news_sm",
        "spanish": "es_core_news_sm",
        "italian": "it_core_news_sm",
        "portuguese": "pt_core_news_sm",
        "chinese": "zh_core_web_sm",
        "russian": "ru_core_news_sm",
    }

    model_name = model_map.get(language, "en_core_web_sm")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Fallback: try loading with xx (multilingual)
        try:
            nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            )

    doc = nlp(text)
    sentences = []
    token_id = 0
    sent_id = 0

    for sent in doc.sents:
        tokens = []
        for tok in sent:
            token_id += 1
            tokens.append(
                Token(
                    begin=tok.idx,
                    end=tok.idx + len(tok.text),
                    text=tok.text,
                    pos=tok.tag_,  # Penn Treebank tags (compatible with TreeTagger for English)
                    token_id=token_id,
                )
            )
        sent_id += 1
        sentences.append(
            Sentence(
                begin=sent.start_char,
                end=sent.end_char,
                text=sent.text,
                tokens=tokens,
                sentence_id=sent_id,
            )
        )

    return sentences


def tokenize_simple(text: str) -> list[Sentence]:
    """
    Simple tokenizer that splits on sentence-ending punctuation.
    No POS tags. Use when spaCy is not available.
    """
    import re

    # Split on sentence boundaries (. ! ?) followed by space and uppercase
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = []
    token_id = 0
    offset = 0

    for sent_id, raw in enumerate(raw_sents, 1):
        # Find the actual start position in the original text
        start = text.find(raw, offset)
        if start == -1:
            start = offset
        end = start + len(raw)

        # Simple whitespace tokenization
        tokens = []
        for m in re.finditer(r"\S+", raw):
            token_id += 1
            tokens.append(
                Token(
                    begin=start + m.start(),
                    end=start + m.end(),
                    text=m.group(),
                    pos="",
                    token_id=token_id,
                )
            )

        sentences.append(
            Sentence(
                begin=start,
                end=end,
                text=raw,
                tokens=tokens,
                sentence_id=sent_id,
            )
        )
        offset = end

    return sentences


def tokenize(text: str, language: str = "english", use_spacy: bool = True) -> list[Sentence]:
    """Tokenize text, preferring spaCy if available."""
    if use_spacy:
        try:
            return tokenize_with_spacy(text, language)
        except (ImportError, RuntimeError):
            pass
    return tokenize_simple(text)
