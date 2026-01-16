import random
from typing import List

import rank_bm25

def fetch_random_paragraphs(question: str, paragraphs: List[str], num_paragraphs: int, use_bm25: bool) -> List[str]:
    if len(paragraphs) <= num_paragraphs:
        return paragraphs
    
    if use_bm25:
        return bm25_most_similar_sentences(question, paragraphs, top_k=num_paragraphs)
    
    return random.sample(paragraphs, num_paragraphs)

def bm25_most_similar_sentences(query_sentence: str, sentences: List[str], top_k: int = 5) -> List[str]:
    if not sentences:
        return []

    tokenized_corpus = [s.lower().split() for s in sentences]
    tokenized_query = query_sentence.lower().split()

    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_sentences: List[str] = [s[0] for s in ranked]

    return top_sentences[:top_k]