"""RAG pipeline for Time-Sensitive QA.

Pairs the trained temporal embedding model with a semantic retriever to select
top-k paragraphs, then asks a generator LLM to extract the answer. Computes
both retrieval (Recall@k, MRR, NDCG@k) and SQuAD-style QA metrics (EM, F1,
containment) so results can go straight into the paper.

This module is intentionally self-contained: benchmark loading, prompt
construction, generator wrapping, and QA metrics live here. The orchestrator
(`run_rag_evaluation.py`) handles retrieval scoring via the existing
`compute_temporal_similarities` / `compute_semantic_similarities` helpers.
"""

from __future__ import annotations

import json
import os
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


@dataclass
class RagItem:
    """One RAG query with retrieval labels, QA answers, and its data split."""

    question: str
    paragraphs: List[str]
    gold_paragraph_indices: List[int]
    gold_answers: List[str] = field(default_factory=list)
    split: Optional[str] = None


def load_rag_benchmark(
    processed_path: Path,
    raw_path: Path,
) -> List[RagItem]:
    """Join the retrieval-ready `processed_human_annotated_test.json` with the
    raw `human_annotated_test.json` to attach gold answer strings."""

    with processed_path.open("r", encoding="utf-8") as f:
        processed = json.load(f)

    with raw_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    question_to_answers: Dict[str, List[str]] = {}
    for entry in raw:
        for question_entry in entry.get("questions", []):
            if not question_entry or len(question_entry) < 2:
                continue
            question_text, answer_spans = question_entry[0], question_entry[1]
            answers = [span["answer"] for span in answer_spans if "answer" in span]
            if not answers:
                continue
            question_to_answers.setdefault(question_text, []).extend(answers)

    items: List[RagItem] = []
    missing = 0
    for entry in processed:
        question = entry["question"]
        gold_answers = question_to_answers.get(question, [])
        if not gold_answers:
            missing += 1
        items.append(
            RagItem(
                question=question,
                paragraphs=list(entry["paragraphs"]),
                gold_paragraph_indices=list(entry["answer"]),
                gold_answers=gold_answers,
                split=entry.get("split"),
            )
        )

    if missing:
        print(
            f"[load_rag_benchmark] warning: {missing}/{len(items)} questions "
            "had no matching gold answer text in the raw file."
        )

    return items


# ---------------------------------------------------------------------------
# Retrieval: pick top-k paragraphs from a precomputed similarity bucket.
# ---------------------------------------------------------------------------


def select_top_k(
    paragraphs: List[str],
    similarity_bucket: Dict[str, float],
    top_k: int,
) -> Tuple[List[int], List[str]]:
    """Return (top_k_indices, top_k_paragraphs) ranked by similarity descending.

    `similarity_bucket` maps paragraph text → score. Indices are returned with
    respect to the candidate list `paragraphs` (so they're comparable to the
    benchmark's gold paragraph indices)."""

    scored: List[Tuple[int, float]] = []
    for idx, paragraph in enumerate(paragraphs):
        scored.append((idx, float(similarity_bucket.get(paragraph, 0.0))))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    top_indices = [idx for idx, _ in top]
    top_paragraphs = [paragraphs[idx] for idx in top_indices]
    return top_indices, top_paragraphs


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. You will be given a "
    "question and several candidate passages retrieved from Wikipedia. Use "
    "ONLY the information in the passages to answer. The question contains "
    "an explicit time constraint — pick the passage(s) whose time range "
    "matches that constraint. Answer with the shortest possible text span "
    "that directly answers the question (a name, place, organization, date, "
    "or number). Do NOT add explanations, qualifiers, or full sentences. If "
    "no passage answers the question, reply exactly with: unknown."
)


def build_messages(
    question: str,
    top_k_paragraphs: List[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    context_lines = [f"[{i + 1}] {p}" for i, p in enumerate(top_k_paragraphs)]
    user_content = (
        "Passages:\n"
        + "\n".join(context_lines)
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# OpenRouter generator.
# ---------------------------------------------------------------------------


class Generator:
    """OpenRouter chat-completion client with concurrent requests and retries."""

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 64,
        batch_size: int = 4,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        request_timeout_seconds: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"Missing {api_key_env}. Add it to the project .env file or "
                "export it before running the RAG evaluation."
            )
        if batch_size <= 0:
            raise ValueError("generator.batch_size must be positive.")
        if max_retries < 0:
            raise ValueError("generator.max_retries must be non-negative.")

        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.base_url = base_url
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "TEMPS RAG Evaluation",
        }
        print(
            f"[Generator] OpenRouter model: {model_name_or_path} "
            f"({batch_size} concurrent requests)"
        )

    def _generate_one(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model_name_or_path,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenRouter request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                time.sleep(min(2**attempt, 30))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenRouter returned HTTP {response.status_code} after "
                        f"{attempt + 1} attempts: {response.text[:500]}"
                    )
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else float(2**attempt)
                except ValueError:
                    delay = float(2**attempt)
                time.sleep(min(delay, 60.0))
                continue

            try:
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid OpenRouter response (HTTP {response.status_code}): "
                    f"{response.text[:500]}"
                ) from exc
            return str(content or "").strip()

        raise AssertionError("OpenRouter retry loop exited unexpectedly.")

    def generate(self, prompts: List[List[Dict[str, str]]]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            return list(
                tqdm(
                    executor.map(self._generate_one, prompts),
                    total=len(prompts),
                    desc="Generating answers via OpenRouter",
                )
            )


# ---------------------------------------------------------------------------
# QA metrics: SQuAD-style EM / F1 plus a containment indicator.
# ---------------------------------------------------------------------------


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, strip articles, strip punctuation,
    collapse whitespace."""

    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLE_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def _strip_prediction(prediction: str) -> str:
    """Remove obvious chat-format leftovers (leading 'Answer:', surrounding
    quotes, citation markers like '[1]')."""

    pred = prediction.strip()
    pred = re.sub(r"^answer\s*[:\-]\s*", "", pred, flags=re.IGNORECASE)
    pred = pred.strip().strip('"').strip("'").strip()
    pred = re.sub(r"\s*\[\d+\]\s*", " ", pred).strip()
    if "\n" in pred:
        pred = pred.split("\n", 1)[0].strip()
    return pred


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    pred_norm = _normalize_answer(_strip_prediction(prediction))
    return float(
        any(pred_norm == _normalize_answer(gold) for gold in gold_answers)
    )


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    pred_tokens = _normalize_answer(_strip_prediction(prediction)).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def containment(prediction: str, gold_answers: List[str]) -> float:
    """1.0 if the normalized gold answer appears as a substring of the
    normalized prediction. Lenient indicator useful when the LLM emits a
    short sentence instead of a bare span."""

    if not gold_answers:
        return 0.0
    pred_norm = _normalize_answer(_strip_prediction(prediction))
    return float(
        any(_normalize_answer(gold) in pred_norm for gold in gold_answers)
    )


def aggregate_qa_metrics(
    predictions: List[str],
    gold_answers_list: List[List[str]],
) -> Dict[str, float]:
    assert len(predictions) == len(gold_answers_list)
    n = len(predictions)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "containment": 0.0, "answered": 0.0}

    em_sum = 0.0
    f1_sum = 0.0
    contain_sum = 0.0
    scored = 0
    for pred, golds in zip(predictions, gold_answers_list):
        if not golds:
            continue
        em_sum += exact_match(pred, golds)
        f1_sum += f1_score(pred, golds)
        contain_sum += containment(pred, golds)
        scored += 1

    denom = max(scored, 1)
    return {
        "em": em_sum / denom,
        "f1": f1_sum / denom,
        "containment": contain_sum / denom,
        "answered": scored / n,
    }
