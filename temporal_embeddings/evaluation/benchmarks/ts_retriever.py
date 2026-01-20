from pathlib import Path
import json
from typing import Dict, List

def create_ts_retriever_benchmark() -> None:
    base: Path = Path("data/evaluation/ts_retriever")
    query_path: Path = base / "query.json"
    doc_path: Path = base / "doc.json"
    output_path: Path = base / "processed_ts_retriever.json"

    with query_path.open("r", encoding="utf-8") as f:
        questions: List[Dict] = [{"query": q["query"], "positive_text": [ps for pt in q["positive_text"] for ps in pt.split("\n") if len(ps.strip().split()) > 3]} for q in json.load(f)]

    with doc_path.open("r", encoding="utf-8") as f:
        paragraphs: List[str] = list(set([s for p in json.load(f) for s in p.split("\n") if len(s.strip().split()) > 3]))

    output: List[Dict] = []

    for i, q in enumerate(questions):
        answer_idx: List[int] = []
        
        for positive_sentence in q["positive_text"]:
            try:
                index: int = paragraphs.index(positive_sentence)
            
            except ValueError:
                print(f"Warning: Positive sentence not found in paragraphs: {positive_sentence[:50]}...")
                paragraphs.append(positive_sentence)
                index = len(paragraphs) - 1
            
            answer_idx.append(index)

        entry = {
            "question": q["query"],
            "paragraphs": paragraphs if i == 0 else q["positive_text"],
            "answer": answer_idx if i == 0 else list(range(len(q["positive_text"]))),
        }

        output.append(entry)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Processed dataset saved to: {output_path}")