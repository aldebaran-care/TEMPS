from pathlib import Path
import json
from typing import Dict, List

def create_ts_retriever_benchmark() -> None:
    base: Path = Path("data/evaluation/ts_retriever")
    query_path: Path = base / "query.json"
    doc_path: Path = base / "doc.json"
    output_path: Path = base / "processed_ts_retriever.json"

    with query_path.open("r", encoding="utf-8") as f:
        questions: List[Dict] = [{"query": q["query"], "positive_text": [pt[:400] for pt in q["positive_text"]]} for q in json.load(f)]

    with doc_path.open("r", encoding="utf-8") as f:
        paragraphs = [p[:400] for p in json.load(f)]

    output: List[Dict] = []

    for _, q in enumerate(questions):
        answer_idx: List[int] = []
        
        for positive_text in q["positive_text"]:
            try:
                index: int = paragraphs.index(positive_text)
            
            except ValueError:
                print(f"Warning: Positive text not found in paragraphs: {positive_text[:50]}...")
                paragraphs.append(positive_text)
                index = len(paragraphs) - 1
                
            answer_idx.append(index)

        answer_idx = []
        
        for positive_text in q["positive_text"]:
            answer_idx.append(paragraphs.index(positive_text))

        entry = {
            "question": q["query"],
            "paragraphs": paragraphs,
            "answer": answer_idx,
        }

        output.append(entry)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Processed dataset saved to: {output_path}")