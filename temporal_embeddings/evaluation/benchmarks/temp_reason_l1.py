from pathlib import Path
import json
from typing import List, Dict

from temporal_embeddings.evaluation.benchmarks.temp_reason_utils import add_query_anchored_l1_negatives


def create_temp_reason_l1_benchmark(
    num_negative_samples: int = 5,
    hard_negative_pool: int = 20,
    seed: int = 42,
) -> None:
    main_folder: Path = Path("data/evaluation/temp_reason_l1")
    input_path: Path = main_folder / "data.json"
    output_path: Path = main_folder / "processed_data.json"
    
    main_folder.mkdir(parents=True, exist_ok=True)
    
    with input_path.open("r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)
        
    processed_data: List[Dict] = []
    
    for item in data:
        paragraphs: List[str] = []
        
        answer_date_str: str = item["text_answers"]["text"][0]
        paragraphs.append(answer_date_str)
        
        question = item["question"]
        
        entry = {
            "question": question,
            "paragraphs": paragraphs,
            "answer": [0]
        }
        
        processed_data.append(entry)

    stats = add_query_anchored_l1_negatives(
        processed_data,
        num_negative_samples=num_negative_samples,
        hard_negative_pool=hard_negative_pool,
        seed=seed,
    )
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed dataset saved to: {output_path}")
    print(
        f"Added query-anchored temporal hard negatives to {stats['query_anchor_count']} L1 items "
        f"({stats['fallback_count']} fell back to random sampling)"
    )
