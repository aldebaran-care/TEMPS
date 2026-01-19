from pathlib import Path
import json
from typing import List, Dict

def create_temp_reason_l1_benchmark() -> None:
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
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed dataset saved to: {output_path}")