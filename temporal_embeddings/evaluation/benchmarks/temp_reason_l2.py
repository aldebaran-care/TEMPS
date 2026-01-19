import json
from pathlib import Path
from typing import List, Dict

def create_temp_reason_l2_benchmark() -> None:
    main_folder: Path = Path("data/evaluation/temp_reason_l2")
    input_path: Path = main_folder / "data.json"
    output_path: Path = main_folder / "processed_data.json"
    
    main_folder.mkdir(parents=True, exist_ok=True)
    
    with input_path.open("r", encoding="utf-8") as f:
        data: List[Dict] = []

        for line in f:
            data.append(json.loads(line))
        
    processed_data: List[Dict] = []
    
    for item in data:
        paragraphs: List[str] = item["fact_context"].split("\n")
        text_answers: List[str] = item["text_answers"]["text"]
        neg_answers: List[str] = item["neg_answers"]

        answers: List[int] = []

        for p in paragraphs:
            is_answer = False
            
            for ans in text_answers:
                if ans in p:
                    is_answer = True

            for neg in neg_answers:
                if neg in p:
                    is_answer = False

            if is_answer:
                answers.append(paragraphs.index(p))
        
        question = item["question"]
        
        entry = {
            "question": question,
            "paragraphs": paragraphs,
            "answer": answers
        }
        
        processed_data.append(entry)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed dataset saved to: {output_path}")