from pathlib import Path
import json
import csv
from typing import List, Dict


def create_temp_rag_eval_dataset() -> None:
    main_folder: Path = Path("data/evaluation/temp_rag_eval")
    input_path: Path = main_folder / "data.csv"
    output_path: Path = main_folder / "processed_data.json"
    
    main_folder.mkdir(parents=True, exist_ok=True)
    
    processed_data: List[Dict] = []
    
    # Try different encodings
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    
    for encoding in encodings:
        print(f"Trying encoding: {encoding}")
        try:
            with input_path.open("r", encoding=encoding, errors="replace") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    question: str = row["question"].strip()
                    gold_evidence_1: str = row["gold_evidence_1"].strip()
                    gold_evidence_2: str = row["gold_evidence_2"].strip()
                    
                    # Combine evidence as paragraphs, filtering out empty strings
                    paragraphs: List[str] = [p for p in [gold_evidence_1, gold_evidence_2] if p]
                    
                    entry = {
                        "question": question,
                        "paragraphs": paragraphs,
                        "answer": [i for i in range(len(paragraphs))]
                    }
                    
                    processed_data.append(entry)
            
            break
        
        except UnicodeDecodeError:
            continue
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Processed {len(processed_data)} entries")
    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":
    create_temp_rag_eval_dataset()
