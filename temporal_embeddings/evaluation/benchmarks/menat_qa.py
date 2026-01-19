from pathlib import Path
import json
from typing import List, Dict

def create_menat_qa_benchmark(dataset_name: str) -> None:
    main_folder: Path = Path("data/evaluation/menat_qa")
    input_path: Path = main_folder / Path("MenatQA.json")
    
    if dataset_name.lower() == "menat_qa":
        output_file: Path = main_folder / "processed_menat_qa.json"
    
    elif dataset_name.lower() == "menat_qa_granularity":
        output_file: Path = main_folder / "processed_menat_qa_granularity.json"
    
    elif dataset_name.lower() == "menat_qa_counterfactual":
        output_file: Path = main_folder / "processed_menat_qa_counterfactual.json"
    
    elif dataset_name.lower() == "menat_qa_expand":
        output_file: Path = main_folder / "processed_menat_qa_expand.json"
    
    elif dataset_name.lower() == "menat_qa_narrow":
        output_file: Path = main_folder / "processed_menat_qa_narrow.json"
    
    else:
        raise ValueError("Unsupported dataset name.")

    main_folder.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile:
        menat_data: List[Dict] = json.load(infile)

    processed_data: List[Dict] = []
    
    for item in menat_data:
        if item["updated_answer"] == "unanswerable" or item["updated_answer"] != item["answer"]:
            continue

        if dataset_name.lower() == "menat_qa_granularity" and item.get("type") != "granularity":
            continue
        
        if dataset_name.lower() == "menat_qa_counterfactual" and item.get("type") != "counterfactual":
            continue
        
        if dataset_name.lower() == "menat_qa_expand" and item.get("type") != "expand":
            continue
        
        if dataset_name.lower() == "menat_qa_narrow" and item.get("type") != "narrow":
            continue

        question: str = item.get("updated_question", "")
        
        paragraphs: List[str] = [ctx["text"] for ctx in item["context"]]
        
        answer: str = item.get("annotated_para", "")
        answer_index: int = next((i for i, p in enumerate(paragraphs) if answer in p), -1)

        entry = {
            "question": question,
            "paragraphs": paragraphs,
            "answer": [answer_index] if answer_index >= 0 else [0]
        }
        
        processed_data.append(entry)

    with output_file.open("w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, indent=2, ensure_ascii=False)

    if dataset_name.lower() == "menat_qa":
        print(f"Processed dataset saved to: {output_file}")
    
    else:
        print(f"Processed filtered dataset saved to: {output_file}")