import json
from pathlib import Path

from temporal_embeddings.evaluation.benchmarks.utils.fetch_random_paragraphs import fetch_random_paragraphs

def create_time_sensitive_qa_benchmark(add_negative_samples: bool) -> None:
    main_folder: Path = Path("data/evaluation/time_sensitive_qa")
    input_path: Path = main_folder / "human_annotated_test.json"
    output_file: Path = main_folder / "processed_human_annotated_test.json"

    main_folder.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    processed_data = []

    all_paragraphs = set()
    
    for item in data:
        paragraphs = [p[:400] for p in item.get("paras", [])]

        all_paragraphs.update(paragraphs)
        
        for q_pair in item.get("questions", []):
            question_text = q_pair[0][:400]
            answers = q_pair[1]
            
            for ans in answers:
                if ans["answer"] == "":
                    continue

                para_idx = ans["para"]
                
                entry = {
                    "question": question_text,
                    "paragraphs": paragraphs + fetch_random_paragraphs(question_text, list(all_paragraphs - set(paragraphs)), 10, use_bm25=True) if add_negative_samples else paragraphs,
                    "answer": [para_idx]
                }
                
                processed_data.append(entry)

    with output_file.open("w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, indent=2, ensure_ascii=False)

    print(f"Processed dataset saved to: {output_file}")