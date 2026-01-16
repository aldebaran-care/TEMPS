import json
from pathlib import Path
import argparse
import random
from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta

from temporal_embeddings.evaluation.benchmarks.time_sensitive_qa import create_time_sensitive_qa_benchmark
from temporal_embeddings.evaluation.benchmarks.ts_retriever import create_ts_retriever_benchmark

def create_evaluation_dataset(dataset_name):
    if dataset_name.lower().startswith("menat_qa"):
        main_folder = Path("data/evaluation/menat_qa")
        input_path = main_folder / Path("MenatQA.json")
        
        if dataset_name.lower() == "menat_qa":
            output_file = main_folder / "processed_menat_qa.json"
        
        elif dataset_name.lower() == "menat_qa_granularity":
            output_file = main_folder / "processed_menat_qa_granularity.json"
        
        elif dataset_name.lower() == "menat_qa_counterfactual":
            output_file = main_folder / "processed_menat_qa_counterfactual.json"
        
        elif dataset_name.lower() == "menat_qa_expand":
            output_file = main_folder / "processed_menat_qa_expand.json"
        
        elif dataset_name.lower() == "menat_qa_narrow":
            output_file = main_folder / "processed_menat_qa_narrow.json"
        
        else:
            print(f"Dataset '{dataset_name}' is not supported.")
            return

        main_folder.mkdir(parents=True, exist_ok=True)

        with input_path.open("r", encoding="utf-8") as infile:
            menat_data = json.load(infile)

        processed_data = []

        all_paragraphs = set([ctx["text"] for item in menat_data for ctx in item["context"]])
        
        for item in menat_data:
            if dataset_name.lower() == "menat_qa_granularity" and item.get("type") != "granularity":
                continue
            
            if dataset_name.lower() == "menat_qa_counterfactual" and item.get("type") != "counterfactual":
                continue
            
            if dataset_name.lower() == "menat_qa_expand" and item.get("type") != "expand":
                continue
            
            if dataset_name.lower() == "menat_qa_narrow" and item.get("type") != "narrow":
                continue

            question = item.get("question", "")
            
            paragraphs = [ctx["text"] for ctx in item["context"]]
            if len(paragraphs) <= 3:
                continue

            additional_paragraphs = list(all_paragraphs - set(paragraphs))
            sampled_paragraphs = random.sample(additional_paragraphs, min(10, len(additional_paragraphs)))
            paragraphs += sampled_paragraphs
            random.shuffle(paragraphs)
            
            answer = item.get("annotated_para", "")
            answer_index = next((i for i, p in enumerate(paragraphs) if answer in p), -1)

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

    elif dataset_name.lower().startswith("ts_retriever"):
        create_ts_retriever_benchmark()

    elif dataset_name.lower().startswith("time_sensitive_qa"):
        create_time_sensitive_qa_benchmark()

    elif dataset_name.lower().startswith("temp_reason_l1"):
        main_folder = Path("data/evaluation/temp_reason_l1")
        input_path = main_folder / "data.json"
        output_path = main_folder / "processed_data.json"
        
        main_folder.mkdir(parents=True, exist_ok=True)
        
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        processed_data = []
        
        for item in data:
            paragraphs = []
            
            answer_date_str = item["text_answers"]["text"][0]
            paragraphs.append(answer_date_str)

            question_date_str = item["date"]
            question_date_obj = datetime.strptime(question_date_str, "%B %d, %Y")
            
            question = item["question"]
            
            for offset_months in range(-2, 3, 1):
                neighbor_date = question_date_obj + relativedelta(months=offset_months)
                neighbor_str = neighbor_date.strftime("%b, %Y")
                neighbor_str = f"{neighbor_str.split(', ')[0]}, {neighbor_str.split(', ')[1]}"
                paragraphs.append(neighbor_str)
            
            random.shuffle(paragraphs)
            correct_idx = paragraphs.index(answer_date_str)
            
            entry = {
                "question": question,
                "paragraphs": paragraphs,
                "answer": [correct_idx]
            }
            
            processed_data.append(entry)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        print(f"Processed dataset saved to: {output_path}")
        
    else:
        print(f"Dataset '{dataset_name}' is not supported.")

def fetch_random_paragraphs(pargraphs: List[str], num_paragraphs: int) -> List[str]:
    if len(pargraphs) <= num_paragraphs:
        return pargraphs
    
    return random.sample(pargraphs, num_paragraphs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation dataset.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to process")
    args = parser.parse_args()
    create_evaluation_dataset(args.dataset_name)