from notion_client import Client
import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NOTION_API_KEY")

notion = Client(auth=api_key)

DATABASE_ID = "2bf4c2eb753080abb961f6435cb303da"

def list_databases():
    data = notion.search(filter={"property": "object", "value": "data_source"})

    for db in data["results"]:
        print(db["title"][0]["plain_text"], " → ", db["id"])

def log_metrics_to_notion(
    id: str,
    model: str,
    benchmark: str,
    metrics: Dict[str, float],
    external_model: str = None,
    command: str = None,
    comment: str = None,
    k: int = None,
    alpha: float = None,
    num_negative_samples: int = None
):
    properties = {
        "ID": {"title": [{"text": {"content": id}}]},
        "Model": {"select": {"name": model}},
        "Benchmark": {"select": {"name": benchmark}},
    }
    
    if external_model:
        properties["External Model"] = {"select": {"name": external_model}}
    
    if command:
        properties["Command"] = {"rich_text": [{"text": {"content": command}}]}
    
    if comment:
        properties["Comment"] = {"rich_text": [{"text": {"content": comment}}]}

    if k is not None:
        properties["K"] = {"number": k}

    if alpha is not None:
        properties["Alpha"] = {"number": alpha}

    if num_negative_samples is not None:
        properties["Num Negative Samples"] = {"number": num_negative_samples}

    for key, value in metrics.items():
        properties[key] = {"number": value}
    
    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties=properties,
    )
