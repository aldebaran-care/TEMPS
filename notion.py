from datetime import date
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
    dataset: str,
    metrics: Dict[str, float],
    external_model: str = None,
    metric: str = None,
    command: str = None,
    rank: str = None,
    comment: str = None,
    merge: str = None,
):
    properties = {
        "ID": {"title": [{"text": {"content": id}}]},
        "Model": {"select": {"name": model}},
        "Dataset": {"select": {"name": dataset}},
    }
    
    if external_model:
        properties["External Model"] = {"select": {"name": external_model}}
    
    if metric:
        properties["Metric"] = {"select": {"name": metric}}
    
    if command:
        properties["Command"] = {"rich_text": [{"text": {"content": command}}]}
    
    if rank is not None:
        properties["Rank"] = {"select": {"name": rank}}
    
    if comment:
        properties["Comment"] = {"rich_text": [{"text": {"content": comment}}]}
    
    if merge is not None:
        properties["Merge"] = {"select": {"name": merge}}

    for key, value in metrics.items():
        properties[key] = {"number": value}
    
    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties=properties,
    )

metrics = {
    "Accuracy": 0.1,
}

log_metrics_to_notion(
    id="Run 18",
    model="gpt-4",
    dataset="test-dataset",
    metrics=metrics,
    comment="The worst run yet",
    time=date.today()
)
