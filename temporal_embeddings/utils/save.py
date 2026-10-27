from pathlib import Path
from typing import Any, Dict
import json

def save_json(data: Dict[Any, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)