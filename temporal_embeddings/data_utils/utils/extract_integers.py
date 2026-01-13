import re
from typing import List

def extract_integers(text: str) -> List[int]:
    pattern = r"-?\d+"
    
    return [int(match) for match in re.findall(pattern, text)]