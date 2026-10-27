from typing import List
import numpy as np

def normalize_list(lst: List[float]) -> List[float]:    
    arr = np.array(lst)

    if arr.max() - arr.min() == 0:
        return [0.0 for _ in arr]

    norm = (arr - arr.min()) / (arr.max() - arr.min())
    
    return norm.tolist()