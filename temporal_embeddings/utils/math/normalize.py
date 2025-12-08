from typing import List
import numpy as np

def normalize_list(lst: List[List[float]]) -> List[List[float]]:
    normalized = []
    
    for sublist in lst:
        arr = np.array(sublist)
        if arr.max() - arr.min() == 0:
            normalized.append([0.0 for _ in arr])
    
        else:
            norm = (arr - arr.min()) / (arr.max() - arr.min())
            normalized.append(norm.tolist())
    
    return normalized