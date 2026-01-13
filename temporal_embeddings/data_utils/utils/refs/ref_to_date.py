from typing import List

def ref_to_date(annotation: str, current_date: str) -> List[str]:
    if annotation in ["PRESENT_REF", "THIS MO", "THIS NI", "TEV", "TMO", "TNI"]:
        return [current_date]
    
    raise ValueError(f"Cannot convert ref to date: {annotation}")