import random
from typing import List

from temporal_embeddings.data_utils.utils.extract_integers import extract_integers

def generate_random_offset(close: bool, value: str, type: str) -> str:
    if not close:
        random_offset: int = random.randint(1, 10) * (-1 if bool(random.getrandbits(1)) else 1)
        random_format: int = random.randint(0, 15)

        immediate_text: str = "NEXT_IMMEDIATE" if bool(random.getrandbits(1)) else "PREV_IMMEDIATE"
        
        if random_format == 0:
            return f"OFFSET P{random_offset}D"
        
        if random_format == 1:
            return f"OFFSET P{random_offset}W"
        
        if random_format == 2:
            return f"OFFSET P{random_offset}M"
        
        if random_format == 3:
            return f"OFFSET P{random_offset}Y"
        
        if random_format == 4:
            return f"THIS P{abs(random_offset)}D OFFSET P{random_offset}D"
        
        if random_format == 5:
            return f"THIS P{abs(random_offset)}W OFFSET P{random_offset}W"
        
        if random_format == 6:
            return f"THIS P{abs(random_offset)}M OFFSET P{random_offset}M"
        
        if random_format == 7:
            return f"THIS P{abs(random_offset)}Y OFFSET P{random_offset}Y"
        
        if random_format == 8:
            return f"THIS P{abs(random_offset)}D"
        
        if random_format == 9:
            return f"THIS P{abs(random_offset)}W"
        
        if random_format == 10:
            return f"THIS P{abs(random_offset)}M"
        
        if random_format == 11:
            return f"THIS P{abs(random_offset)}Y"
        
        if random_format == 12:
            return f"{immediate_text} P{abs(random_offset)}D"
        
        if random_format == 13:
            return f"{immediate_text} P{abs(random_offset)}W"
        
        if random_format == 14:
            return f"{immediate_text} P{abs(random_offset)}M"

        return f"{immediate_text} P{abs(random_offset)}Y"
    
    else:
        if type in ["td", "tw", "tm", "ty"]:
            extracted_value: int = extract_integers(value)[1]
        
        else:
            extracted_value: int = extract_integers(value)[0]
            
        is_negative = True if extracted_value < 0 else False
        
        extracted_value = abs(extracted_value)
        min_value = max(1, extracted_value - 4)
        max_value = min(15, extracted_value + 4)
        
        random_offset: int = random.randint(min_value, max_value)

        if is_negative:
            random_offset *= -1

        immediate_text: str = "NEXT_IMMEDIATE" if bool(random.getrandbits(1)) else "PREV_IMMEDIATE"
        
        if type == "d":
            return f"OFFSET P{random_offset}D"
        
        if type == "w":
            return f"OFFSET P{random_offset}W"
        
        if type == "m":
            return f"OFFSET P{random_offset}M"
        
        if type == "y":
            return f"OFFSET P{random_offset}Y"
        
        if type == "td":
            return f"THIS P{abs(random_offset)}D OFFSET P{random_offset}D"
        
        if type == "tw":
            return f"THIS P{abs(random_offset)}W OFFSET P{random_offset}W"
        
        if type == "tm":
            return f"THIS P{abs(random_offset)}M OFFSET P{random_offset}M"
        
        if type == "ty":
            return f"THIS P{abs(random_offset)}Y OFFSET P{random_offset}Y"
        
        if type == "thisd":
            return f"THIS P{random_offset}D"
        
        if type == "thisw":
            return f"THIS P{random_offset}W"
        
        if type == "thism":
            return f"THIS P{random_offset}M"
        
        if type == "thisy":
            return f"THIS P{random_offset}Y"
        
        if type == "immediated":
            return f"{immediate_text} P{random_offset}D"
        
        if type == "immediatew":
            return f"{immediate_text} P{random_offset}W"
        
        if type == "immediatem":
            return f"{immediate_text} P{random_offset}M"

        return f"{immediate_text} P{random_offset}Y"