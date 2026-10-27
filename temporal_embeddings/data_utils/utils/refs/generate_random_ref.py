import random

def generate_random_ref() -> str:
    rand_int: int = random.randint(0, 2)
    
    annotations = ["PRESENT_REF", "THIS MO", "THIS NI"]

    return annotations[rand_int]