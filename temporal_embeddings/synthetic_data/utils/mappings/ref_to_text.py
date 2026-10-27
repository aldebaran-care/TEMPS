import random

def ref_to_text(annotation):
    mappings = {
        "PRESENT_REF": ["now", "at this moment", "currently", "right now", "at present", "as of now", "at this time", "in this moment", "at the moment", "presently"],
        "THIS MO": ["this morning", "earlier today", "in the morning", "during the morning", "this a.m.", "this dawn", "this early day", "this sunrise", "this forenoon", "this early hour"],
        "THIS NI": ["tonight", "this evening", "later tonight", "this night", "this p.m.", "this late hour", "this dusk", "this twilight", "this dark", "this nocturnal time"],
    }

    return random.choice(mappings[annotation])