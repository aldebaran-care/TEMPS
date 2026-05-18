"""
HeidelTime Python - A Python port of the HeidelTime temporal expression tagger.

Usage:
    from heideltime_py import HeidelTime

    ht = HeidelTime(language="english", document_type="news")
    result = ht.process("He arrived on January 15, 2024.", dct="2024-01-20")
    print(result)  # TimeML XML output
"""

from .heideltime import HeidelTime
from .models import Document, Timex3

__version__ = "1.0.0"
__all__ = ["HeidelTime", "Timex3", "Document"]
