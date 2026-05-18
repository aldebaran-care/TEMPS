"""Command-line interface for HeidelTime Python."""

import argparse
import sys

from .heideltime import HeidelTime


def main():
    parser = argparse.ArgumentParser(
        description="HeidelTime Python - Temporal Expression Tagger and Normalizer",
        prog="heideltime_py",
    )
    parser.add_argument("input", nargs="?", help="Input file (stdin if omitted)")
    parser.add_argument("-l", "--language", default="english", help="Language (default: english)")
    parser.add_argument(
        "-t",
        "--document-type",
        default="news",
        choices=["news", "narrative", "narratives", "colloquial", "scientific"],
        help="Document type (default: news)",
    )
    parser.add_argument(
        "-dct", "--dct", default=None, help="Document Creation Time (YYYY-MM-DD or YYYYMMDD)"
    )
    parser.add_argument("-r", "--resources", default=None, help="Path to resources directory")
    parser.add_argument("--no-dates", action="store_true", help="Don't extract DATE expressions")
    parser.add_argument("--no-times", action="store_true", help="Don't extract TIME expressions")
    parser.add_argument(
        "--no-durations", action="store_true", help="Don't extract DURATION expressions"
    )
    parser.add_argument("--no-sets", action="store_true", help="Don't extract SET expressions")
    parser.add_argument("--temponyms", action="store_true", help="Extract TEMPONYM expressions")
    parser.add_argument(
        "--no-spacy", action="store_true", help="Don't use spaCy (use simple tokenizer)"
    )
    parser.add_argument("-o", "--output", default=None, help="Output file (stdout if omitted)")

    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    # Initialize HeidelTime
    ht = HeidelTime(
        language=args.language,
        document_type=args.document_type,
        resources_path=args.resources,
        find_dates=not args.no_dates,
        find_times=not args.no_times,
        find_durations=not args.no_durations,
        find_sets=not args.no_sets,
        find_temponyms=args.temponyms,
        use_spacy=not args.no_spacy,
    )

    # Process
    result = ht.process(text, dct=args.dct)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
