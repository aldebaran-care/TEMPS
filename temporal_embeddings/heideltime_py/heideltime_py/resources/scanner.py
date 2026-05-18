"""Resource scanner: discovers resource files on the filesystem."""

import os


def find_resources_path(language: str, custom_path: str | None = None) -> str:
    """Find the resources directory for a given language."""
    if custom_path:
        lang_path = os.path.join(custom_path, language)
        if os.path.isdir(lang_path):
            return lang_path
        raise FileNotFoundError(f"Resource directory not found: {lang_path}")

    # Try relative to this package, then relative to project root
    candidates = [
        # resources/resources/english — data dir is nested inside this Python package
        os.path.join(os.path.dirname(__file__), "resources", language),
        # Legacy paths kept for backwards compatibility
        os.path.join(os.path.dirname(__file__), "..", "..", "resources", language),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources", language),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.realpath(c)
    raise FileNotFoundError(
        f"Resource directory for language '{language}' not found. Searched: {candidates}"
    )


def scan_resources(
    language: str, resource_type: str, custom_path: str | None = None
) -> dict[str, str]:
    """
    Scan a language resource directory for files of a given type.

    Returns dict mapping resource name -> file path.
    E.g. for repattern: {"reMonthLong": "/path/to/resources_repattern_reMonthLong.txt"}
    """
    lang_path = find_resources_path(language, custom_path)
    type_dir = os.path.join(lang_path, resource_type)

    if not os.path.isdir(type_dir):
        return {}

    resources = {}
    prefix = f"resources_{resource_type}_"
    for fname in sorted(os.listdir(type_dir)):
        if fname.startswith(prefix) and fname.endswith(".txt"):
            name = fname[len(prefix) : -4]  # strip prefix and .txt
            resources[name] = os.path.join(type_dir, fname)
    return resources
