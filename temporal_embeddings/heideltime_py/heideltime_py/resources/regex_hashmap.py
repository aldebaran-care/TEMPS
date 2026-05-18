"""RegexHashMap: a dictionary with regex-key fallback and caching."""

import re


class RegexHashMap:
    """Dict where get() first tries direct match, then regex match on keys."""

    def __init__(self):
        self._container = {}
        self._cache = {}

    def put(self, key, value):
        self._container[key] = value

    def get(self, key, default=None):
        if key is None:
            return default
        # check cache
        if key in self._cache:
            return self._cache[key]
        # check direct
        if key in self._container:
            return self._container[key]
        # check regex keys
        for regex_key, val in self._container.items():
            try:
                if re.fullmatch(regex_key, key):
                    self._cache[key] = val
                    return val
            except re.error:
                continue
        return default

    def contains_key(self, key):
        if key in self._cache or key in self._container:
            return True
        for regex_key in self._container:
            try:
                if re.fullmatch(regex_key, key):
                    return True
            except re.error:
                continue
        return False

    def __contains__(self, key):
        return self.contains_key(key)

    def __getitem__(self, key):
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key, value):
        self.put(key, value)

    def __len__(self):
        return len(self._container) + len(self._cache)

    def keys(self):
        return set(self._container.keys()) | set(self._cache.keys())

    def values(self):
        return list(self._container.values()) + list(self._cache.values())

    def items(self):
        items = list(self._container.items())
        items.extend(self._cache.items())
        return items
