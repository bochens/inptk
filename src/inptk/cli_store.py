"""Saved files for terminal commands; named in-memory results for client sessions."""

from pathlib import Path

from .io import load, save


class ResultStore:
    def __init__(self, *, memory=False):
        self.memory = memory
        self.results = {}

    def _is_reference(self, name):
        if str(name).startswith("@"):
            if not self.memory:
                raise ValueError("In-memory @references require 'inptk serve'")
            if len(str(name)) == 1:
                raise ValueError("An in-memory result requires a name after @")
            return True
        return False

    def exists(self, name):
        return name in self.results if self._is_reference(name) else Path(name).exists()

    def load(self, name):
        if self._is_reference(name):
            if name not in self.results:
                raise ValueError(f"Unknown in-memory result {name!r}")
            return self.results[name]
        return load(name)

    def save(self, value, name):
        if self._is_reference(name):
            if name in self.results:
                raise FileExistsError(f"Output already exists: {name}")
            self.results[name] = value
        else:
            save(value, name)

    def output_name(self, name):
        return name if self._is_reference(name) else str(Path(name).resolve())

    def release(self, names):
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or name not in self.results for name in names)
        ):
            raise ValueError("release requires a nonempty list of existing @references")
        for name in set(names):
            del self.results[name]
