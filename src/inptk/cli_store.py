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

    def import_counts(self, payload):
        """Validate native JSON observations once and retain the experiment."""
        from .readers import read_counts

        required = {"out", "counts", "metadata"}
        optional = {"run_id", "water_blank_map"}
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise ValueError("import requires out, counts and metadata")
        if payload.keys() - required - optional:
            raise ValueError("Unknown import fields")
        name = payload["out"]
        if not isinstance(name, str) or not self._is_reference(name):
            raise ValueError("import out must be an in-memory @reference")
        if self.exists(name):
            raise FileExistsError(f"Output already exists: {name}")
        for key in ("counts", "metadata"):
            data = payload[key]
            if not isinstance(data, (dict, list)):
                raise TypeError(f"import {key} must contain records or column arrays")
            if isinstance(data, dict) and any(not isinstance(v, list) for v in data.values()):
                raise TypeError(f"import {key} column values must be arrays")
            if isinstance(data, list) and any(not isinstance(v, dict) for v in data):
                raise TypeError(f"import {key} rows must be objects")
        experiment = read_counts(
            payload["counts"],
            metadata=payload["metadata"],
            run_id=payload.get("run_id", "1"),
            water_blank_map=payload.get("water_blank_map"),
        )
        self.save(experiment, name)
        return experiment

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
