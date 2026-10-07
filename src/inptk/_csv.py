"""Read CSV tables following an optional commented metadata preamble."""

import io
from itertools import dropwhile
from pathlib import Path

import pandas as pd


def read_csv_with_preamble(source, **kwargs):
    """Skip leading metadata lines while preserving hashes inside CSV values."""
    text = (Path(source).read_text(encoding="utf-8-sig")
            if isinstance(source, (str, Path)) else source.read())
    lines = dropwhile(lambda line: line.startswith("#"), text.splitlines(keepends=True))
    return pd.read_csv(io.StringIO("".join(lines)), **kwargs)
