"""Standalone command: retain the ordinary CLI, stdin and stdout."""

from inptk.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
