#!/usr/bin/env python3
"""Stable deployment entrypoint for the canonical A800 preflight.

The implementation remains in ``server_run/preflight_a800.py`` so the legacy
server launchers and the requested ``scripts/`` layout execute exactly the
same read-only checks.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(
        str(root / "server_run" / "preflight_a800.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
