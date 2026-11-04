#!/usr/bin/env python3
"""Stable deployment entrypoint for the canonical three-level A800 smoke.

Run this file directly with ``--level 1|2|3``.  Keeping the implementation in
``server_run/a800_smoke.py`` prevents two smoke definitions from drifting.
Importing this module during test discovery does not allocate a GPU.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(
        str(root / "server_run" / "a800_smoke.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
