#!/usr/bin/env python3
"""Repository-root entry point for ``server_run/run_baseline.py``."""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "server_run" / "run_baseline.py"),
               run_name="__main__")

