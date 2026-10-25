#!/usr/bin/env python3
"""Run the complete test suite sequentially with a terminal result per case."""

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    # Ignore inherited options that could enable parallelism or stop on failure.
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        "tests",
        "-vv",
        "-ra",
        "--tb=short",
        "--maxfail=0",
        "-p", "no:xdist",
        "-o", "addopts=",
        "-o", "console_output_style=classic",
    ]
    print("Running all tests one by one. Each case prints its result below.\n", flush=True)
    # Inherit terminal streams so results appear immediately, without buffering
    # them in this runner. Preserve pytest's exit code for shell/CI callers.
    return subprocess.call(command, cwd=project_dir, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
