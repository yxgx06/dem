from __future__ import annotations

import platform
import subprocess
from typing import Any

import numpy as np
import scipy

from egga.config import CONFIG_DIR, REPO_ROOT, config_hash


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def provenance() -> dict[str, Any]:
    names = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    dirty = _git("status", "--porcelain", "--", "src", "configs", "tests") != ""
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "config_hash": config_hash(names),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
