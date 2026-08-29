"""Reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def get_git_revision() -> str | None:
    """Get current git revision, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of configuration for reproducibility."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
