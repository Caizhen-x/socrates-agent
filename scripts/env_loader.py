"""
env_loader.py — Shared .env file loader. Replaces duplicated parsing in each script.
"""

import os
from pathlib import Path


def load_env(project_root: Path):
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
