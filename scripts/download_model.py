"""
download_model.py — Pre-cache the embedding model for offline use.
Run once after cloning, before scripts/ingest.py or scripts/retrieve.py.
Usage: python scripts/download_model.py
"""

import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent

with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
    settings = yaml.safe_load(f)

model_name = settings["embedding"]["model"]
print(f"Downloading/verifying model: {model_name}")
model = SentenceTransformer(model_name)
print(f"Model cached. Ready for offline use.")
