"""Load train/val/test id sets for data_split explicit_lists mode."""

from __future__ import annotations

import json
from pathlib import Path


def load_id_set(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ID list not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return {str(x) for x in data}
        raise ValueError("JSON id file must be a list of strings")
    return {line.strip() for line in text.splitlines() if line.strip()}
