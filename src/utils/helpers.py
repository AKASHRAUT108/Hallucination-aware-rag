"""Shared helpers: project root, config loading, JSONL io."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
