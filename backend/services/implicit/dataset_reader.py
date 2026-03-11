# proto/backend/services/implicit/dataset_reader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from services.implicit.config import CONFIG


VALID_SPLITS = {"train", "val", "test"}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {file_path}: {exc}"
                ) from exc
    return rows


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {file_path}: {exc}"
                ) from exc


def _validate_split(split: str) -> str:
    clean = split.strip().lower()
    if clean not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got: {split}")
    return clean


def _filter_by_declared_split(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        row_split = str(row.get("split", "")).strip().lower()
        if row_split and row_split != split:
            continue
        filtered.append(row)
    return filtered


def get_reviewlevel_path(split: str) -> Path:
    split = _validate_split(split)
    if split == "train":
        return CONFIG.review_train_path
    if split == "val":
        return CONFIG.review_val_path
    return CONFIG.review_test_path


def get_episode_path(split: str) -> Path:
    split = _validate_split(split)
    if split == "train":
        return CONFIG.episode_train_path
    if split == "val":
        return CONFIG.episode_val_path
    return CONFIG.episode_test_path


def load_reviewlevel_split(split: str) -> List[Dict[str, Any]]:
    clean_split = _validate_split(split)
    rows = load_jsonl(get_reviewlevel_path(clean_split))
    return _filter_by_declared_split(rows, clean_split)


def load_episode_split(split: str) -> List[Dict[str, Any]]:
    clean_split = _validate_split(split)
    rows = load_jsonl(get_episode_path(clean_split))
    return _filter_by_declared_split(rows, clean_split)


def summarize_dataset(rows: List[Dict[str, Any]], label_key: str) -> Dict[str, Any]:
    label_counts: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}

    for row in rows:
        domain = str(row.get("domain", "unknown")).strip().lower()
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        label = str(row.get(label_key, "unknown")).strip()
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "num_rows": len(rows),
        "num_domains": len(domain_counts),
        "domains": dict(sorted(domain_counts.items())),
        "num_labels": len(label_counts),
        "labels": dict(sorted(label_counts.items())),
    }