from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".xml"}


def flatten_one_level(payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}_{inner_key}"] = inner_value
        else:
            flat[key] = value
    return flat


def list_input_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_file_to_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frame = pd.DataFrame([flatten_one_level(row) if isinstance(row, dict) else {"value": row} for row in payload])
        elif isinstance(payload, dict) and "records" in payload and isinstance(payload["records"], list):
            frame = pd.DataFrame([flatten_one_level(row) if isinstance(row, dict) else {"value": row} for row in payload["records"]])
        else:
            frame = pd.DataFrame([flatten_one_level(payload if isinstance(payload, dict) else {"value": payload})])
    elif suffix == ".jsonl":
        rows = [flatten_one_level(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
    elif suffix == ".xml":
        root = ET.parse(path).getroot()
        rows = []
        for child in list(root):
            record = {f"attr_{k}": v for k, v in child.attrib.items()}
            for sub in list(child):
                record[sub.tag] = (sub.text or "").strip()
            if not record and (child.text or "").strip():
                record[child.tag] = (child.text or "").strip()
            rows.append(record)
        frame = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported input file: {path}")

    if not frame.empty:
        frame = frame.copy()
        frame["source_file"] = path.name
    return frame


def load_inputs(input_dir: Path) -> pd.DataFrame:
    frames = [load_file_to_dataframe(path) for path in list_input_files(input_dir)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

