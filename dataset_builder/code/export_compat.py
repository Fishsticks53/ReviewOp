from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils import write_jsonl


def implicit_to_episode_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for row in rows:
        implicit = row.get("implicit", {})
        aspects = implicit.get("aspects", []) or []
        sentiments = implicit.get("aspect_sentiments", {}) or {}
        confidences = implicit.get("aspect_confidence", {}) or {}
        for index, aspect in enumerate(aspects, start=1):
            episodes.append(
                {
                    "example_id": f"{row.get('id')}_e{index}",
                    "parent_review_id": row.get("id"),
                    "review_text": row.get("source_text", ""),
                    "evidence_sentence": row.get("source_text", ""),
                    "domain": "mixed",
                    "aspect": aspect,
                    "implicit_aspect": aspect,
                    "sentiment": sentiments.get(aspect, implicit.get("dominant_sentiment", "neutral")),
                    "label_type": "implicit",
                    "confidence": float(confidences.get(aspect, implicit.get("avg_confidence", 0.5))),
                    "split": row.get("split", "train"),
                    "source": "dataset_builder",
                }
            )
    return episodes


def implicit_to_reviewlevel_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    review_rows: List[Dict[str, Any]] = []
    for row in rows:
        implicit = row.get("implicit", {})
        labels = []
        for aspect in implicit.get("aspects", []) or []:
            labels.append(
                {
                    "aspect": aspect,
                    "implicit_aspect": aspect,
                    "sentiment": implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral")),
                    "confidence": float(implicit.get("aspect_confidence", {}).get(aspect, implicit.get("avg_confidence", 0.5))),
                    "type": "implicit",
                    "evidence_sentence": row.get("source_text", ""),
                }
            )
        review_rows.append(
            {
                "id": row.get("id"),
                "review_text": row.get("source_text", ""),
                "source_file": row.get("source_file", "dataset_builder"),
                "split": row.get("split", "train"),
                "domain": "mixed",
                "labels": labels,
                "target_text": " ;; ".join(
                    f"{item['aspect']} | {item['sentiment']} | {item['evidence_sentence']}" for item in labels
                ),
            }
        )
    return review_rows


def export_compatibility_artifacts(
    *,
    explicit_by_split: Dict[str, List[Dict[str, Any]]],
    implicit_by_split: Dict[str, List[Dict[str, Any]]],
    protonet_export_dir: Path,
    backend_export_dir: Path,
    protonet_input_dir: Path,
    backend_implicit_dir: Path,
) -> Dict[str, List[Path]]:
    written: Dict[str, List[Path]] = {"protonet": [], "backend": []}
    protonet_input_dir.mkdir(parents=True, exist_ok=True)
    backend_implicit_dir.mkdir(parents=True, exist_ok=True)

    protonet_rows = {split: implicit_to_episode_examples(rows) for split, rows in implicit_by_split.items()}
    for split, rows in protonet_rows.items():
        path = protonet_export_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        write_jsonl(protonet_input_dir / f"{split}.jsonl", rows)
        written["protonet"].append(path)

    backend_explicit_names = {
        "train": "explicit_train.jsonl",
        "val": "explicit_val.jsonl",
        "test": "explicit_test.jsonl",
    }

    backend_review_names = {
        "train": "implicit_reviewlevel_train.jsonl",
        "val": "implicit_reviewlevel_val.jsonl",
        "test": "implicit_reviewlevel_test.jsonl",
    }
    backend_episode_names = {
        "train": "implicit_episode_train.jsonl",
        "val": "implicit_episode_val.jsonl",
        "test": "implicit_episode_test.jsonl",
    }

    reviewlevel_rows = {split: implicit_to_reviewlevel_rows(rows) for split, rows in implicit_by_split.items()}

    for split, rows in explicit_by_split.items():
        path = backend_export_dir / backend_explicit_names[split]
        write_jsonl(path, rows)
        written["backend"].append(path)

    for split, rows in protonet_rows.items():
        path = backend_implicit_dir / backend_episode_names[split]
        write_jsonl(path, rows)
        written["backend"].append(path)

    for split, rows in reviewlevel_rows.items():
        path = backend_implicit_dir / backend_review_names[split]
        write_jsonl(path, rows)
        written["backend"].append(path)

    return written
