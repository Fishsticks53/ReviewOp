from __future__ import annotations

from datetime import datetime
from typing import Optional

from models.tables import Prediction


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except Exception:
            return None


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().replace("_", " ").replace("-", " ").split())


def aspect_label(aspect: str) -> str:
    return " ".join(part.capitalize() for part in aspect.replace("-", " ").replace("_", " ").split()) or aspect


def canonical_aspect(prediction: Prediction) -> str:
    return (prediction.aspect_cluster or prediction.aspect_raw or "unknown").strip() or "unknown"


def infer_origin(aspect: str, snippet: str | None) -> str:
    aspect_terms = set(normalize_text(aspect).split())
    snippet_terms = set(normalize_text(snippet or "").split())
    if aspect_terms and aspect_terms.issubset(snippet_terms):
        return "explicit"
    return "implicit"
