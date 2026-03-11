# proto/backend/services/implicit/label_maps.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from services.implicit.config import CONFIG


@dataclass
class LabelMaps:
    aspect_to_id: Dict[str, int]
    id_to_aspect: Dict[int, str]
    domain_to_aspects: Dict[str, List[str]]
    aspect_to_domain: Dict[str, str]


def _normalize_domain_aspects(payload: dict) -> Dict[str, List[str]]:
    """
    Supports:
    {
      "domains": {
        "telecom": ["network", "call_quality"]
      }
    }

    {
      "domains": {
        "telecom": {
          "aspects": ["network", "call_quality"]
        }
      }
    }

    {
      "telecom": ["network", "call_quality"]
    }
    """
    if "domains" in payload and isinstance(payload["domains"], dict):
        raw_domain_map = payload["domains"]
    else:
        raw_domain_map = payload

    normalized: Dict[str, List[str]] = {}

    for domain, value in raw_domain_map.items():
        clean_domain = str(domain).strip().lower()

        aspects: List[str] = []

        if isinstance(value, list):
            aspects = [str(a).strip() for a in value if str(a).strip()]

        elif isinstance(value, dict):
            raw_aspects = value.get("aspects", [])
            if isinstance(raw_aspects, list):
                aspects = [str(a).strip() for a in raw_aspects if str(a).strip()]

        if aspects:
            normalized[clean_domain] = sorted(set(aspects))

    return normalized

def load_ontology(path: Path | None = None) -> dict:
    ontology_path = path or CONFIG.ontology_path
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    with ontology_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(path: Path | None = None) -> LabelMaps:
    payload = load_ontology(path)
    domain_to_aspects = _normalize_domain_aspects(payload)

    all_aspects = sorted(
        {
            aspect
            for aspects in domain_to_aspects.values()
            for aspect in aspects
        }
    )

    aspect_to_id = {aspect: idx for idx, aspect in enumerate(all_aspects)}
    id_to_aspect = {idx: aspect for aspect, idx in aspect_to_id.items()}

    aspect_to_domain: Dict[str, str] = {}
    for domain, aspects in domain_to_aspects.items():
        for aspect in aspects:
            aspect_to_domain[aspect] = domain

    return LabelMaps(
        aspect_to_id=aspect_to_id,
        id_to_aspect=id_to_aspect,
        domain_to_aspects=domain_to_aspects,
        aspect_to_domain=aspect_to_domain,
    )


def save_label_encoder(path: Path | None = None) -> None:
    target_path = path or CONFIG.label_encoder_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    label_maps = build_label_maps()
    payload = {
        "aspect_to_id": label_maps.aspect_to_id,
        "id_to_aspect": {str(k): v for k, v in label_maps.id_to_aspect.items()},
        "domain_to_aspects": label_maps.domain_to_aspects,
        "aspect_to_domain": label_maps.aspect_to_domain,
    }

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_label_encoder(path: Path | None = None, force_rebuild: bool = False) -> LabelMaps:
    target_path = path or CONFIG.label_encoder_path

    if force_rebuild or not target_path.exists():
        label_maps = build_label_maps()
        save_label_encoder(target_path)
        return label_maps

    with target_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return LabelMaps(
        aspect_to_id={str(k): int(v) for k, v in payload["aspect_to_id"].items()},
        id_to_aspect={int(k): str(v) for k, v in payload["id_to_aspect"].items()},
        domain_to_aspects={
            str(k): [str(x) for x in v]
            for k, v in payload["domain_to_aspects"].items()
        },
        aspect_to_domain={
            str(k): str(v) for k, v in payload["aspect_to_domain"].items()
        },
    )