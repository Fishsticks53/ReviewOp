# proto/backend/services/implicit/review_aggregator.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence


@dataclass
class AggregatedImplicitAspect:
    aspect: str
    confidence: float
    sentiment_hint: str
    evidence_sentence: str
    sentence_index: int
    domain_hint: str | None = None
    support_count: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)


def aggregate_implicit_candidates(
    candidates: Sequence[Dict],
    max_aspects: int = 5,
) -> List[Dict]:
    """
    Merge sentence-level implicit candidates into review-level aspect predictions.

    Rules:
    - keep one record per aspect
    - retain the highest-confidence evidence sentence as primary evidence
    - count repeated support for the same aspect
    """
    merged: Dict[str, AggregatedImplicitAspect] = {}

    for item in candidates:
        aspect = str(item.get("aspect", "")).strip()
        if not aspect:
            continue

        confidence = float(item.get("confidence", 0.0))
        sentiment_hint = str(item.get("sentiment_hint", "negative")).strip() or "negative"
        evidence_sentence = str(item.get("evidence_sentence", "")).strip()
        sentence_index = int(item.get("sentence_index", -1))
        domain_hint = item.get("domain_hint")

        current = merged.get(aspect)
        if current is None:
            merged[aspect] = AggregatedImplicitAspect(
                aspect=aspect,
                confidence=confidence,
                sentiment_hint=sentiment_hint,
                evidence_sentence=evidence_sentence,
                sentence_index=sentence_index,
                domain_hint=domain_hint,
                support_count=1,
            )
            continue

        current.support_count += 1
        if confidence > current.confidence:
            current.confidence = confidence
            current.sentiment_hint = sentiment_hint
            current.evidence_sentence = evidence_sentence
            current.sentence_index = sentence_index
            current.domain_hint = domain_hint

    results = sorted(
        [obj.to_dict() for obj in merged.values()],
        key=lambda x: (x["confidence"], x["support_count"]),
        reverse=True,
    )

    return results[:max_aspects]


def flatten_sentence_predictions(
    sentence_predictions: Sequence[Sequence[Dict]],
) -> List[Dict]:
    flat: List[Dict] = []
    for group in sentence_predictions:
        for item in group:
            flat.append(dict(item))
    return flat


def build_review_level_output(
    review_text: str,
    sentences: Sequence[str],
    candidates: Sequence[Dict],
    max_aspects: int = 5,
) -> Dict:
    aggregated = aggregate_implicit_candidates(
        candidates=candidates,
        max_aspects=max_aspects,
    )

    return {
        "review_text": review_text,
        "sentences": list(sentences),
        "num_implicit_aspects": len(aggregated),
        "implicit_aspects": aggregated,
    }