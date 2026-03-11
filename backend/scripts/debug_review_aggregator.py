# proto/backend/scripts/debug_review_aggregator.py
from __future__ import annotations

import json

from services.implicit.review_aggregator import build_review_level_output


def main() -> None:
    review = (
        "I used it for a week. Calls keep showing busy even when the line is free. "
        "By evening the battery is already nearly dead. Later I noticed calls still fail."
    )
    sentences = [
        "I used it for a week.",
        "Calls keep showing busy even when the line is free.",
        "By evening the battery is already nearly dead.",
        "Later I noticed calls still fail.",
    ]
    candidates = [
        {
            "aspect": "network",
            "confidence": 0.87,
            "sentiment_hint": "negative",
            "evidence_sentence": "Calls keep showing busy even when the line is free.",
            "sentence_index": 1,
            "domain_hint": "telecom",
        },
        {
            "aspect": "battery",
            "confidence": 0.82,
            "sentiment_hint": "negative",
            "evidence_sentence": "By evening the battery is already nearly dead.",
            "sentence_index": 2,
            "domain_hint": "electronics",
        },
        {
            "aspect": "network",
            "confidence": 0.79,
            "sentiment_hint": "negative",
            "evidence_sentence": "Later I noticed calls still fail.",
            "sentence_index": 3,
            "domain_hint": "telecom",
        },
    ]

    out = build_review_level_output(
        review_text=review,
        sentences=sentences,
        candidates=candidates,
        max_aspects=5,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()