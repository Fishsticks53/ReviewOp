# proto/backend/services/implicit/sentence_splitter.py
from __future__ import annotations

import re
from typing import List


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_review_into_sentences(review_text: str) -> List[str]:
    text = (review_text or "").strip()
    if not text:
        return []

    parts = _SENT_SPLIT_RE.split(text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences