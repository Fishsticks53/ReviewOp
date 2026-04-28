from __future__ import annotations
import logging
from rapidfuzz import process, fuzz
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

class FuzzyMatcher:
    @staticmethod
    def find_best_match(
        query: str, 
        candidates: Iterable[str], 
        threshold: float = 85.0
    ) -> tuple[str, float] | None:
        """
        Find the best fuzzy match for a query among a set of candidates.
        Returns (matched_candidate, score) or None if below threshold.
        """
        if not query or not candidates:
            return None
        
        query_clean = str(query).lower().strip()
        # candidate list may be generators, so convert to list for process.extractOne
        candidate_list = list(candidates)
        if not candidate_list:
            return None

        # scorer=fuzz.WRatio is generally good for different lengths and word order
        result = process.extractOne(
            query_clean, 
            candidate_list, 
            scorer=fuzz.WRatio
        )
        
        if result:
            matched, score, index = result
            if score >= threshold:
                return matched, score
        
        return None
