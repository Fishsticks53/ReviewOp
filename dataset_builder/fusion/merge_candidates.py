from __future__ import annotations

from ..schemas.interpretation import Interpretation


def merge_explicit_implicit(explicit: list[Interpretation], implicit: list[Interpretation]) -> list[Interpretation]:
    return dedupe_merged_candidates([*explicit, *implicit])


def dedupe_merged_candidates(items: list[Interpretation]) -> list[Interpretation]:
    # Sort by confidence, then by source strength (learned > explicit > json)
    def source_rank(i: Interpretation) -> int:
        if i.source_type == "implicit_learned": return 3
        if i.source_type == "explicit": return 2
        if i.source_type == "implicit_json": return 1
        return 0

    sorted_items = sorted(items, key=lambda i: (i.canonical_confidence, source_rank(i)), reverse=True)
    out: list[Interpretation] = []
    
    for item in sorted_items:
        is_duplicate = False
        for existing in out:
            # If same canonical and evidence overlaps significantly
            if item.aspect_canonical == existing.aspect_canonical:
                s1, e1 = item.evidence_span
                s2, e2 = existing.evidence_span
                # Check for any overlap
                overlap = max(0, min(e1, e2) - max(s1, s2))
                if overlap > 0:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            out.append(item)
    return out
