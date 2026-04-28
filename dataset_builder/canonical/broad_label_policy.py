from __future__ import annotations

from ..schemas.interpretation import Interpretation


from .domain_registry import DomainRegistry


def is_broad_label(label: str, domain: str | None = None) -> bool:
    broad_labels = DomainRegistry.get_broad_labels(domain)
    return str(label or "").strip().lower() in broad_labels


def prune_broad_labels(items: list[Interpretation], domain: str | None = None) -> tuple[list[Interpretation], dict[str, int]]:
    if len(items) <= 1:
        return items, {"dropped_broad_labels": 0}
    
    # We only want to prune 'broad' labels if we have more 'specific' ones 
    # of the SAME label type (usually explicit).
    # Implicit findings are often broad-ish but high-value, so we keep them.
    
    explicit_specific = [i for i in items if i.label_type == "explicit" and not is_broad_label(i.aspect_canonical, domain)]
    
    kept = []
    for item in items:
        # 1. Always keep implicit findings (they are likely learned symptoms)
        if item.label_type == "implicit":
            kept.append(item)
            continue
            
        # 2. Keep explicit findings if they are not broad
        if not is_broad_label(item.aspect_canonical, domain):
            kept.append(item)
            continue
            
        # 3. It is an explicit broad label. Keep it ONLY if we don't have explicit specific ones.
        if not explicit_specific:
            kept.append(item)
            continue
            
        # Otherwise, prune it (it's redundant with the specific explicit findings)
    
    return kept, {"dropped_broad_labels": len(items) - len(kept)}
