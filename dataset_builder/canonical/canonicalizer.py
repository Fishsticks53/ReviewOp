from __future__ import annotations
from dataclasses import replace
from .domain_maps import lookup_domain_map
from .open_world_fallback import classify_unmapped_candidate, keep_open_world_candidate, mark_provisional_canonical, strip_sentiment_modifiers
from ..schemas.interpretation import Interpretation

def canonicalize_label(target: str | Interpretation, domain: str = "unknown", domain_mode: str = "full") -> str:
    res = lookup_domain_map(domain, target, domain_mode=domain_mode)
    if res.aspect_canonical:
        return res.aspect_canonical
    
    label = target.aspect_raw if isinstance(target, Interpretation) else str(target)
    provisional = mark_provisional_canonical(label)
    return provisional if provisional else "unknown"

def canonicalize_interpretation(
    item: Interpretation,
    domain: str = "unknown",
    *,
    domain_mode: str = "full",
    provisional_policy: str = "strict",
) -> Interpretation:
    """Canonicalize an interpretation using multi-step lookup."""
    res = lookup_domain_map(domain, item, domain_mode=domain_mode)
    new_canonical = res.aspect_canonical
    mapping_source = res.mapping_source
    mapping_scope = res.mapping_scope
    mapping_layers = res.mapping_layers
    confidence = res.mapping_confidence
    
    if not new_canonical:
        decision = classify_unmapped_candidate(
            item.aspect_raw,
            item.evidence_text,
            support_count=1,
            provisional_policy=provisional_policy,
        )
        provisional = mark_provisional_canonical(item.aspect_raw)
        if decision.bucket == "provisional" and provisional:
            new_canonical = provisional
            mapping_source = "provisional"
            mapping_scope = "provisional"
            mapping_layers = ("provisional",)
            confidence = 0.35
        elif decision.bucket == "open_world" and keep_open_world_candidate(item.aspect_raw, 0.0):
            cleaned = strip_sentiment_modifiers(item.aspect_raw)
            new_canonical = str(cleaned or item.aspect_raw or "open_world").strip().lower().replace(" ", "_")
            mapping_source = "open_world"
            mapping_scope = "open_world"
            mapping_layers = ("open_world",)
            confidence = 0.25
        else:
            new_canonical = "unknown"
            if decision.bucket == "dropped_noise":
                mapping_source = "dropped_noise"
                mapping_scope = "dropped_noise"
                mapping_layers = ("dropped_noise",)
            else:
                mapping_source = "memory_candidate"
                mapping_scope = "memory_candidate"
                mapping_layers = ("memory_candidate",)
            confidence = 0.0
        
    return replace(
        item, 
        aspect_canonical=new_canonical,
        mapping_source=mapping_source,
        canonical_confidence=confidence,
        mapping_scope=mapping_scope,
        mapping_layers=tuple(mapping_layers),
    )
