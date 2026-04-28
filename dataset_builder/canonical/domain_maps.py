from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from .domain_registry import DomainRegistry
from .fuzzy_match import FuzzyMatcher
from ..schemas.interpretation import Interpretation

@dataclass
class CanonicalMappingResult:
    aspect_canonical: str | None
    latent_family: str | None = None
    mapping_source: str = "unknown"
    mapping_confidence: float = 0.0
    matched_key: str | None = None
    ambiguity_flag: bool = False
    mapping_layers: tuple[str, ...] = field(default_factory=tuple)
    mapping_scope: str = "unknown"

def lookup_domain_map(domain: str | None, target: Any, config_dir: Path | None = None, domain_mode: str = "full") -> CanonicalMappingResult:
    """
    Look up a canonical aspect from a domain map with multi-step precedence.
    """
    if target is None:
        return CanonicalMappingResult(None)

    # Pre-step: Extract info from Interpretation if provided
    raw_phrase = None
    anchor = None
    modifiers = ()
    source_type = "unknown"
    existing_canonical = None

    if isinstance(target, Interpretation):
        raw_phrase = target.aspect_raw
        anchor = target.aspect_anchor
        modifiers = target.modifier_terms
        source_type = target.source_type
        existing_canonical = target.aspect_canonical
        latent_family = target.latent_family
    else:
        raw_phrase = str(target)

    # 1. Preserve trusted canonical (learned store)
    if source_type == "implicit_learned" and existing_canonical and existing_canonical != "unknown":
        return CanonicalMappingResult(
            aspect_canonical=existing_canonical,
            mapping_source="trusted_learned",
            mapping_confidence=1.0,
            mapping_scope="learned_store",
            mapping_layers=("learned_store",),
        )

    # Load config hierarchy
    effective_domain = "generic" if domain_mode in {"generic_only", "generic_plus_learned"} else str(domain or "generic").lower()
    source_cfg = DomainRegistry.get_source_config(effective_domain, config_dir=config_dir) if config_dir else DomainRegistry.get_source_config(effective_domain)

    generic_map = source_cfg.generic.get("domain_maps", {})
    domain_map = source_cfg.domain_raw.get("domain_maps", {})
    full_map = source_cfg.merged.get("domain_maps", {})
    generic_modifiers = source_cfg.generic.get("modifier_maps", {})
    domain_modifiers = source_cfg.domain_raw.get("modifier_maps", {})
    full_modifiers = source_cfg.merged.get("modifier_maps", {})

    # Helper to build result with layers
    def make_result(aspect, source, confidence=1.0, key=None, ambiguity=False):
        key_l = str(key or "").lower()
        layers: list[str] = []
        if key_l:
            in_generic = key_l in generic_map or key_l in generic_modifiers
            in_domain = key_l in domain_map or key_l in domain_modifiers
            if in_generic:
                layers.append("generic")
            if in_domain:
                layers.append("domain_specific")
        if not layers and source in {"trusted_learned"}:
            layers = ["learned_store"]

        if domain_mode == "generic_only":
            scope = "generic"
            layers = ["generic"]
        elif layers == ["generic"]:
            scope = "generic"
        elif layers == ["domain_specific"]:
            scope = "domain_specific"
        elif "generic" in layers and "domain_specific" in layers:
            scope = "generic+domain_specific"
        elif layers == ["learned_store"]:
            scope = "learned_store"
        else:
            scope = "generic" if effective_domain == "generic" else "domain_specific"
        if not layers:
            if scope == "generic":
                layers = ["generic"]
            elif scope == "domain_specific":
                layers = ["domain_specific"]
            elif scope == "generic+domain_specific":
                layers = ["generic", "domain_specific"]
            elif scope == "learned_store":
                layers = ["learned_store"]

        return CanonicalMappingResult(
            aspect_canonical=aspect,
            mapping_source=source,
            mapping_confidence=confidence,
            matched_key=key,
            ambiguity_flag=ambiguity,
            mapping_layers=tuple(layers),
            mapping_scope=scope
        )

    # 2. Anchor + Modifier contextual match
    if anchor:
        lookup_anchor = anchor.lower().strip()
        if lookup_anchor in full_modifiers:
            sub_map = full_modifiers[lookup_anchor]
            found_canonicals = []
            for mod in modifiers:
                mod_clean = mod.lower().strip()
                if mod_clean in sub_map:
                    found_canonicals.append((sub_map[mod_clean], mod_clean))
            
            if found_canonicals:
                unique_canons = sorted(list(set(c[0] for c in found_canonicals)))
                mod_key = found_canonicals[0][1]
                in_generic = lookup_anchor in generic_modifiers and mod_key in generic_modifiers.get(lookup_anchor, {})
                in_domain = lookup_anchor in domain_modifiers and mod_key in domain_modifiers.get(lookup_anchor, {})
                layers = tuple([x for x, ok in (("generic", in_generic), ("domain_specific", in_domain)) if ok]) or ("generic",)
                scope = "generic+domain_specific" if set(layers) == {"generic", "domain_specific"} else ("domain_specific" if layers == ("domain_specific",) else "generic")
                return CanonicalMappingResult(
                    aspect_canonical=unique_canons[0],
                    mapping_source="anchor_modifier",
                    mapping_confidence=0.7 if len(unique_canons) > 1 else 0.85,
                    matched_key=f"{lookup_anchor}.{mod_key}",
                    ambiguity_flag=len(unique_canons) > 1,
                    mapping_layers=layers,
                    mapping_scope=scope,
                )

    # 3. Exact phrase match
    if raw_phrase:
        lookup_phrase = raw_phrase.lower().strip()
        if lookup_phrase in full_map:
            return make_result(full_map[lookup_phrase], "exact_phrase", 1.0, lookup_phrase)

    # 4. Simple anchor match
    if anchor:
        lookup_anchor = anchor.lower().strip()
        if lookup_anchor in full_map:
            return make_result(full_map[lookup_anchor], "anchor_only", 0.8, lookup_anchor)

    # 5. Controlled whole-token fallback
    if raw_phrase:
        tokens = [t.strip().lower() for t in raw_phrase.split() if len(t.strip()) > 2]
        for token in tokens:
            if token in full_map:
                return make_result(full_map[token], "token_fallback", 0.55, token)
    
    # 6. Fuzzy Semantic match
    if raw_phrase:
        broad_labels = source_cfg.merged.get("broad_labels", [])
        fuzzy_res = FuzzyMatcher.find_best_match(raw_phrase, broad_labels, threshold=85.0)
        if fuzzy_res:
            matched, score = fuzzy_res
            return make_result(matched, "fuzzy_canonical", score / 100.0, matched)
            
        all_aliases = list(full_map.keys())
        fuzzy_res = FuzzyMatcher.find_best_match(raw_phrase, all_aliases, threshold=80.0)
        if fuzzy_res:
            matched_key, score = fuzzy_res
            return make_result(full_map[matched_key], "fuzzy_alias", (score / 100.0) * 0.9, matched_key)

    # 7. Latent Family Inference
    if isinstance(target, Interpretation) and target.latent_family and target.latent_family != "unknown":
        broad_labels = source_cfg.merged.get("broad_labels", [])
        family_clean = target.latent_family.lower().strip()
        
        if family_clean in broad_labels:
            return make_result(family_clean, "latent_family_inference", 0.55, family_clean)
            
        if family_clean in full_map:
            return make_result(full_map[family_clean], "latent_family_map", 0.5, family_clean)
            
        fuzzy_res = FuzzyMatcher.find_best_match(family_clean, broad_labels, threshold=90.0)
        if fuzzy_res:
            matched, score = fuzzy_res
            return make_result(matched, "fuzzy_family_inference", (score / 100.0) * 0.45, matched)

    return CanonicalMappingResult(None, mapping_source="no_match", mapping_scope="unmapped_internal")
