from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from ..canonical.domain_registry import DomainRegistry

@dataclass(frozen=True)
class FamilyScore:
    latent_family: str
    confidence: float
    matched_terms: tuple[str, ...] = ()

def load_latent_families(domain: str | None = None) -> dict[str, list[str]]:
    """Load latent families from registry."""
    return DomainRegistry.get_latent_families(domain)

def score_all_families(
    text: str, 
    families: dict[str, list[str]] | None = None,
    domain: str | None = None
) -> list[FamilyScore]:
    """Score a text against all latent families and return all matches with confidence > 0."""
    families = families or load_latent_families(domain)
    text_norm = str(text or "").lower()
    results: list[FamilyScore] = []
    
    for family, terms in families.items():
        matches = [term for term in terms if term.lower() in text_norm]
        if matches:
            confidence = min(0.95, 0.4 + 0.1 * len(matches))
            results.append(FamilyScore(family, confidence, tuple(matches)))
            
    return sorted(results, key=lambda x: -x.confidence)

def score_family_match(
    text: str, 
    families: dict[str, list[str]] | None = None,
    symptom_prior: str | None = None,
    domain: str | None = None
) -> FamilyScore:
    """
    Score a text against latent families. 
    Returns the single best match.
    """
    all_matches = score_all_families(text, families, domain)
    
    if symptom_prior and symptom_prior != "unknown":
        # If we have a prior, see if it's in all_matches
        for match in all_matches:
            if match.latent_family == symptom_prior:
                # Boost confidence
                new_conf = min(1.0, match.confidence + 0.2)
                return FamilyScore(match.latent_family, new_conf, match.matched_terms)
        return FamilyScore(symptom_prior, 0.3, ())

    if not all_matches:
        return FamilyScore("unknown", 0.0, ())
        
    return all_matches[0]
