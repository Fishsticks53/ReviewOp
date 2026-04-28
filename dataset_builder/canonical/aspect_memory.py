from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

@dataclass
class MemoryEntry:
    aspect_raw: str
    status: str = "detected" # detected, candidate, validated, promoted
    support_count: int = 0
    unique_reviews: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    evidence_samples: list[str] = field(default_factory=list)
    aspect_canonical: Optional[str] = None
    validation_status: str = "unverified"
    unique_review_count: int = 0
    evidence_examples: list[str] = field(default_factory=list)
    generic_parent: Optional[str] = None
    domain_specific_aspect: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unique_reviews"] = list(self.unique_reviews)
        d["domains"] = list(self.domains)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        d["unique_reviews"] = set(d.get("unique_reviews", []))
        d["domains"] = set(d.get("domains", []))
        return cls(**d)

class AspectMemory:
    def __init__(self, storage_path: str | Path, auto_promote: bool = False):
        self.storage_path = Path(storage_path)
        self.auto_promote = auto_promote
        self.entries: dict[str, MemoryEntry] = {}
        self.load()

    def add_evidence(self, aspect_raw: str, review_id: str, context: str, domain: str):
        raw_key = aspect_raw.lower().strip()
        if raw_key not in self.entries:
            self.entries[raw_key] = MemoryEntry(aspect_raw=aspect_raw)
        
        entry = self.entries[raw_key]
        entry.support_count += 1
        entry.unique_reviews.add(review_id)
        entry.domains.add(domain)
        entry.unique_review_count = len(entry.unique_reviews)
        
        if len(entry.evidence_samples) < 5:
            entry.evidence_samples.append(context)
        if len(entry.evidence_examples) < 5:
            entry.evidence_examples.append(context)
            
        self._update_status(entry)

    def _update_status(self, entry: MemoryEntry):
        unique_count = len(entry.unique_reviews)
        bad_tokens = {"good", "bad", "great", "terrible", "excellent", "poor", "thing", "item", "product", "stuff", "use", "works"}
        canonicals = {e.aspect_canonical for e in self.entries.values() if e.aspect_canonical}
        if not str(entry.aspect_raw or "").strip():
            entry.status = "rejected"
            entry.validation_status = "failed_validation"
            return
        if entry.aspect_raw.lower().strip() in bad_tokens:
            entry.status = "rejected"
            entry.validation_status = "failed_validation"
            return
        if not entry.evidence_examples:
            entry.status = "review_queue"
            return
        
        if self.auto_promote and unique_count >= 3 and entry.support_count >= 5:
            entry.status = "promoted"
            entry.validation_status = "auto_validated"
            entry.aspect_canonical = entry.aspect_raw.lower().strip().replace(" ", "_")
            if entry.aspect_canonical in canonicals and canonicals:
                entry.status = "review_queue"
                entry.validation_status = "unverified"
        elif unique_count >= 1:
            entry.status = "review_queue"
            entry.validation_status = "unverified"
        else:
            entry.status = "detected"

    def write_review_queue(self, output_path: str | Path) -> None:
        items = []
        for k, entry in self.entries.items():
            if entry.status != "review_queue":
                continue
            items.append({
                "aspect_id": k.replace(" ", "_"),
                "domain": next(iter(entry.domains), "unknown"),
                "surface_forms": [entry.aspect_raw],
                "suggested_canonical": entry.aspect_canonical or entry.aspect_raw.lower().strip().replace(" ", "_"),
                "generic_parent": entry.generic_parent,
                "support_count": entry.support_count,
                "unique_review_count": entry.unique_review_count,
                "evidence_examples": [{"review_id": "", "evidence_text": s} for s in entry.evidence_examples[:3]],
                "reason": "needs_manual_validation",
            })
        payload = {"created_at": "", "total": len(items), "items": items}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_entry(self, aspect_raw: str) -> Optional[MemoryEntry]:
        return self.entries.get(aspect_raw.lower().strip())

    def match_promoted(self, text: str) -> list[MemoryEntry]:
        text_norm = str(text or "").lower()
        matches: list[MemoryEntry] = []
        for entry in self.entries.values():
            if entry.status != "promoted":
                continue
            if entry.validation_status not in {"auto_validated", "manual_validated"}:
                continue
            if str(entry.aspect_raw or "").lower() in text_norm:
                matches.append(entry)
        return matches

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": {k: v.to_dict() for k, v in self.entries.items()}
        }
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self):
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.entries = {k: MemoryEntry.from_dict(v) for k, v in data.get("entries", {}).items()}
        except Exception:
            self.entries = {}
