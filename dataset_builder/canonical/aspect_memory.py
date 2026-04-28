from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryEntry:
    aspect_raw: str
    status: str = "detected"
    support_count: int = 0
    unique_reviews: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    evidence_samples: list[str] = field(default_factory=list)
    aspect_canonical: Optional[str] = None
    validation_status: str = "unverified"
    unique_review_count: int = 0
    evidence_examples: list[dict[str, Any]] = field(default_factory=list)
    generic_parent: Optional[str] = None
    domain_specific_aspect: Optional[str] = None
    confidence: float = 0.0
    cue_type: Optional[str] = None
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unique_reviews"] = list(self.unique_reviews)
        d["domains"] = list(self.domains)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        d["unique_reviews"] = set(d.get("unique_reviews", []))
        d["domains"] = set(d.get("domains", []))
        raw_examples = list(d.get("evidence_examples", []) or [])
        normalized: list[dict[str, Any]] = []
        for e in raw_examples:
            if isinstance(e, str):
                normalized.append({"review_id": "", "domain": "", "evidence_text": e, "sentiment": "unknown", "cue_type": "unknown", "run_id": ""})
            elif isinstance(e, dict):
                normalized.append({
                    "review_id": str(e.get("review_id", "")),
                    "domain": str(e.get("domain", "")),
                    "evidence_text": str(e.get("evidence_text", "")),
                    "sentiment": str(e.get("sentiment", "unknown")),
                    "cue_type": str(e.get("cue_type", "unknown")),
                    "run_id": str(e.get("run_id", "")),
                })
        d["evidence_examples"] = normalized
        return cls(**d)


class AspectMemory:
    BROAD_OBJECT_BLOCKLIST = {"computer", "restaurant", "product", "thing", "place", "dinner", "table", "bar", "glass"}
    STRONG_EVENT_SET = {
        "failure_event", "delay_event", "durability_event", "reliability_event",
        "support_problem", "usability_problem", "comfort_problem", "quality_judgment",
    }

    def __init__(
        self,
        storage_path: str | Path,
        auto_promote: bool = False,
        review_queue_min_support: int = 5,
        review_queue_min_reviews: int = 3,
        review_queue_min_surface_forms: int = 2,
    ):
        self.storage_path = Path(storage_path)
        self.auto_promote = auto_promote
        self.review_queue_min_support = int(review_queue_min_support)
        self.review_queue_min_reviews = int(review_queue_min_reviews)
        self.review_queue_min_surface_forms = int(review_queue_min_surface_forms)
        self.entries: dict[str, MemoryEntry] = {}
        self.load()

    def _make_run_id(self, seed: int = 42) -> str:
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"

    def _infer_generic_parent(self, aspect: str) -> Optional[str]:
        low = str(aspect or "").lower()
        hints = {
            "quality": ("quality", "build", "craft"),
            "performance": ("speed", "processor", "ram", "memory"),
            "storage": ("storage", "drive", "disk", "space"),
            "display": ("screen", "display", "lcd"),
            "durability": ("durable", "broke", "ripped", "wear"),
            "service": ("service", "staff", "support", "wait"),
            "delivery": ("delivery", "shipping", "arrived", "late"),
            "comfort": ("comfort", "noisy", "fit"),
        }
        for parent, keys in hints.items():
            if any(k in low for k in keys):
                return parent
        return None

    def add_evidence(
        self,
        aspect_raw: str,
        review_id: str,
        context: str,
        domain: str,
        *,
        sentiment: str = "unknown",
        cue_type: str = "unknown",
        run_id: Optional[str] = None,
    ):
        raw_key = aspect_raw.lower().strip()
        if raw_key not in self.entries:
            self.entries[raw_key] = MemoryEntry(aspect_raw=aspect_raw)

        entry = self.entries[raw_key]
        entry.support_count += 1
        entry.unique_reviews.add(review_id)
        entry.domains.add(domain)
        entry.unique_review_count = len(entry.unique_reviews)
        entry.cue_type = cue_type or entry.cue_type
        if not entry.run_id:
            entry.run_id = run_id or self._make_run_id()
        if not entry.generic_parent:
            entry.generic_parent = self._infer_generic_parent(aspect_raw)
        if len(entry.evidence_samples) < 5:
            entry.evidence_samples.append(context)
        if len(entry.evidence_examples) < 5:
            entry.evidence_examples.append({
                "review_id": review_id,
                "domain": domain,
                "evidence_text": context,
                "sentiment": sentiment,
                "cue_type": cue_type,
                "run_id": run_id or entry.run_id or self._make_run_id(),
            })
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
            entry.status = "candidate"
            return

        if self.auto_promote and unique_count >= 3 and entry.support_count >= 5:
            entry.status = "promoted"
            entry.validation_status = "auto_validated"
            entry.aspect_canonical = entry.aspect_raw.lower().strip().replace(" ", "_")
            if entry.aspect_canonical in canonicals and canonicals:
                entry.status = "review_queue"
                entry.validation_status = "unverified"
        elif (
            entry.support_count >= self.review_queue_min_support
            and entry.unique_review_count >= self.review_queue_min_reviews
            and len(set(e.get("evidence_text", "").lower().strip() for e in entry.evidence_examples if isinstance(e, dict))) >= self.review_queue_min_surface_forms
            and str(entry.aspect_raw or "").lower().strip() not in self.BROAD_OBJECT_BLOCKLIST
            and (entry.generic_parent is not None or (entry.cue_type in self.STRONG_EVENT_SET))
        ):
            entry.status = "review_queue"
            entry.validation_status = "unverified"
        elif unique_count >= 1:
            entry.status = "candidate"
            entry.validation_status = "unverified"
        else:
            entry.status = "detected"

    def write_summary(self, output_path: str | Path) -> None:
        total = len(self.entries)
        review_queue_count = sum(1 for e in self.entries.values() if e.status == "review_queue")
        promoted_count = sum(1 for e in self.entries.values() if e.status == "promoted")
        rejected_count = sum(1 for e in self.entries.values() if e.status == "rejected")
        top = sorted(self.entries.values(), key=lambda e: e.support_count, reverse=True)[:20]
        payload = {
            "total_entries": total,
            "review_queue_count": review_queue_count,
            "promoted_count": promoted_count,
            "rejected_count": rejected_count,
            "top_candidates": [
                {
                    "aspect_raw": e.aspect_raw,
                    "support_count": e.support_count,
                    "unique_review_count": e.unique_review_count,
                    "domains": sorted(list(e.domains)),
                }
                for e in top
            ],
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
                "evidence_examples": entry.evidence_examples[:3],
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
        data = {"entries": {k: v.to_dict() for k, v in self.entries.items()}}
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self):
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.entries = {k: MemoryEntry.from_dict(v) for k, v in data.get("entries", {}).items()}
        except Exception:
            self.entries = {}
