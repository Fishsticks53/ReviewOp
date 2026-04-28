from __future__ import annotations
import abc
from abc import ABC, abstractmethod
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from ..schemas.benchmark_row import BenchmarkRow
from ..schemas.interpretation import Interpretation
from ..config import BuilderConfig
from ..explicit.phrase_rules import extract_noun_chunks, extract_dependency_phrases
from ..explicit.phrase_cleaning import is_noisy_label
from ..implicit.symptom_store import SymptomPatternStore
from ..implicit.latent_families import score_family_match
from ..canonical.domain_registry import DomainRegistry
from ..canonical.domain_maps import lookup_domain_map
from ..benchmark.novelty import detect_novelty, aggregate_row_novelty
import logging

logger = logging.getLogger(__name__)


def _find_sentence_span(text: str, sentence: str) -> tuple[int, int]:
    start = str(text or "").find(str(sentence or ""))
    if start < 0:
        return -1, -1
    return start, start + len(sentence)

def _extract_phrase_window(text: str, cue: str, window_tokens: int = 8) -> tuple[str, list[int]]:
    words = str(text or "").split()
    cue_tokens = str(cue or "").split()
    if not words or not cue_tokens:
        return str(text or ""), [0, len(str(text or ""))]
    low = [w.lower() for w in words]
    cue_low = [w.lower() for w in cue_tokens]
    start_idx = -1
    for i in range(0, len(low) - len(cue_low) + 1):
        if low[i:i + len(cue_low)] == cue_low:
            start_idx = i
            break
    if start_idx < 0:
        return str(text or ""), [0, len(str(text or ""))]
    left = max(0, start_idx - max(0, int(window_tokens)))
    right = min(len(words), start_idx + len(cue_low) + max(0, int(window_tokens)))
    snippet = " ".join(words[left:right]).strip()
    abs_start = str(text).lower().find(snippet.lower())
    if abs_start < 0:
        return snippet, [0, len(snippet)]
    return snippet, [abs_start, abs_start + len(snippet)]

def _span_hint_from_review_id(review_id: str) -> tuple[str, int, int] | None:
    parts = str(review_id or "").split(":")
    if len(parts) < 4:
        return None
    try:
        hint_term = str(parts[-3] or "").strip().lower()
        start = int(parts[-2]); end = int(parts[-1])
        if start >= 0 and end > start:
            return (hint_term, start, end)
    except Exception:
        return None
    return None

def _canonical_cue_aliases(canonical: str) -> list[str]:
    aliases = {
        "food_quality": ["food", "dish", "meal", "flavor", "taste"],
        "service_quality": ["service", "staff", "server", "waiter", "waitress"],
        "display": ["screen", "display", "lcd", "monitor"],
        "performance": ["processor", "memory", "speed", "hard drive", "ram"],
        "value": ["price", "cost", "worth", "value"],
        "battery_life": ["battery", "charge", "charging", "lasted", "drain"],
        "cleanliness": ["clean", "dirty", "smell", "sanitary"],
        "delivery": ["delivery", "arrived", "shipping", "late"],
        "customer_support": ["support", "help", "response", "agent"],
        "support": ["support", "help", "response", "agent"],
        "comfort": ["comfort", "comfortable", "noise", "fit"],
        "durability": ["durable", "broke", "worn", "lasting"],
        "reliability": ["reliable", "disconnect", "drop", "crash"],
        "quality": ["quality", "build", "craftsmanship"],
        "availability": ["available", "stock", "reservation"],
        "design": ["design", "style", "look"],
        "usability": ["easy", "use", "usability", "interface"],
        "storage": ["storage", "space", "drive", "disk"],
        "audio": ["audio", "sound", "speaker", "volume"],
        "camera": ["camera", "webcam", "photo", "video"],
        "connectivity": ["wifi", "bluetooth", "network", "connect"],
    }
    key = str(canonical or "").lower().strip()
    return aliases.get(key, [])

def _narrow_final_interpretation_evidence(row_text: str, row_id: str, interp: Interpretation, window_tokens: int = 8) -> Interpretation:
    if interp.evidence_scope not in {"sentence", "full_review"} and str(interp.evidence_text or "").strip() != str(row_text or "").strip():
        return interp
    cues = []
    cues.extend(list(interp.matched_terms or ()))
    cues.extend(list(interp.modifier_terms or ()))
    cues.append(str(interp.aspect_raw or "").replace("_", " "))
    cues.extend(_canonical_cue_aliases(interp.aspect_canonical))
    cues.append(str(interp.latent_family or "").replace("_", " "))
    cues.extend(str(interp.latent_family or "").replace("_", " ").split())
    cues.extend(str(interp.aspect_canonical or "").replace("_", " ").split())
    seen = set()
    for cue in [c.strip() for c in cues if str(c).strip()]:
        low = cue.lower()
        if low in seen:
            continue
        seen.add(low)
        if low in row_text.lower():
            txt, span = _extract_phrase_window(row_text, cue, window_tokens)
            if txt and txt.strip() and txt.strip().lower() != row_text.strip().lower():
                return replace(interp, evidence_text=txt, evidence_span=span, evidence_scope="phrase_window")
    hint = _span_hint_from_review_id(row_id)
    if hint:
        hint_term, s, e = hint
        compat_cues = [str(c).strip().lower() for c in cues if str(c).strip()]
        compat = any(hint_term == c or hint_term in c or c in hint_term for c in compat_cues)
        if compat and 0 <= s < e <= len(row_text):
            txt, span = _extract_phrase_window(row_text, row_text[s:e], window_tokens)
            if txt and txt.strip() and txt.strip().lower() != row_text.strip().lower():
                return replace(interp, evidence_text=txt, evidence_span=span, evidence_scope="phrase_window")
    opinion_cues = ("good", "great", "excellent", "poor", "bad", "slow", "fast", "friendly", "rude")
    for cue in opinion_cues:
        if cue in row_text.lower():
            txt, span = _extract_phrase_window(row_text, cue, window_tokens)
            if txt and txt.strip() and txt.strip().lower() != row_text.strip().lower():
                return replace(interp, evidence_text=txt, evidence_span=span, evidence_scope="phrase_window")
    return interp

class PipelineStage(ABC):
    @abstractmethod
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        """Process a list of rows and return the modified list."""
        pass

def _extract_for_row(row: BenchmarkRow, domain_mode: str = "full", provisional_policy: str = "strict") -> BenchmarkRow:
    """Helper function for ProcessPoolExecutor."""
    chunks = extract_noun_chunks(row.review_text)
    phrases = extract_dependency_phrases(row.review_text)
    
    from ..canonical.canonicalizer import canonicalize_interpretation
    new_interps = []
    for c in chunks:
        temp = Interpretation(
            aspect_raw=c["text"],
            aspect_canonical="unknown",
            latent_family="unknown",
            label_type="explicit",
            sentiment="unknown",
            evidence_text=c["text"],
            evidence_span=c["span"],
            source="spacy_noun_chunk",
            support_type="exact",
            source_type="explicit",
            aspect_anchor=c["aspect_anchor"],
            modifier_terms=tuple(c["modifier_terms"]),
            anchor_source=c["anchor_source"],
            evidence_scope="exact_phrase"
        )
        new_interps.append(canonicalize_interpretation(temp, domain=row.domain, domain_mode=domain_mode, provisional_policy=provisional_policy))
        
    for p in phrases:
        temp = Interpretation(
            aspect_raw=p["text"],
            aspect_canonical="unknown",
            latent_family="unknown",
            label_type="explicit",
            sentiment="unknown",
            evidence_text=p["text"],
            evidence_span=p["span"],
            source=f"spacy_{p['type']}",
            support_type="exact",
            source_type="explicit",
            aspect_anchor=p["aspect_anchor"],
            modifier_terms=tuple(p["modifier_terms"]),
            anchor_source=p["anchor_source"],
            evidence_scope="exact_phrase"
        )
        new_interps.append(canonicalize_interpretation(temp, domain=row.domain, domain_mode=domain_mode, provisional_policy=provisional_policy))
    
    return replace(
        row,
        explicit_interpretations=tuple(new_interps)
    )

class ExtractionStage(PipelineStage):
    """Stage A: Explicit Extraction using spaCy and Multiprocessing."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        if not rows:
            return rows
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        max_workers = getattr(cfg, "max_workers", 4)
        processed = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_extraction_candidates_with_modifiers"] = 0
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                from concurrent.futures import as_completed
                futures = [executor.submit(_extract_for_row, r, cfg.domain_mode, cfg.provisional_policy) for r in rows]
                for future in as_completed(futures):
                    row = future.result()
                    cfg.__dict__["_anchor_modifier_debug"]["after_extraction_candidates_with_modifiers"] += sum(1 for i in row.explicit_interpretations if tuple(getattr(i, "modifier_terms", ()) or ()))
                    processed.append(row)
                    GLOBAL_STATS.record_row_processed()
        except PermissionError:
            for r in rows:
                row = _extract_for_row(r, cfg.domain_mode, cfg.provisional_policy)
                cfg.__dict__["_anchor_modifier_debug"]["after_extraction_candidates_with_modifiers"] += sum(1 for i in row.explicit_interpretations if tuple(getattr(i, "modifier_terms", ()) or ()))
                processed.append(row)
                GLOBAL_STATS.record_row_processed()
        return processed

class InferenceStage(PipelineStage):
    """Stage B: Implicit Inference (Learned Patterns + JSON Fallback)."""
    _store_cache: dict[str, SymptomPatternStore] = {}

    def _get_store(self, path: str | None, cfg: Any = None) -> SymptomPatternStore | None:
        # Default fallback if no path provided
        if not path:
            default_path = Path("dataset_builder/config/symptom_stores/symptoms_v001.json")
            if default_path.exists():
                path = str(default_path)
            else:
                return None
                
        if path in self._store_cache:
            return self._store_cache[path]
        
        try:
            store = SymptomPatternStore.load(path)
            self._store_cache[path] = store
            return store
        except Exception as e:
            if cfg and getattr(cfg, "strict", False):
                raise RuntimeError(f"Strict Mode Failure: Failed to load symptom store from {path}: {e}")
            return None

    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        store = self._get_store(cfg.symptom_store_path, cfg=cfg)
        new_rows = []
        from ..implicit.latent_families import score_all_families
        from ..canonical.canonicalizer import canonicalize_interpretation
        from ..canonical.aspect_memory import AspectMemory
        from ..evidence.sentence_selector import select_best_sentence
        memory = AspectMemory(
            cfg.aspect_memory_path,
            auto_promote=cfg.aspect_memory_auto_promote,
            review_queue_min_support=cfg.aspect_memory_review_queue_min_support,
            review_queue_min_reviews=cfg.aspect_memory_review_queue_min_reviews,
            review_queue_min_surface_forms=cfg.aspect_memory_review_queue_min_surface_forms,
        ) if cfg.aspect_memory_path else None
        
        for row in rows:
            try:
                logger.debug(f"Processing row {row.review_id}, text='{row.review_text[:50]}...', domain='{row.domain}'")
                implicits = []
                matched_any_learned = False
                seen_canonicals = set()
                
                # 1. Learned Detection
                if store and cfg.domain_mode in {"generic_plus_learned", "full"}:
                    matches = store.match(row.review_text, domain=row.domain)
                    logger.debug(f"Store found {len(matches)} matches for {row.review_id}")
                    for match in matches:
                        matched_any_learned = True
                        family_score = score_family_match(match.matched_pattern, domain=row.domain)
                        latent_family = match.latent_family or family_score.latent_family
                        
                        temp_interp = Interpretation(
                            aspect_raw=match.matched_pattern,
                            aspect_canonical=match.aspect_canonical or "unknown",
                            latent_family=latent_family,
                            label_type="implicit",
                            sentiment="unknown",
                            evidence_text=row.review_text[match.start_char:match.end_char],
                            evidence_span=[match.start_char, match.end_char],
                            source="symptom_store",
                            support_type="contextual",
                            matched_pattern=match.matched_pattern,
                            pattern_id=match.pattern_id,
                            pattern_confidence=match.confidence,
                            evidence_scope="exact_phrase" if match.match_type == "exact" else "phrase_window",
                            source_type="implicit_learned"
                        )
                        
                        canonicalized = canonicalize_interpretation(temp_interp, domain=row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy)
                        if memory:
                            mem_matches = memory.match_promoted(row.review_text)
                            for mem in mem_matches:
                                if mem.aspect_raw.lower() in match.matched_text.lower() or match.matched_text.lower() in mem.aspect_raw.lower():
                                    canonical = canonicalized.aspect_canonical
                                    resolution = "symptom_store_preferred"
                                    mem_conf = float(getattr(mem, "confidence", 0.0) or 0.0)
                                    if (
                                        mem.validation_status == "manual_validated"
                                        and mem.aspect_canonical
                                        and mem.aspect_canonical != canonicalized.aspect_canonical
                                        and mem_conf >= float(match.confidence or 0.0)
                                    ):
                                        canonical = mem.aspect_canonical
                                        resolution = "manual_validated_aspect_memory_preferred"
                                        cfg.__dict__.setdefault("_aspect_memory_metrics", {}).setdefault("promoted_matches_used", 0)
                                        cfg.__dict__["_aspect_memory_metrics"]["promoted_matches_used"] += 1
                                    canonicalized = replace(
                                        canonicalized,
                                        aspect_canonical=canonical,
                                        mapping_scope="learned_store",
                                        mapping_layers=("symptom_store", "aspect_memory"),
                                        conflict_resolution=resolution,
                                    )
                        implicits.append(canonicalized)
                        seen_canonicals.add(canonicalized.aspect_canonical)
                
                # 2. JSON Fallback Detection (Broad Keyword Matching)
                # 2.1 AspectMemory promoted matching
                if memory and cfg.domain_mode in {"generic_plus_learned", "full"}:
                    memory_matches = memory.match_promoted(row.review_text)
                    for mem in memory_matches:
                        evidence_text = mem.aspect_raw
                        start = row.review_text.lower().find(mem.aspect_raw.lower())
                        end = start + len(mem.aspect_raw) if start >= 0 else -1
                        temp_interp = Interpretation(
                            aspect_raw=mem.aspect_raw,
                            latent_family="unknown",
                            aspect_canonical=mem.aspect_canonical or "unknown",
                            label_type="implicit",
                            sentiment="unknown",
                            evidence_text=evidence_text if start >= 0 else row.review_text,
                            evidence_span=[start, end] if start >= 0 else [0, len(row.review_text)],
                            source="aspect_memory",
                            support_type="contextual",
                            source_type="implicit_learned",
                            matched_pattern=mem.aspect_raw,
                            pattern_id=f"aspect_memory:{mem.aspect_raw.lower().replace(' ', '_')}",
                            evidence_scope="exact_phrase" if start >= 0 else "full_review",
                            mapping_scope="learned_store",
                            mapping_layers=("aspect_memory",),
                        )
                        canonicalized = canonicalize_interpretation(temp_interp, domain=row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy)
                        if canonicalized.aspect_canonical not in seen_canonicals:
                            implicits.append(canonicalized)
                            seen_canonicals.add(canonicalized.aspect_canonical)

                # 2.2 JSON Fallback Detection (Broad Keyword Matching)
                json_scores = score_all_families(row.review_text, domain=row.domain)
                for score in json_scores:
                    cue_candidates = []
                    if score.matched_terms:
                        cue_candidates.extend(list(score.matched_terms))
                    cue_candidates.append(score.latent_family.replace("_", " "))
                    cue = next((c for c in cue_candidates if c and c.lower() in row.review_text.lower()), score.latent_family)
                    sentence = select_best_sentence(row.review_text, cue)
                    sent_span = _find_sentence_span(row.review_text, sentence)
                    evidence_scope = "sentence"
                    evidence_text = sentence
                    evidence_span = [sent_span[0], sent_span[1]]
                    if sent_span == (-1, -1) and score.matched_terms:
                        token = score.matched_terms[0].lower()
                        start = row.review_text.lower().find(token)
                        if start >= 0:
                            end = min(len(row.review_text), start + max(len(token), 32))
                            evidence_text = row.review_text[start:end]
                            evidence_span = [start, end]
                            evidence_scope = "phrase_window"
                        else:
                            evidence_text = row.review_text
                            evidence_span = [0, len(row.review_text)]
                            evidence_scope = "full_review"
                    elif sent_span == (-1, -1):
                        evidence_text = row.review_text
                        evidence_span = [0, len(row.review_text)]
                        evidence_scope = "full_review"

                    temp_interp = Interpretation(
                        aspect_raw=score.latent_family,
                        latent_family=score.latent_family,
                        aspect_canonical="unknown",
                        label_type="implicit",
                        sentiment="unknown",
                        evidence_text=evidence_text,
                        evidence_span=evidence_span,
                        source="latent_family_matcher",
                        support_type="contextual",
                        source_type="implicit_json",
                        evidence_scope=evidence_scope,
                        matched_terms=tuple(score.matched_terms),
                        implicit_trigger="latent_family_match",
                    )
                    canonicalized = canonicalize_interpretation(temp_interp, domain=row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy)
                    if canonicalized.evidence_scope in {"sentence", "full_review"}:
                        post_cues = [
                            *(list(getattr(canonicalized, "matched_terms", ()) or [])),
                            str(canonicalized.aspect_raw or "").replace("_", " "),
                            str(canonicalized.aspect_canonical or "").replace("_", " "),
                            str(canonicalized.latent_family or "").replace("_", " "),
                        ]
                        for pcue in post_cues:
                            if pcue and pcue.lower() in row.review_text.lower():
                                win_text, win_span = _extract_phrase_window(row.review_text, pcue, cfg.evidence_window_tokens)
                                if win_text and win_span != [0, len(row.review_text)]:
                                    canonicalized = replace(
                                        canonicalized,
                                        evidence_text=win_text,
                                        evidence_span=win_span,
                                        evidence_scope="phrase_window",
                                    )
                                    break
                    
                    if canonicalized.aspect_canonical not in seen_canonicals:
                        implicits.append(canonicalized)
                        seen_canonicals.add(canonicalized.aspect_canonical)
                
                row = replace(row, implicit_interpretations=tuple(implicits))
                new_rows.append(row)
            finally:
                GLOBAL_STATS.record_row_processed()
            
        return new_rows

class EvidenceStage(PipelineStage):
    """Stage C: Evidence Grounding and Span Validation."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..evidence.sentence_selector import select_best_sentence
        from ..evidence.span_extractor import extract_span_from_sentence
        new_rows = []
        for row in rows:
            new_gold = []
            for i in row.gold_interpretations:
                if not i.evidence_text or i.evidence_span == [-1, -1] or i.evidence_span == (0, 0) or i.evidence_span == [0, 0]:
                    sentence = select_best_sentence(row.review_text, i.aspect_raw)
                    span = extract_span_from_sentence(row.review_text, sentence)
                    i = replace(i, evidence_text=sentence, evidence_span=tuple(span), evidence_scope="sentence")
                new_gold.append(i)
            new_rows.append(replace(row, gold_interpretations=tuple(new_gold)))
        return new_rows

class PostVerificationEvidenceStage(PipelineStage):
    """Stage D2: Grounding specifically for verifier-added interpretations."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..evidence.sentence_selector import select_best_sentence
        from ..evidence.span_extractor import extract_span_from_sentence
        new_rows = []
        for row in rows:
            new_gold = []
            for i in row.gold_interpretations:
                # Only ground if it's from the verifier or has missing span
                if i.source == "llm_verifier" or i.evidence_span == [0, len(row.review_text)]:
                    sentence = select_best_sentence(row.review_text, i.evidence_text or i.aspect_raw)
                    span = extract_span_from_sentence(row.review_text, sentence)
                    if span == [-1, -1]:
                        # If we can't ground it, drop it (as per design)
                        continue
                    i = replace(i, evidence_text=sentence, evidence_span=tuple(span), evidence_scope="sentence")
                new_gold.append(i)
            new_rows.append(replace(row, gold_interpretations=tuple(new_gold)))
        return new_rows

class VerificationStage(PipelineStage):
    """Stage D: LLM-based Verification (Keep/Drop/Merge)."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..verify.llm_verifier import LLMVerifier
        
        if cfg.llm_provider == "none":
            new_rows = []
            for row in rows:
                filtered_gold = []
                for i in row.gold_interpretations:
                    if is_noisy_label(i.aspect_raw):
                        logger.info(f"Verification {row.review_id} DROPPING noisy label: {i.aspect_raw}")
                        continue
                    filtered_gold.append(i)
                new_rows.append(replace(row, gold_interpretations=tuple(filtered_gold)))
            return new_rows

        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        verifier = LLMVerifier(cfg)

        def process_row(row: BenchmarkRow) -> BenchmarkRow:
            try:
                # First pass: deterministic noisy filter
                filtered_gold = [i for i in row.gold_interpretations if not is_noisy_label(i.aspect_raw)]
                if not filtered_gold:
                    return replace(row, gold_interpretations=tuple())
                
                try:
                    v_row = verifier.verify_row(replace(row, gold_interpretations=tuple(filtered_gold)))
                    # Second pass: ensure NO noisy labels survive
                    final_gold = [i for i in v_row.gold_interpretations if not is_noisy_label(i.aspect_raw)]
                    return replace(v_row, gold_interpretations=tuple(final_gold))
                except Exception:
                    return replace(row, gold_interpretations=tuple(filtered_gold))
            finally:
                GLOBAL_STATS.record_row_processed()

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            return list(executor.map(process_row, rows))

class FusionStage(PipelineStage):
    """Stage E: Fusion of Explicit and Implicit Candidates."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..fusion.merge_candidates import merge_explicit_implicit
        new_rows = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_fusion_candidates_with_modifiers"] = 0
        for row in rows:
            merged = merge_explicit_implicit(list(row.explicit_interpretations), list(row.implicit_interpretations))
            cfg.__dict__["_anchor_modifier_debug"]["after_fusion_candidates_with_modifiers"] += sum(1 for i in merged if tuple(getattr(i, "modifier_terms", ()) or ()))
            logger.debug(f"Fusion {row.review_id}: explicit={len(row.explicit_interpretations)}, implicit={len(row.implicit_interpretations)} -> merged={len(merged)}")
            for i in merged:
                if i.label_type == "implicit":
                    logger.debug(f"  - Merged implicit: {i.aspect_canonical} ({i.aspect_raw})")
            new_rows.append(replace(row, gold_interpretations=tuple(merged)))
        return new_rows

class CanonicalizationStage(PipelineStage):
    """Stage F: Canonical Mapping and Pruning."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..canonical.canonicalizer import canonicalize_interpretation
        from ..canonical.broad_label_policy import prune_broad_labels
        from ..canonical.fragment_collapse import collapse_same_evidence_fragments
        from ..canonical.aspect_memory import AspectMemory
        memory = AspectMemory(
            cfg.aspect_memory_path,
            auto_promote=cfg.aspect_memory_auto_promote,
            review_queue_min_support=cfg.aspect_memory_review_queue_min_support,
            review_queue_min_reviews=cfg.aspect_memory_review_queue_min_reviews,
            review_queue_min_surface_forms=cfg.aspect_memory_review_queue_min_surface_forms,
        ) if cfg.aspect_memory_path else None
        
        new_rows = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_canonicalization"] = 0
        cfg.__dict__["_anchor_modifier_debug"]["after_pruning"] = 0
        for row in rows:
            # 1. Canonicalize
            canons = [canonicalize_interpretation(i, row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy) for i in row.gold_interpretations]
            cfg.__dict__["_anchor_modifier_debug"]["after_canonicalization"] += sum(1 for i in canons if i.mapping_source == "anchor_modifier")
            for i in canons:
                if i.mapping_source == "memory_candidate":
                    if memory:
                        before = memory.get_entry(i.aspect_raw)
                        memory.add_evidence(i.aspect_raw, row.review_id, i.evidence_text or row.review_text, row.domain)
                        cfg.__dict__.setdefault("_aspect_memory_metrics", {})
                        if before is None:
                            cfg.__dict__["_aspect_memory_metrics"]["candidates_added"] = cfg.__dict__["_aspect_memory_metrics"].get("candidates_added", 0) + 1
                        else:
                            cfg.__dict__["_aspect_memory_metrics"]["candidates_updated"] = cfg.__dict__["_aspect_memory_metrics"].get("candidates_updated", 0) + 1
                        cfg.__dict__["_aspect_memory_metrics"]["memory_candidate_count"] = cfg.__dict__["_aspect_memory_metrics"].get("memory_candidate_count", 0) + 1
                    else:
                        cfg.__dict__.setdefault("_rejection_reason_counts", {})
                        cfg.__dict__["_rejection_reason_counts"]["memory_candidate_no_store"] = cfg.__dict__["_rejection_reason_counts"].get("memory_candidate_no_store", 0) + 1
            dropped_noise = sum(1 for i in canons if i.mapping_source == "dropped_noise")
            if dropped_noise:
                cfg.__dict__.setdefault("_rejection_reason_counts", {})
                cfg.__dict__["_rejection_reason_counts"]["all_candidates_noisy"] = (
                    cfg.__dict__["_rejection_reason_counts"].get("all_candidates_noisy", 0) + dropped_noise
                )
            canons = [i for i in canons if i.mapping_source not in {"dropped_noise", "memory_candidate"}]
            # 2. Collapse fragments
            collapsed, _ = collapse_same_evidence_fragments(canons)
            # 3. Prune broad labels
            final_gold, _ = prune_broad_labels(collapsed, row.domain)
            final_gold = [_narrow_final_interpretation_evidence(row.review_text, row.review_id, i, cfg.evidence_window_tokens) for i in final_gold]
            cfg.__dict__["_anchor_modifier_debug"]["after_pruning"] += sum(1 for i in final_gold if i.mapping_source == "anchor_modifier")
            new_rows.append(replace(row, gold_interpretations=tuple(final_gold)))
        if memory:
            memory.save()
            memory.write_review_queue(Path(cfg.output_dir) / "aspect_memory_review_queue.json")
            memory.write_summary(Path(cfg.output_dir) / "aspect_memory_summary.json")
        return new_rows

class SentimentStage(PipelineStage):
    """Stage G: Aspect-Conditioned Sentiment Analysis."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..sentiment.classifier import SentimentClassifier
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        classifier = SentimentClassifier(cfg)
        
        def process_row(row: BenchmarkRow) -> BenchmarkRow:
            try:
                if not row.gold_interpretations:
                    return row
                new_gold = classifier.classify_batch(row.review_text, list(row.gold_interpretations))
                return replace(row, gold_interpretations=tuple(new_gold))
            finally:
                GLOBAL_STATS.record_row_processed()

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            return list(executor.map(process_row, rows))

class BenchmarkStage(PipelineStage):
    """Stage H: Hardness Scoring and Finalization."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..benchmark.hardness_scorer import score_row_hardness
        from ..benchmark.novelty import detect_novelty
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        seen_texts = set()
        unique_rows = []
        
        from ..canonical.aspect_memory import AspectMemory
        memory = AspectMemory(
            cfg.aspect_memory_path,
            auto_promote=cfg.aspect_memory_auto_promote,
            review_queue_min_support=cfg.aspect_memory_review_queue_min_support,
            review_queue_min_reviews=cfg.aspect_memory_review_queue_min_reviews,
            review_queue_min_surface_forms=cfg.aspect_memory_review_queue_min_surface_forms,
        ) if cfg.aspect_memory_path else None
        cfg.__dict__.setdefault("_aspect_memory_metrics", {})
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_final"] = 0
        for row in rows:
            try:
                # 1. Dedupe by text
                if row.review_text in seen_texts:
                    continue
                seen_texts.add(row.review_text)
                
                # 2. Filter empty gold
                if not row.gold_interpretations:
                    continue
                    
                # 3. Cap interpretations
                final_gold = sorted(list(row.gold_interpretations), key=lambda i: i.canonical_confidence, reverse=True)[:8]
                final_gold = [_narrow_final_interpretation_evidence(row.review_text, row.review_id, i, cfg.evidence_window_tokens) for i in final_gold]
                cfg.__dict__["_anchor_modifier_debug"]["after_final"] += sum(1 for i in final_gold if i.mapping_source == "anchor_modifier")
                
                # 4. Calculate Ambiguity Score
                sentiments = {i.sentiment for i in final_gold if i.sentiment != "unknown"}
                ambiguity = min(1.0, (len(final_gold) / 10.0) + (0.3 if len(sentiments) > 1 else 0.0))
                
                # 5. Determine Novelty Status
                domain_cfg = DomainRegistry.get_config(row.domain)
                known_from_map = set(domain_cfg.get("domain_maps", {}).values())
                known_from_families = set(domain_cfg.get("latent_families", {}).keys())
                known_canonicals = known_from_map | known_from_families

                scored_gold = []
                for i in final_gold:
                    # Open-World Learning: capture unmapped/provisional explicit aspects
                    if i.mapping_source in ["unmapped", "provisional"] and i.label_type == "explicit" and memory:
                        before_status = memory.get_entry(i.aspect_raw).status if memory.get_entry(i.aspect_raw) else None
                        memory.add_evidence(i.aspect_raw, row.review_id, row.review_text, row.domain)
                        cfg.__dict__["_aspect_memory_metrics"]["candidates_added"] = cfg.__dict__["_aspect_memory_metrics"].get("candidates_added", 0) + 1
                        after = memory.get_entry(i.aspect_raw)
                        if before_status != "promoted" and after and after.status == "promoted":
                            cfg.__dict__["_aspect_memory_metrics"]["candidates_promoted_this_run"] = cfg.__dict__["_aspect_memory_metrics"].get("candidates_promoted_this_run", 0) + 1
                        if after and after.status == "rejected":
                            cfg.__dict__["_aspect_memory_metrics"]["rejected_candidates_this_run"] = cfg.__dict__["_aspect_memory_metrics"].get("rejected_candidates_this_run", 0) + 1

                    novelty_status = detect_novelty(
                        i.aspect_canonical, 
                        known_canonicals,
                        mapping_confidence=i.canonical_confidence or 0.0,
                        mapping_source=i.mapping_source or "none"
                    )
                    i = replace(i, novelty_status=novelty_status)
                    scored_gold.append(i)
                
                row_novelty = aggregate_row_novelty(scored_gold)
                
                unique_rows.append(replace(row, 
                    gold_interpretations=tuple(scored_gold),
                    hardness_tier=(h := score_row_hardness(replace(row, gold_interpretations=tuple(scored_gold)))),
                    abstain_acceptable=(h in ["H2", "H3"]),
                    novelty_status=row_novelty,
                    ambiguity_score=ambiguity
                ))
            finally:
                GLOBAL_STATS.record_row_processed()
            
        if memory:
            memory.save()
            memory.write_review_queue(Path(cfg.output_dir) / "aspect_memory_review_queue.json")
            memory.write_summary(Path(cfg.output_dir) / "aspect_memory_summary.json")
            cfg.__dict__["_aspect_memory_metrics"]["promoted_entries_total"] = sum(1 for e in memory.entries.values() if e.status == "promoted")
            cfg.__dict__["_aspect_memory_metrics"]["review_queue_count"] = sum(1 for e in memory.entries.values() if e.status == "review_queue")
        return unique_rows
