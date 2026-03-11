# proto/backend/services/implicit/implicit_infer.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from services.implicit.config import CONFIG
from services.implicit.encoder import SentenceEncoder, build_encoder
from services.implicit.label_maps import LabelMaps, load_label_encoder
from services.implicit.maml_model import ImplicitMAMLModel, build_maml_model
from services.implicit.maml_train import load_checkpoint
from services.implicit.sentence_splitter import split_review_into_sentences


@dataclass
class ImplicitCandidate:
    aspect: str
    confidence: float
    sentiment_hint: str
    evidence_sentence: str
    sentence_index: int
    domain_hint: str | None = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ImplicitInferenceService:
    def __init__(
        self,
        encoder: SentenceEncoder,
        model: ImplicitMAMLModel,
        label_maps: LabelMaps,
        device: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.model = model
        self.label_maps = label_maps
        self.device = device or CONFIG.device

        self.model.eval()
        self.encoder.eval()

    @torch.no_grad()
    def score_sentences(
        self,
        sentences: Sequence[str],
        allowed_aspects: Sequence[str] | None = None,
    ) -> List[List[Dict]]:
        clean_sentences = [str(s).strip() for s in sentences if str(s).strip()]
        if not clean_sentences:
            return []

        embeddings = self.encoder.encode(clean_sentences).to(self.device)
        logits = self.model(embeddings)
        probs = F.softmax(logits, dim=-1)

        allowed_ids = None
        if allowed_aspects:
            allowed_ids = {
                self.label_maps.aspect_to_id[a]
                for a in allowed_aspects
                if a in self.label_maps.aspect_to_id
            }

        scored: List[List[Dict]] = []
        for row_probs in probs:
            items: List[Dict] = []
            for class_id, score in enumerate(row_probs.tolist()):
                aspect = self.label_maps.id_to_aspect.get(class_id)
                if aspect is None:
                    continue
                if allowed_ids is not None and class_id not in allowed_ids:
                    continue
                items.append(
                    {
                        "aspect": aspect,
                        "confidence": float(score),
                    }
                )
            items.sort(key=lambda x: x["confidence"], reverse=True)
            scored.append(items)

        return scored

    def infer_review(
        self,
        review_text: str,
        allowed_aspects: Sequence[str] | None = None,
        max_predictions_per_sentence: int = 2,
        max_predictions_per_review: int | None = None,
        threshold: float | None = None,
    ) -> Dict:
        sentences = split_review_into_sentences(review_text)
        if not sentences:
            return {
                "review_text": review_text,
                "sentences": [],
                "implicit_predictions": [],
            }

        scored = self.score_sentences(
            sentences=sentences,
            allowed_aspects=allowed_aspects,
        )

        threshold = threshold if threshold is not None else CONFIG.implicit_score_threshold
        max_predictions_per_review = (
            max_predictions_per_review
            if max_predictions_per_review is not None
            else CONFIG.max_predictions_per_review
        )

        merged: Dict[str, ImplicitCandidate] = {}

        for sent_idx, (sentence, sentence_scores) in enumerate(zip(sentences, scored)):
            top_scores = sentence_scores[:max_predictions_per_sentence]

            for item in top_scores:
                aspect = item["aspect"]
                conf = float(item["confidence"])

                if conf < threshold:
                    continue

                candidate = ImplicitCandidate(
                    aspect=aspect,
                    confidence=conf,
                    sentiment_hint="negative",
                    evidence_sentence=sentence,
                    sentence_index=sent_idx,
                    domain_hint=self.label_maps.aspect_to_domain.get(aspect),
                )

                prev = merged.get(aspect)
                if prev is None or candidate.confidence > prev.confidence:
                    merged[aspect] = candidate

        final_predictions = sorted(
            [c.to_dict() for c in merged.values()],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:max_predictions_per_review]

        return {
            "review_text": review_text,
            "sentences": sentences,
            "implicit_predictions": final_predictions,
        }


def build_inference_service(
    device: str | None = None,
    freeze_backbone: bool = True,
) -> ImplicitInferenceService:
    runtime_device = device or CONFIG.device
    label_maps = load_label_encoder()

    encoder = build_encoder(
        freeze_backbone=freeze_backbone,
        device=runtime_device,
    )
    model = build_maml_model(
        input_dim=encoder.hidden_size,
        hidden_dim=CONFIG.classifier_hidden_dim,
        device=runtime_device,
    )

    load_checkpoint(
        model=model,
        optimizer=None,
        map_location=runtime_device,
    )

    return ImplicitInferenceService(
        encoder=encoder,
        model=model,
        label_maps=label_maps,
        device=runtime_device,
    )