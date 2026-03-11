# proto/backend/services/implicit/encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from services.implicit.config import CONFIG


@dataclass
class EncodedBatch:
    texts: List[str]
    embeddings: torch.Tensor  # [batch, hidden_dim]


class SentenceEncoder(nn.Module):
    """
    Lightweight sentence encoder using a transformer backbone + mean pooling.
    Good starting point for episodic implicit-aspect training.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int | None = None,
        device: str | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.model_name = model_name or CONFIG.encoder_name
        self.max_length = max_length or CONFIG.max_length
        self.device_name = device or CONFIG.device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.to(self.device_name)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, CONFIG.embedding_dim), device=self.device_name)

        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokenized = {k: v.to(self.device_name) for k, v in tokenized.items()}

        outputs = self.backbone(**tokenized)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

        embeddings = mean_pooling(
            token_embeddings=last_hidden_state,
            attention_mask=tokenized["attention_mask"],
        )
        embeddings = l2_normalize(embeddings)
        return embeddings

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        was_training = self.training
        self.eval()
        embeddings = self.forward(texts)
        if was_training:
            self.train()
        return embeddings

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)


def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling over non-masked tokens.
    token_embeddings: [batch, seq_len, hidden]
    attention_mask: [batch, seq_len]
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked_embeddings = token_embeddings * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def build_encoder(
    freeze_backbone: bool = False,
    device: str | None = None,
) -> SentenceEncoder:
    return SentenceEncoder(
        model_name=CONFIG.encoder_name,
        max_length=CONFIG.max_length,
        device=device or CONFIG.device,
        freeze_backbone=freeze_backbone,
    )


def encode_texts(
    encoder: SentenceEncoder,
    texts: Sequence[str],
) -> EncodedBatch:
    embeddings = encoder.encode(texts)
    return EncodedBatch(
        texts=list(texts),
        embeddings=embeddings,
    )


def encode_episode_rows(
    encoder: SentenceEncoder,
    rows: Sequence[dict],
    text_key: str = "evidence_sentence",
    label_key: str = "local_label",
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    texts: List[str] = []
    labels: List[int] = []

    for row in rows:
        text = str(row.get(text_key, "")).strip()
        label = row.get(label_key)
        if not text or label is None:
            continue
        texts.append(text)
        labels.append(int(label))

    embeddings = encoder.encode(texts)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=embeddings.device)
    return embeddings, label_tensor, texts