# proto/backend/services/implicit/maml_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from services.implicit.config import CONFIG


@dataclass
class EpisodeLogits:
    support_logits: torch.Tensor
    query_logits: torch.Tensor
    support_loss: torch.Tensor
    query_loss: torch.Tensor
    query_preds: torch.Tensor
    query_acc: float


class EpisodicClassifierHead(nn.Module):
    """
    Small classification head placed on top of sentence embeddings.
    This is the part we adapt in the inner loop first.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        input_dim = input_dim or CONFIG.embedding_dim
        hidden_dim = hidden_dim or CONFIG.classifier_hidden_dim
        dropout = dropout if dropout is not None else CONFIG.dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, CONFIG.n_way)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.out(h)

    def functional_forward(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = F.linear(x, params["net.0.weight"], params["net.0.bias"])
        x = F.relu(x)
        x = F.dropout(x, p=CONFIG.dropout, training=self.training)

        x = F.linear(x, params["net.3.weight"], params["net.3.bias"])
        x = F.relu(x)
        x = F.dropout(x, p=CONFIG.dropout, training=self.training)

        x = F.linear(x, params["out.weight"], params["out.bias"])
        return x


class ImplicitMAMLModel(nn.Module):
    """
    MAML-ready episodic classifier.
    Version 1 adapts only the classifier head on top of sentence embeddings.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim or CONFIG.embedding_dim
        self.hidden_dim = hidden_dim or CONFIG.classifier_hidden_dim
        self.head = EpisodicClassifierHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)

    def clone_head_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.clone()
            for name, param in self.head.named_parameters()
        }

    def inner_adapt(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        inner_steps: int | None = None,
        inner_lr: float | None = None,
        create_graph: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        steps = inner_steps or CONFIG.inner_steps
        lr = inner_lr or CONFIG.inner_lr

        fast_weights = {
            name: param
            for name, param in self.head.named_parameters()
        }

        support_loss = torch.tensor(0.0, device=support_embeddings.device)

        for _ in range(steps):
            support_logits = self.head.functional_forward(support_embeddings, fast_weights)
            support_loss = F.cross_entropy(support_logits, support_labels)

            grads = torch.autograd.grad(
                support_loss,
                list(fast_weights.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
            )

            fast_weights = {
                name: weight - lr * grad
                for (name, weight), grad in zip(fast_weights.items(), grads)
            }

        return fast_weights, support_loss

    def episode_forward(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        inner_steps: int | None = None,
        inner_lr: float | None = None,
        create_graph: bool = True,
    ) -> EpisodeLogits:
        fast_weights, support_loss = self.inner_adapt(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            create_graph=create_graph,
        )

        support_logits = self.head.functional_forward(support_embeddings, fast_weights)
        query_logits = self.head.functional_forward(query_embeddings, fast_weights)

        query_loss = F.cross_entropy(query_logits, query_labels)
        query_preds = torch.argmax(query_logits, dim=-1)
        query_acc = compute_accuracy(query_preds, query_labels)

        return EpisodeLogits(
            support_logits=support_logits,
            query_logits=query_logits,
            support_loss=support_loss,
            query_loss=query_loss,
            query_preds=query_preds,
            query_acc=query_acc,
        )

    @torch.no_grad()
    def predict_from_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backup / debug path:
        compute class prototypes directly and classify by distance.
        Useful for sanity-checking episodic quality.
        """
        prototypes = compute_prototypes(support_embeddings, support_labels)
        dists = pairwise_squared_euclidean(query_embeddings, prototypes)
        return -dists


def compute_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    embeddings: [num_support, dim]
    labels: [num_support]
    returns: [num_classes, dim]
    """
    num_classes = int(labels.max().item()) + 1
    prototypes = []

    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_vectors = embeddings[class_mask]
        proto = class_vectors.mean(dim=0)
        prototypes.append(proto)

    return torch.stack(prototypes, dim=0)


def pairwise_squared_euclidean(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    x: [n, d]
    y: [m, d]
    returns: [n, m]
    """
    n = x.size(0)
    m = y.size(0)

    x_exp = x.unsqueeze(1).expand(n, m, -1)
    y_exp = y.unsqueeze(0).expand(n, m, -1)
    return torch.pow(x_exp - y_exp, 2).sum(dim=2)


def compute_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    if labels.numel() == 0:
        return 0.0
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return float(correct / max(1, total))


def build_maml_model(
    input_dim: int | None = None,
    hidden_dim: int | None = None,
    device: str | None = None,
) -> ImplicitMAMLModel:
    model = ImplicitMAMLModel(
        input_dim=input_dim or CONFIG.embedding_dim,
        hidden_dim=hidden_dim or CONFIG.classifier_hidden_dim,
    )
    model.to(device or CONFIG.device)
    return model