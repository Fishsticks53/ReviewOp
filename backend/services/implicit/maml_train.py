# proto/backend/services/implicit/maml_train.py
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


import numpy as np
import torch
from torch.optim import AdamW

from services.implicit.config import CONFIG, ensure_implicit_dirs
from services.implicit.encoder import SentenceEncoder, build_encoder, encode_episode_rows
from services.implicit.episode_sampler import EpisodeBatch, build_episode_sampler
from services.implicit.maml_model import ImplicitMAMLModel, build_maml_model


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: ImplicitMAMLModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    path: Path | None = None,
) -> None:
    ckpt_path = path or CONFIG.best_ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(CONFIG),
        },
        ckpt_path,
    )


def save_metrics(metrics: List[EpochMetrics], path: Path | None = None) -> None:
    target_path = path or CONFIG.metrics_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [asdict(m) for m in metrics]
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_checkpoint(
    model: ImplicitMAMLModel,
    optimizer: torch.optim.Optimizer | None = None,
    path: Path | None = None,
    map_location: str | None = None,
) -> Dict[str, Any]:
    ckpt_path = path or CONFIG.best_ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=map_location or CONFIG.device)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state


def move_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device)


def run_episode_step(
    model: ImplicitMAMLModel,
    encoder: SentenceEncoder,
    episode: EpisodeBatch,
    device: str,
    training: bool,
) -> tuple[torch.Tensor, float]:
    support_embeddings, support_labels, _ = encode_episode_rows(
        encoder=encoder,
        rows=episode.support_rows,
        text_key="evidence_sentence",
        label_key="local_label",
    )
    query_embeddings, query_labels, _ = encode_episode_rows(
        encoder=encoder,
        rows=episode.query_rows,
        text_key="evidence_sentence",
        label_key="local_label",
    )

    support_embeddings = move_to_device(support_embeddings, device)
    query_embeddings = move_to_device(query_embeddings, device)
    support_labels = move_to_device(support_labels, device)
    query_labels = move_to_device(query_labels, device)

    out = model.episode_forward(
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        inner_steps=CONFIG.inner_steps,
        inner_lr=CONFIG.inner_lr,
        create_graph=training,
    )
    return out.query_loss, out.query_acc


def train_one_epoch(
    model: ImplicitMAMLModel,
    encoder: SentenceEncoder,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    encoder.train()

    sampler = build_episode_sampler("train", seed=CONFIG.random_seed + epoch)

    total_loss = 0.0
    total_acc = 0.0
    num_steps = CONFIG.episodes_per_epoch

    for _ in range(num_steps):
        episode = sampler.sample_episode()

        optimizer.zero_grad()
        loss, acc = run_episode_step(
            model=model,
            encoder=encoder,
            episode=episode,
            device=device,
            training=True,
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += acc

    avg_loss = total_loss / max(1, num_steps)
    avg_acc = total_acc / max(1, num_steps)
    return avg_loss, avg_acc


#@torch.no_grad()
def evaluate(
    model: ImplicitMAMLModel,
    encoder: SentenceEncoder,
    split: str,
    device: str,
    seed_offset: int = 0,
) -> tuple[float, float]:
    model.eval()
    encoder.eval()

    sampler = build_episode_sampler(split, seed=CONFIG.random_seed + seed_offset)

    total_loss = 0.0
    total_acc = 0.0
    num_steps = CONFIG.eval_episodes

    for _ in range(num_steps):
        episode = sampler.sample_episode()
        loss, acc = run_episode_step(
            model=model,
            encoder=encoder,
            episode=episode,
            device=device,
            training=False,
        )
        total_loss += float(loss.item())
        total_acc += acc

    avg_loss = total_loss / max(1, num_steps)
    avg_acc = total_acc / max(1, num_steps)
    return avg_loss, avg_acc


def build_optimizer(
    model: ImplicitMAMLModel,
    encoder: SentenceEncoder,
) -> torch.optim.Optimizer:
    params = list(model.parameters()) + [
        p for p in encoder.parameters() if p.requires_grad
    ]
    return AdamW(
        params,
        lr=CONFIG.meta_lr,
        weight_decay=CONFIG.weight_decay,
    )


def train_implicit_maml(
    device: str | None = None,
    freeze_backbone: bool = True,
) -> List[EpochMetrics]:
    ensure_implicit_dirs()
    set_seed(CONFIG.random_seed)

    runtime_device = device or CONFIG.device
    encoder = build_encoder(
        freeze_backbone=freeze_backbone,
        device=runtime_device,
    )
    model = build_maml_model(
        input_dim=encoder.hidden_size,
        hidden_dim=CONFIG.classifier_hidden_dim,
        device=runtime_device,
    )
    optimizer = build_optimizer(model=model, encoder=encoder)

    metrics: List[EpochMetrics] = []
    best_val_acc = -1.0

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            device=runtime_device,
            epoch=epoch,
        )
        val_loss, val_acc = evaluate(
            model=model,
            encoder=encoder,
            split="val",
            device=runtime_device,
            seed_offset=1000 + epoch,
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        metrics.append(epoch_metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
            )

        save_metrics(metrics)

        print(
            f"[epoch {epoch}/{CONFIG.num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return metrics