# proto/backend/services/implicit/maml_eval.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from services.implicit.config import CONFIG, ARTIFACT_DIR, ensure_implicit_dirs
from services.implicit.encoder import SentenceEncoder, build_encoder, encode_episode_rows
from services.implicit.episode_sampler import build_episode_sampler
from services.implicit.label_maps import load_label_encoder
from services.implicit.maml_model import ImplicitMAMLModel, build_maml_model
from services.implicit.maml_train import load_checkpoint, set_seed


@dataclass
class EvalMetrics:
    split: str
    num_episodes: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def _run_eval_episode(
    model: ImplicitMAMLModel,
    encoder: SentenceEncoder,
    episode,
    device: str,
) -> Dict[str, Any]:
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

    support_embeddings = support_embeddings.to(device)
    support_labels = support_labels.to(device)
    query_embeddings = query_embeddings.to(device)
    query_labels = query_labels.to(device)

    out = model.episode_forward(
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        inner_steps=CONFIG.inner_steps,
        inner_lr=CONFIG.inner_lr,
        create_graph=False,
    )

    return {
        "preds": out.query_preds.detach().cpu().tolist(),
        "labels": query_labels.detach().cpu().tolist(),
        "selected_aspects": list(episode.selected_aspects),
        "query_acc": float(out.query_acc),
        "query_loss": float(out.query_loss.item()),
    }


def _accumulate_per_aspect_stats(
    y_true_all: List[int],
    y_pred_all: List[int],
    aspect_id_to_name: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    labels_sorted = sorted(aspect_id_to_name.keys())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        labels=labels_sorted,
        average=None,
        zero_division=0,
    )

    report: Dict[str, Dict[str, float]] = {}
    for idx, aspect_id in enumerate(labels_sorted):
        aspect_name = aspect_id_to_name[aspect_id]
        report[aspect_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return report


def evaluate_implicit_maml(
    split: str = "test",
    device: str | None = None,
    freeze_backbone: bool = True,
    num_episodes: int | None = None,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    ensure_implicit_dirs()
    set_seed(CONFIG.random_seed)

    runtime_device = device or CONFIG.device
    label_maps = load_label_encoder(force_rebuild=True)

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

    model.eval()
    encoder.eval()

    sampler = build_episode_sampler(split, seed=CONFIG.random_seed + 5000)
    steps = num_episodes or CONFIG.eval_episodes

    episode_records: List[Dict[str, Any]] = []
    all_true_global: List[int] = []
    all_pred_global: List[int] = []
    total_loss = 0.0
    total_acc = 0.0

    for _ in range(steps):
        episode = sampler.sample_episode()
        result = _run_eval_episode(
            model=model,
            encoder=encoder,
            episode=episode,
            device=runtime_device,
        )

        local_to_global = {
            local_id: label_maps.aspect_to_id[aspect]
            for local_id, aspect in enumerate(episode.selected_aspects)
        }

        true_global = [local_to_global[int(x)] for x in result["labels"]]
        pred_global = [local_to_global[int(x)] for x in result["preds"]]

        all_true_global.extend(true_global)
        all_pred_global.extend(pred_global)
        total_loss += float(result["query_loss"])
        total_acc += float(result["query_acc"])

        episode_records.append(
            {
                "selected_aspects": result["selected_aspects"],
                "query_loss": result["query_loss"],
                "query_acc": result["query_acc"],
                "num_queries": len(result["labels"]),
            }
        )

    accuracy = float(accuracy_score(all_true_global, all_pred_global))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_true_global,
        all_pred_global,
        average="macro",
        zero_division=0,
    )

    metrics = EvalMetrics(
        split=split,
        num_episodes=steps,
        accuracy=accuracy,
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
    )

    per_aspect = _accumulate_per_aspect_stats(
        y_true_all=all_true_global,
        y_pred_all=all_pred_global,
        aspect_id_to_name=label_maps.id_to_aspect,
    )

    payload: Dict[str, Any] = {
        "summary": asdict(metrics),
        "avg_query_loss": float(total_loss / max(1, steps)),
        "avg_query_acc": float(total_acc / max(1, steps)),
        "per_aspect": per_aspect,
        "episode_records": episode_records,
    }

    target_path = output_path or (ARTIFACT_DIR / f"implicit_eval_{split}.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload