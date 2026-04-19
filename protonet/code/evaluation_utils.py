from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import importlib.util
import sys
from typing import Any, Dict, List
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

try:
    from .config import ProtonetConfig
    from .novelty_utils import compute_novelty_score
    from .quality_signals import prediction_error_buckets, top_aspect_confusions
    from .progress import task_bar
except ImportError:
    _config_path = Path(__file__).resolve().with_name("config.py")
    _config_spec = importlib.util.spec_from_file_location("protonet_local_config", _config_path)
    if _config_spec is None or _config_spec.loader is None:
        raise
    _config_module = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = _config_module
    _config_spec.loader.exec_module(_config_module)
    ProtonetConfig = _config_module.ProtonetConfig
    from novelty_utils import compute_novelty_score
    from quality_signals import prediction_error_buckets, top_aspect_confusions
    from progress import task_bar


def aspect_from_joint(label: str, separator: str) -> str:
    return label.split(separator, 1)[0]


def expected_calibration_error(confidences: List[float], correct: List[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (conf >= left) & (conf < right if i < bins - 1 else conf <= right)
        if not mask.any():
            continue
        bucket_conf = conf[mask].mean()
        bucket_acc = corr[mask].mean()
        ece += abs(bucket_acc - bucket_conf) * (mask.sum() / len(conf))
    return float(ece)


def diagnostic_summary(rows: List[Dict[str, Any]], separator: str) -> Dict[str, Any]:
    return {
        "error_buckets": prediction_error_buckets(rows),
        "top_aspect_confusions": top_aspect_confusions(rows, separator=separator, limit=5),
    }


def stable_cluster_id(*, domain: str, hint: str) -> str:
    basis = f"v2-novel-cluster|{str(domain).strip().lower()}|{str(hint).strip().lower()}"
    return f"novel_{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:12]}"


def decode_post_aspect_prediction(
    probabilities: np.ndarray,
    ordered_labels: List[str],
    *,
    separator: str,
    multi_label_margin: float,
) -> dict[str, Any]:
    aspect_best: dict[str, tuple[float, int]] = {}
    for idx, label in enumerate(ordered_labels):
        aspect = aspect_from_joint(label, separator)
        score = float(probabilities[idx])
        current = aspect_best.get(aspect)
        if current is None or score > current[0]:
            aspect_best[aspect] = (score, idx)
    if not aspect_best:
        return {"pred_label": "", "pred_labels": [], "confidence": 0.0, "selected_aspects": [], "aspect": "", "sentiment": "neutral"}

    ranked = sorted(aspect_best.items(), key=lambda item: item[1][0], reverse=True)
    top_score = ranked[0][1][0]
    selected = [item for item in ranked if item[1][0] >= top_score - float(multi_label_margin)]
    selected.sort(key=lambda item: item[1][0], reverse=True)
    pred_labels = [ordered_labels[idx] for _, (_, idx) in selected]
    best_aspect, (_, best_index) = ranked[0]
    best_label = ordered_labels[best_index]
    aspect = aspect_from_joint(best_label, separator)
    sentiment = best_label.split(separator, 1)[1] if separator in best_label else "neutral"
    return {
        "pred_label": best_label,
        "pred_labels": pred_labels,
        "confidence": float(ranked[0][1][0]),
        "selected_aspects": [aspect_name for aspect_name, _ in selected],
        "aspect": aspect or best_aspect,
        "sentiment": sentiment,
    }


def project_prediction_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "joint":
        return list(rows)
    if mode != "post_aspect":
        return list(rows)
    projected: List[Dict[str, Any]] = []
    for row in rows:
        pred_label = str(row.get("post_aspect_pred_label") or row.get("pred_label") or "")
        pred_labels = list(row.get("post_aspect_pred_labels") or ([pred_label] if pred_label else []))
        projected.append(
            {
                **row,
                "pred_label": pred_label,
                "pred_labels": pred_labels,
                "confidence": float(row.get("post_aspect_confidence", row.get("confidence", 0.0))),
                "correct": bool(row.get("post_aspect_correct", row.get("correct", False))),
                "flex_correct": bool(row.get("post_aspect_flex_correct", row.get("flex_correct", False))),
                "multi_label_overlap": float(row.get("post_aspect_multi_label_overlap", row.get("multi_label_overlap", 0.0))),
                "abstained": bool(row.get("post_aspect_abstained", row.get("abstained", False))),
                "low_confidence": bool(row.get("post_aspect_low_confidence", row.get("low_confidence", False))),
            }
        )
    return projected


def compact_mode_metrics(
    rows: List[Dict[str, Any]],
    cfg: ProtonetConfig,
    split_name: str,
    *,
    mode: str,
    elapsed: float,
    episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        return {
            "split": split_name,
            "num_episodes": len(episodes),
            "num_queries": 0,
            "accuracy": 0.0,
            "aspect_only_accuracy": 0.0,
            "macro_f1": 0.0,
            "flexible_match_score": 0.0,
            "multi_label_overlap_score": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "abstention_f1": 0.0,
            "coverage": 0.0,
            "risk": 0.0,
            "low_confidence_rate": 0.0,
            "selected_mode": mode,
        }
    y_true = [str(row.get("true_label") or "") for row in rows]
    y_pred = [str(row.get("pred_label") or "") for row in rows]
    y_true_aspect = [aspect_from_joint(label, cfg.joint_label_separator) for label in y_true]
    y_pred_aspect = [aspect_from_joint(label, cfg.joint_label_separator) for label in y_pred]
    correctness = [int(row.get("correct", False)) for row in rows]
    low_confidence_count = sum(1 for row in rows if bool(row.get("abstained", False)))
    abstain_true_positive = sum(1 for row in rows if bool(row.get("abstained", False)) and not bool(row.get("correct", False)))
    abstain_false_positive = sum(1 for row in rows if bool(row.get("abstained", False)) and bool(row.get("correct", False)))
    abstain_false_negative = sum(1 for row in rows if not bool(row.get("abstained", False)) and not bool(row.get("correct", False)))
    abstain_precision = abstain_true_positive / max(1, abstain_true_positive + abstain_false_positive)
    abstain_recall = abstain_true_positive / max(1, abstain_true_positive + abstain_false_negative)
    coverage = 1.0 - (low_confidence_count / max(1, len(rows)))
    risk = 1.0 - (sum(correctness) / max(1, len(rows)))
    return {
        "split": split_name,
        "num_episodes": len(episodes),
        "num_queries": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "aspect_only_accuracy": float(accuracy_score(y_true_aspect, y_pred_aspect)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0,
        "flexible_match_score": float(np.mean([1.0 if row.get("flex_correct") else 0.0 for row in rows])) if rows else 0.0,
        "multi_label_overlap_score": float(np.mean([float(row.get("multi_label_overlap", 0.0)) for row in rows])) if rows else 0.0,
        "abstention_precision": abstain_precision,
        "abstention_recall": abstain_recall,
        "abstention_f1": (2 * abstain_precision * abstain_recall / max(1e-12, abstain_precision + abstain_recall)),
        "coverage": float(coverage),
        "risk": float(risk),
        "low_confidence_rate": low_confidence_count / max(1, len(rows)),
        "calibration_ece": float(expected_calibration_error([float(row.get("confidence", 0.0)) for row in rows], correctness)),
        **diagnostic_summary(rows, cfg.joint_label_separator),
        "selected_mode": mode,
    }
