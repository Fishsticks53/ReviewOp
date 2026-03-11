# proto/backend/services/implicit/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "implicit"
RAW_DIR = DATA_DIR / "raw"
ONTOLOGY_DIR = DATA_DIR / "ontology"
MODEL_DIR = BASE_DIR / "models" / "implicit"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
ARTIFACT_DIR = DATA_DIR / "artifacts"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class ImplicitConfig:
    # data
    review_train_path: Path = RAW_DIR / "implicit_reviewlevel_train.jsonl"
    review_val_path: Path = RAW_DIR / "implicit_reviewlevel_val.jsonl"
    review_test_path: Path = RAW_DIR / "implicit_reviewlevel_test.jsonl"

    episode_train_path: Path = RAW_DIR / "implicit_episode_train.jsonl"
    episode_val_path: Path = RAW_DIR / "implicit_episode_val.jsonl"
    episode_test_path: Path = RAW_DIR / "implicit_episode_test.jsonl"

    ontology_path: Path = ONTOLOGY_DIR / "implicit_aspect_ontology.json"
    label_encoder_path: Path = MODEL_DIR / "label_encoder.json"
    metrics_path: Path = MODEL_DIR / "training_metrics.json"
    best_ckpt_path: Path = CHECKPOINT_DIR / "implicit_maml_best.pt"

    # model
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 128
    embedding_dim: int = 384
    classifier_hidden_dim: int = 256
    dropout: float = 0.1

    # episodic training
    n_way: int = 3
    k_shot: int = 3
    q_query: int = 2
    episodes_per_epoch: int = 20
    eval_episodes: int = 10

    # optimization
    inner_lr: float = 1e-2
    meta_lr: float = 2e-4
    weight_decay: float = 1e-5
    inner_steps: int = 1
    num_epochs: int = 3
    batch_size_hint: int = 1  # episodic, usually 1 episode step at a time

    # inference
    implicit_score_threshold: float = 0.55
    max_predictions_per_review: int = 5

    # runtime
    random_seed: int = 42
    device: str = "cpu"


CONFIG = ImplicitConfig()


def ensure_implicit_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DIR,
        ONTOLOGY_DIR,
        MODEL_DIR,
        CHECKPOINT_DIR,
        ARTIFACT_DIR,
        PROCESSED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)