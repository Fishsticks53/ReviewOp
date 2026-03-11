# proto/backend/scripts/train_implicit_maml.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.implicit.maml_train import train_implicit_maml


def main() -> None:
    metrics = train_implicit_maml(
        freeze_backbone=True,
    )
    print(f"Training complete. Epochs: {len(metrics)}")


if __name__ == "__main__":
    main()