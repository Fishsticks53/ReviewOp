# proto/backend/scripts/debug_encoder.py
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.implicit.encoder import build_encoder, encode_texts


def main() -> None:
    encoder = build_encoder(freeze_backbone=True)

    texts = [
        "Calls keep showing busy even when the line is free.",
        "The battery drains before the day ends with light use.",
        "The food took too long to arrive at the table.",
    ]

    batch = encode_texts(encoder, texts)

    print("Num texts:", len(batch.texts))
    print("Embeddings shape:", tuple(batch.embeddings.shape))
    print("First vector slice:", batch.embeddings[0][:8].tolist())


if __name__ == "__main__":
    main()