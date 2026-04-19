from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class _Encoder:
    trainable = True


class _Model:
    def __init__(self) -> None:
        self.encoder = _Encoder()
        self.precomputed_embeddings = {"stale": torch.ones(2)}


class TrainerCacheTests(unittest.TestCase):
    def test_clear_embedding_cache_removes_stale_embeddings_for_trainable_encoder(self) -> None:
        from trainer import _clear_embedding_cache_if_trainable

        model = _Model()

        self.assertTrue(_clear_embedding_cache_if_trainable(model))
        self.assertEqual(model.precomputed_embeddings, {})

    def test_clear_embedding_cache_keeps_frozen_encoder_cache(self) -> None:
        from trainer import _clear_embedding_cache_if_trainable

        model = _Model()
        model.encoder.trainable = False

        self.assertFalse(_clear_embedding_cache_if_trainable(model))
        self.assertIn("stale", model.precomputed_embeddings)


if __name__ == "__main__":
    unittest.main()
