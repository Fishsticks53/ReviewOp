from __future__ import annotations

import sys
import unittest
from pathlib import Path
import uuid

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class RuntimeInferBundleLoadTests(unittest.TestCase):
    def test_runtime_load_moves_modules_to_cfg_device_and_parses_path_fields(self) -> None:
        from runtime_infer import ProtonetRuntime
        from projection_head import ProjectionHead

        tmp_dir = Path(__file__).resolve().parent / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = tmp_dir / f"bundle_{uuid.uuid4().hex}.pt"

        # Small BoW bundle so we don't depend on a transformer cache for this unit test.
        bow_dim = 16
        proj_dim = 8
        projection = ProjectionHead(bow_dim, proj_dim, dropout=0.0)

        novelty_path = Path(__file__).resolve().parents[1] / "metadata" / "novelty_calibration_v2.json"

        payload = {
            "config": {
                "encoder_backend": "bow",
                "bow_dim": bow_dim,
                "dropout": 0.0,
                "novelty_calibration_path": str(novelty_path),
            },
            "encoder": {"backend": "bow", "hidden_size": bow_dim},
            "projection_state_dict": projection.state_dict(),
            "prototype_bank": {
                "labels": ["aspect__pos", "aspect__neg"],
                "prototypes": torch.randn(2, proj_dim, dtype=torch.float32),
                "counts": {"aspect__pos": 1, "aspect__neg": 1},
                "mean_confidence": {"aspect__pos": 1.0, "aspect__neg": 1.0},
            },
            "temperature": 1.0,
        }

        try:
            torch.save(payload, bundle_path)
            runtime = ProtonetRuntime.load(bundle_path)

            self.assertIsInstance(runtime.cfg.novelty_calibration_path, Path)
            self.assertEqual(Path(runtime.cfg.novelty_calibration_path), novelty_path)

            expected_device = runtime.cfg.device
            proj_param_device = next(runtime.projection.parameters()).device
            self.assertEqual(proj_param_device.type, expected_device.type)
            self.assertEqual(runtime.prototypes.device.type, expected_device.type)

            # Smoke: should not throw device-mismatch errors.
            rows = runtime.score_text("battery life is great", "battery life", domain="electronics")
            self.assertTrue(isinstance(rows, list))
            self.assertTrue(rows)
        finally:
            try:
                bundle_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()

