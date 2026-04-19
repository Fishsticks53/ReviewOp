from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class PipelineRunnerTests(unittest.TestCase):
    def test_run_pipeline_sync_awaits_async_pipeline_result(self) -> None:
        from pipeline_runner import run_pipeline_sync

        async def async_pipeline(cfg):
            await asyncio.sleep(0)
            return {"cfg": cfg, "status": "ok"}

        self.assertEqual(
            run_pipeline_sync("demo-cfg", pipeline=async_pipeline),
            {"cfg": "demo-cfg", "status": "ok"},
        )

    def test_run_pipeline_sync_accepts_sync_pipeline_result(self) -> None:
        from pipeline_runner import run_pipeline_sync

        def sync_pipeline(cfg):
            return {"cfg": cfg, "status": "ok"}

        self.assertEqual(
            run_pipeline_sync("demo-cfg", pipeline=sync_pipeline),
            {"cfg": "demo-cfg", "status": "ok"},
        )


if __name__ == "__main__":
    unittest.main()
