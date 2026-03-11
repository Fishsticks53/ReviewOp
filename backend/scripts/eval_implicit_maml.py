# proto/backend/scripts/eval_implicit_maml.py
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.implicit.maml_eval import evaluate_implicit_maml


def main() -> None:
    report = evaluate_implicit_maml(
        split="test",
        freeze_backbone=True,
    )

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Per-aspect entries: {len(report['per_aspect'])}")


if __name__ == "__main__":
    main()