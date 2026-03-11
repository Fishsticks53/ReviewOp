# proto/backend/scripts/run_implicit_demo.py
from __future__ import annotations

import json

from services.implicit.implicit_infer import build_inference_service


def main() -> None:
    service = build_inference_service(
        freeze_backbone=True,
    )

    review = (
        "I used it for a week. Calls keep showing busy even when the other person "
        "says the line is free. By evening the battery is already nearly dead."
    )

    result = service.infer_review(review)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()