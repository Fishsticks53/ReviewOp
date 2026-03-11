# proto/backend/scripts/debug_episode_sampler.py
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path so we can import services
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.implicit.episode_sampler import (
    build_episode_sampler,
    describe_sampler,
    episode_rows_to_texts_and_labels,
)


def main() -> None:
    print(json.dumps(describe_sampler("train"), indent=2, ensure_ascii=False))

    sampler = build_episode_sampler("train")
    episode = sampler.sample_episode()

    print("\nSelected aspects:")
    print(episode.selected_aspects)

    print("\nSupport size:", len(episode.support_rows))
    print("Query size:", len(episode.query_rows))

    support_texts, support_labels = episode_rows_to_texts_and_labels(episode.support_rows)
    query_texts, query_labels = episode_rows_to_texts_and_labels(episode.query_rows)

    print("\nFirst 3 support samples:")
    for i, row in enumerate(episode.support_rows[:3], start=1):
        print(
            {
                "i": i,
                "aspect": row["implicit_aspect"],
                "local_label": row["local_label"],
                "text": row["evidence_sentence"],
            }
        )

    print("\nSupport texts:", len(support_texts), "Support labels:", len(support_labels))
    print("Query texts:", len(query_texts), "Query labels:", len(query_labels))


if __name__ == "__main__":
    main()