# proto/backend/scripts/debug_maml_model.py
from __future__ import annotations

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.implicit.config import CONFIG
from services.implicit.maml_model import build_maml_model


def main() -> None:
    model = build_maml_model()

    num_support = CONFIG.n_way * CONFIG.k_shot
    num_query = CONFIG.n_way * CONFIG.q_query

    support_embeddings = torch.randn(num_support, CONFIG.embedding_dim)
    query_embeddings = torch.randn(num_query, CONFIG.embedding_dim)

    support_labels = torch.tensor(
        [i for i in range(CONFIG.n_way) for _ in range(CONFIG.k_shot)],
        dtype=torch.long,
    )
    query_labels = torch.tensor(
        [i for i in range(CONFIG.n_way) for _ in range(CONFIG.q_query)],
        dtype=torch.long,
    )

    out = model.episode_forward(
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        create_graph=True,
    )

    print("Support logits:", tuple(out.support_logits.shape))
    print("Query logits:", tuple(out.query_logits.shape))
    print("Support loss:", float(out.support_loss.item()))
    print("Query loss:", float(out.query_loss.item()))
    print("Query acc:", out.query_acc)


if __name__ == "__main__":
    main()