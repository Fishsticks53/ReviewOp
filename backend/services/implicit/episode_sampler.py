# proto/backend/services/implicit/episode_sampler.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from services.implicit.config import CONFIG
from services.implicit.dataset_reader import load_episode_split
from services.implicit.label_maps import LabelMaps, load_label_encoder


@dataclass
class EpisodeBatch:
    selected_aspects: List[str]
    aspect_to_local_id: Dict[str, int]
    support_rows: List[Dict[str, Any]]
    query_rows: List[Dict[str, Any]]

    def to_training_dict(self) -> Dict[str, Any]:
        return {
            "selected_aspects": self.selected_aspects,
            "aspect_to_local_id": self.aspect_to_local_id,
            "support_rows": self.support_rows,
            "query_rows": self.query_rows,
        }


class EpisodeSampler:
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        label_maps: LabelMaps | None = None,
        seed: int | None = None,
    ) -> None:
        self.rows = rows
        self.label_maps = label_maps or load_label_encoder(force_rebuild=True)
        self.rng = random.Random(seed if seed is not None else CONFIG.random_seed)

        self.rows_by_aspect: Dict[str, List[Dict[str, Any]]] = self._group_rows_by_aspect(rows)
        self.eligible_aspects: List[str] = sorted(
            [
                aspect
                for aspect, aspect_rows in self.rows_by_aspect.items()
                if len(aspect_rows) >= (CONFIG.k_shot + CONFIG.q_query)
            ]
        )

        if len(self.eligible_aspects) < CONFIG.n_way:
            raise ValueError(
                f"Not enough eligible aspects for {CONFIG.n_way}-way sampling. "
                f"Eligible={len(self.eligible_aspects)} aspects={self.eligible_aspects}"
            )

    @staticmethod
    def _group_rows_by_aspect(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            aspect = str(row.get("implicit_aspect", "")).strip()
            if not aspect:
                continue
            grouped.setdefault(aspect, []).append(row)
        return grouped

    def sample_episode(
        self,
        n_way: int | None = None,
        k_shot: int | None = None,
        q_query: int | None = None,
    ) -> EpisodeBatch:
        n_way = n_way or CONFIG.n_way
        k_shot = k_shot or CONFIG.k_shot
        q_query = q_query or CONFIG.q_query

        eligible_aspects = [
            aspect
            for aspect, rows in self.rows_by_aspect.items()
            if len(rows) >= (k_shot + q_query)
        ]
        eligible_aspects = sorted(eligible_aspects)

        if len(eligible_aspects) < n_way:
            raise ValueError(
                f"Cannot sample {n_way}-way episode. "
                f"Only {len(eligible_aspects)} eligible aspects available."
            )

        selected_aspects = self.rng.sample(eligible_aspects, n_way)
        aspect_to_local_id = {aspect: idx for idx, aspect in enumerate(selected_aspects)}

        support_rows: List[Dict[str, Any]] = []
        query_rows: List[Dict[str, Any]] = []

        for aspect in selected_aspects:
            aspect_rows = list(self.rows_by_aspect[aspect])
            sampled_rows = self.rng.sample(aspect_rows, k_shot + q_query)

            support_part = sampled_rows[:k_shot]
            query_part = sampled_rows[k_shot:]

            for row in support_part:
                support_rows.append(self._attach_labels(row, aspect_to_local_id[aspect]))

            for row in query_part:
                query_rows.append(self._attach_labels(row, aspect_to_local_id[aspect]))

        self.rng.shuffle(support_rows)
        self.rng.shuffle(query_rows)

        return EpisodeBatch(
            selected_aspects=selected_aspects,
            aspect_to_local_id=aspect_to_local_id,
            support_rows=support_rows,
            query_rows=query_rows,
        )

    def sample_many(
        self,
        num_episodes: int,
        n_way: int | None = None,
        k_shot: int | None = None,
        q_query: int | None = None,
    ) -> List[EpisodeBatch]:
        return [
            self.sample_episode(n_way=n_way, k_shot=k_shot, q_query=q_query)
            for _ in range(num_episodes)
        ]

    def _attach_labels(self, row: Dict[str, Any], local_label: int) -> Dict[str, Any]:
        aspect = str(row["implicit_aspect"]).strip()
        global_label = self.label_maps.aspect_to_id.get(aspect)

        payload = dict(row)
        payload["local_label"] = local_label
        payload["global_label"] = global_label
        return payload


def build_episode_sampler(
    split: str,
    seed: int | None = None,
) -> EpisodeSampler:
    rows = load_episode_split(split)
    return EpisodeSampler(
        rows=rows,
        label_maps=load_label_encoder(),
        seed=seed,
    )


def episode_rows_to_texts_and_labels(
    rows: Sequence[Dict[str, Any]],
    text_key: str = "evidence_sentence",
    label_key: str = "local_label",
) -> tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    for row in rows:
        text = str(row.get(text_key, "")).strip()
        if not text:
            continue

        label = row.get(label_key)
        if label is None:
            continue

        texts.append(text)
        labels.append(int(label))

    return texts, labels


def describe_sampler(split: str) -> Dict[str, Any]:
    rows = load_episode_split(split)
    sampler = EpisodeSampler(rows=rows, label_maps=load_label_encoder())

    aspect_counts = {
        aspect: len(aspect_rows)
        for aspect, aspect_rows in sorted(sampler.rows_by_aspect.items())
    }

    return {
        "split": split,
        "num_rows": len(rows),
        "num_total_aspects": len(sampler.rows_by_aspect),
        "num_eligible_aspects": len(sampler.eligible_aspects),
        "eligible_aspects": sampler.eligible_aspects,
        "aspect_counts": aspect_counts,
        "config": {
            "n_way": CONFIG.n_way,
            "k_shot": CONFIG.k_shot,
            "q_query": CONFIG.q_query,
        },
    }