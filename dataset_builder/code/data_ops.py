from __future__ import annotations

import hashlib
import random
from typing import Dict, List


def _stable_group_key(record: Dict) -> str:
    text = str(record.get("clean_text", "")).strip().lower()
    if text:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]
    base = record.get("group_id") or record.get("review_id") or ""
    return hashlib.sha1(str(base).encode("utf-8")).hexdigest()[:14]


def assign_splits(records: List[Dict], preserve_official_splits: bool, ratios: Dict[str, float], seed: int = 42) -> List[Dict]:
    if preserve_official_splits and any(str(r.get("official_split", "")).lower() in {"train", "val", "test"} for r in records):
        for r in records:
            sp = str(r.get("official_split", "")).lower()
            r["split"] = sp if sp in {"train", "val", "test"} else "train"
        return records

    keys = sorted({_stable_group_key(r) for r in records})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * ratios.get("train", 0.8))
    n_val = int(n * ratios.get("val", 0.1))
    train = set(keys[:n_train])
    val = set(keys[n_train : n_train + n_val])

    for r in records:
        k = _stable_group_key(r)
        r["split"] = "train" if k in train else "val" if k in val else "test"
    return records


def split_map(records: List[Dict]) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    for r in records:
        s = r.get("split", "train")
        if s not in out:
            s = "train"
        out[s].append(r)
    return out


def leakage_report(split_rows: Dict[str, List[Dict]]) -> Dict:
    ids = {k: {r.get("review_id") for r in v} for k, v in split_rows.items()}
    overlaps = {}
    keys = ["train", "val", "test"]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = sorted(list(ids[a].intersection(ids[b])))
            overlaps[f"{a}_{b}"] = common[:100]
    text_hashes = {
        k: {hashlib.sha1(r.get("clean_text", "").lower().encode("utf-8")).hexdigest() for r in v}
        for k, v in split_rows.items()
    }
    hash_overlap = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            hash_overlap[f"{a}_{b}"] = len(text_hashes[a].intersection(text_hashes[b]))
    return {"id_overlap_samples": overlaps, "near_exact_text_overlap": hash_overlap}
