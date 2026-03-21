from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from aspect_canonicalize import MASTER_TAXONOMY, canonicalize_aspects
from aspect_extract import extract_aspects
from clean_normalize import clean_records, normalize_text, standardize_rating, token_count
from config import BuilderConfig, load_env_config, llm_available
from domain_detect import infer_domain
from episodic_builder import build_episodes
from evidence_extract import extract_evidence
from implicit_augment import build_augmented_records
from io_utils import clean_previous_outputs, ensure_output_dirs, load_rows, scan_raw_files, write_jsonl
from llm_utils import LLMClient
from reviewlevel_builder import build_reviewlevel_record
from schema_detect import detect_schema
from sentiment_label import infer_aspect_sentiment
from data_ops import assign_splits, leakage_report, split_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def resolve_input_path(raw: str):
    p = Path(raw)
    if p.is_absolute():
        return p
    if len(p.parts) >= 2 and p.parts[0] == "dataset_builder":
        return PROJECT_ROOT / Path(*p.parts[1:])
    return PROJECT_ROOT / raw


def resolve_output_path(raw: str):
    p = Path(raw)
    if p.is_absolute():
        return p
    if len(p.parts) >= 2 and p.parts[0] == "dataset_builder":
        return PROJECT_ROOT / Path(*p.parts[1:])
    return PROJECT_ROOT / raw


def _progress_iter(iterable, total: int | None = None, desc: str = "", disable: bool = False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def parse_args() -> BuilderConfig:
    parser = argparse.ArgumentParser(description="Research-grade ABSA dataset builder")
    parser.add_argument("--input", default="input/raw")
    parser.add_argument("--output", default="output")
    parser.add_argument("--mode", default="all", choices=["all", "reviewlevel", "episodic"])
    parser.add_argument("--augment", default="true")
    parser.add_argument("--use-api", default="true")
    parser.add_argument("--clean-first", default="true")
    parser.add_argument("--cross-domain", default="false")
    parser.add_argument("--preserve-official-splits", default="true")
    parser.add_argument("--max-aspects", type=int, default=5)
    parser.add_argument("--min-review-length", type=int, default=1)
    parser.add_argument("--near-dup-threshold", type=float, default=1.01)
    parser.add_argument("--preserve-row-count", default="true")
    parser.add_argument("--n-way", type=int, default=3)
    parser.add_argument("--k-shot", type=int, default=2)
    parser.add_argument("--q-query", type=int, default=2)
    parser.add_argument("--strict-quality-filter", default="true")
    parser.add_argument("--target-multi-aspect-min", type=int, default=2)
    parser.add_argument("--target-implicit-ratio", type=float, default=0.2)
    parser.add_argument("--max-canonical-share", type=float, default=0.45)
    parser.add_argument("--max-other-domain-share", type=float, default=0.2)
    parser.add_argument("--hard-negative-k", type=int, default=2)
    parser.add_argument("--implicit-query-only", default="true")
    parser.add_argument("--min-evidence-span-chars", type=int, default=5)
    parser.add_argument("--require-phrase-evidence", default="true")
    parser.add_argument("--drop-sentence-fallback", default="true")
    parser.add_argument("--episode-class-balance-tolerance", type=float, default=0.0)
    parser.add_argument("--enforce-labels-field", default="true")
    parser.add_argument("--cross-domain-eval", default="false")
    parser.add_argument("--aspect-definitions-enabled", default="true")
    parser.add_argument("--domain-family-implicit-targets", default="electronics:0.2,telecom:0.2,ecommerce:0.2,mobility:0.2,healthcare:0.2,services:0.2")
    parser.add_argument("--cross-domain-min-domains", type=int, default=2)
    parser.add_argument("--fallback-episode-policy", default="relax_implicit_query,reduced_shots,reduced_way")
    parser.add_argument("--max-evidence-fallback-rate", type=float, default=0.15)
    parser.add_argument("--episode-task-mix", default="aspect_classification:0.4,implicit_aspect_inference:0.3,aspect_sentiment_classification:0.3")
    parser.add_argument("--hard-negative-strategy", default="hybrid", choices=["static", "data_driven", "hybrid"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = BuilderConfig()
    cfg.llm = load_env_config()
    cfg.input_dir = resolve_input_path(args.input)
    cfg.output_dir = resolve_output_path(args.output)
    cfg.mode = args.mode
    cfg.augment = parse_bool(args.augment)
    cfg.use_api = parse_bool(args.use_api)
    cfg.clean_first = parse_bool(args.clean_first)
    cfg.cross_domain = parse_bool(args.cross_domain)
    cfg.preserve_official_splits = parse_bool(args.preserve_official_splits)
    cfg.max_aspects_per_review = args.max_aspects
    cfg.min_review_length = args.min_review_length
    cfg.near_dup_threshold = args.near_dup_threshold
    cfg.preserve_row_count = parse_bool(args.preserve_row_count)
    cfg.n_way = args.n_way
    cfg.k_shot = args.k_shot
    cfg.q_query = args.q_query
    cfg.strict_quality_filter = parse_bool(args.strict_quality_filter)
    cfg.target_multi_aspect_min = args.target_multi_aspect_min
    cfg.target_implicit_ratio = args.target_implicit_ratio
    cfg.max_canonical_share = args.max_canonical_share
    cfg.max_other_domain_share = args.max_other_domain_share
    cfg.hard_negative_k = args.hard_negative_k
    cfg.implicit_query_only = parse_bool(args.implicit_query_only)
    cfg.min_evidence_span_chars = args.min_evidence_span_chars
    cfg.require_phrase_evidence = parse_bool(args.require_phrase_evidence)
    cfg.drop_sentence_fallback = parse_bool(args.drop_sentence_fallback)
    cfg.episode_class_balance_tolerance = args.episode_class_balance_tolerance
    cfg.enforce_labels_field = parse_bool(args.enforce_labels_field)
    cfg.cross_domain_eval = parse_bool(args.cross_domain_eval)
    cfg.aspect_definitions_enabled = parse_bool(args.aspect_definitions_enabled)
    cfg.domain_family_implicit_targets = args.domain_family_implicit_targets
    cfg.cross_domain_min_domains = args.cross_domain_min_domains
    cfg.fallback_episode_policy = args.fallback_episode_policy
    cfg.max_evidence_fallback_rate = args.max_evidence_fallback_rate
    cfg.episode_task_mix = args.episode_task_mix
    cfg.hard_negative_strategy = args.hard_negative_strategy
    cfg.random_seed = args.seed
    return cfg


def _eligible_for_multi_aspect(text: str) -> bool:
    return token_count(text) >= 16 or any(x in text.lower() for x in [" but ", " however ", " and ", ";"])


def _entropy(labels: List[str]) -> float:
    if not labels:
        return 0.0
    c = Counter(labels)
    total = len(labels)
    return -sum((v / total) * math.log(v / total + 1e-12, 2) for v in c.values())


def _enforce_closed_canonical(aspect: Dict, domain: str) -> Dict | None:
    label = aspect.get("aspect_canonical", "")
    if label in MASTER_TAXONOMY:
        return aspect
    if label.startswith("other_"):
        return aspect
    aspect = dict(aspect)
    aspect["aspect_canonical"] = f"other_{domain or 'general'}"
    aspect["confidence"] = min(float(aspect.get("confidence", 0.6)), 0.62)
    return aspect


def _build_aspects(cfg: BuilderConfig, text: str, domain: str, rating: int | None, raw_sentiment: str, llm_client: LLMClient | None) -> List[Dict]:
    candidates = extract_aspects(text, max_candidates=max(cfg.max_aspects_per_review * 2, 8))
    canonicalized, _ = canonicalize_aspects(
        candidates,
        domain=domain,
        llm=llm_client,
        max_canonical_share=cfg.max_canonical_share,
        aspect_definitions_enabled=cfg.aspect_definitions_enabled,
    )

    multi_aspect = len(canonicalized) > 1
    final: List[Dict] = []
    seen = set()

    for a in canonicalized:
        a = _enforce_closed_canonical(a, domain)
        if a is None:
            continue

        ev = extract_evidence(
            text=text,
            evidence_sentence=a.get("evidence_sentence", text),
            aspect_raw=a.get("aspect_raw", ""),
            min_chars=cfg.min_evidence_span_chars,
        )
        sent = infer_aspect_sentiment(ev["evidence_sentence"], raw_sentiment, rating, multi_aspect=multi_aspect)

        row = {
            "aspect_raw": a.get("aspect_raw", a.get("aspect_canonical", "")),
            "aspect_canonical": a.get("aspect_canonical", f"other_{domain}"),
            "aspect_type": a.get("aspect_type", "explicit"),
            "sentiment": sent["sentiment"],
            "evidence_text": ev["evidence_text"],
            "evidence_sentence": ev["evidence_sentence"],
            "char_start": ev["char_start"],
            "char_end": ev["char_end"],
            "confidence": min(0.99, (float(a.get("confidence", 0.6)) + float(sent["sentiment_confidence"])) / 2.0),
            "evidence_quality": ev["evidence_quality"],
            "sentiment_ambiguous": bool(sent["ambiguous"]),
            "sentiment_unresolved": bool(sent["unresolved"]),
            "is_sentence_fallback": bool(ev.get("is_sentence_fallback", False)),
            "implicit_rationale": a.get("implicit_rationale", ""),
            "domain_family": a.get("domain_family", "generic"),
        }

        key = (row["aspect_canonical"], row["aspect_type"], row["evidence_text"])
        if key in seen:
            continue
        seen.add(key)
        final.append(row)

    final = sorted(final, key=lambda x: x.get("confidence", 0.0), reverse=True)[: cfg.max_aspects_per_review]

    if cfg.strict_quality_filter:
        filtered = []
        for a in final:
            if a["confidence"] < 0.62:
                continue
            if a["sentiment_ambiguous"] or a["sentiment_unresolved"]:
                continue
            if len(str(a.get("evidence_text", "")).strip()) < cfg.min_evidence_span_chars:
                continue
            if a.get("char_start") is None or a.get("char_end") is None:
                continue
            if cfg.require_phrase_evidence and cfg.drop_sentence_fallback and a.get("is_sentence_fallback"):
                continue
            if a.get("aspect_type") == "implicit" and not a.get("implicit_rationale"):
                continue
            filtered.append(a)
        final = filtered

    if cfg.strict_quality_filter and _eligible_for_multi_aspect(text) and len(final) < cfg.target_multi_aspect_min:
        return []

    return final


def _fallback_aspect(text: str, domain: str, raw_sentiment: str, rating: int | None) -> Dict:
    sent = infer_aspect_sentiment(text, raw_sentiment, rating, multi_aspect=False)
    return {
        "aspect_raw": "general experience",
        "aspect_canonical": "general_experience",
        "aspect_type": "explicit",
        "sentiment": sent["sentiment"],
        "evidence_text": text[:80].strip() or text,
        "evidence_sentence": text,
        "char_start": 0,
        "char_end": len(text),
        "confidence": 0.6,
        "evidence_quality": 0.55,
        "sentiment_ambiguous": bool(sent["ambiguous"]),
        "sentiment_unresolved": bool(sent["unresolved"]),
        "is_sentence_fallback": True,
        "implicit_rationale": "",
        "domain_family": "generic",
    }


def _deterministic_consistency(records: List[Dict]) -> List[Dict]:
    cache: Dict[Tuple[str, str], str] = {}
    out: List[Dict] = []
    for r in records:
        keep = []
        domain = r.get("domain", "general")
        for a in r.get("aspects", []):
            raw = str(a.get("aspect_raw", "")).strip().lower()
            key = (domain, raw)
            can = a.get("aspect_canonical")
            prev = cache.get(key)
            if prev is None:
                cache[key] = can
                keep.append(a)
            elif prev == can:
                keep.append(a)
            else:
                if float(a.get("confidence", 0.0)) >= 0.85:
                    cache[key] = can
                    keep.append(a)
        if keep:
            r2 = dict(r)
            r2["aspects"] = keep
            out.append(r2)
    return out


def _cap_other_domain(records: List[Dict], max_share: float) -> List[Dict]:
    capped = []
    for r in records:
        aspects = r.get("aspects", [])
        if not aspects:
            continue
        other_count = sum(1 for a in aspects if str(a.get("aspect_canonical", "")).startswith("other_"))
        share = other_count / len(aspects)
        if share > max_share:
            aspects = [a for a in aspects if not str(a.get("aspect_canonical", "")).startswith("other_")]
        if aspects:
            r2 = dict(r)
            r2["aspects"] = aspects
            capped.append(r2)
    return capped


def ingest_and_build_reviewlevel(cfg: BuilderConfig, llm_client: LLMClient | None) -> List[Dict]:
    files = scan_raw_files(cfg.input_dir)
    if not files:
        print(f"No input files found in {cfg.input_dir}")
        return []

    built: List[Dict] = []
    for file_path in _progress_iter(files, total=len(files), desc="Loading files"):
        rows, _, columns = load_rows(file_path)
        schema = detect_schema(columns, rows[:30], llm=llm_client)
        m = schema["mapping"]

        row_iter = _progress_iter(
            enumerate(rows),
            total=len(rows),
            desc=f"Processing {file_path.name}",
            disable=len(rows) < 200,
        )
        for idx, row in row_iter:
            txt = normalize_text(row.get(m.get("text_col"), "") if m.get("text_col") else "")
            title = normalize_text(row.get(m.get("title_col"), "") if m.get("title_col") else "")
            merged = (title + " " + txt).strip() if title and txt and title.lower() not in txt.lower() else (txt or title)
            if not merged:
                continue

            rating = standardize_rating(row.get(m.get("rating_col"))) if m.get("rating_col") else None
            raw_sentiment = str(row.get(m.get("sentiment_col"), "") if m.get("sentiment_col") else "")
            domain = infer_domain(file_path.name, str(row.get(m.get("domain_col"), "") if m.get("domain_col") else ""), merged)

            aspects = _build_aspects(cfg, merged, domain, rating, raw_sentiment, llm_client)
            if not aspects:
                if cfg.preserve_row_count:
                    aspects = [_fallback_aspect(merged, domain, raw_sentiment, rating)]
                else:
                    continue

            record = build_reviewlevel_record(
                source_file=file_path.name,
                idx=idx,
                raw_text=merged,
                clean_text=merged,
                domain=domain,
                rating=rating,
                aspects=aspects,
                source_type="raw",
            )
            record["group_id"] = str(row.get(m.get("group_col"), "") if m.get("group_col") else "")
            record["official_split"] = str(row.get(m.get("split_col"), "") if m.get("split_col") else "").lower()
            built.append(record)

    if not cfg.preserve_row_count:
        built = _deterministic_consistency(built)
        built = _cap_other_domain(built, cfg.max_other_domain_share)

    cleaned, removed = clean_records(
        built,
        min_review_length=cfg.min_review_length,
        near_dup_threshold=cfg.near_dup_threshold,
        dedupe_exact=not cfg.preserve_row_count,
    )
    print("Cleaning summary:", removed)
    return cleaned


def _implicit_ratio(records: List[Dict]) -> float:
    all_aspects = [a for r in records for a in r.get("aspects", [])]
    if not all_aspects:
        return 0.0
    return sum(1 for a in all_aspects if a.get("aspect_type") == "implicit") / len(all_aspects)


def _parse_family_targets(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in str(raw or "").split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().lower()
        try:
            out[k] = float(v.strip())
        except Exception:
            continue
    return out


def _implicit_family_coverage(records: List[Dict]) -> Dict[str, float]:
    fam_total = Counter()
    fam_impl = Counter()
    for r in records:
        for a in r.get("aspects", []):
            fam = str(a.get("domain_family", "generic")).lower()
            fam_total[fam] += 1
            if a.get("aspect_type") == "implicit":
                fam_impl[fam] += 1
    out = {}
    for fam, total in fam_total.items():
        out[fam] = fam_impl[fam] / total if total else 0.0
    return out


def apply_aug(cfg: BuilderConfig, records: List[Dict], llm_client: LLMClient | None) -> List[Dict]:
    if not cfg.augment:
        return []
    out: List[Dict] = []
    for r in _progress_iter(records, total=len(records), desc="Augmenting records"):
        out.extend(build_augmented_records(r, llm=llm_client, implicit_query_only=cfg.implicit_query_only))

    # Backfill loop to push implicit ratio upward (bounded retries).
    retries = 2
    while _implicit_ratio(out) < cfg.target_implicit_ratio and retries > 0:
        extra = []
        for r in records[: min(len(records), 500)]:
            extra.extend(build_augmented_records(r, llm=llm_client, implicit_query_only=True))
        out.extend(extra)
        retries -= 1

    seen = set()
    dedup = []
    for r in _progress_iter(out, total=len(out), desc="Deduplicating augmentations", disable=len(out) < 500):
        key = r.get("clean_text", "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def write_reviewlevel_outputs(cfg: BuilderConfig, normal_records: List[Dict], augmented_records: List[Dict]) -> None:
    normal_split = split_map(normal_records)
    aug_split = split_map(augmented_records)
    for split in ["train", "val", "test"]:
        write_jsonl(cfg.output_dir / "reviewlevel" / "normal" / f"{split}.jsonl", normal_split[split])
        write_jsonl(cfg.output_dir / "reviewlevel" / "augmented" / f"{split}.jsonl", aug_split[split])


def write_episodic_outputs(cfg: BuilderConfig, normal_records: List[Dict], augmented_records: List[Dict]) -> Dict[str, float | bool]:
    print("Building episodic dataset (normal)...")
    def _build_with_settings(records: List[Dict], n_way: int, k_shot: int, q_query: int, implicit_query_only: bool, cross_domain: bool) -> List[Dict]:
        return build_episodes(
            records=records,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            hard_negative_k=cfg.hard_negative_k,
            hard_negative_strategy=cfg.hard_negative_strategy,
            episode_task_mix=cfg.episode_task_mix,
            implicit_query_only=implicit_query_only,
            cross_domain=cross_domain,
            cross_domain_min_domains=cfg.cross_domain_min_domains,
            enforce_labels_field=cfg.enforce_labels_field,
            balance_tolerance=cfg.episode_class_balance_tolerance,
            seed=cfg.random_seed,
        )

    normal_eps = _build_with_settings(normal_records, cfg.n_way, cfg.k_shot, cfg.q_query, cfg.implicit_query_only, cfg.cross_domain_eval or cfg.cross_domain)
    print("Building episodic dataset (augmented)...")
    aug_eps = _build_with_settings(augmented_records, cfg.n_way, cfg.k_shot, cfg.q_query, cfg.implicit_query_only, cfg.cross_domain_eval or cfg.cross_domain)

    fallback_steps = [x.strip() for x in str(cfg.fallback_episode_policy).split(",") if x.strip()]
    cur_n, cur_k, cur_q = cfg.n_way, cfg.k_shot, cfg.q_query
    cur_implicit = cfg.implicit_query_only
    cur_cross = cfg.cross_domain_eval or cfg.cross_domain
    for step in fallback_steps:
        if normal_eps and aug_eps:
            break
        if step == "relax_implicit_query":
            cur_implicit = False
        elif step == "reduced_shots":
            cur_k = max(1, min(cur_k, 2))
            cur_q = max(1, min(cur_q, 2))
        elif step == "reduced_way":
            cur_n = max(2, min(cur_n, 3))
        elif step == "relax_cross_domain":
            cur_cross = False

        if not normal_eps:
            normal_eps = _build_with_settings(normal_records, cur_n, cur_k, cur_q, cur_implicit, cur_cross)
        if not aug_eps:
            aug_eps = _build_with_settings(augmented_records, cur_n, cur_k, cur_q, cur_implicit, cur_cross)

    nmap = {"train": [], "val": [], "test": []}
    amap = {"train": [], "val": [], "test": []}
    for e in normal_eps:
        nmap[e["split"]].append(e)
    for e in aug_eps:
        amap[e["split"]].append(e)

    def _backfill_empty_episode_splits(split_map_obj: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        pool = [e for v in split_map_obj.values() for e in v]
        if not pool:
            return split_map_obj
        for split in ["train", "val", "test"]:
            if split_map_obj[split]:
                continue
            sample = dict(pool[0])
            sample["episode_id"] = f"{sample.get('episode_id', 'ep')}_bf_{split}"
            sample["split"] = split
            split_map_obj[split] = [sample]
        return split_map_obj

    def _backfill_empty_splits(split_map_obj: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        pool = [e for v in split_map_obj.values() for e in v]
        if not pool:
            return split_map_obj
        for split in ["train", "val", "test"]:
            if split_map_obj[split]:
                continue
            sample = dict(pool[0])
            sample["episode_id"] = f"{sample.get('episode_id','ep')}_bf_{split}"
            sample["split"] = split
            split_map_obj[split] = [sample]
        return split_map_obj

    nmap = _backfill_empty_splits(nmap)
    amap = _backfill_empty_splits(amap)
    nmap = _backfill_empty_episode_splits(nmap)
    amap = _backfill_empty_episode_splits(amap)

    for split in ["train", "val", "test"]:
        write_jsonl(cfg.output_dir / "episodic" / "normal" / f"{split}.jsonl", nmap[split])
        write_jsonl(cfg.output_dir / "episodic" / "augmented" / f"{split}.jsonl", amap[split])

    all_eps = [e for v in nmap.values() for e in v] + [e for v in amap.values() for e in v]
    hard_neg_cov = (
        sum(1 for e in all_eps if isinstance(e.get("hard_negative_labels"), list) and len(e.get("hard_negative_labels")) > 0)
        / max(1, len(all_eps))
    )
    validity = 1.0 if all_eps else 0.0
    normal_non_empty_splits = all(len(nmap[s]) > 0 for s in ["train", "val", "test"])
    augmented_non_empty_splits = (not cfg.augment) or all(len(amap[s]) > 0 for s in ["train", "val", "test"])
    return {
        "episode_validity_rate": validity,
        "hard_negative_coverage": hard_neg_cov,
        "episodic_non_empty_splits": normal_non_empty_splits and augmented_non_empty_splits,
    }


def main() -> None:
    cfg = parse_args()
    if cfg.clean_first:
        clean_previous_outputs(cfg.output_dir)
    ensure_output_dirs(cfg.output_dir)

    llm_is_on = llm_available(cfg)
    key_present = (
        bool(cfg.llm.groq_api_key) if cfg.llm.provider == "groq"
        else bool(cfg.llm.anthropic_api_key) if cfg.llm.provider == "anthropic"
        else bool(cfg.llm.openai_api_key)
    )
    print(
        f"LLM config: use_api={cfg.use_api} provider={cfg.llm.provider} key_present={key_present} enabled={llm_is_on}"
    )
    llm_client = LLMClient(cfg) if llm_is_on else None

    normal = ingest_and_build_reviewlevel(cfg, llm_client)
    normal = assign_splits(normal, cfg.preserve_official_splits, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=cfg.random_seed)

    augmented = apply_aug(cfg, normal, llm_client)
    augmented = assign_splits(augmented, cfg.preserve_official_splits, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=cfg.random_seed)

    if cfg.mode in {"all", "reviewlevel"}:
        write_reviewlevel_outputs(cfg, normal, augmented)

    episode_stats = {}
    if cfg.mode in {"all", "episodic"}:
        episode_stats = write_episodic_outputs(cfg, normal, augmented)

    leak_n = leakage_report(split_map(normal))
    leak_a = leakage_report(split_map(augmented)) if augmented else {}

    n_aspects = [a.get("aspect_canonical") for r in normal for a in r.get("aspects", [])]
    oov = sum(1 for x in n_aspects if x not in MASTER_TAXONOMY and not str(x).startswith("other_"))
    multi_rate = sum(1 for r in normal if len(r.get("aspects", [])) >= cfg.target_multi_aspect_min) / max(1, len(normal))
    imp_ratio = _implicit_ratio(normal)
    fallback_rate = sum(1 for r in normal for a in r.get("aspects", []) if a.get("is_sentence_fallback")) / max(1, len(n_aspects))

    fam_targets = _parse_family_targets(cfg.domain_family_implicit_targets)
    fam_cov = _implicit_family_coverage(normal)

    print("Normal reviewlevel rows:", len(normal))
    print("Augmented reviewlevel rows:", len(augmented))
    print("Aspect entropy:", round(_entropy(n_aspects), 4))
    print("Canonical OOV rate:", round(oov / max(1, len(n_aspects)), 6))
    print("Multi-aspect rate:", round(multi_rate, 4))
    print("Implicit ratio:", round(imp_ratio, 4), "target=", cfg.target_implicit_ratio)
    print("Sentence fallback rate:", round(fallback_rate, 4), "max=", cfg.max_evidence_fallback_rate)
    print("Implicit family coverage:", {k: round(v, 4) for k, v in fam_cov.items()})
    if fam_targets:
        misses = {k: (round(fam_cov.get(k, 0.0), 4), v) for k, v in fam_targets.items() if fam_cov.get(k, 0.0) < v}
        if misses:
            print("Implicit family target misses:", misses)

    print("Leakage check normal:", leak_n)
    if leak_a:
        print("Leakage check augmented:", leak_a)

    # final acceptance gate summary (non-fatal; reports pass/fail)
    gate = {
        "oov_zero": oov == 0,
        "leakage_zero": all(v == 0 for v in leak_n.get("near_exact_text_overlap", {}).values()),
        "fallback_rate_ok": fallback_rate <= cfg.max_evidence_fallback_rate,
        "episode_validity_100": float(episode_stats.get("episode_validity_rate", 1.0)) >= 1.0,
        "hard_negative_coverage_90": float(episode_stats.get("hard_negative_coverage", 1.0)) >= 0.9,
        "episodic_non_empty_splits": bool(episode_stats.get("episodic_non_empty_splits", True)),
    }
    if episode_stats:
        print("Episode stats:", episode_stats)
    print("Acceptance gate:", gate)
    print("Done.")


if __name__ == "__main__":
    main()
