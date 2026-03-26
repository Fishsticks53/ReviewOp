from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from config import BuilderConfig
from explicit_features import build_explicit_row, fit_explicit_artifacts
from implicit_features import build_implicit_row, collect_implicit_diagnostics, discover_aspects, learn_aspect_seed_vocab
from io_utils import load_inputs
from schema_detect import detect_schema
from splitter import choose_stratify_values, preliminary_split, split_holdout
from utils import stable_id, utc_now_iso, write_json, write_jsonl
from mappings import GENERIC_REVIEW_ASPECT_SEEDS


def _clean_frame(frame: pd.DataFrame, cfg: BuilderConfig) -> tuple[pd.DataFrame, Dict[str, int]]:
    cleaned = frame.copy()
    log = {"rows_in": len(cleaned), "rows_out": len(cleaned), "duplicates_removed": 0, "short_text_dropped": 0}
    if cleaned.empty:
        return cleaned, log

    # Journal Worthiness: Aggressive source-level deduplication
    # We normalize to lowercase and remove non-alphanumeric chars for the check
    def _norm(txt):
        return re.sub(r'[^a-z0-9]', '', str(txt).lower())

    # Identify the text column
    from schema_detect import detect_schema
    temp_schema = detect_schema(cleaned, text_column_override=cfg.text_column_override)
    t_col = temp_schema.primary_text_column
    
    if t_col and t_col in cleaned.columns:
        initial_count = len(cleaned)
        cleaned['__norm_text__'] = cleaned[t_col].map(_norm)
        cleaned = cleaned.drop_duplicates(subset=['__norm_text__'])
        cleaned = cleaned.drop(columns=['__norm_text__'])
        log["duplicates_removed"] = initial_count - len(cleaned)

    cleaned = cleaned.reset_index(drop=True)
    log["rows_out"] = len(cleaned)
    return cleaned, log


def _drop_short_text_rows(frame: pd.DataFrame, text_column: str | None, min_tokens: int) -> tuple[pd.DataFrame, int]:
    if not text_column or text_column not in frame.columns:
        return frame, 0
    mask = frame[text_column].fillna("").astype(str).map(lambda value: len(value.split()) >= min_tokens)
    kept = frame.loc[mask].reset_index(drop=True)
    return kept, int((~mask).sum())


def _assign_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ids: List[str] = []
    for idx, row in out.iterrows():
        ids.append(stable_id(row.get("source_file", "source"), idx, row.to_json()))
    out["id"] = ids
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(list(value))
    from collections import Counter
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _reset_output_tree(cfg: BuilderConfig) -> None:
    for path in [cfg.explicit_dir, cfg.implicit_dir, cfg.reports_dir]:
        if path.exists():
            for child in path.iterdir():
                if child.is_dir():
                    import shutil
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
    cfg.ensure_dirs()


def _row_text(row: Dict[str, Any], text_column: str | None) -> str:
    return str(row.get(text_column, "")).strip() if text_column else ""


def run_pipeline(cfg: BuilderConfig) -> Dict[str, Any]:
    _reset_output_tree(cfg)
    frame = load_inputs(cfg.input_dir)
    if frame.empty:
        raise ValueError(f"No supported input files found under {cfg.input_dir}")

    schema = detect_schema(frame, text_column_override=cfg.text_column_override)
    cleaned, cleaning_log = _clean_frame(frame, cfg)
    cleaned = _assign_row_ids(cleaned)

    text_column = schema.primary_text_column
    if text_column and text_column not in cleaned.columns:
        text_column = None
    feature_numeric_columns = [column for column in schema.numeric_columns if column not in {schema.target_column, text_column}]
    feature_categorical_columns = [column for column in schema.categorical_columns if column not in {schema.target_column, text_column}]
    feature_datetime_columns = [column for column in schema.datetime_columns if column not in {schema.target_column, text_column}]
    cleaned, short_text_dropped = _drop_short_text_rows(cleaned, text_column, cfg.min_text_tokens)
    cleaning_log["short_text_dropped"] = short_text_dropped
    cleaning_log["rows_out"] = len(cleaned)

    stratify_key, stratify_values = choose_stratify_values(
        cleaned.to_dict(orient="records"),
        preferred_key=schema.target_column,
        fallback_key=text_column,
    )
    train_frame, holdout_frame = preliminary_split(
        cleaned,
        train_ratio=cfg.train_ratio,
        random_seed=cfg.random_seed,
        stratify_values=stratify_values,
    )

    train_rows = train_frame.to_dict(orient="records")
    learned_seed_info = learn_aspect_seed_vocab(train_rows, text_column=text_column or "", vocab_size=cfg.aspect_vocab_size) if text_column else {
        "learned_seed_vocab": [],
        "learned_seed_support": {},
        "learned_seed_total_docs": 0,
    }
    learned_seed_vocab = set(learned_seed_info["learned_seed_vocab"])
    fallback_seed_vocab = set(GENERIC_REVIEW_ASPECT_SEEDS)
    seed_vocab = learned_seed_vocab or fallback_seed_vocab
    candidate_aspects = discover_aspects(
        train_rows,
        text_column=text_column or "",
        vocab_size=cfg.aspect_vocab_size,
        seed_vocab=seed_vocab or fallback_seed_vocab,
    ) if text_column else []
    candidate_aspects = list(set(candidate_aspects or []) | set(fallback_seed_vocab))
    artifacts = fit_explicit_artifacts(train_frame, feature_numeric_columns, feature_categorical_columns)
    llm_settings = cfg.llm
    implicit_diagnostics = collect_implicit_diagnostics(
        train_rows,
        text_column=text_column or "",
        candidate_aspects=candidate_aspects,
        seed_vocab=seed_vocab or fallback_seed_vocab,
        confidence_threshold=cfg.confidence_threshold,
        learned_seed_vocab=learned_seed_info["learned_seed_vocab"],
    ) if text_column else {}

    explicit_train: List[Dict[str, Any]] = []
    explicit_holdout: List[Dict[str, Any]] = []
    implicit_train: List[Dict[str, Any]] = []
    implicit_holdout: List[Dict[str, Any]] = []

    for idx, row in enumerate(tqdm(train_rows, desc="train", leave=False)):
        explicit_train.append(build_explicit_row(row, artifacts=artifacts, numeric_columns=feature_numeric_columns, categorical_columns=feature_categorical_columns, datetime_columns=feature_datetime_columns, text_column=text_column))
        if text_column:
            # Only enable LLM fallback for the first N rows to save cost
            llm_enabled_for_row = cfg.enable_llm_fallback and (idx < cfg.llm_sample_size)
            imp_row = build_implicit_row(row, text_column=text_column, candidate_aspects=candidate_aspects, seed_vocab=seed_vocab or fallback_seed_vocab, confidence_threshold=cfg.confidence_threshold, llm_enabled=llm_enabled_for_row, llm_settings=llm_settings)
            implicit_train.append(imp_row)

    # Journal Worthiness Improvement: Promote novel aspects from LLM for subsequent passes
    from collections import Counter
    novel_candidates: Counter[str] = Counter()
    for row in implicit_train:
        for novel in row.get("implicit", {}).get("novel_aspects", []):
            novel_candidates[novel] += 1
    
    # Promote aspects found >= 3 times to the candidate list
    promotion_count = 0
    for aspect, count in novel_candidates.items():
        if count >= 3 and aspect not in candidate_aspects:
            candidate_aspects.append(aspect)
            promotion_count += 1
    if promotion_count > 0:
        tqdm.write(f"Promoted {promotion_count} novel aspects from LLM: {candidate_aspects[-promotion_count:]}")

    holdout_rows = holdout_frame.to_dict(orient="records")
    val_rows, test_rows = split_holdout(
        holdout_rows,
        val_ratio_within_holdout=cfg.val_ratio / max(cfg.val_ratio + cfg.test_ratio, 1e-9),
        random_seed=cfg.random_seed + 1,
        stratify_values=[str(row.get(stratify_key, "unknown")) for row in holdout_rows] if stratify_key else None,
    )

    val_ids = {str(row.get("id")) for row in val_rows}
    for idx, row in enumerate(tqdm(val_rows + test_rows, desc="holdout", leave=False)):
        split_name = "val" if str(row.get("id")) in val_ids else "test"
        explicit_row = build_explicit_row(row, artifacts=artifacts, numeric_columns=feature_numeric_columns, categorical_columns=feature_categorical_columns, datetime_columns=feature_datetime_columns, text_column=text_column)
        explicit_holdout.append({**explicit_row, "split": split_name})
        if text_column:
            llm_enabled_for_row = cfg.enable_llm_fallback and (idx < cfg.llm_sample_size)
            implicit_row = build_implicit_row(row, text_column=text_column, candidate_aspects=candidate_aspects, seed_vocab=seed_vocab or fallback_seed_vocab, confidence_threshold=cfg.confidence_threshold, llm_enabled=llm_enabled_for_row, llm_settings=llm_settings)
            implicit_holdout.append({**implicit_row, "split": split_name})

    explicit_by_split = {
        "train": [dict(row, split="train") for row in explicit_train],
        "val": [row for row in explicit_holdout if row["split"] == "val"],
        "test": [row for row in explicit_holdout if row["split"] == "test"],
    }
    implicit_by_split = {
        "train": [dict(row, split="train") for row in implicit_train],
        "val": [row for row in implicit_holdout if row["split"] == "val"],
        "test": [row for row in implicit_holdout if row["split"] == "test"],
    }

    # Phase 5: Hard Explicit-Implicit Separation (REMOVED)
    # This is now handled natively by clausal routing in build_implicit_row.

    for split, rows in explicit_by_split.items():
        tqdm.write(f"writing explicit/{split}.jsonl ({len(rows)} rows)")
        write_jsonl(cfg.explicit_dir / f"{split}.jsonl", rows)
    for split, rows in implicit_by_split.items():
        tqdm.write(f"writing implicit/{split}.jsonl ({len(rows)} rows)")
        write_jsonl(cfg.implicit_dir / f"{split}.jsonl", rows)

    build_report = {
        "pipeline_version": "3.0",
        "generated_at": utc_now_iso(),
        "config": _json_safe(asdict(cfg)),
        "schema": _json_safe(asdict(schema)),
        "cleaning_log": cleaning_log,
        "split_sizes": {
            "explicit": {split: len(rows) for split, rows in explicit_by_split.items()},
            "implicit": {split: len(rows) for split, rows in implicit_by_split.items()},
        },
        "discovered_aspects": candidate_aspects,
        "learned_seed_vocab": sorted(learned_seed_vocab),
        "learned_seed_support": dict(learned_seed_info["learned_seed_support"]),
        "seed_vocab_fallback": sorted(fallback_seed_vocab),
        "implicit_diagnostics": implicit_diagnostics,
        "text_column": text_column,
        "stratify_key": stratify_key,
    }
    write_json(cfg.reports_dir / "build_report.json", build_report)
    write_json(cfg.reports_dir / "data_quality_report.json", {
        "rows": len(cleaned),
        "columns": list(cleaned.columns),
        "text_column": text_column,
        "candidate_aspects": candidate_aspects,
    })

    return build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain-agnostic explicit/implicit dataset builder")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--aspect-vocab-size", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--enable-llm-fallback", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = BuilderConfig(
        input_dir=args.input_dir or BuilderConfig().input_dir,
        output_dir=args.output_dir or BuilderConfig().output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        aspect_vocab_size=args.aspect_vocab_size,
        confidence_threshold=args.confidence_threshold,
        enable_llm_fallback=args.enable_llm_fallback,
    )
    report = run_pipeline(cfg)
    print(f"Build complete: {report['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
