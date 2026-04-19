from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

try:
    from .contracts import BuilderConfig
    from .experiment_policy import (
        QUALITY_GATES,
        build_sweep_summary,
        candidate_grid,
        meets_quality_gates,
        metrics_from_report,
        run_ablation_matrix,
    )
    from .pipeline_runner import run_pipeline_sync
    from .runtime_options import load_runtime_defaults, optional_env, resolve_artifact_mode, resolve_domain_conditioning_modes
    from .experiments import run_experiments
    from .research_stack import build_experiment_plan, benchmark_registry_payload, model_registry_payload
    from .utils import stable_id, utc_now_iso, write_json, compress_output_folder
except ImportError:  # pragma: no cover
    from contracts import BuilderConfig
    from experiment_policy import (
        QUALITY_GATES,
        build_sweep_summary,
        candidate_grid,
        meets_quality_gates,
        metrics_from_report,
        run_ablation_matrix,
    )
    from pipeline_runner import run_pipeline_sync
    from runtime_options import load_runtime_defaults, optional_env, resolve_artifact_mode, resolve_domain_conditioning_modes
    from experiments import run_experiments
    from research_stack import build_experiment_plan, benchmark_registry_payload, model_registry_payload
    from utils import stable_id, utc_now_iso, write_json, compress_output_folder

try:
    from .llm_utils import flush_llm_cache
except Exception:  # pragma: no cover
    try:
        from llm_utils import flush_llm_cache
    except Exception:  # pragma: no cover
        def flush_llm_cache() -> None:
            return None

def _parse_bounded_int_list(raw: str, *, minimum: int, maximum: int, fallback: list[int]) -> list[int]:
    values: list[int] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            continue
        if minimum <= value <= maximum:
            values.append(value)
    unique_values = sorted(set(values))
    return unique_values or fallback


def _execute_v4_sweep(
    cfg: BuilderConfig,
    run_dir: Path,
    *,
    include_coref: bool,
    implicit_min_tokens_values: list[int],
    min_text_tokens_values: list[int],
    quality_gates: dict[str, Any] = QUALITY_GATES,
) -> dict[str, Any]:
    candidates = candidate_grid(
        include_coref=include_coref,
        implicit_min_tokens_values=implicit_min_tokens_values,
        min_text_tokens_values=min_text_tokens_values,
    )
    candidate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_output_dir = run_dir / "candidates" / candidate["candidate_id"]
        candidate_cfg = replace(
            cfg,
            output_dir=candidate_output_dir,
            implicit_mode=candidate["implicit_mode"],
            confidence_threshold=candidate["confidence_threshold"],
            llm_fallback_threshold=candidate["llm_fallback_threshold"],
            use_coref=candidate["use_coref"],
            implicit_min_tokens=candidate["implicit_min_tokens"],
            min_text_tokens=candidate["min_text_tokens"],
            dry_run=False,
            preview_only=False,
        )
        report = run_pipeline_sync(candidate_cfg)
        metrics = metrics_from_report(report)
        candidate_results.append({
            **candidate,
            "output_dir": str(candidate_output_dir),
            "report_path": str(candidate_output_dir / "reports" / "build_report.json"),
            "metrics": metrics,
            "meets_quality_gates": meets_quality_gates(metrics, quality_gates=quality_gates),
            "generated_at": report.get("generated_at"),
            "validation": report.get("validation", {}),
        })

    summary = build_sweep_summary(
        cfg=cfg,
        run_dir=run_dir,
        candidate_results=candidate_results,
        include_coref=include_coref,
        ablation_summary=run_ablation_matrix(cfg, run_dir),
        quality_gates=quality_gates,
    )
    write_json(run_dir / "v4_sweep_results.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    runtime_defaults = load_runtime_defaults()
    parser = argparse.ArgumentParser(description="Dataset builder research experiment runner")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--run-profile", type=str, default="research", choices=["research", "debug"])
    parser.add_argument("--artifact-mode", type=str, default="auto", choices=["auto", "debug_artifacts", "research_release"])
    parser.add_argument("--confidence-threshold", type=float, default=float(runtime_defaults.get("confidence_threshold", 0.6)))
    parser.add_argument("--max-aspects", type=int, default=20)
    parser.add_argument("--min-text-tokens", type=int, default=4)
    parser.add_argument("--implicit-min-tokens", type=int, default=8)
    parser.add_argument("--implicit-mode", type=str, default=str(runtime_defaults.get("implicit_mode", "zeroshot")), choices=["zeroshot", "supervised", "hybrid", "heuristic", "benchmark"])
    parser.add_argument("--multilingual-mode", type=str, default="shared_vocab")
    parser.add_argument("--use-coref", dest="use_coref", action="store_true")
    parser.add_argument("--no-use-coref", dest="use_coref", action="store_false")
    parser.set_defaults(use_coref=bool(runtime_defaults.get("use_coref", False)))
    parser.add_argument("--language-detection-mode", type=str, default="heuristic")
    parser.add_argument("--no-drop", action="store_true")
    parser.add_argument("--enable-llm-fallback", dest="enable_llm_fallback", action="store_true")
    parser.add_argument("--no-enable-llm-fallback", dest="enable_llm_fallback", action="store_false")
    parser.set_defaults(enable_llm_fallback=bool(runtime_defaults.get("enable_llm_fallback", True)))
    parser.add_argument("--llm-fallback-threshold", type=float, default=float(runtime_defaults.get("llm_fallback_threshold", 0.65)))
    parser.add_argument("--benchmark-key", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="heuristic_latent")
    parser.add_argument("--augmentation-mode", type=str, default="none")
    parser.add_argument("--prompt-mode", type=str, default="constrained")
    parser.add_argument("--gold-annotations-path", type=Path, default=None)
    parser.add_argument("--evaluation-protocol", type=str, default="random", choices=["random", "loo", "source-free"])
    parser.add_argument("--domain-holdout", type=str, default=None)
    parser.add_argument("--no-enforce-grounding", dest="enforce_grounding", action="store_false")
    parser.add_argument("--no-domain-conditioning", dest="use_domain_conditioning", action="store_false")
    parser.add_argument("--no-strict-domain-conditioning", dest="strict_domain_conditioning", action="store_false")
    parser.add_argument("--domain-conditioning-mode", type=str, default="adaptive_soft", choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--train-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.add_argument("--eval-domain-conditioning-mode", type=str, default=None, choices=["adaptive_soft", "strict_hard", "off"])
    parser.set_defaults(enforce_grounding=True, use_domain_conditioning=True, strict_domain_conditioning=False)
    parser.add_argument("--domain-prior-boost", type=float, default=0.05)
    parser.add_argument("--domain-prior-penalty", type=float, default=0.08)
    parser.add_argument("--weak-domain-support-row-threshold", type=int, default=80)
    parser.add_argument("--unseen-non-general-coverage-min", type=float, default=0.55)
    parser.add_argument("--unseen-implicit-not-ready-rate-max", type=float, default=0.35)
    parser.add_argument("--unseen-domain-leakage-row-rate-max", type=float, default=0.02)
    parser.add_argument("--train-fallback-general-policy", type=str, default="cap", choices=["keep", "cap", "drop"])
    parser.add_argument("--enable-reasoned-recovery", dest="enable_reasoned_recovery", action="store_true")
    parser.add_argument("--no-enable-reasoned-recovery", dest="enable_reasoned_recovery", action="store_false")
    # Backward-compatible alias for older scripts.
    parser.add_argument("--no-enable-reasoned_recovery", dest="enable_reasoned_recovery", action="store_false")
    parser.set_defaults(enable_reasoned_recovery=bool(runtime_defaults.get("enable_reasoned_recovery", True)))
    parser.add_argument("--processor", type=str, default=optional_env("DATASET_BUILDER_PROCESSOR", default="local"), choices=["local", "runpod"])
    parser.add_argument(
        "--llm-provider",
        "--llm-option",
        dest="llm_provider",
        type=str,
        default=None,
        choices=["auto", "none", "openai", "runpod", "ollama", "mock", "claude", "anthropic"],
        help="LLM provider for fallback/reasoned recovery. Defaults to runpod when --processor runpod is used; otherwise disabled. --llm-option is a backward-compatible alias.",
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=optional_env("LLM_MODEL_NAME", "GROQ_MODEL", "OPENAI_MODEL", "OLLAMA_MODEL"),
    )
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--train-fallback-general-cap-ratio", type=float, default=0.15)
    parser.add_argument("--train-review-filter-mode", type=str, default="reasoned_strict", choices=["keep", "drop_needs_review", "reasoned_strict"])
    parser.add_argument("--train-salvage-mode", type=str, default="recover_non_general", choices=["off", "recover_non_general"])
    parser.add_argument("--train-salvage-confidence-threshold", type=float, default=0.56)
    parser.add_argument("--train-salvage-accepted-support-types", type=str, default="exact,near_exact,gold")
    parser.add_argument(
        "--train-sentiment-balance-mode",
        type=str,
        default="cap_neutral_with_dual_floor",
        choices=["none", "cap_neutral", "cap_neutral_with_negative_floor", "cap_neutral_with_dual_floor"],
    )
    parser.add_argument("--train-neutral-cap-ratio", type=float, default=0.5)
    parser.add_argument("--train-min-negative-ratio", type=float, default=0.12)
    parser.add_argument("--train-min-positive-ratio", type=float, default=0.12)
    parser.add_argument("--train-max-positive-ratio", type=float, default=0.5)
    parser.add_argument("--train-neutral-max-ratio", type=float, default=0.58)
    parser.add_argument("--train-topup-recovery-mode", type=str, default="strict_topup", choices=["off", "strict_topup"])
    parser.add_argument("--train-topup-confidence-threshold", type=float, default=0.58)
    parser.add_argument("--train-topup-staged-recovery", dest="train_topup_staged_recovery", action="store_true")
    parser.add_argument("--no-train-topup-staged-recovery", dest="train_topup_staged_recovery", action="store_false")
    parser.set_defaults(train_topup_staged_recovery=True)
    parser.add_argument("--train-topup-stage-b-confidence-threshold", type=float, default=0.54)
    parser.add_argument("--train-topup-allow-weak-support-in-stage-c", dest="train_topup_allow_weak_support_in_stage_c", action="store_true")
    parser.add_argument("--no-train-topup-allow-weak-support-in-stage-c", dest="train_topup_allow_weak_support_in_stage_c", action="store_false")
    parser.set_defaults(train_topup_allow_weak_support_in_stage_c=True)
    parser.add_argument("--train-topup-stage-c-confidence-threshold", type=float, default=0.52)
    parser.add_argument("--train-topup-allowed-support-types", type=str, default="exact,near_exact,gold")
    parser.add_argument("--train-target-min-rows", type=int, default=2200)
    parser.add_argument("--train-target-max-rows", type=int, default=2500)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-baseline", action="store_true")
    parser.add_argument("--execute-v4-sweep", action="store_true")
    parser.add_argument("--include-coref", action="store_true")
    parser.add_argument("--apply-best-defaults", action="store_true")
    parser.add_argument("--sweep-implicit-min-tokens", type=str, default="6,8")
    parser.add_argument("--sweep-min-text-tokens", type=str, default="3,4")
    parser.add_argument("--gold-min-rows-for-promotion", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    artifact_mode = resolve_artifact_mode(run_profile=args.run_profile, artifact_mode=args.artifact_mode)
    processor = str(args.processor or "").strip().lower()
    if processor not in {"local", "runpod"}:
        raise ValueError(f"Unsupported processor: {args.processor}")
    llm_provider_choice = str(args.llm_provider or "").strip().lower()
    if llm_provider_choice in {"", "none"}:
        llm_provider = "runpod" if processor == "runpod" else None
    elif llm_provider_choice == "auto":
        llm_provider = "runpod" if processor == "runpod" else optional_env("DEFAULT_LLM_PROVIDER")
    else:
        llm_provider = llm_provider_choice
    train_domain_conditioning_mode, eval_domain_conditioning_mode = resolve_domain_conditioning_modes(
        domain_conditioning_mode=args.domain_conditioning_mode,
        use_domain_conditioning=bool(args.use_domain_conditioning),
        strict_domain_conditioning=bool(args.strict_domain_conditioning),
        train_domain_conditioning_mode=args.train_domain_conditioning_mode,
        eval_domain_conditioning_mode=args.eval_domain_conditioning_mode,
    )
    cfg = BuilderConfig(
        input_dir=args.input_dir or BuilderConfig().input_dir,
        output_dir=args.output_dir or BuilderConfig().output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        run_profile=args.run_profile,
        artifact_mode=artifact_mode,
        confidence_threshold=args.confidence_threshold,
        max_aspects=args.max_aspects,
        min_text_tokens=args.min_text_tokens,
        implicit_min_tokens=args.implicit_min_tokens,
        implicit_mode=args.implicit_mode,
        multilingual_mode=args.multilingual_mode,
        use_coref=args.use_coref,
        language_detection_mode=args.language_detection_mode,
        no_drop=args.no_drop,
        enable_llm_fallback=args.enable_llm_fallback,
        llm_fallback_threshold=args.llm_fallback_threshold,
        benchmark_key=args.benchmark_key,
        model_family=args.model_family,
        augmentation_mode=args.augmentation_mode,
        prompt_mode=args.prompt_mode,
        gold_annotations_path=args.gold_annotations_path,
        evaluation_protocol=args.evaluation_protocol,
        domain_holdout=args.domain_holdout,
        enforce_grounding=args.enforce_grounding,
        use_domain_conditioning=args.use_domain_conditioning,
        strict_domain_conditioning=args.strict_domain_conditioning,
        domain_conditioning_mode=domain_conditioning_mode,
        train_domain_conditioning_mode=str(train_domain_conditioning_mode),
        eval_domain_conditioning_mode=str(eval_domain_conditioning_mode),
        domain_prior_boost=args.domain_prior_boost,
        domain_prior_penalty=args.domain_prior_penalty,
        weak_domain_support_row_threshold=args.weak_domain_support_row_threshold,
        unseen_non_general_coverage_min=args.unseen_non_general_coverage_min,
        unseen_implicit_not_ready_rate_max=args.unseen_implicit_not_ready_rate_max,
        unseen_domain_leakage_row_rate_max=args.unseen_domain_leakage_row_rate_max,
        enable_reasoned_recovery=args.enable_reasoned_recovery,
        processor=processor,
        llm_provider=llm_provider,
        llm_model_name=args.llm_model_name,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_max_retries=args.llm_max_retries,
        train_fallback_general_policy=args.train_fallback_general_policy,
        train_fallback_general_cap_ratio=args.train_fallback_general_cap_ratio,
        train_review_filter_mode=args.train_review_filter_mode,
        train_salvage_mode=args.train_salvage_mode,
        train_salvage_confidence_threshold=args.train_salvage_confidence_threshold,
        train_salvage_accepted_support_types=tuple(part.strip() for part in str(args.train_salvage_accepted_support_types).split(",") if part.strip()),
        train_sentiment_balance_mode=args.train_sentiment_balance_mode,
        train_neutral_cap_ratio=args.train_neutral_cap_ratio,
        train_min_negative_ratio=args.train_min_negative_ratio,
        train_min_positive_ratio=args.train_min_positive_ratio,
        train_max_positive_ratio=args.train_max_positive_ratio,
        train_neutral_max_ratio=args.train_neutral_max_ratio,
        train_topup_recovery_mode=args.train_topup_recovery_mode,
        train_topup_confidence_threshold=args.train_topup_confidence_threshold,
        train_topup_staged_recovery=args.train_topup_staged_recovery,
        train_topup_stage_b_confidence_threshold=args.train_topup_stage_b_confidence_threshold,
        train_topup_allow_weak_support_in_stage_c=args.train_topup_allow_weak_support_in_stage_c,
        train_topup_stage_c_confidence_threshold=args.train_topup_stage_c_confidence_threshold,
        train_topup_allowed_support_types=tuple(part.strip() for part in str(args.train_topup_allowed_support_types).split(",") if part.strip()),
        train_target_min_rows=args.train_target_min_rows,
        train_target_max_rows=args.train_target_max_rows,
    )

    run_id = stable_id(
        cfg.input_dir,
        cfg.output_dir,
        cfg.benchmark_key or "auto",
        cfg.model_family,
        cfg.random_seed,
        cfg.sample_size,
        cfg.chunk_size,
        cfg.chunk_offset,
        cfg.implicit_mode,
    )
    run_dir = cfg.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "benchmark_registry.json", benchmark_registry_payload())
    write_json(run_dir / "model_registry.json", model_registry_payload())
    write_json(run_dir / "experiment_plan.json", [asdict(item) for item in build_experiment_plan()])
    write_json(run_dir / "base_config.json", asdict(cfg))

    if args.plan_only:
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "planned",
            "config": asdict(cfg),
        })
        flush_llm_cache()
        if not cfg.dry_run:
            zip_path = compress_output_folder(cfg.output_dir)
            if zip_path:
                print(f"Output compressed: {zip_path}")
        return 0

    if args.execute_baseline:
        baseline_cfg = replace(cfg)
        report = run_pipeline_sync(baseline_cfg)
        write_json(run_dir / "baseline_report.json", report)
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "completed",
            "config": asdict(cfg),
            "report": report,
        })
        flush_llm_cache()
        if not cfg.dry_run:
            zip_path = compress_output_folder(cfg.output_dir)
            if zip_path:
                print(f"Output compressed: {zip_path}")
        return 0

    if args.execute_v4_sweep:
        quality_gates = dict(QUALITY_GATES)
        quality_gates["gold_min_rows_for_promotion"] = max(1, int(args.gold_min_rows_for_promotion))
        quality_gates["unseen_non_general_coverage_min"] = float(args.unseen_non_general_coverage_min)
        quality_gates["unseen_implicit_not_ready_rate_max"] = float(args.unseen_implicit_not_ready_rate_max)
        quality_gates["unseen_domain_leakage_row_rate_max"] = float(args.unseen_domain_leakage_row_rate_max)
        implicit_min_tokens_values = _parse_bounded_int_list(
            args.sweep_implicit_min_tokens,
            minimum=4,
            maximum=16,
            fallback=[6, 8],
        )
        min_text_tokens_values = _parse_bounded_int_list(
            args.sweep_min_text_tokens,
            minimum=2,
            maximum=12,
            fallback=[3, 4],
        )
        sweep = _execute_v4_sweep(
            cfg,
            run_dir,
            include_coref=args.include_coref,
            implicit_min_tokens_values=implicit_min_tokens_values,
            min_text_tokens_values=min_text_tokens_values,
            quality_gates=quality_gates,
        )
        promoted_defaults = sweep.get("promoted_defaults")
        if args.apply_best_defaults and promoted_defaults:
            defaults_path = Path(__file__).resolve().parent / "runtime_defaults.json"
            write_json(defaults_path, {
                "generated_at": utc_now_iso(),
                "source_run_id": run_id,
                "quality_gates": quality_gates,
                "defaults": promoted_defaults,
            })
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "completed_v4_sweep",
            "config": asdict(cfg),
            "quality_gates": quality_gates,
            "best_candidate_id": sweep.get("best_candidate_id"),
            "promoted_defaults": promoted_defaults if args.apply_best_defaults else None,
            "defaults_applied": bool(args.apply_best_defaults and promoted_defaults),
        })
        flush_llm_cache()
        if not cfg.dry_run:
            zip_path = compress_output_folder(cfg.output_dir)
            if zip_path:
                print(f"Output compressed: {zip_path}")
        return 0

    run_experiments(
        cfg,
        [{
            "model_family": cfg.model_family,
            "benchmark_key": cfg.benchmark_key,
            "implicit_mode": cfg.implicit_mode,
            "multilingual_mode": cfg.multilingual_mode,
            "use_coref": cfg.use_coref,
        }],
        run_dir,
    )
    write_json(run_dir / "manifest.json", {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "status": "configured",
        "config": asdict(cfg),
    })
    flush_llm_cache()
    if not cfg.dry_run:
        zip_path = compress_output_folder(cfg.output_dir)
        if zip_path:
            print(f"Output compressed: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
