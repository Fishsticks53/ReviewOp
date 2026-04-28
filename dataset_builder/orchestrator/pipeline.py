from __future__ import annotations

from pathlib import Path
import shutil
import json

from rich.progress import Progress

from ..config import BuilderConfig
from ..export.archive import write_artifact_zip
from ..export.jsonl_export import write_split_jsonl
from ..export.manifest import write_manifest
from ..export.sidecars import write_sidecar
from ..reports.quality_report import build_quality_report
from ..schemas.artifact_manifest import ArtifactManifest
from ..split.leakage_checks import check_cross_split_leakage
from .release_gate import assert_release_ready
from .exceptions import QualityGateError
from .stages import (
    ExtractionStage,
    InferenceStage,
    EvidenceStage,
    VerificationStage,
    PostVerificationEvidenceStage,
    CanonicalizationStage,
    FusionStage,
    SentimentStage,
    BenchmarkStage,
)
from ..schemas.benchmark_row import BenchmarkRow
from ..schemas.raw_review import RawReview
from ..split.grouped_split import grouped_train_val_test_split


def run_builder_pipeline(
    cfg: BuilderConfig, 
    raw_reviews: list[RawReview] | None = None,
    rows_by_split: dict[str, list[BenchmarkRow]] | None = None, 
    profile_summary: dict[str, object] | None = None
) -> dict[str, object]:
    """
    Main entry point for the builder pipeline.
    If raw_reviews are provided, it runs Stages A-F and then splits.
    If rows_by_split are provided directly, it skips to checks and export.
    """
    output_dir = Path(cfg.output_dir)
    if not cfg.aspect_memory_path:
        cfg = BuilderConfig(**{**cfg.__dict__, "aspect_memory_path": str(output_dir / "aspect_memory_candidates.json")})
    if not cfg.dry_run:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not cfg.overwrite:
                raise FileExistsError(f"output directory is not empty: {output_dir} (use --overwrite to clear)")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if raw_reviews is not None:
        requested_rows = cfg.sample_size
        loaded_rows = len(raw_reviews)
        
        # Step 1: Initial Conversion
        rows = [
            BenchmarkRow(
                review_id=r.review_id,
                group_id=r.group_id,
                domain=r.domain,
                domain_family=r.domain_family,
                review_text=r.text,
                gold_interpretations=[], # Will be filled by stages
                provenance={"source_name": r.source_name, "source_split": r.source_split}
            ) for r in raw_reviews
        ]
        
        # Step 2: Run Stages A-F
        from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
        from .telemetry import GLOBAL_STATS
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[stats]}"),
        ) as progress:
            t_stages = progress.add_task("[cyan]Building Benchmark...", total=9, stats="")
            
            stages = [
                ExtractionStage(),
                InferenceStage(),
                FusionStage(),
                EvidenceStage(),
                VerificationStage(),
                PostVerificationEvidenceStage(),
                CanonicalizationStage(),
                SentimentStage(),
                BenchmarkStage(),
            ]
            
            import threading
            import time
            
            stop_event = threading.Event()
            
            def update_stats():
                while not stop_event.is_set():
                    stats_str = (
                        f"LLM: {GLOBAL_STATS.llm_calls} | "
                        f"Cache: {GLOBAL_STATS.cached_llm_calls} | "
                        f"Fallback: {GLOBAL_STATS.fallback_calls}"
                    )
                    row_progress = ""
                    if GLOBAL_STATS.current_stage_total > 0:
                        row_progress = f" | Rows: {GLOBAL_STATS.current_stage_processed}/{GLOBAL_STATS.current_stage_total}"
                    
                    progress.update(t_stages, stats=f"{stats_str}{row_progress}")
                    time.sleep(0.5)
            
            updater_thread = threading.Thread(target=update_stats, daemon=True)
            updater_thread.start()
            
            try:
                for stage in stages:
                    stage_name = stage.__class__.__name__
                    progress.update(t_stages, description=f"[cyan]Running {stage_name}...")
                    rows = stage.process(rows, cfg)
                    progress.update(t_stages, advance=1)
            finally:
                stop_event.set()
                updater_thread.join(timeout=1.0)
                
        processed_rows = len(rows)
        rejected_rows = loaded_rows - processed_rows
        discarded_rows = 0 # Future expansion
        
        # Step 3: Split
        rows_by_split = grouped_train_val_test_split(
            rows,
            seed=cfg.random_seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
    else:
        # If provided directly via rows_by_split
        requested_rows = sum(len(s) for s in rows_by_split.values())
        loaded_rows = requested_rows
        processed_rows = requested_rows
        rejected_rows = 0
        discarded_rows = 0

    if rows_by_split is None:
        raise ValueError("Either raw_reviews or rows_by_split must be provided")

    with Progress() as progress:
        t1 = progress.add_task("[green]Quality & Leakage Checks...", total=3)
        quality = build_quality_report(
            rows_by_split, 
            requested_rows=requested_rows,
            loaded_rows=loaded_rows,
            processed_rows=processed_rows,
            rejected_rows=rejected_rows,
            discarded_rows=discarded_rows,
            runtime_reason_counts=getattr(cfg, "_rejection_reason_counts", {}) or {},
        )
        progress.update(t1, advance=1)
        leakage_results = check_cross_split_leakage(rows_by_split)
        progress.update(t1, advance=2)
        
        leakage = {
            "grouped_leakage": int(leakage_results["grouped_leakage"]),
            "exact_text_leakage": int(leakage_results["exact_text_leakage"]),
            "near_duplicate_leakage": int(leakage_results.get("near_duplicate_leakage", 0)),
        }
        
        profile = "diagnostic_strict" if getattr(cfg, "strict", False) else "research_default"
        metrics = {
            "counts": quality.export_counts,
            "quality": quality.__dict__ if hasattr(quality, "__dict__") else quality,
            "leakage": leakage,
            "profile": profile
        }
        aspect_memory_metrics = {
            "candidates_added": 0,
            "promoted_matches_used": 0,
            "candidates_promoted_this_run": 0,
            "promoted_entries_total": 0,
            "review_queue_count": 0,
            "rejected_candidates_this_run": 0,
        }
        if cfg.aspect_memory_path:
            runtime_metrics = getattr(cfg, "_aspect_memory_metrics", {}) or {}
            try:
                from ..canonical.aspect_memory import AspectMemory
                memory = AspectMemory(cfg.aspect_memory_path)
                promoted_total = sum(1 for e in memory.entries.values() if e.status == "promoted")
                review_queue_total = sum(1 for e in memory.entries.values() if e.status == "review_queue")
                aspect_memory_metrics["promoted_entries_total"] = promoted_total
                aspect_memory_metrics["review_queue_count"] = review_queue_total
            except Exception:
                pass
            for key in ("candidates_added", "promoted_matches_used", "candidates_promoted_this_run", "rejected_candidates_this_run"):
                if key in runtime_metrics:
                    aspect_memory_metrics[key] = runtime_metrics[key]
        metrics["aspect_memory"] = aspect_memory_metrics
        
        try:
            gate_results = assert_release_ready(rows_by_split, reports={"quality": quality}, leakage=leakage, profile=profile)
            status_map = {"PASS": "passed", "WARNING": "warning", "FAIL": "failed", "FATAL": "failed"}
            release_status = status_map.get(gate_results.get("status"), "unknown")
        except QualityGateError as e:
            gate_results = e.gate_results
            release_status = "failed"
        
        metrics["gate_results"] = gate_results
        
        # Mandatory export of metrics_summary.json
        if not cfg.dry_run:
            with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
                # Need to handle non-serializable objects in quality report
                def d_ser(obj):
                    if hasattr(obj, "to_dict"): return obj.to_dict()
                    if hasattr(obj, "__dict__"): return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
                    return str(obj)
                json.dump(metrics, f, indent=2, default=d_ser)
        
        if cfg.dry_run:
            return {"counts": quality.export_counts, "quality": quality, "leakage": leakage, "dry_run": True}
            
        t2 = progress.add_task("[yellow]Exporting Artifacts...", total=4)
        counts = write_split_jsonl(output_dir, rows_by_split)
        progress.update(t2, advance=1)
        
        manifest = ArtifactManifest(
            version="dataset_builder_p0",
            dataset_inputs=[str(path) for path in cfg.input_paths],
            profile_summary=profile_summary or {},
            policies_used={
                "release_gate": profile,
                "llm_provider": cfg.llm_provider,
                "llm_model": cfg.llm_model,
                "random_seed": cfg.random_seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "sample_size": cfg.sample_size,
                "chunk_size": cfg.chunk_size,
                "chunk_offset": cfg.chunk_offset,
                "strict": cfg.strict,
                "domain_mode": cfg.domain_mode,
                "provisional_policy": cfg.provisional_policy,
                "evidence_window_tokens": cfg.evidence_window_tokens,
                "aspect_memory_path": cfg.aspect_memory_path,
                "aspect_memory_auto_promote": cfg.aspect_memory_auto_promote,
                "symptom_store_path": cfg.symptom_store_path,
                "max_workers": cfg.max_workers,
            },
            split_summary=counts,
            release_status=release_status,
        )
        write_manifest(output_dir / "manifest.json", manifest)
        progress.update(t2, advance=1)
        
        write_sidecar(output_dir / "quality_report.json", quality)
        progress.update(t2, advance=1)
        
        archive_path = write_artifact_zip(output_dir)
        progress.update(t2, advance=1)
        
    return {"counts": counts, "quality": quality, "leakage": leakage, "archive_path": archive_path}
