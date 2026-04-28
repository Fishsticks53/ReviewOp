from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_builder.benchmark.ambiguity import compute_ambiguity_score
from dataset_builder.benchmark.hardness_scorer import score_row_hardness
from dataset_builder.benchmark.novelty import assess_novelty
from dataset_builder.canonical.canonicalizer import canonicalize_interpretation
from dataset_builder.canonical.domain_maps import CanonicalMappingResult
from dataset_builder.config import BuilderConfig
from dataset_builder.implicit.symptom_store import SymptomPatternStore
from dataset_builder.orchestrator.stages import InferenceStage
from dataset_builder.orchestrator.release_gate import assert_release_ready
from dataset_builder.reports.quality_report import build_quality_report
from dataset_builder.scripts.run_diagnostics import DiagnosticConfig, DiagnosticRunner
from dataset_builder.scripts.build_benchmark import build_arg_parser, build_config_from_args
from dataset_builder.schemas.benchmark_row import BenchmarkRow
from dataset_builder.schemas.interpretation import Interpretation
from unittest.mock import patch


def interp(**overrides) -> Interpretation:
    payload = {
        "aspect_raw": "battery",
        "aspect_canonical": "battery_life",
        "latent_family": "battery",
        "label_type": "explicit",
        "sentiment": "unknown",
        "evidence_text": "battery",
        "evidence_span": [4, 11],
        "source": "test",
        "support_type": "exact",
        "source_type": "explicit",
        "mapping_source": "exact_phrase",
    }
    payload.update(overrides)
    return Interpretation(**payload)


class InterpretationContractTests(unittest.TestCase):
    def test_rejects_invalid_source_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid source_type"):
            interp(source_type="unknown")

    def test_implicit_learned_requires_pattern_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched_pattern"):
            interp(label_type="implicit", source_type="implicit_learned", matched_pattern=None, pattern_id="p1")
        with self.assertRaisesRegex(ValueError, "pattern_id"):
            interp(label_type="implicit", source_type="implicit_learned", matched_pattern="keeps crashing", pattern_id=None)

    def test_explicit_rejects_pattern_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit interpretations cannot include pattern metadata"):
            interp(matched_pattern="battery issue", pattern_id="p1")


class SymptomStoreTests(unittest.TestCase):
    def write_store(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with tmp:
            json.dump(rows, tmp)
        return Path(tmp.name)

    def test_store_rejects_missing_pattern_id(self) -> None:
        path = self.write_store([
            {"phrase": "keeps crashing", "aspect_canonical": "performance", "status": "promoted"}
        ])
        with self.assertRaisesRegex(ValueError, "pattern_id"):
            SymptomPatternStore.load(path)

    def test_store_rejects_duplicate_pattern_id(self) -> None:
        path = self.write_store([
            {"pattern_id": "p1", "phrase": "keeps crashing", "aspect_canonical": "performance", "status": "promoted"},
            {"pattern_id": "p1", "phrase": "kept crashing", "aspect_canonical": "performance", "status": "promoted"},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate pattern_id"):
            SymptomPatternStore.load(path)

    def test_exact_and_normalized_matches_return_spans(self) -> None:
        path = self.write_store([
            {
                "pattern_id": "electronics_performance_crash_001",
                "phrase": "keeps crashing",
                "aspect_canonical": "performance",
                "latent_family": "performance",
                "status": "promoted",
                "confidence": 0.87,
            }
        ])
        store = SymptomPatternStore.load(path)

        exact = store.match("This app keeps crashing on launch.", domain="electronics")
        self.assertEqual(exact[0].pattern_id, "electronics_performance_crash_001")
        self.assertEqual(exact[0].matched_text, "keeps crashing")
        self.assertEqual(exact[0].start_char, 9)
        self.assertEqual(exact[0].end_char, 23)

        normalized = store.match("This app kept crashing on launch.", domain="electronics")
        self.assertEqual(normalized[0].matched_text, "kept crashing")
        self.assertNotEqual(normalized[0].matched_text, "This app kept crashing on launch.")


class InferenceAndCanonicalizationTests(unittest.TestCase):
    def test_implicit_json_prefers_sentence_evidence_when_term_matches(self) -> None:
        row = BenchmarkRow(
            review_id="r-json-1",
            group_id="g1",
            domain="laptop",
            domain_family="electronics",
            review_text="The keyboard feels great. But battery life is weak and dies fast.",
        )
        [processed] = InferenceStage().process([row], BuilderConfig())
        implicit_json = [i for i in processed.implicit_interpretations if i.source_type == "implicit_json"]
        self.assertTrue(implicit_json)
        self.assertTrue(any(i.evidence_scope in {"sentence", "phrase_window"} for i in implicit_json))
        self.assertTrue(any(i.implicit_trigger == "latent_family_match" for i in implicit_json))

    def test_inference_uses_learned_pattern_id_and_span_evidence(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump([
                {
                    "pattern_id": "electronics_battery_life_001",
                    "phrase": "battery doesn't last",
                    "aspect_canonical": "battery_life",
                    "latent_family": "battery",
                    "status": "promoted",
                }
            ], tmp)
            store_path = tmp.name

        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The screen is bright, but the battery doesn't last through lunch.",
        )
        [processed] = InferenceStage().process([row], BuilderConfig(symptom_store_path=store_path))
        learned = processed.implicit_interpretations[0]

        self.assertEqual(learned.source_type, "implicit_learned")
        self.assertEqual(learned.pattern_id, "electronics_battery_life_001")
        self.assertEqual(learned.matched_pattern, "battery doesn't last")
        self.assertEqual(learned.evidence_text, processed.review_text[learned.evidence_span[0]:learned.evidence_span[1]])
        self.assertNotEqual(learned.evidence_span, [0, len(processed.review_text)])

    def test_canonicalization_preserves_learned_canonical(self) -> None:
        learned = interp(
            aspect_raw="battery doesn't last",
            aspect_canonical="battery_life",
            label_type="implicit",
            source_type="implicit_learned",
            matched_pattern="battery doesn't last",
            pattern_id="electronics_battery_life_001",
        )

        result = canonicalize_interpretation(learned, "electronics")

        self.assertEqual(result.aspect_canonical, "battery_life")

    def test_canonicalization_propagates_mapping_scope_and_layers(self) -> None:
        item = interp(aspect_raw="service quick")
        with patch("dataset_builder.canonical.canonicalizer.lookup_domain_map") as mocked_lookup:
            mocked_lookup.return_value = CanonicalMappingResult(
                aspect_canonical="service_speed",
                mapping_source="anchor_modifier",
                mapping_confidence=0.85,
                mapping_scope="generic+domain_specific",
                mapping_layers=("generic", "domain_specific"),
            )
            result = canonicalize_interpretation(item, "restaurant")

        self.assertEqual(result.mapping_scope, "generic+domain_specific")
        self.assertEqual(result.mapping_layers, ("generic", "domain_specific"))


class BenchmarkQualityTests(unittest.TestCase):
    def test_novelty_has_known_boundary_and_novel_states(self) -> None:
        self.assertEqual(assess_novelty("battery_life", {"battery_life"}).status, "known")
        self.assertEqual(assess_novelty("screen_eye_strain", {"display"}, mapping_confidence=0.35, evidence_supported=True).status, "boundary")
        self.assertEqual(assess_novelty("hinge_sparks", {"display"}, mapping_confidence=0.0, evidence_supported=True).status, "novel")

    def test_h3_can_be_emitted_for_novel_ambiguous_rows(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The hinge sparks and the screen hurts my eyes.",
            gold_interpretations=[
                interp(label_type="implicit", source_type="implicit_json", aspect_raw="hinge sparks", aspect_canonical="unknown"),
                interp(label_type="implicit", source_type="implicit_json", aspect_raw="screen hurts", aspect_canonical="display"),
            ],
            ambiguity_score=0.8,
            novelty_status="novel",
        )

        self.assertGreater(compute_ambiguity_score(list(row.gold_interpretations)), 0)
        self.assertEqual(score_row_hardness(row), "H3")

    def test_quality_report_contains_diagnostic_distributions(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery doesn't last.",
            gold_interpretations=[
                interp(label_type="implicit", source_type="implicit_learned", matched_pattern="battery doesn't last", pattern_id="p1")
            ],
            novelty_status="known",
            hardness_tier="H1",
        )

        report = build_quality_report({"train": [row], "val": [], "test": []}, loaded_rows=1, processed_rows=1)

        self.assertEqual(report.source_type_distribution["implicit_learned"], 1)
        self.assertEqual(report.novelty_distribution["known"], 1)
        self.assertEqual(report.hardness_distribution["H1"], 1)
        self.assertIn("mapping_scope_distribution", report.__dict__)
        self.assertTrue(report.accounting_valid)


class ReleaseGateTests(unittest.TestCase):
    def test_research_default_thresholds_for_provisional_and_anchor_modifier(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
            gold_interpretations=[
                interp(
                    mapping_source="provisional",
                    source_type="explicit",
                    label_type="explicit",
                    mapping_scope="provisional",
                    evidence_text="Battery",
                    evidence_span=[0, 7],
                )
            ],
        )
        report = build_quality_report({"train": [row], "val": [row], "test": [row]}, loaded_rows=3, processed_rows=3)
        result = assert_release_ready(
            {"train": [row], "val": [row], "test": [row]},
            reports={"quality": report},
            leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            profile="research_default",
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("provisional" in f for f in result["failures"]))

    def test_gate_fails_when_mapping_scope_unknown_exists(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The battery is weak.",
            gold_interpretations=[interp(mapping_scope="unknown")],
        )
        report = build_quality_report({"train": [row], "val": [row], "test": [row]}, loaded_rows=3, processed_rows=3)
        result = assert_release_ready(
            {"train": [row], "val": [row], "test": [row]},
            reports={"quality": report},
            leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            profile="research_default",
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("mapping_scope" in f for f in result["failures"]))

    def test_learned_run_fails_without_implicit_learned_output(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The battery is weak.",
            gold_interpretations=[interp()],
        )
        report = build_quality_report({"train": [row], "val": [row], "test": [row]}, loaded_rows=3, processed_rows=3)

        with self.assertRaisesRegex(ValueError, "implicit_learned"):
            assert_release_ready(
                {"train": [row], "val": [row], "test": [row]},
                reports={"quality": report, "require_learned": True},
                leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
                profile="diagnostic_strict"
            )

    def test_gate_fails_on_evidence_mismatch_and_accounting_mismatch(self) -> None:
        bad = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
            gold_interpretations=[interp(evidence_text="not present", evidence_span=[0, 7])],
        )
        report = build_quality_report({"train": [bad], "val": [bad], "test": [bad]}, loaded_rows=9, processed_rows=3, rejected_rows=6)

        with self.assertRaisesRegex(ValueError, "evidence exact-match"):
            assert_release_ready(
                {"train": [bad], "val": [bad], "test": [bad]},
                reports={"quality": report},
                leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
                profile="diagnostic_strict"
            )

    def test_gate_fails_on_invalid_source_type_in_dict_payload(self) -> None:
        splits = {
            "train": [{"review_text": "x", "gold_interpretations": [{"source_type": "unknown", "evidence_text": "x", "evidence_span": [0, 1]}]}],
            "val": [{"review_text": "y", "gold_interpretations": [{"source_type": "explicit", "evidence_text": "y", "evidence_span": [0, 1]}]}],
            "test": [{"review_text": "z", "gold_interpretations": [{"source_type": "explicit", "evidence_text": "z", "evidence_span": [0, 1]}]}],
        }

        with self.assertRaisesRegex(ValueError, "invalid source_type"):
            assert_release_ready(
                splits,
                reports={"quality": {"total_exported": 3, "evidence": {"exact_match_rate": 1.0}, "accounting_valid": True}},
                leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            )


class BuildBenchmarkCliTests(unittest.TestCase):
    def test_cli_exposes_new_flags_and_wires_builder_config(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([
            "dataset_builder/input",
            "--strict",
            "--domain-mode",
            "generic_only",
            "--aspect-memory",
            "dataset_builder/config/aspect_memory/memory_v001.json",
        ])
        cfg = build_config_from_args(args, [Path("dataset_builder/input/reviews.jsonl")])
        self.assertTrue(cfg.strict)
        self.assertEqual(cfg.domain_mode, "generic_only")
        self.assertEqual(Path(cfg.aspect_memory_path), Path("dataset_builder/config/aspect_memory/memory_v001.json"))


class MappingAndUnseenDomainTests(unittest.TestCase):
    def test_anchor_modifier_expected_cases(self) -> None:
        cases = [
            ("laptop", "screen", ("dim",), "display"),
            ("laptop", "battery", ("weak",), "battery_life"),
            ("laptop", "laptop", ("portable",), "portability"),
            ("restaurant", "service", ("quick",), "service_speed"),
            ("restaurant", "food", ("fresh",), "food_quality"),
            ("laptop", "speaker", ("loud",), "audio"),
        ]
        for domain, anchor, mods, expected in cases:
            out = canonicalize_interpretation(
                interp(
                    aspect_raw=f"{anchor} {' '.join(mods)}",
                    aspect_anchor=anchor,
                    modifier_terms=mods,
                    anchor_source="test",
                ),
                domain,
            )
            self.assertEqual(out.aspect_canonical, expected)

    def test_unseen_domain_generic_fallback_no_crash(self) -> None:
        row = BenchmarkRow(
            review_id="fashion_001",
            group_id="g_fashion_1",
            domain="fashion",
            domain_family="unknown",
            review_text="The stitching came apart after one wash, but the fabric felt soft.",
        )
        [processed] = InferenceStage().process([row], BuilderConfig())
        self.assertIsNotNone(processed)


class DiagnosticsAndMemoryTddTests(unittest.TestCase):
    def test_diagnostics_include_unseen_domain_flag_filters_fixture_loading(self) -> None:
        root = Path("dataset_builder/output/_tmp_diag_test")
        if root.exists():
            import shutil
            shutil.rmtree(root)
        fixtures = root / "fixtures"
        fixtures.mkdir(parents=True)
        (fixtures / "regular.jsonl").write_text(
            json.dumps({"id": "r1", "fixture_type": "regular", "review_text": "ok"}) + "\n",
            encoding="utf-8",
        )
        (fixtures / "unseen_domain_fashion.jsonl").write_text(
            json.dumps({"id": "u1", "fixture_type": "unseen_domain", "review_text": "ok"}) + "\n",
            encoding="utf-8",
        )
        exp = root / "exp.yaml"
        exp.write_text("{}", encoding="utf-8")
        cfg = DiagnosticConfig(fixtures, exp, root / "out")
        runner = DiagnosticRunner(cfg)
        rows = runner.load_fixtures()
        self.assertEqual(len(rows), 1)

    def test_metrics_summary_contains_aspect_memory_block(self) -> None:
        from dataset_builder.orchestrator.pipeline import run_builder_pipeline
        from dataset_builder.schemas.raw_review import RawReview

        root = Path("dataset_builder/output/_tmp_metrics_test")
        if root.exists():
            import shutil
            shutil.rmtree(root)
        out = root / "out"
        memory = root / "memory.json"
        cfg = BuilderConfig(
            input_paths=(),
            output_dir=out,
            llm_provider="none",
            overwrite=True,
            aspect_memory_path=str(memory),
        )
        raws = [
            RawReview(
                review_id="r1",
                group_id="g1",
                text="Thing works but stitching came apart and battery weak",
                domain="fashion",
                domain_family="fashion",
                source_name="test",
            )
        ]
        with patch("dataset_builder.orchestrator.pipeline.assert_release_ready") as gate:
            gate.return_value = {"status": "PASS", "failures": [], "warnings": [], "metrics": {}}
            with patch("dataset_builder.orchestrator.pipeline.ExtractionStage") as mock_ext, \
                 patch("dataset_builder.orchestrator.pipeline.InferenceStage") as mock_inf, \
                 patch("dataset_builder.orchestrator.pipeline.FusionStage") as mock_fus, \
                 patch("dataset_builder.orchestrator.pipeline.EvidenceStage") as mock_ev, \
                 patch("dataset_builder.orchestrator.pipeline.VerificationStage") as mock_ver, \
                 patch("dataset_builder.orchestrator.pipeline.PostVerificationEvidenceStage") as mock_pv, \
                 patch("dataset_builder.orchestrator.pipeline.CanonicalizationStage") as mock_can, \
                 patch("dataset_builder.orchestrator.pipeline.SentimentStage") as mock_sen, \
                 patch("dataset_builder.orchestrator.pipeline.BenchmarkStage") as mock_ben:
                for m in [mock_ext, mock_inf, mock_fus, mock_ev, mock_ver, mock_pv, mock_can, mock_sen, mock_ben]:
                    m.return_value.process.side_effect = lambda rows, _cfg: rows
                run_builder_pipeline(cfg, raw_reviews=raws)
        payload = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
        self.assertIn("aspect_memory", payload)
        for key in (
            "candidates_added",
            "promoted_matches_used",
            "candidates_promoted_this_run",
            "promoted_entries_total",
            "review_queue_count",
            "rejected_candidates_this_run",
        ):
            self.assertIn(key, payload["aspect_memory"])

    def test_conflict_resolution_prefers_symptom_store_by_default(self) -> None:
        memory_path = Path("dataset_builder/output/_tmp_memory_conflict/memory.json")
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_payload = {
            "entries": {
                "battery weak": {
                    "aspect_raw": "battery weak",
                    "status": "promoted",
                    "validation_status": "auto_validated",
                    "support_count": 5,
                    "unique_reviews": ["r1"],
                    "domains": ["electronics"],
                    "evidence_samples": ["battery weak"],
                    "evidence_examples": ["battery weak"],
                    "aspect_canonical": "battery_health",
                    "unique_review_count": 1,
                    "generic_parent": "quality",
                    "domain_specific_aspect": "battery_health",
                }
            }
        }
        memory_path.write_text(json.dumps(memory_payload), encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump([{"pattern_id":"p1","phrase":"battery weak","aspect_canonical":"battery_life","status":"promoted","confidence":0.9}], tmp)
            store_path = tmp.name
        row = BenchmarkRow(review_id="r1", group_id="g1", domain="electronics", domain_family="electronics", review_text="battery weak")
        [processed] = InferenceStage().process([row], BuilderConfig(symptom_store_path=store_path, aspect_memory_path=str(memory_path)))
        learned = [i for i in processed.implicit_interpretations if i.source_type == "implicit_learned"][0]
        self.assertEqual(learned.aspect_canonical, "battery_life")
        self.assertIn("symptom_store", learned.mapping_layers)
        self.assertIn("aspect_memory", learned.mapping_layers)
        self.assertEqual(getattr(learned, "conflict_resolution", "none"), "symptom_store_preferred")

    def test_conflict_resolution_allows_manual_validated_memory_override(self) -> None:
        memory_path = Path("dataset_builder/output/_tmp_memory_conflict2/memory.json")
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_payload = {
            "entries": {
                "battery weak": {
                    "aspect_raw": "battery weak",
                    "status": "promoted",
                    "validation_status": "manual_validated",
                    "support_count": 5,
                    "unique_reviews": ["r1"],
                    "domains": ["electronics"],
                    "evidence_samples": ["battery weak"],
                    "evidence_examples": ["battery weak"],
                    "aspect_canonical": "battery_health",
                    "unique_review_count": 1,
                    "generic_parent": "quality",
                    "domain_specific_aspect": "battery_health",
                    "confidence": 0.95
                }
            }
        }
        memory_path.write_text(json.dumps(memory_payload), encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump([{"pattern_id":"p1","phrase":"battery weak","aspect_canonical":"battery_life","status":"promoted","confidence":0.8}], tmp)
            store_path = tmp.name
        row = BenchmarkRow(review_id="r1", group_id="g1", domain="electronics", domain_family="electronics", review_text="battery weak")
        [processed] = InferenceStage().process([row], BuilderConfig(symptom_store_path=store_path, aspect_memory_path=str(memory_path)))
        learned = [i for i in processed.implicit_interpretations if i.source_type == "implicit_learned"][0]
        self.assertEqual(learned.aspect_canonical, "battery_health")
        self.assertEqual(getattr(learned, "conflict_resolution", "none"), "manual_validated_aspect_memory_preferred")

    def test_review_queue_file_contains_required_keys(self) -> None:
        from dataset_builder.canonical.aspect_memory import AspectMemory
        root = Path("dataset_builder/output/_tmp_review_queue")
        root.mkdir(parents=True, exist_ok=True)
        memory = AspectMemory(root / "memory.json")
        memory.add_evidence("thing", "r1", "thing broke", "fashion")
        queue_path = root / "aspect_memory_review_queue.json"
        memory.write_review_queue(queue_path)
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        self.assertIn("items", payload)
        if payload["items"]:
            item = payload["items"][0]
            for k in ("aspect_id", "domain", "surface_forms", "suggested_canonical", "support_count", "unique_review_count", "evidence_examples", "reason"):
                self.assertIn(k, item)


if __name__ == "__main__":
    unittest.main()
