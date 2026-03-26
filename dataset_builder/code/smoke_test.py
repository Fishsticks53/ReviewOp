from __future__ import annotations

from pathlib import Path
import shutil

from build_dataset import run_pipeline
from config import BuilderConfig
from utils import read_jsonl


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    scratch_root = repo_root / "dataset_builder" / "output" / "domain_agnostic_smoke_v2"
    input_dir = scratch_root / "input"
    output_dir = scratch_root / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = input_dir / "sample.csv"
    sample_csv.write_text(
        "review_text,rating,brand\n"
        "The battery lasts all day and the screen is bright,5,Acme\n"
        "Great food but service was slow,4,Bistro\n"
        "The camera is sharp and the performance is fast,5,Acme\n"
        "The price is high but the quality feels solid,4,Bistro\n"
        "The delivery was late and the packaging was damaged,2,Generic\n"
        "Battery never lasts long and the display is dim,2,Acme\n"
        "The food was tasty and the service was friendly,5,Generic\n"
        "The comfort is good but the price is too high,3,Generic\n"
        "It wouldn't even turn on and the pointer is stuck,1,Acme\n"
        "The deliveries take forever and it never arrived,1,Bistro\n",
        encoding="utf-8",
    )

    cfg = BuilderConfig(input_dir=input_dir, output_dir=output_dir, text_column_override="review_text", enable_llm_fallback=False)
    report = run_pipeline(cfg)

    assert (output_dir / "explicit" / "train.jsonl").exists()
    assert (output_dir / "implicit" / "train.jsonl").exists()
    assert (output_dir / "reports" / "build_report.json").exists()

    explicit_rows = read_jsonl(output_dir / "explicit" / "train.jsonl")
    implicit_rows = read_jsonl(output_dir / "implicit" / "train.jsonl")
    assert explicit_rows
    assert implicit_rows
    assert "discovered_aspects" in report
    assert "learned_seed_vocab" in report
    assert "implicit_diagnostics" in report
    assert report["implicit_diagnostics"]["top_implicit_aspects"]
    assert "sentiment_lexicon_coverage" in report["implicit_diagnostics"]

    shutil.rmtree(scratch_root, ignore_errors=True)
    print("dataset_builder smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
