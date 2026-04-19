# Dataset Builder

`dataset_builder` converts raw review data into benchmark artifacts used to train and evaluate ReviewOp's implicit-aspect classifier. It combines deterministic preprocessing, linguistic normalization, optional LLM-backed ambiguity resolution, quality gates, and experiment reporting.

## Outputs

The primary benchmark output is written under:

```text
dataset_builder/output/benchmark/<dataset>/
```

Expected files:

| File | Purpose |
| --- | --- |
| `train.jsonl` | Training split consumed by ProtoNet. |
| `val.jsonl` | Validation split consumed by ProtoNet evaluation and threshold tuning. |
| `test.jsonl` | Held-out test split for reporting. |
| `metadata.json` | Dataset statistics, label inventory, configuration, and provenance. |

Build reports are written under `dataset_builder/output/reports/` when reporting is enabled.

## Data Flow

```text
input reviews
    -> source loading
    -> normalization and language utilities
    -> target/aspect extraction
    -> optional coreference and LLM fallback
    -> policy and quality filters
    -> split generation
    -> benchmark JSONL + metadata + reports
```

The pipeline keeps LLM calls behind explicit runtime options so deterministic builds can run without network/model-provider access.

## Code Map

| Path | Responsibility |
| --- | --- |
| `code/build_dataset.py` | Main CLI entry point for building benchmark datasets. |
| `code/pipeline_runner.py` | Coordinates high-level pipeline execution. |
| `code/pipeline_helpers.py` | Shared helpers used by the build pipeline. |
| `code/pipeline_state.py` | Runtime state containers used during dataset construction. |
| `code/runtime_options.py` | CLI/runtime option parsing and mode handling. |
| `code/experiment_policy.py` | Dataset quality and experiment policy decisions. |
| `code/language_utils.py` | Text normalization, token handling, and linguistic helpers. |
| `code/coref.py` | Coreference-related helpers used during example construction. |
| `code/evaluation.py` | Dataset/evaluation utilities for generated artifacts. |
| `code/report_context.py` | Report context assembly. |
| `code/report_payload.py` | Report payload serialization. |
| `code/report_blockers.py` | Build blocker and warning reporting. |
| `code/run_experiment.py` | Recommended runner for LLM-backed experiment sweeps. |

## Common Commands

Run the commands in this section from the repository root. If your terminal is already inside `dataset_builder\code`, use `python build_dataset.py ...` instead of prefixing the path with `dataset_builder\code\`.

Run a deterministic local build:

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output --run-profile research --sample-size 50 --no-enable-llm-fallback
```

Run with a small LLM-backed fallback budget:

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output --run-profile research --sample-size 50 --enable-llm-fallback --llm-option openai
```

Run an experiment sweep:

```powershell
python dataset_builder\code\run_experiment.py --input-dir dataset_builder\input --output-dir dataset_builder\output --run-profile research --sample-size 50 --enable-llm-fallback --llm-option openai
```

Inspect all supported flags:

```powershell
python dataset_builder\code\build_dataset.py --help
python dataset_builder\code\run_experiment.py --help
```

## Recommended Workflow

1. Run a deterministic build with `--no-enable-llm-fallback`.
2. Review the generated metadata and build report.
3. Run a bounded LLM fallback build only if deterministic extraction leaves too many unresolved examples.
4. Compare reports between deterministic and LLM-backed runs.
5. Promote only the benchmark directory that passes quality gates and produces stable ProtoNet validation metrics.

## LLM Usage

LLM integration is optional and should be treated as an experiment-time fallback, not as a required build dependency.

Operational rules:

- Use explicit `--enable-llm-fallback` or `--no-enable-llm-fallback` settings.
- Keep `--sample-size` bounded for LLM-backed runs to control reproducibility and cost.
- Prefer cached/repeated experiment runs through `run_experiment.py`.
- Do not store provider API keys in this repository.
- Provider failures should degrade to neutral/no-aspect output instead of inserting raw review text into labels.

## Quality and Failure Behavior

The builder is designed to make failures visible:

- Row-level failures are isolated so one bad review does not poison the full dataset.
- Reports record blockers, warnings, and split statistics.
- Generated examples should be normalized before writing JSONL.
- Empty, malformed, or ambiguous rows should be filtered or marked explicitly by policy.

## Downstream Consumers

`protonet` consumes the benchmark directory directly:

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --output-dir protonet\output
```

The backend does not consume raw builder outputs directly. It consumes the exported ProtoNet bundle produced after training.
