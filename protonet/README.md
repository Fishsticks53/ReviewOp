# ProtoNet

`protonet` trains, evaluates, exports, and serves the prototype-network model used by ReviewOp for implicit aspect classification. It consumes benchmark JSONL artifacts from `dataset_builder` and exports a runtime bundle that the backend can load.

## Inputs and Outputs

Input dataset directory:

```text
dataset_builder/output/benchmark/<dataset>/
```

Expected input files:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `metadata.json`

Primary exported model bundle:

```text
protonet/metadata/model_bundle.pt
```

Intermediate training outputs are usually written under `protonet/output/` or the path passed through `--output-dir`.

## Code Map

| Path | Responsibility |
| --- | --- |
| `code/cli.py` | Main command-line interface for train, evaluate, and export workflows. |
| `code/training.py` | Training loop and model optimization logic. |
| `code/training_utils.py` | Shared helpers for training setup and reusable training operations. |
| `code/evaluation.py` | Evaluation flow and metrics generation. |
| `code/evaluation_utils.py` | Shared helpers for evaluation and reporting. |
| `code/export.py` | Model bundle export logic. |
| `runtime/` | Runtime bundle loading and inference helpers. |
| `http_api.py` | Optional local HTTP API for ProtoNet inference. |
| `metadata/` | Default location for exported runtime bundle metadata. |

## Training Workflow

Run the commands in this section from the repository root.

Train a model:

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --output-dir protonet\output
```

Evaluate the trained model:

```powershell
python protonet\code\cli.py eval --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --checkpoint protonet\output\checkpoints\best.pt --split test
```

Export a backend-loadable bundle:

```powershell
python protonet\code\cli.py export --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --checkpoint protonet\output\checkpoints\best.pt
```

Inspect all supported flags:

```powershell
python protonet\code\cli.py --help
```

## Runtime Usage

The backend loads ProtoNet through the implicit-aspect client when implicit inference is enabled.

Relevant environment variables:

| Variable | Purpose |
| --- | --- |
| `REVIEWOP_ENABLE_IMPLICIT` | Enables or disables implicit-aspect inference in the backend. |
| `REVIEWOP_TRUSTED_BUNDLE_ROOTS` | Adds trusted filesystem roots for bundle loading. |

Bundle loading is intentionally restricted to trusted roots because PyTorch model bundles may require object deserialization. Only load bundles produced by this project or from a trusted source.

## Optional Local HTTP Service

Run the ProtoNet HTTP API directly:

```powershell
python -m uvicorn protonet.http_api:app --host 127.0.0.1 --port 8010
```

This service is useful for isolated model testing. The main ReviewOp backend can also load the exported bundle directly without running this separate service.

## Validation

Basic import check:

```powershell
python -c "from protonet.runtime import load_bundle; print('protonet import ok')"
```

Recommended validation before using a new bundle:

1. Train on the target benchmark directory.
2. Evaluate on validation and test splits.
3. Export the bundle.
4. Confirm the backend can import and load the bundle with the expected environment settings.
5. Run one backend inference request that exercises implicit-aspect predictions.

## Relationship to Other Folders

- `dataset_builder` produces the benchmark data used here.
- `backend` loads the exported bundle for hybrid explicit/implicit inference.
- `frontend` displays the inference and analytics output returned by the backend.
