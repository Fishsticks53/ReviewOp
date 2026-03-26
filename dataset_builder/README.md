# ReviewOp Dataset Builder

This folder now contains the new domain-agnostic dataset pipeline.

## Outputs

- `output/explicit/train.jsonl`
- `output/explicit/val.jsonl`
- `output/explicit/test.jsonl`
- `output/implicit/train.jsonl`
- `output/implicit/val.jsonl`
- `output/implicit/test.jsonl`
- `output/reports/build_report.json`
- `output/reports/data_quality_report.json`

## Run

From the repo root:

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

Optional flags:

- `--seed`
- `--text-column`
- `--aspect-vocab-size`
- `--confidence-threshold`
- `--enable-llm-fallback`

LLM priority when fallback is enabled:

1. Groq
2. OpenAI
