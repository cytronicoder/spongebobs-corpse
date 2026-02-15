# Batch Reproduction

Canonical command (single source of truth):

```bash
python -m spongebobs_corpse pipeline run-batch --input batch/data.csv --out batch/outputs
```

Input data:

- Primary path: `batch/data.csv`
- Backward-compatible fallback supported by pipeline: `data/batch/data.csv`

Outputs:

- `batch/outputs/` for canonical figures/tables/reports

Parity verification:

```bash
python tools/verify_outputs.py --manifest batch/outputs/manifest_expected.json --repo-root .
```

The verifier fails with non-zero exit if any required artifact is missing or empty.
