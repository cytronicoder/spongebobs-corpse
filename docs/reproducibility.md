### Reproducibility

- Seed: default `42` (`analysis.seed` in `src/spongebobs_corpse/config/defaults.yaml`)
- Deterministic command:

```bash
python -m spongebobs_corpse pipeline run-batch --input batch/data.csv --out batch/outputs
python tools/verify_outputs.py --manifest batch/outputs/manifest_expected.json
```

Expected output root:

- `batch/outputs` (tables, reports, figures)
- `batch/outputs/final` (publication copies)
