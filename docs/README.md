### Documentation

1. [Getting Started](getting-started.md) - Installation and setup instructions
2. [API Reference](api-reference.md) - Function and class documentation
3. [Analysis Methods](analysis-methods.md) - Mathematical and statistical methods used
4. [Troubleshooting](troubleshooting.md) - Common issues and solutions

#### Quick Links

- [Running the Analysis](getting-started.md#running-the-analysis)
- [Contact Duration Methods](analysis-methods.md#contact-duration-detection)
- [Statistical Analysis](analysis-methods.md#statistical-methods)

#### Regenerate Figures

Use the canonical batch regeneration command from the repository root:

```bash
python make_all_figures.py
```

Verify parity against the expected artifact manifest:

```bash
python tools/verify_outputs.py
```

Outputs are written to `batch/outputs/` and `batch/outputs/final/`.

Final figure package format:

- `<stem>.png` (400 dpi)
- `<stem>.pdf`
- `<stem>_caption.txt`

Key figure stems:

- `batch_analysis_plot`
- `batch_cv_plot`
- `residual_plots`
