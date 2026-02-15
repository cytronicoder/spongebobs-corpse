from pathlib import Path

from spongebobs_corpse.pipeline.run import run_pipeline


def test_pipeline_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    artifacts = run_pipeline(
        input_csv=Path("batch/data.csv"),
        out_dir=out_dir,
        verify=False,
    )
    assert artifacts
    assert (out_dir / "batch_analysis_plot.png").exists()
