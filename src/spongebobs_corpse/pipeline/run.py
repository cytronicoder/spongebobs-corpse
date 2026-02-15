from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config
from ..io.paths import OutputManager
from .steps import run_steps


def _verify_manifest(repo_root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for artifact in manifest.get("artifacts", []):
        path = repo_root / artifact["path"]
        formats = artifact.get("required_formats", [])
        for fmt in formats:
            if fmt == "caption_txt":
                candidate = path.with_name(f"{path.stem}_caption.txt")
            elif fmt == "png":
                candidate = path if path.suffix == ".png" else path.with_suffix(".png")
            elif fmt == "pdf":
                candidate = path.with_suffix(".pdf")
            elif fmt == "csv":
                candidate = path if path.suffix == ".csv" else path.with_suffix(".csv")
            elif fmt == "txt":
                candidate = path if path.suffix == ".txt" else path.with_suffix(".txt")
            else:
                candidate = path
            if not candidate.exists():
                missing.append(str(candidate))
    if missing:
        raise RuntimeError("Manifest parity failed. Missing artifacts:\n" + "\n".join(missing))


def _resolve_input_csv(repo_root: Path, requested: Path) -> Path:
    if requested.is_absolute() and requested.exists():
        return requested

    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(repo_root / requested)
    candidates.append(repo_root / "batch" / "data.csv")
    candidates.append(repo_root / "data" / "batch" / "data.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Batch input data not found. Searched:\n{searched}")


def run_pipeline(
    input_csv: Path = Path("batch/data.csv"),
    out_dir: Path = Path("batch/outputs"),
    config_path: Path | None = None,
    seed_override: int | None = None,
    verify: bool = True,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    config = load_config(config_path)
    if seed_override is not None:
        config.analysis.seed = seed_override

    resolved_input = _resolve_input_csv(repo_root, input_csv)

    manager = OutputManager(repo_root=repo_root, out_dir=out_dir, create_final_dir=config.io.create_final_dir)
    artifacts = run_steps(config=config, input_csv=resolved_input, manager=manager)

    manifest_path = repo_root / config.io.manifest_path
    if verify:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Required manifest not found: {manifest_path}")
        _verify_manifest(repo_root, manifest_path)

    return artifacts
