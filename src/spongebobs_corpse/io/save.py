from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.image as mpimg
import pandas as pd

from ..viz.captions import build_caption


def validate_saved_figure(path: Path, min_dpi: int, figsize: tuple[float, float]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected output missing: {path}")
    if path.suffix.lower() == ".png":
        image = mpimg.imread(path)
        min_height = int(figsize[1] * min_dpi * 0.85)
        if image.shape[0] < min_height:
            raise ValueError(f"PNG resolution too low for {path.name}: {image.shape[0]} px tall")


def save_table(frame: pd.DataFrame, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def save_text(text: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target


def save_figure(
    fig,
    stem: str,
    out_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 400,
    write_caption: bool = True,
    caption: str | None = None,
    caption_metadata: Mapping | None = None,
    create_final: bool = True,
    final_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    figsize = tuple(fig.get_size_inches())

    for fmt in formats:
        target = out_dir / f"{stem}.{fmt.lower()}"
        fig.savefig(target, dpi=dpi, facecolor="white", bbox_inches="tight")
        validate_saved_figure(target, min_dpi=300, figsize=figsize)
        saved[target.name] = target

    if create_final and final_dir is not None:
        final_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            target = final_dir / f"{stem}.{fmt.lower()}"
            fig.savefig(target, dpi=dpi, facecolor="white", bbox_inches="tight")
            validate_saved_figure(target, min_dpi=300, figsize=figsize)
            saved[f"final/{target.name}"] = target

    if write_caption:
        caption_text = caption or build_caption(caption_metadata or {})
        caption_target = out_dir / f"{stem}_caption.txt"
        caption_target.write_text(caption_text + "\n", encoding="utf-8")
        saved[caption_target.name] = caption_target
        if create_final and final_dir is not None:
            final_caption = final_dir / f"{stem}_caption.txt"
            final_caption.write_text(caption_text + "\n", encoding="utf-8")
            saved[f"final/{final_caption.name}"] = final_caption

    return saved
