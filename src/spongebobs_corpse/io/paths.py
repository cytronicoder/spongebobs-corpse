from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OutputManager:
    repo_root: Path
    out_dir: Path
    create_final_dir: bool = True

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        if not self.out_dir.is_absolute():
            self.out_dir = (self.repo_root / self.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.create_final_dir:
            self.final_dir.mkdir(parents=True, exist_ok=True)

    @property
    def final_dir(self) -> Path:
        return self.out_dir / "final"

    @property
    def batch_dir(self) -> Path:
        return self.repo_root / "batch"

    def output_file(self, name: str) -> Path:
        return self.out_dir / name

    def final_file(self, name: str) -> Path:
        return self.final_dir / name

    def batch_file(self, name: str) -> Path:
        return self.batch_dir / name
