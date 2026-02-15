#!/usr/bin/env python3
"""Verify output parity against batch/outputs/manifest_expected.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _paths_for_entry(entry_path: Path, required_formats: list[str]) -> list[Path]:
    checks: list[Path] = []
    for fmt in required_formats:
        if fmt == "caption_txt":
            checks.append(entry_path.with_name(f"{entry_path.stem}_caption.txt"))
        elif fmt == "png":
            checks.append(entry_path if entry_path.suffix.lower() == ".png" else entry_path.with_suffix(".png"))
        elif fmt == "pdf":
            checks.append(entry_path.with_suffix(".pdf"))
        elif fmt == "csv":
            checks.append(entry_path if entry_path.suffix.lower() == ".csv" else entry_path.with_suffix(".csv"))
        elif fmt == "txt":
            checks.append(entry_path if entry_path.suffix.lower() == ".txt" else entry_path.with_suffix(".txt"))
        else:
            checks.append(entry_path.with_suffix(f".{fmt}"))
    return checks


def verify(manifest_path: Path, repo_root: Path, check_non_empty: bool = True) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])

    missing: list[str] = []
    empty: list[str] = []
    checked = 0

    for artifact in artifacts:
        rel_path = artifact["path"]
        required_formats = artifact.get("required_formats", [])
        path = repo_root / rel_path

        checks = _paths_for_entry(path, required_formats)
        for candidate in checks:
            checked += 1
            if not candidate.exists():
                missing.append(str(candidate))
                continue
            if check_non_empty and candidate.is_file() and candidate.stat().st_size == 0:
                empty.append(str(candidate))

    print(f"Checked files: {checked}")
    if missing:
        print("Missing files:")
        for item in missing:
            print(f"  - {item}")
    if empty:
        print("Empty files:")
        for item in empty:
            print(f"  - {item}")

    if missing or empty:
        print("PARITY CHECK: FAIL")
        return 1

    print("PARITY CHECK: PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify batch output parity manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="batch/outputs/manifest_expected.json",
        help="Path to expected artifact manifest",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root used for resolving relative paths",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero-byte files",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    repo_root = Path(args.repo_root).resolve()

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    exit_code = verify(manifest_path, repo_root, check_non_empty=not args.allow_empty)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
