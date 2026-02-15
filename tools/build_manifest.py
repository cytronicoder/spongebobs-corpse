#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(outputs_dir: Path, batch_dir: Path) -> dict:
    artifacts = []
    for file in sorted(outputs_dir.glob("*")):
        if file.is_dir() or file.name == "manifest_expected.json":
            continue

        ext = file.suffix.lower()
        if ext in {".png", ".pdf"}:
            kind = "figure"
        elif ext == ".csv":
            kind = "table"
        else:
            kind = "report"

        required = []
        if ext == ".png":
            required = ["png", "pdf", "caption_txt"]
        elif ext == ".csv":
            required = ["csv"]
        elif ext == ".txt":
            required = ["txt"]

        if required:
            rel_path = str(file.relative_to(batch_dir.parent))
            artifacts.append({"path": rel_path, "type": kind, "required_formats": required})

    # legacy top-level artifacts intentionally omitted — manifest reflects canonical batch/outputs/
    # only

    return {"artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output parity manifest")
    parser.add_argument("--outputs", default="batch/outputs")
    parser.add_argument("--batch", default="batch")
    parser.add_argument("--out", default="batch/outputs/manifest_expected.json")
    args = parser.parse_args()

    payload = build_manifest(Path(args.outputs), Path(args.batch))
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest with {len(payload['artifacts'])} entries: {out}")


if __name__ == "__main__":
    main()
