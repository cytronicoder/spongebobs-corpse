import json
from pathlib import Path


def test_manifest_expected_has_artifacts() -> None:
    manifest = Path("batch/outputs/manifest_expected.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert payload["artifacts"]
    assert all("path" in item for item in payload["artifacts"])
