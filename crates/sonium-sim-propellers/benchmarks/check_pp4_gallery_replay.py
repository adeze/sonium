#!/usr/bin/env python3
"""Strict checker for native PP4 production-gallery replay evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


EXPECTED_FILES = {
    "gallery.json",
    "execution_evidence.json",
    "optimized_blade.vtu",
    "baseline_blade.vtu",
    "source_spectrum.csv",
    "blade_load_distribution.csv",
    "optimized-blade-loading.png",
    "axial-wake-proxy.png",
    "spectrum.svg",
    "thrust-versus-operating-point.svg",
    "torque-versus-operating-point.svg",
    "blade-load-distribution.svg",
    "conventional-toroidal-comparison.svg",
    "rotation.gif",
}
FIGURES = {
    "source_spectrum.csv",
    "blade_load_distribution.csv",
    "optimized-blade-loading.png",
    "axial-wake-proxy.png",
    "spectrum.svg",
    "thrust-versus-operating-point.svg",
    "torque-versus-operating-point.svg",
    "blade-load-distribution.svg",
    "conventional-toroidal-comparison.svg",
    "rotation.gif",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def check_gallery(root: Path) -> None:
    assert root.is_dir()
    names = {entry.name for entry in root.iterdir() if entry.is_file()}
    assert names == EXPECTED_FILES
    manifest = json.loads((root / "gallery.json").read_text())
    assert manifest["schema_version"] == 1
    assert (manifest["width_px"], manifest["height_px"]) == (1920, 1080)
    assert {figure["filename"] for figure in manifest["figures"]} == FIGURES
    assert all("fidelity=PanelGeometrySurrogate" in figure["caption"] for figure in manifest["figures"])
    for figure in manifest["figures"]:
        path = root / figure["filename"]
        assert path.is_file() and path.stat().st_size > 0
    evidence = json.loads((root / "execution_evidence.json").read_text())
    assert evidence["schema"] == "sonium.propeller.execution-evidence.v1"
    checksums = evidence["artifact_checksums_sha256"]
    assert set(checksums) == EXPECTED_FILES - {"execution_evidence.json"}
    for name, expected in checksums.items():
        assert safe_relative(name)
        assert len(expected) == 64
        assert digest(root / name) == expected


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact_path = Path(sys.argv[1]).resolve()
    root = artifact_path.parent
    artifact = json.loads(artifact_path.read_text())

    assert artifact["schema"] == "sonium.propellers.pp4-gallery-replay-evidence.v1"
    assert artifact["status"] == "pass"
    assert artifact["scope"] == "native Linux x86_64 production CLI CPU/f64; Windows/Arm64/AVX512 waived"
    assert artifact["claim_class"] == "native production gallery byte-identical clean replay"
    assert "two clean invocations" in artifact["reference"]
    canonical_input = artifact["canonical_input"]
    assert isinstance(canonical_input, str) and canonical_input
    assert hashlib.sha256(canonical_input.encode()).hexdigest() == artifact["canonical_input_sha256"]
    assert artifact["run_count"] == 2
    assert artifact["deterministic"] is True
    assert artifact["excluded_files"] == []

    first = root / "replay-one"
    second = root / "replay-two"
    check_gallery(first)
    check_gallery(second)
    files = artifact["files"]
    assert len(files) == len(EXPECTED_FILES)
    assert {record["path"] for record in files} == EXPECTED_FILES
    for record in files:
        name = record["path"]
        assert safe_relative(name) and name in EXPECTED_FILES
        first_path = first / name
        second_path = second / name
        first_sha256 = digest(first_path)
        second_sha256 = digest(second_path)
        first_bytes = first_path.stat().st_size
        second_bytes = second_path.stat().st_size
        assert record["first_sha256"] == first_sha256
        assert record["second_sha256"] == second_sha256
        assert record["first_bytes"] == first_bytes
        assert record["second_bytes"] == second_bytes
        assert record["first_sha256"] == record["second_sha256"]
        assert record["first_bytes"] == record["second_bytes"]
        assert record["byte_identical"] is True

    canonical_output = artifact["canonical_output"]
    assert isinstance(canonical_output, str) and canonical_output
    assert hashlib.sha256(canonical_output.encode()).hexdigest() == artifact["output_sha256"]
    assert json.loads(canonical_output) == files
    checksum_lines = [line for line in (root / "checksums.sha256").read_text().splitlines() if line]
    assert len(checksum_lines) == 1
    expected_sha256, name = checksum_lines[0].split("  ", 1)
    assert name == artifact_path.name
    assert expected_sha256 == digest(artifact_path)
    print(json.dumps({"files": len(files), "runs": 2, "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
