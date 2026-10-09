#!/usr/bin/env python3
"""Strictly validate the host-applicable PP4 release fan-in bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "sonium.propellers.pp4-release-fanin.v1"
SCOPE = "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
SHARDS = {
    "qa": ("sonium.propellers.pp4-qa-fan-in.v1", "gated", "check_pp4_qa.py"),
    "release": ("sonium.propellers.native-x86-pp4.v1", "gated", "check_pp4_release.py"),
    "gallery-replay": (
        "sonium.propellers.pp4-gallery-replay-evidence.v1",
        "pass",
        "check_pp4_gallery_replay.py",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def check_manifest(root: Path, artifact: Path, expected_summary: Path | None) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert len(expected) == 64 and safe(name)
        assert name not in entries
        assert (root / name).is_file()
        assert digest(root / name) == expected
        entries[name] = expected
    assert artifact.name in entries
    if expected_summary is not None:
        assert expected_summary.name in entries
    return entries


def run_nested_checker(root: Path, path: Path, checker: str) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name(checker)), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, f"{checker} failed: {result.stdout}{result.stderr}"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FANIN_JSON")
    artifact = Path(sys.argv[1]).resolve()
    root = artifact.parent
    record = json.loads(artifact.read_text())
    assert record["schema"] == SCHEMA
    assert record["status"] == "gated"
    assert record["scope"] == SCOPE
    assert record["claim_class"] == "host-applicable PP4 shard fan-in; no external/full-fidelity promotion"
    assert record["promotion_eligible"] is False

    inventory = record["inventory"]
    assert inventory["source"] == "crates/sonium-qa/validation-manifest.toml"
    assert len(inventory["manifest_sha256"]) == 64
    assert inventory["registered_case_count"] == 29
    assert len(inventory["registered_case_ids"]) == 29
    assert all(case.startswith("propeller.") for case in inventory["registered_case_ids"])
    assert inventory["executed"] == 29
    assert inventory["passed"] == 29
    assert inventory["unsupported"] == 0

    shards = record["shards"]
    assert len(shards) == len(SHARDS)
    seen: set[str] = set()
    for shard in shards:
        shard_id = shard["id"]
        assert shard_id in SHARDS and shard_id not in seen
        seen.add(shard_id)
        expected_schema, expected_status, checker = SHARDS[shard_id]
        assert shard["schema"] == expected_schema
        assert shard["status"] == expected_status
        path_name = shard["path"]
        assert safe(path_name)
        shard_path = root / path_name
        assert shard_path.is_file()
        assert shard_path.parent.name in {"qa", "release", "gallery-replay"}
        run_nested_checker(root, shard_path, checker)
    assert seen == set(SHARDS)

    gate = record["final_release_gate"]
    assert gate["status"] == "blocked"
    assert gate["official_recipe"] == "just qa-release-critical"
    assert "PP2 and PP3 remain partial" in gate["reason"]
    assert gate["physical_claim"] == "not promoted"
    assert set(gate["blocked_lanes"]) == {
        "PP2 external/full-fidelity CUDA propeller validation",
        "PP3 external/full-fidelity resolved-flow/aeroacoustic validation",
    }

    checksums = record["artifact_checksums_sha256"]
    assert checksums
    for name, expected in checksums.items():
        assert safe(name) and len(expected) == 64
        assert (root / name).is_file()
        assert digest(root / name) == expected
    summary_path = root / "pp4-fanin-checker.json"
    summary = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        assert summary["schema"] == "sonium.propellers.pp4-release-fanin-checker.v1"
        assert summary["status"] == "pass"
        assert summary["artifact_sha256"] == digest(artifact)
    entries = check_manifest(root, artifact, summary_path if summary is not None else None)
    expected_entries = set(checksums) | {artifact.name}
    if summary is not None:
        expected_entries.add(summary_path.name)
    assert set(entries) == expected_entries
    assert entries[artifact.name] == digest(artifact)

    print(
        json.dumps(
            {
                "status": "pass",
                "registered": 29,
                "shards": sorted(seen),
                "promotion": "blocked-PP2-PP3",
                "manifest_entries": len(entries),
                "finalized": summary is not None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
