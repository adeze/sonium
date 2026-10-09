#!/usr/bin/env python3
"""Validate the propeller-only PP4 QA shard archive."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FAN_IN_JSON")
    evidence_path = Path(sys.argv[1]).resolve()
    root = evidence_path.parent
    evidence = json.loads(evidence_path.read_text())
    assert evidence["schema"] == "sonium.propellers.pp4-qa-fan-in.v1"
    assert evidence["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
    assert evidence["registered_case_count"] == 29
    assert len(evidence["registered_case_ids"]) == 29
    assert all(case.startswith("propeller.") for case in evidence["registered_case_ids"])
    assert evidence["coverage"]["executed"] == 29
    assert evidence["coverage"]["unsupported"] == 0
    assert evidence["release_join"]["status"] == "blocked"
    assert evidence["release_join"]["official_recipe"] == "just qa-release-critical"
    assert "PP2--PP3 remain partial" in evidence["release_join"]["reason"]

    results = evidence["case_results"]
    assert len(results) == 29
    assert {result.get("case_id") for result in results} == set(evidence["registered_case_ids"])
    for result in results:
        assert result["status"] == "passed"
        assert result["case_id"] in evidence["registered_case_ids"]
        for metric in result.get("metrics", {}).values():
            if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                assert math.isfinite(metric)

    checksums = evidence["artifact_checksums_sha256"]
    assert checksums
    for name, expected in checksums.items():
        assert safe(name)
        assert len(expected) == 64
        assert digest(root / name) == expected
    checksum_file = root / "checksums.sha256"
    parsed = {}
    for line in checksum_file.read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert safe(name) and len(expected) == 64
        parsed[name] = expected
        assert digest(root / name) == expected
    assert "qa-propeller-fan-in.json" in parsed
    print(json.dumps({"registered": 29, "passed": 29, "release_join": "blocked", "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
