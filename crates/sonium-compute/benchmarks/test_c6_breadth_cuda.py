"""Focused tamper tests for the strict C6 breadth CUDA checker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = ROOT / "crates/sonium-compute/benchmarks/check_c6_breadth_cuda.py"
ARTIFACT_PATH = ROOT / ".target-c6-breadth-fresh/artifact/cuda_c6_breadth_bem_quadrature.json"
spec = importlib.util.spec_from_file_location("check_c6_breadth_cuda", CHECKER_PATH)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def artifact_copy(name: str) -> tuple[Path, dict]:
    if not ARTIFACT_PATH.is_file():
        pytest.skip("native C6 breadth artifact has not been produced yet")
    path = ARTIFACT_PATH.parent / f"test-{name}.json"
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path, artifact


def test_native_artifact_passes_strict_checker() -> None:
    if not ARTIFACT_PATH.is_file():
        pytest.skip("native C6 breadth artifact has not been produced yet")
    result = checker.check(ARTIFACT_PATH)
    assert result["status"] == "passed"
    assert [row["id"] for row in result["tiers"]] == ["small_odd", "medium_odd", "large_odd"]


def test_timing_median_tamper_is_rejected() -> None:
    path, artifact = artifact_copy("median-tamper")
    try:
        artifact["rows"][0]["cuda"]["median_ms"] *= 1.25
        path.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(ValueError, match="median_ms"):
            checker.check(path)
    finally:
        path.unlink(missing_ok=True)


def test_roofline_claim_tamper_is_rejected() -> None:
    path, artifact = artifact_copy("roofline-tamper")
    try:
        artifact["claims"]["roofline"] = "measured"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(ValueError, match="roofline"):
            checker.check(path)
    finally:
        path.unlink(missing_ok=True)
