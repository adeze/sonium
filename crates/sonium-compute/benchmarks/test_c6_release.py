#!/usr/bin/env python3
"""Focused tests for the checked-in C6 release artifact contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = ROOT / "crates/sonium-compute/benchmarks/check_c6_release.py"
FIXTURE_PATH = ROOT / "crates/sonium-compute/benchmarks/artifacts/cuda-c6-release-v1.json"


def checker_module():
    spec = importlib.util.spec_from_file_location("check_c6_release", CHECKER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_checked_in_release_fixture_and_sidecar_pass():
    checker = checker_module()
    result = checker.check_release_document(fixture())
    assert result["status"] == "passed"
    assert result["canonical_payload_sha256"] == fixture()["canonical_payload_sha256"]
    assert checker.verify_release_sidecar(FIXTURE_PATH) == "41ada2893d7e4031e0579899e471aadef898eff97b1431880089a3d8e1c70827"


def test_runtime_observations_are_excluded_from_canonical_payload():
    checker = checker_module()
    mutated = copy.deepcopy(fixture())
    mutated["observations"]["bem"]["workload_elapsed_ms"] = 1
    mutated["observations"]["bem"]["throughput_flops_per_second"] = 2
    mutated["observations"]["bem"]["roofline_fraction"] = 3e-12
    assert checker.check_release_document(mutated)["status"] == "passed"


def test_stable_release_claim_mutation_is_rejected():
    checker = checker_module()
    mutated = copy.deepcopy(fixture())
    mutated["claims"]["energy"] = "measured"
    with pytest.raises(ValueError, match="energy must remain explicitly unclaimed"):
        checker.check_release_document(mutated)

