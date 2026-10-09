#!/usr/bin/env python3
"""Strict checker for the native Linux x86 C5 sanitizer corpus."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


POSITIVE_CASES = tuple(
    f"spmv-row-{scalar}-len-{length}"
    for length in (0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33)
    for scalar in ("f32", "f64")
)
NEGATIVE_CASES = (
    ("gemv-invalid-lda", "gemv", "InvalidArgument"),
    ("gemm-invalid-ldc", "gemm", "InvalidArgument"),
    ("prebake-nonmonotone-row-pointer", "csr_row_buckets", "InvalidArgument"),
    ("spmv-nonfinite-input", "spmv_prebaked", "OperationFailed"),
    ("spmm-invalid-input-stride", "spmm_prebaked", "InvalidArgument"),
    ("gather-out-of-range-index", "gather4_weighted", "InvalidArgument"),
    ("cast-short-input", "copy_h2d_cast", "InvalidArgument"),
    ("bem-degenerate-triangle", "bem_regular_pair_c64_batch", "InvalidArgument"),
    ("bem-nonfinite-wave-number", "bem_regular_pair_c64_batch", "OperationFailed"),
    ("view-out-of-bounds", "buffer_view", "InvalidArgument"),
)
EXPECTED_KEYS = {
    "schema",
    "artifact_kind",
    "fixture_version",
    "deterministic_fixture",
    "status",
    "target_arch",
    "target_os",
    "cpu_model",
    "microcode",
    "compiler",
    "thread_policy",
    "affinity_policy",
    "numa_policy",
    "runtime_avx2",
    "runtime_fma",
    "sanitizer",
    "sanitizer_options",
    "leak_detection",
    "positive_case_count",
    "positive_cases",
    "negative_case_count",
    "negative_cases",
    "cases_failed",
    "corpus_sha256",
    "unsupported_scope",
}
NEGATIVE_KEYS = {
    "id",
    "operation",
    "expected_error_kind",
    "observed_error_kind",
    "observed_operation",
    "destination_preserved",
    "retry_succeeded",
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"C5 sanitizer artifact rejected: {message}")


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() in {
        "",
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} provenance is missing")
    return value


def integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        fail(f"{label} is not a non-negative integer")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    sidecar = Path(f"{path}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{sidecar} is missing or unreadable: {error}")
    expected = next(
        (
            fields[0].lower()
            for row in rows
            if len(fields := row.split()) == 2 and fields[1].lstrip("*") == path.name
        ),
        None,
    )
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"{sidecar} has no valid digest for {path.name}")
    actual = digest(path)
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")
    return value


def check_parity(artifact: dict) -> dict:
    if artifact.get("schema") != 2 or artifact.get("status") != "pass":
        fail("native parity schema/status is invalid")
    if artifact.get("target_arch") != "x86_64" or artifact.get("os") != "linux-x86_64":
        fail("native parity is not Linux x86_64")
    if artifact.get("selected_kernel") != "avx2-fma" or artifact.get("force_scalar") is not False:
        fail("native parity did not select AVX2/FMA")
    if artifact.get("cases_passed") != 5 or artifact.get("cases_skipped") != 0 or artifact.get("cases_failed") != 0:
        fail("native parity case counts are incomplete")
    observables = artifact.get("observables")
    if not isinstance(observables, list) or len(observables) != 1634:
        fail("native parity observable corpus is incomplete")
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in observables):
        fail("native parity contains a non-finite observable")
    features = text(artifact.get("features"), "native parity.features")
    for token in ("avx2=true", "fma=true", "force_scalar=false"):
        if token not in features:
            fail(f"native parity lacks {token}")
    for key in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy", "deterministic_mode"):
        text(artifact.get(key), f"native parity.{key}")
    return {
        "observable_count": len(observables),
        "max_norm_error": artifact.get("max_norm_error"),
        "sha256": None,
    }


def corpus_digest(positive: list[str], negative: list[dict]) -> str:
    payload = json.dumps([positive, negative], separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def check_sanitizer(artifact: dict) -> dict:
    if set(artifact) != EXPECTED_KEYS:
        fail(f"sanitizer artifact keys differ: missing={EXPECTED_KEYS - set(artifact)}, extra={set(artifact) - EXPECTED_KEYS}")
    if artifact.get("schema") != 1 or artifact.get("artifact_kind") != "sonium-compute.c5.native-x86-sanitizer-negative":
        fail("sanitizer artifact identity is invalid")
    if artifact.get("fixture_version") != 1 or artifact.get("deterministic_fixture") != "c5-native-x86-sanitizer-negative-v1":
        fail("sanitizer fixture identity is invalid")
    if artifact.get("status") != "passed" or artifact.get("target_arch") != "x86_64" or artifact.get("target_os") != "linux":
        fail("sanitizer status/target is invalid")
    if artifact.get("runtime_avx2") is not True or artifact.get("runtime_fma") is not True:
        fail("sanitizer runtime feature gate is incomplete")
    if artifact.get("sanitizer") != "address" or artifact.get("leak_detection") != "disabled_by_runner_in_traced_container":
        fail("sanitizer mode or leak policy is invalid")
    options = text(artifact.get("sanitizer_options"), "sanitizer_options")
    if options != "ASAN_OPTIONS=detect_leaks=0;LSAN_OPTIONS=detect_leaks=0":
        fail("sanitizer options are not the pinned leak-safe runner contract")
    for key in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy", "unsupported_scope"):
        text(artifact.get(key), f"sanitizer.{key}")
    if artifact["thread_policy"] != "single" or artifact["affinity_policy"] != "unbound" or artifact["numa_policy"] != "default":
        fail("sanitizer execution policy is not pinned")
    positive = artifact.get("positive_cases")
    if artifact.get("positive_case_count") != len(POSITIVE_CASES) or positive != list(POSITIVE_CASES):
        fail("sanitizer positive tail corpus is incomplete or reordered")
    negative = artifact.get("negative_cases")
    if artifact.get("negative_case_count") != len(NEGATIVE_CASES) or not isinstance(negative, list):
        fail("sanitizer negative corpus count is invalid")
    if [probe.get("id") for probe in negative] != [case[0] for case in NEGATIVE_CASES]:
        fail("sanitizer negative corpus IDs are incomplete or reordered")
    for index, (probe, (case_id, operation, kind)) in enumerate(zip(negative, NEGATIVE_CASES)):
        if set(probe) != NEGATIVE_KEYS:
            fail(f"negative probe {index} fields are incomplete")
        if probe.get("id") != case_id or probe.get("operation") != operation or probe.get("expected_error_kind") != kind:
            fail(f"negative probe {index} identity is invalid")
        if probe.get("observed_error_kind") != kind or probe.get("observed_operation") != operation:
            fail(f"negative probe {index} did not preserve the typed operation")
        if probe.get("destination_preserved") is not True or probe.get("retry_succeeded") is not True:
            fail(f"negative probe {index} lacks preservation/retry proof")
    if artifact.get("cases_failed") != 0:
        fail("sanitizer corpus reports failed cases")
    corpus = text(artifact.get("corpus_sha256"), "corpus_sha256").lower()
    if len(corpus) != 64 or any(c not in "0123456789abcdef" for c in corpus):
        fail("sanitizer corpus digest is malformed")
    if corpus != corpus_digest(positive, negative):
        fail("sanitizer corpus digest does not match deterministic rows")
    return {
        "positive_case_count": len(positive),
        "negative_case_count": len(negative),
        "corpus_sha256": corpus,
        "sha256": None,
    }


def main() -> None:
    if len(sys.argv) not in {3, 5} or (len(sys.argv) == 5 and sys.argv[3] != "--output"):
        raise SystemExit(f"usage: {sys.argv[0]} PARITY.json SANITIZER.json [--output SUMMARY.json]")
    parity_path = Path(sys.argv[1])
    sanitizer_path = Path(sys.argv[2])
    parity = load(parity_path)
    sanitizer = load(sanitizer_path)
    parity_summary = check_parity(parity)
    sanitizer_summary = check_sanitizer(sanitizer)
    for key in ("cpu_model", "microcode", "compiler", "thread_policy", "affinity_policy", "numa_policy"):
        if parity.get(key) != sanitizer.get(key):
            fail(f"parity/sanitizer provenance differs for {key}")
    parity_summary["sha256"] = digest(parity_path)
    sanitizer_summary["sha256"] = digest(sanitizer_path)
    summary = {
        "schema": "sonium-compute.c5.native-x86-sanitizer-comparison/v1",
        "status": "passed",
        "target_arch": "x86_64",
        "target_os": "linux",
        "parity": parity_summary,
        "sanitizer": sanitizer_summary,
        "negative_paths": "typed_rejection;destination_preserved;retry_succeeds",
        "vendor_baseline": "not_applicable_to_sanitizer_corpus",
        "unsupported_scope": "no AVX512, Arm64, Windows, external CI, multi-socket, or vendor claim",
    }
    if len(sys.argv) == 5:
        output = Path(sys.argv[4])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
