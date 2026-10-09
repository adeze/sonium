#!/usr/bin/env python3
"""Validate the checked-in C6 CUDA release evidence.

The CUDA producers intentionally include runtime measurements in their raw
reports.  Those values (timing, throughput, derived roofline, and the raw
report digest) are observations and are therefore kept out of the canonical
release payload.  This checker validates both parts independently:

* the checked-in release document has a stable schema, provenance, and
  canonical-payload digest;
* optional fresh raw producer reports pass the normal C6 evidence contract.

No vendor comparison or energy result is inferred from the bounded workload
reports.  The `observations` throughput and roofline figures are
bounded-workload accounting values (operation counts from `perf.rs` divided
by wall time on one QA workload), not representative kernel throughput, and
the roofline fraction uses the provisioned device peak without Nsight
attribution — do not quote them as utilization or whole-solver speedups.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

try:
    import summarize_cuda_evidence as raw_checker
except ModuleNotFoundError:  # Loaded by the focused test through importlib.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import summarize_cuda_evidence as raw_checker


SCHEMA = "sonium-compute-cuda-c6-release/v1"
RELEASE_FILENAME = "cuda-c6-release-v1.json"
VOLATILE_FIELDS = (
    "observations.*.workload_elapsed_ms",
    "observations.*.throughput_flops_per_second",
    "observations.*.roofline_fraction",
    "observations.*.raw_sha256",
    "observations.*.summary_sha256",
)
RAW_CASES = {
    "bem": (
        "sonium-compute-cuda-bem-evidence/v1",
        "bem.cuda-capability-and-assembly-matrix.v1",
    ),
    "fem": (
        "sonium-compute-cuda-fem-adjoint/v1",
        "compute.fem-adjoint-cuda.v1",
    ),
}
HARDWARE_FIELDS = (
    "device",
    "driver",
    "toolkit",
    "compiler",
    "topology",
    "peak_flops",
    "memory_bandwidth",
    "os",
    "target_features",
    "runtime_dispatch",
    "deterministic_mode",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> NoReturn:
    raise ValueError(f"C6 release artifact rejected: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the stable part of a release artifact.

    ``observations`` is deliberately removed as a whole.  Keeping the
    exclusion at one top-level boundary prevents a future timing field nested
    in a raw observation from silently changing the release digest.
    """

    return {
        key: value
        for key, value in artifact.items()
        if key not in {"observations", "canonical_payload_sha256"}
    }


def required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{field} must be non-empty text")
    return value


def positive_number(value: Any, field: str) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            fail(f"{field} must be numeric")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        fail(f"{field} must be finite and positive")
    return number


def nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{field} must be a non-negative integer")
    return value


def integral_measurement(value: Any, field: str) -> int:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            fail(f"{field} must be numeric")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        fail(f"{field} must be finite and non-negative")
    if not number.is_integer():
        fail(f"{field} must be integral")
    return int(number)


def check_observations(artifact: dict[str, Any]) -> None:
    observations = artifact.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(RAW_CASES):
        fail("observations must contain exactly BEM and FEM entries")
    for name in RAW_CASES:
        observation = observations[name]
        if not isinstance(observation, dict):
            fail(f"observations.{name} must be an object")
        for field in ("raw_sha256", "summary_sha256"):
            value = required_text(observation.get(field), f"observations.{name}.{field}")
            if not SHA256_RE.fullmatch(value):
                fail(f"observations.{name}.{field} is not a SHA-256 digest")
        positive_number(observation.get("workload_elapsed_ms"), f"observations.{name}.workload_elapsed_ms")
        positive_number(
            observation.get("throughput_flops_per_second"),
            f"observations.{name}.throughput_flops_per_second",
        )
        roofline = positive_number(
            observation.get("roofline_fraction"),
            f"observations.{name}.roofline_fraction",
        )
        if roofline > 1.0:
            fail(f"observations.{name}.roofline_fraction exceeds one")


def check_hardware(provenance: dict[str, Any]) -> None:
    hardware = provenance.get("reference_hardware")
    if not isinstance(hardware, dict):
        fail("reference_hardware provenance is missing")
    for field in HARDWARE_FIELDS:
        required_text(hardware.get(field), f"provenance.reference_hardware.{field}")
    positive_number(hardware["peak_flops"], "provenance.reference_hardware.peak_flops")
    positive_number(
        hardware["memory_bandwidth"],
        "provenance.reference_hardware.memory_bandwidth",
    )
    if hardware["os"] != "linux-x86_64":
        fail("reference hardware is not Linux x86_64")
    if hardware["runtime_dispatch"] != "cuda-native":
        fail("reference hardware is not native CUDA")
    if hardware["deterministic_mode"] != "enabled":
        fail("reference run is not marked deterministic")


def check_workloads(artifact: dict[str, Any]) -> None:
    workloads = artifact.get("workloads")
    if not isinstance(workloads, list) or any(not isinstance(item, dict) for item in workloads):
        fail("workloads must be an array of objects")
    if [item.get("id") for item in workloads] != list(RAW_CASES):
        fail("workloads must be ordered BEM then FEM")
    for workload in workloads:
        name = workload["id"]
        expected_schema, expected_case = RAW_CASES[name]
        if workload.get("raw_schema") != expected_schema:
            fail(f"workloads.{name}.raw_schema is invalid")
        if workload.get("case_id") != expected_case:
            fail(f"workloads.{name}.case_id is invalid")
        required_text(workload.get("producer"), f"workloads.{name}.producer")
        required_text(workload.get("raw_artifact"), f"workloads.{name}.raw_artifact")
        contract = workload.get("stable_contract")
        if not isinstance(contract, dict) or contract.get("status") != "passed":
            fail(f"workloads.{name}.stable_contract is not a passed contract")
        for field in (
            "operation_count",
            "kernel_launches",
            "library_operations",
            "h2d_operations",
            "d2h_operations",
            "allocation_high_water_bytes",
        ):
            nonnegative_integer(contract.get(field), f"workloads.{name}.stable_contract.{field}")
        if contract["operation_count"] <= 0:
            fail(f"workloads.{name}.stable_contract.operation_count must be positive")
        capability_counts = contract.get("capability_counts")
        if not isinstance(capability_counts, dict) or not capability_counts:
            fail(f"workloads.{name}.stable_contract.capability_counts is missing")
        for field, value in capability_counts.items():
            required_text(field, f"workloads.{name}.capability_counts key")
            nonnegative_integer(value, f"workloads.{name}.capability_counts.{field}")


def raw_stable_contract(artifact: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    runtime = artifact["runtime"]
    transfers = runtime.get("transfers") if isinstance(runtime.get("transfers"), dict) else runtime
    capabilities = {
        key: value
        for key, value in (artifact.get("environment") or {}).items()
        if key.startswith("capability.")
    }
    counts = {
        "native": sum(value == "native" for value in capabilities.values()),
        "native_real_equivalent": sum(
            value == "native_real_equivalent" for value in capabilities.values()
        ),
        "unsupported": sum(value == "unsupported" for value in capabilities.values()),
        "host_shared_fallback": sum(
            value == "host_shared_fallback" for value in capabilities.values()
        ),
        "native_or_supported": sum(
            value in {"native", "native_real_equivalent", "supported"}
            for value in capabilities.values()
        ),
    }
    return {
        "operation_count": integral_measurement(summary["operation_count"], "operation_count"),
        "kernel_launches": integral_measurement(summary["kernel_launches"], "kernel_launches"),
        "library_operations": integral_measurement(summary["library_operations"], "library_operations"),
        "h2d_operations": integral_measurement(transfers.get("h2d_operations", 0), "h2d_operations"),
        "d2h_operations": integral_measurement(transfers.get("d2h_operations", 0), "d2h_operations"),
        "allocation_high_water_bytes": integral_measurement(
            runtime["allocation_high_water_bytes"], "allocation_high_water_bytes"
        ),
        "capability_counts": counts,
    }


def check_raw_reports(raw_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, path in raw_paths.items():
        if name not in RAW_CASES:
            fail(f"unknown raw workload {name!r}")
        try:
            artifact = raw_checker.load(str(path))
            summary = raw_checker.validate(artifact)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            fail(f"raw {name} report failed the C6 contract: {error}")
        expected_schema, expected_case = RAW_CASES[name]
        if artifact.get("schema") != expected_schema or artifact.get("case_id") != expected_case:
            fail(f"raw {name} report identity does not match the release workload")
        hardware = artifact.get("hardware")
        if not isinstance(hardware, dict) or hardware.get("runtime_dispatch") != "cuda-native":
            fail(f"raw {name} report is not native CUDA")
        results[name] = {
            "path": path.as_posix(),
            "sha256": digest(path),
            "summary": summary,
            "stable_contract": raw_stable_contract(artifact, summary),
            "hardware": hardware,
        }
    return results


def check_release_document(
    artifact: dict[str, Any], raw_paths: dict[str, Path] | None = None
) -> dict[str, Any]:
    if artifact.get("schema") != SCHEMA or artifact.get("version") != 1:
        fail("schema/version is invalid")
    if artifact.get("status") != "passed":
        fail("status is not passed")
    required_text(artifact.get("scope"), "scope")
    claims = artifact.get("claims")
    if not isinstance(claims, dict):
        fail("claims are missing")
    if claims.get("vendor_comparison") != "not_claimed":
        fail("vendor comparison must remain explicitly unclaimed")
    if claims.get("energy") != "not_claimed":
        fail("energy must remain explicitly unclaimed")
    required_text(claims.get("roofline_basis"), "claims.roofline_basis")
    excluded = artifact.get("volatile_fields_excluded")
    if excluded != list(VOLATILE_FIELDS):
        fail("volatile field exclusion list is not the checked contract")
    check_hardware(artifact.get("provenance") or {})
    check_workloads(artifact)
    check_observations(artifact)
    microkernel = artifact.get("microkernel_roofline")
    if not isinstance(microkernel, dict):
        fail("microkernel_roofline is missing")
    if microkernel.get("status") != "accounting_only" or microkernel.get("measured") is not False:
        fail("microkernel roofline must remain accounting-only")
    required_text(microkernel.get("source"), "microkernel_roofline.source")
    models = microkernel.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(item, str) for item in models):
        fail("microkernel_roofline.models must be a non-empty list")
    canonical_hash = artifact.get("canonical_payload_sha256")
    if not isinstance(canonical_hash, str) or not SHA256_RE.fullmatch(canonical_hash):
        fail("canonical_payload_sha256 is invalid")
    actual_hash = hashlib.sha256(canonical_bytes(canonical_payload(artifact))).hexdigest()
    if actual_hash != canonical_hash:
        fail("canonical payload checksum is stale")
    raw_results = check_raw_reports(raw_paths or {}) if raw_paths else {}
    if raw_results:
        expected_hardware = artifact["provenance"]["reference_hardware"]
        workload_by_id = {workload["id"]: workload for workload in artifact["workloads"]}
        for name, result in raw_results.items():
            contract = workload_by_id[name]["stable_contract"]
            actual = result["stable_contract"]
            for field in (
                "operation_count",
                "kernel_launches",
                "library_operations",
                "h2d_operations",
                "d2h_operations",
                "allocation_high_water_bytes",
            ):
                if actual[field] != contract[field]:
                    fail(f"raw {name} report differs from its stable release contract: {field}")
            expected_counts = contract["capability_counts"]
            if any(actual["capability_counts"].get(key, 0) != value for key, value in expected_counts.items()):
                fail(f"raw {name} capability counts differ from the stable release contract")
            for field in HARDWARE_FIELDS:
                if str(result["hardware"].get(field)) != str(expected_hardware[field]):
                    fail(f"raw {name} hardware differs from the stable release provenance: {field}")
    return {
        "status": "passed",
        "schema": SCHEMA,
        "canonical_payload_sha256": actual_hash,
        "raw_reports": raw_results,
        "volatile_fields_excluded": list(VOLATILE_FIELDS),
    }


def verify_release_sidecar(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"release checksum sidecar is missing or unreadable: {error}")
    matches = [row.split() for row in rows if len(row.split()) == 2 and row.split()[1] == path.name]
    if len(matches) != 1 or not SHA256_RE.fullmatch(matches[0][0].lower()):
        fail("release checksum sidecar must contain exactly one valid digest")
    actual = digest(path)
    if actual != matches[0][0].lower():
        fail("release artifact checksum does not match its sidecar")
    return actual


def load_release(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read release artifact: {error}")
    if not isinstance(value, dict):
        fail("release artifact must be a JSON object")
    return value


def main() -> int:
    if len(sys.argv) not in {2, 4}:
        print(f"usage: {sys.argv[0]} RELEASE.json [BEM_RAW.json FEM_RAW.json]", file=sys.stderr)
        return 2
    release_path = Path(sys.argv[1])
    try:
        release_digest = verify_release_sidecar(release_path)
        raw_paths = (
            {"bem": Path(sys.argv[2]), "fem": Path(sys.argv[3])}
            if len(sys.argv) == 4
            else {}
        )
        result = check_release_document(load_release(release_path), raw_paths)
        result["release_sha256"] = release_digest
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
