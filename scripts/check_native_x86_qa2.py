#!/usr/bin/env python3
"""Build the native Linux/x86 QA2 fan-in without inventing missing evidence.

The producer lanes intentionally use different artifact schemas.  This
checker preserves each producer artifact, records its SHA-256 digest, and
normalizes only the fan-in status and host metadata.  A missing or failed
required producer keeps the fan-in ineligible; it is never represented as a
skip-as-pass result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def cpu_info() -> tuple[str, str, str]:
    vendor = model = "unknown"
    flags = ""
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.strip()
        if key.strip() == "vendor_id" and vendor == "unknown":
            vendor = value
        elif key.strip() == "model name" and model == "unknown":
            model = value
        elif key.strip() in {"flags", "Features"} and not flags:
            flags = value
    return vendor, model, flags


def compiler() -> str:
    try:
        completed = subprocess.run(
            ["rustc", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def microcode() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/microcode/version")
    try:
        return path.read_text().strip() or "unavailable"
    except OSError:
        return "unavailable"


def validate_fastmath(docs: list[dict[str, Any]]) -> tuple[str, str]:
    native, scalar, gate = docs
    if native.get("correctness_gate") != "passed" or scalar.get("correctness_gate") != "passed":
        return "failed", "FastMath correctness gate did not pass"
    if gate.get("status") != "pass":
        failures = gate.get("failures") or ["statistical gate failed"]
        return "failed", "FastMath statistical gate: " + "; ".join(map(str, failures))
    return "passed", "native AVX2 and forced-scalar accuracy plus timing gate"


def validate_fft(docs: list[dict[str, Any]]) -> tuple[str, str]:
    artifact = docs[0]
    rows = artifact.get("rows")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("architecture") != "x86_64"
        or artifact.get("status") != "passed"
        or artifact.get("avx2_available") is not True
        or not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) and row.get("passed") is True for row in rows)
    ):
        return "failed", "FFT runtime dispatch artifact failed its AVX2/row gates"
    return "passed", "scalar/AVX2 Complex32/Complex64 forward/inverse matrix"


def validate_compute(docs: list[dict[str, Any]]) -> tuple[str, str]:
    native, scalar = docs
    if native.get("status") != "pass" or native.get("selected_kernel") != "avx2-fma":
        return "failed", "compute native artifact is not a passing AVX2/FMA result"
    if scalar.get("status") != "scalar-baseline" or scalar.get("selected_kernel") != "scalar-fallback":
        return "failed", "compute scalar artifact is not explicitly labelled scalar"
    if any(doc.get("cases_failed") != 0 or len(doc.get("observables", [])) != 1634 for doc in docs):
        return "failed", "compute case or observable count gate failed"
    return "passed", "native AVX2/FMA and forced-scalar BLAS/SpMV/BEM parity"


def validate_sparse(docs: list[dict[str, Any]]) -> tuple[str, str]:
    native, scalar, thresholds = docs
    if thresholds.get("status") != "passed":
        return "failed", "sparse planner threshold artifact is not passing"
    for document in (native, scalar):
        hardware = document.get("hardware") or {}
        if hardware.get("arch") != "x86_64" or hardware.get("os") != "linux":
            return "failed", "sparse artifact has unexpected host identity"
        if not document.get("runs") or any(
            run.get("correctness_status") != "passed" for run in document["runs"]
        ):
            return "failed", "sparse action matrix contains a non-passing correctness row"
    return "passed", "CSR/compact/row-bucketed/SELL/BSR/NUMA action and threshold records"


def validate_fem(docs: list[dict[str, Any]]) -> tuple[str, str]:
    artifact = docs[0]
    target = artifact.get("target") or {}
    runtime = target.get("runtime_features") or {}
    if artifact.get("status") != "pass" or artifact.get("scope") != "native-linux-x86_64":
        return "failed", "FEM native artifact is not a passing Linux/x86 result"
    if runtime.get("avx2") is not True or runtime.get("fma") is not True:
        return "failed", "FEM native artifact does not record AVX2/FMA"
    if not all(case.get("status") == "pass" for case in artifact.get("cases", [])):
        return "failed", "FEM reference case failed"
    return "passed", "portable FEM reference, assembly/BC/interpolation budgets"


def validate_room(docs: list[dict[str, Any]]) -> tuple[str, str]:
    native, scalar, check = docs
    if native.get("host_architecture") != "x86_64" or scalar.get("host_architecture") != "x86_64":
        return "failed", "room artifact host architecture mismatch"
    if native.get("cross_architecture", {}).get("status") != "waived":
        return "failed", "room cross-architecture waiver is not explicit"
    if check.get("status") != "pass":
        return "failed", "room artifact checker did not pass"
    return "passed", "native/scalar room topology, field, residual, reciprocity, and ledger gates"


def validate_propeller_pp3(docs: list[dict[str, Any]]) -> tuple[str, str]:
    artifact = docs[0]
    if artifact.get("status") != "pass" or not artifact.get("cases"):
        return "failed", "PP3 fidelity artifact is not passing"
    return "passed", "bounded native-x86 fidelity/reference ladder"


def validate_propeller_pp1(docs: list[dict[str, Any]]) -> tuple[str, str]:
    if len(docs) != 2:
        return "failed", "PP1 requires both single-worker and configured-Rayon artifacts"
    workers: set[int] = set()
    for artifact in docs:
        if (
            artifact.get("schema") != "sonium.propellers.native-x86-pp1.v1"
            or artifact.get("status") != "pass"
            or not artifact.get("scope", "").startswith("native Linux x86_64")
            or artifact.get("deterministic_mode") is not True
        ):
            return "failed", "PP1 artifact schema/status/determinism gate failed"
        execution = artifact.get("execution") or {}
        if (
            execution.get("requested_backend") != "cpu"
            or execution.get("actual_backend") != "cpu"
            or execution.get("precision") != "f64"
            or execution.get("scalar_or_simd") != "scalar"
            or execution.get("cancelled") is not False
            or execution.get("device_lost") is not False
            or execution.get("allocation_failures") != 0
            or len(execution.get("input_checksum_sha256", "")) != 64
            or len(execution.get("output_checksum_sha256", "")) != 64
        ):
            return "failed", "PP1 CPU/f64 execution provenance or failure counters failed"
        host = execution.get("host") or {}
        if host.get("os") != "linux" or not host.get("target", "").startswith("x86_64-"):
            return "failed", "PP1 host provenance is not native Linux/x86_64"
        workers.add(host.get("rayon_threads"))
        cases = artifact.get("cases")
        if not isinstance(cases, list) or len(cases) != 12 or any(
            not isinstance(case, dict) or case.get("status") != "pass" for case in cases
        ):
            return "failed", "PP1 case matrix is incomplete or contains a failed row"
        cross_architecture = artifact.get("cross_architecture") or {}
        if cross_architecture.get("status") != "waived":
            return "failed", "PP1 cross-architecture waiver is not explicit"
        capability = artifact.get("capability") or {}
        if capability.get("total_cells") != 648 or capability.get("device_requests_without_fallback") is not True:
            return "failed", "PP1 capability boundary is incomplete"
    if workers != {1, 2}:
        return "failed", "PP1 worker-count replay does not cover 1 and 2 Rayon workers"
    return "passed", "single-worker/configured-Rayon native CPU/f64 case and capability matrix"


def validate_speaker_modeling_sm3(docs: list[dict[str, Any]]) -> tuple[str, str]:
    if len(docs) != 1:
        return "failed", "SM3 requires one combined native modeling artifact"
    artifact = docs[0]
    if (
        artifact.get("schema") != "sonium.speaker-modeling.native-x86-sm3.v1"
        or artifact.get("status") != "pass"
        or not artifact.get("scope", "").startswith("native Linux x86_64")
    ):
        return "failed", "SM3 artifact schema/status/scope gate failed"
    producer = artifact.get("producer") or {}
    if (
        producer.get("schema_version") != 1
        or producer.get("architecture") != "x86_64"
        or producer.get("backend") != "cpu"
        or not producer.get("cpu_vendor")
        or not producer.get("cpu_model")
        or not producer.get("compiler", "").startswith("rustc ")
        or len(producer.get("driver_artifact_sha256", "")) != 64
    ):
        return "failed", "SM3 native CPU provenance or DriverModel checksum is incomplete"
    features = producer.get("simd_features_available") or {}
    if not all(isinstance(features.get(key), bool) for key in ("sse2", "avx2", "fma")):
        return "failed", "SM3 SIMD feature metadata is incomplete"
    axis = producer.get("axisymmetric") or {}
    sector = producer.get("sector") or {}
    monolithic = producer.get("monolithic") or {}
    if axis.get("field_dofs", 0) <= 0 or sector.get("field_dofs", 0) <= 0 or sector.get("mode_field_dofs", 0) <= 0:
        return "failed", "SM3 axisymmetric or sector field counts are empty"
    if monolithic.get("interior_field_dofs", 0) <= 0 or monolithic.get("aperture_field_dofs", 0) <= 0:
        return "failed", "SM3 coupled FEM--BEM field counts are empty"
    numeric_values = [
        axis.get("input_power_w"),
        axis.get("power_residual_w"),
        sector.get("eddy_loss_w"),
        sector.get("periodic_residual"),
        monolithic.get("relative_residual"),
        monolithic.get("interface_residual"),
    ]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric_values):
        return "failed", "SM3 physical or residual measurements are non-finite"
    replay = artifact.get("replay") or {}
    if replay.get("status") != "pass" or replay.get("meaningful_drift_rejected") is not True:
        return "failed", "SM3 replay policy does not reject meaningful drift"
    cross_architecture = artifact.get("cross_architecture") or {}
    if cross_architecture.get("status") != "waived":
        return "failed", "SM3 cross-architecture waiver is not explicit"
    return "passed", "DriverModel/axisymmetric/sector/FEM--BEM native replay artifact"


def validate_propeller_pp4(docs: list[dict[str, Any]]) -> tuple[str, str]:
    artifact = docs[0]
    if artifact.get("status") != "gated":
        return "failed", "PP4 artifact has unexpected status"
    capability = artifact.get("capability") or {}
    if (
        capability.get("renderer") != "sonium_xem::io::GalleryRenderer"
        or capability.get("renderer_status") != "available"
        or not capability.get("locale")
        or "locale-independent" not in capability.get("locale_policy", "")
        or capability.get("pre_publication_hook")
        != "SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION"
        or capability.get("renderer_fault_hook")
        != "SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE"
        or capability.get("fault_directory") != ".target-unix-pp4-faults"
        or "/tmp" in capability.get("fault_directory", "")
    ):
        return "failed", "PP4 renderer/locale fault capability metadata is incomplete"
    probes = artifact.get("failure_probes") or []
    required_faults = {
        "child-process.pre-publication-interruption",
        "child-process.renderer-unavailable",
    }
    fault_probes = {probe.get("id"): probe for probe in probes if probe.get("id") in required_faults}
    if set(fault_probes) != required_faults:
        return "failed", "PP4 child-process fault probes are incomplete"
    for probe in fault_probes.values():
        if (
            not isinstance(probe.get("exit_code"), int)
            or isinstance(probe.get("exit_code"), bool)
            or probe["exit_code"] == 0
            or "ErrorKind::" not in probe.get("typed_error", "")
            or probe.get("sentinel_preserved") is not True
            or probe.get("retry_succeeded") is not True
            or probe.get("published_artifact_count") != 0
            or probe.get("partial_publish") is not False
        ):
            return "failed", "PP4 child-process fault probe did not fail closed and retry"
    return "partially-implemented", str(artifact.get("final_release_gate", "PP4 release gate remains blocked"))


def validate_speaker(docs: list[dict[str, Any]]) -> tuple[str, str]:
    artifact = docs[0]
    if (
        artifact.get("schema_version") != 3
        or artifact.get("target_arch") != "x86_64"
        or len(artifact.get("directivity", [])) != 2
        or artifact.get("driver_model_reference", {}).get("schema_version") != 2
        or artifact.get("projected_driver_model_reference", {}).get("schema_version") != 2
    ):
        return "failed", "speaker schema or shape gate failed"
    dispatch = artifact.get("dispatch", [])
    required = {"scalar=f64", "avx2=true", "fma=true"}
    if not required.issubset(dispatch) or not any(
        value.startswith("vendor=") and value[7:].strip() and value[7:].strip().lower() != "unknown"
        for value in dispatch
    ):
        return "failed", "speaker scalar/AVX/vendor provenance is incomplete"
    return "passed", "CEA/directivity/configuration/reduced/full/projected DriverModel artifact"


def copy_lane_artifacts(
    output: Path,
    lane: str,
    relative_paths: list[str],
    required: bool,
    validator: Callable[[list[dict[str, Any]]], tuple[str, str]] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    documents: list[dict[str, Any]] = []
    copied: dict[str, str] = {}
    missing: list[str] = []
    destination = output / "artifacts" / lane
    destination.mkdir(parents=True, exist_ok=True)
    for relative in relative_paths:
        source = ROOT / relative
        if not source.is_file():
            missing.append(relative)
            continue
        document = read_json(source)
        if document is None:
            missing.append(relative + " (invalid JSON)")
            continue
        # Include the producer directory so native/scalar companions with the
        # same basename cannot overwrite one another in the fan-in bundle.
        source_path = Path(relative)
        target = destination / f"{source_path.parent.name}-{source_path.name}"
        shutil.copy2(source, target)
        digest = sha256(target)
        copied[str(target.relative_to(output))] = f"sha256:{digest}"
        documents.append(document)
    if missing:
        status = "not-started" if required else "partially-implemented"
        reason = "missing producer artifact(s): " + ", ".join(missing)
    elif validator is None:
        status, reason = "passed", "artifact copied and checksum-bound"
    else:
        status, reason = validator(documents)
    return {
        "lane": lane,
        "required": required,
        "status": status,
        "reason": reason,
        "source_artifacts": relative_paths,
        "copied_artifacts": sorted(copied),
        "evidence_class": "native-linux-x86-runtime" if required else "native-linux-x86-supplemental",
    }, copied


def main() -> int:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        print("QA2 requires native Linux x86_64", file=sys.stderr)
        return 2
    vendor, model, flags = cpu_info()
    flag_set = set(flags.split())
    if "avx2" not in flag_set or "fma" not in flag_set:
        print("QA2 requires AVX2 and FMA; refusing scalar fallback", file=sys.stderr)
        return 2
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "target/sonium-unix-qa2"
    output.mkdir(parents=True, exist_ok=True)
    artifact_root = output / "artifacts"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir()
    lanes = [
        ("fastmath", [
            "target/sonium-unix-f3/native-avx2.json",
            "target/sonium-unix-f3/scalar.json",
            "target/sonium-unix-f3/statistical-gate.json",
        ], True, validate_fastmath),
        ("fft", ["target/sonium-unix-qa2/fft-x86-dispatch-v1.json"], True, validate_fft),
        ("compute", [
            "crates/sonium-compute/target/sonium-unix-c5/native/sonium_native_x86_evidence.json",
            "crates/sonium-compute/target/sonium-unix-c5/scalar/sonium_native_x86_evidence.json",
        ], True, validate_compute),
        ("sparse", [
            "target/sonium-unix-ss2/spmv-types-native.json",
            "target/sonium-unix-ss2/spmv-types-scalar.json",
            "target/sonium-unix-ss2/plan-thresholds.json",
        ], True, validate_sparse),
        ("fem", ["target/sonium-unix-fem4/native-x86_64-fem4-v1.json"], True, validate_fem),
        ("room", [
            "target/sonium-unix-sr3/native/native-x86-sr3-v2.json",
            "target/sonium-unix-sr3/scalar/scalar-sr3-v2.json",
            "target/sonium-unix-sr3/native/native-x86-sr3-check-v1.json",
        ], True, validate_room),
        ("propeller-pp1", [
            "target/sonium-unix-pp1/single/native-x86-pp1-v1.json",
            "target/sonium-unix-pp1/configured/native-x86-pp1-v1.json",
        ], True, validate_propeller_pp1),
        ("propeller-pp3", ["target/sonium-unix-pp3/native-x86-pp3-fidelity-v1.json"], False, validate_propeller_pp3),
        ("propeller-pp4", ["target/sonium-unix-pp4/native-x86-pp4-release-v1.json"], False, validate_propeller_pp4),
        ("speaker", ["target/sonium-unix-sk3/speaker_native_x86_golden.json"], True, validate_speaker),
        ("speaker-modeling-sm3", ["target/sonium-unix-sm3/native-x86-sm3-v1.json"], True, validate_speaker_modeling_sm3),
    ]
    records: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    for lane, paths, required, validator in lanes:
        record, lane_checksums = copy_lane_artifacts(output, lane, paths, required, validator)
        records.append(record)
        checksums.update(lane_checksums)

    host = {
        "cpu_vendor": vendor,
        "cpu_model": model,
        "microcode": microcode(),
        "os": platform.platform(),
        "compiler": compiler(),
        "target_arch": platform.machine(),
        "target_features": sorted(feature for feature in ("avx2", "fma", "avx512f") if feature in flag_set),
        "runtime_dispatch": "avx2-fma",
        "thread_policy": os.environ.get("SONIUM_QA_THREAD_POLICY", "not-recorded-per-producer"),
        "affinity_policy": os.environ.get("SONIUM_QA_AFFINITY_POLICY", "not-recorded-per-producer"),
        "numa_policy": os.environ.get("SONIUM_QA_NUMA_POLICY", "not-recorded-per-producer"),
        "deterministic_mode": os.environ.get("SONIUM_QA_DETERMINISTIC_MODE", "not-recorded-per-producer"),
        "avx512f_available": "avx512f" in flag_set,
    }
    required_records = [record for record in records if record["required"]]
    eligible = all(record["status"] == "passed" for record in required_records)
    report = {
        "schema": "sonium-qa-native-x86-fanin/v1",
        "status": "passed" if eligible else "blocked",
        "promotion_eligible": eligible,
        "scope": "native-linux-x86_64",
        "host": host,
        "required_features": ["avx2", "fma"],
        "unsupported_or_waived": {
            "avx512f": "unsupported-not-required",
            "aarch64": "waived-by-user",
            "windows-x86_64": "waived-by-user",
            "rosetta": "excluded-from-native-evidence",
        },
        "tolerance_policy": "producer-specific tolerances are retained in each artifact; no cross-producer numeric comparison is inferred",
        "evidence_classes": {
            "runtime": "native producer executed on this Linux/x86_64 host",
            "reference": "analytic or independent reference gates emitted by a producer",
            "performance": "timing is accepted only where that producer's correctness gate passed",
            "waiver": "platform comparison waived and never counted as a pass",
        },
        "lanes": records,
        "artifact_checksums": checksums,
    }
    report_path = output / "native-x86-qa2-fanin-v1.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    checksums["native-x86-qa2-fanin-v1.json"] = f"sha256:{sha256(report_path)}"
    (output / "checksums.sha256").write_text(
        "".join(f"{digest.removeprefix('sha256:')}  {path}\n" for path, digest in sorted(checksums.items()))
    )
    print(json.dumps({"status": report["status"], "required_lanes": len(required_records), "artifact_count": len(checksums) - 1}))
    if not eligible:
        for record in required_records:
            if record["status"] != "passed":
                print(f"QA2 blocked: {record['lane']}: {record['reason']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
