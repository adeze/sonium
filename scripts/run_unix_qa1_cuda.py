#!/usr/bin/env python3
"""Run and validate the required Linux/NVIDIA CUDA QA1 lane.

This runner deliberately has no dependency on the shared manifest generator.
It consumes the immutable CUDA shard produced by the release lane, validates
its lineage, and runs three small native producers (operation, solver, and
physical) in a stable repository-local Cargo target/output tree.  The two
fault probes are required to fail; a zero exit status is itself a QA1 failure.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import tomllib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = pathlib.Path(
    os.environ.get("SONIUM_QA1_OUTPUT_DIR", ROOT / "target" / "sonium-unix-qa1")
).resolve()
CARGO_TARGET = pathlib.Path(
    os.environ.get("CARGO_TARGET_DIR", ROOT / ".target-unix-qa1")
).resolve()
SOURCE_LANE = pathlib.Path(
    os.environ.get(
        "SONIUM_QA1_SOURCE_LANE",
        ROOT / "target" / "sonium-qa-accelerators" / "nvidia-cuda-final",
    )
).resolve()
COMMAND = [
    "cargo",
    "run",
    "-p",
    "sonium-qa",
    "--release",
    "--features",
    "qa-binaries cuda",
]

PLACEHOLDER = {"", "unknown", "unavailable", "unspecified", "n/a"}
REQUIRED_LIBRARIES = ("cuda", "cudart", "cublas", "cusparse", "cusolver")
PRODUCERS = (
    {
        "id": "compute.operation-matrix.cuda.v1",
        "kind": "operation",
        "bin": "qa-compute-operation-matrix",
        "case_id": "compute.operation-matrix.v1",
    },
    {
        "id": "solvers.preconditioner-cuda-native.v1",
        "kind": "solver",
        "bin": "qa-solvers-preconditioner-cuda",
        "case_id": "solvers.preconditioner-cuda-native.v1",
    },
    {
        "id": "compute.fem-adjoint-cuda.v1",
        "kind": "physical",
        "bin": "qa-compute-fem-adjoint",
        "case_id": "compute.fem-adjoint-cuda.v1",
    },
)


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run_command(
    command: list[str],
    env: dict[str, str],
    log_path: pathlib.Path,
    timeout: int = 900,
) -> tuple[int, str, int]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + f"\nTIMEOUT after {timeout}s\n"
        returncode = 124
    elapsed_ms = round((time.monotonic() - started) * 1000)
    log_path.write_text(output, encoding="utf-8")
    return returncode, output, elapsed_ms


def qa_result(output: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        if not line.startswith("QA_RESULT:"):
            continue
        try:
            result = json.loads(line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            return {"status": "failed", "message": "invalid QA_RESULT JSON"}
        if isinstance(result, dict):
            return result
    return None


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in PLACEHOLDER


def preflight() -> tuple[dict[str, Any], list[str]]:
    """Validate the labelled NVIDIA device, toolkit, and vendor libraries."""

    errors: list[str] = []
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "os": platform.system().lower() + "-" + platform.machine(),
        "compiler": None,
        "device": None,
        "compute_capability": None,
        "driver": None,
        "toolkit": None,
        "topology": "GPU ordinal 0; single-device QA1 lane",
        "vendor_libraries": {},
    }

    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        gpu = None
        errors.append(f"nvidia-smi could not be executed: {error}")
    if gpu is not None:
        rows = list(csv.reader(line for line in gpu.stdout.splitlines() if line.strip()))
        if gpu.returncode or not rows:
            errors.append(f"nvidia-smi did not report a device: {gpu.stdout.strip()}")
        else:
            row = [field.strip() for field in rows[0]]
            if len(row) < 4:
                errors.append(f"nvidia-smi returned malformed device metadata: {row!r}")
            else:
                metadata.update(
                    {
                        "device": row[0],
                        "compute_capability": row[1],
                        "driver": row[2],
                        "memory_total": row[3],
                        "device_count": len(rows),
                    }
                )
                if not all(nonempty(metadata[key]) for key in ("device", "compute_capability", "driver")):
                    errors.append("nvidia-smi returned incomplete device identity")

    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        release = re.search(r"release\s+([0-9.]+)", nvcc.stdout)
        # Match the provenance spelling used by the existing CUDA producer
        # artifacts while retaining the exact nvcc release number.
        metadata["toolkit"] = f"CUDA UMD {release.group(1)}" if release else None
        if nvcc.returncode or not release:
            errors.append(f"nvcc did not report a CUDA toolkit release: {nvcc.stdout.strip()}")
    except OSError as error:
        errors.append(f"nvcc could not be executed: {error}")

    try:
        compiler = subprocess.run(
            ["rustc", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        metadata["compiler"] = compiler.stdout.strip() or None
        if compiler.returncode:
            errors.append("rustc --version failed")
    except OSError as error:
        errors.append(f"rustc could not be executed: {error}")

    try:
        libraries = subprocess.run(
            ["ldconfig", "-p"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except OSError as error:
        libraries = ""
        errors.append(f"ldconfig could not be executed: {error}")
    for name in REQUIRED_LIBRARIES:
        found = bool(re.search(rf"\blib{name}\.so(?:\.|\s|$)", libraries))
        metadata["vendor_libraries"][f"lib{name}"] = "present" if found else "missing"
        if not found:
            errors.append(f"required CUDA vendor library lib{name}.so is missing")

    return metadata, errors


def common_environment(metadata: dict[str, Any], output: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CARGO_TARGET_DIR": str(CARGO_TARGET),
            "SONIUM_OUTPUT_DIR": str(output / "producer-artifacts"),
            "SONIUM_QA_RESULT_DIR": str(output / "cases"),
            "SONIUM_REQUIRE_CUDA": "1",
            "SONIUM_REQUIRE_BACKEND": "cuda",
            "SONIUM_QA_ACCELERATOR_LANE": "nvidia-cuda",
            "SONIUM_QA_CUDA_DEVICE_NAME": str(metadata.get("device") or "unavailable"),
            "SONIUM_QA_CUDA_DRIVER_VERSION": str(metadata.get("driver") or "unavailable"),
            "SONIUM_QA_CUDA_TOOLKIT_VERSION": str(metadata.get("toolkit") or "unavailable"),
            "SONIUM_QA_CUDA_TOPOLOGY": str(metadata.get("topology") or "unavailable"),
            "SONIUM_QA_TARGET_FEATURES": "cuda",
            "SONIUM_QA_RUNTIME_DISPATCH": "cuda-native",
            "SONIUM_QA_THREAD_POLICY": "runner-default",
            "SONIUM_QA_AFFINITY_POLICY": "runner-default",
            "SONIUM_QA_NUMA_POLICY": "runner-default",
            "SONIUM_QA_DETERMINISTIC_MODE": "enabled",
        }
    )
    output.joinpath("producer-artifacts").mkdir(parents=True, exist_ok=True)
    output.joinpath("cases").mkdir(parents=True, exist_ok=True)
    return env


def artifact_checksums(result: dict[str, Any] | None, output: pathlib.Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not result:
        return checksums
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        declared = evidence.get("artifact_checksums")
        if isinstance(declared, dict):
            checksums.update({str(key): str(value) for key, value in declared.items()})
    for artifact in result.get("artifacts", []):
        if not isinstance(artifact, str):
            continue
        path = output / "producer-artifacts" / artifact
        if path.is_file():
            checksums[artifact] = "sha256:" + digest(path)
    return checksums


def run_producer(
    producer: dict[str, str], env: dict[str, str], output: pathlib.Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    case_env = dict(env)
    if producer["kind"] == "operation":
        # This producer predates SONIUM_REQUIRE_BACKEND and selects the
        # checked backend through SONIUM_BACKEND.  Keep both declarations so
        # the runner still rejects a result that is not native CUDA.
        case_env["SONIUM_BACKEND"] = "cuda"
    log = output / "cases" / f"{producer['id']}.log"
    returncode, text, elapsed_ms = run_command(
        COMMAND + ["--bin", producer["bin"]], case_env, log
    )
    result = qa_result(text)
    actual = result.get("actual_backend") if result else None
    requested = result.get("requested_backend") if result else case_env.get("SONIUM_REQUIRE_BACKEND")
    native_pass = returncode == 0 and result is not None and result.get("status") == "passed" and actual == "cuda"
    runtime = result.get("runtime") if result else {}
    counts = runtime.get("counts") if isinstance(runtime, dict) else None
    if not isinstance(counts, dict):
        counts = {
            "executed": int(result is not None),
            "passed": int(native_pass),
            "failed": int(not native_pass),
            "unsupported": 0,
        }
    return {
        "id": producer["id"],
        "kind": producer["kind"],
        "producer_case_id": result.get("case_id") if result else producer["case_id"],
        "command": " ".join(COMMAND + ["--bin", producer["bin"]]),
        "status": "passed" if native_pass else "failed",
        "returncode": returncode,
        "requested_backend": requested,
        "actual_backend": actual,
        "device_toolchain": {
            key: metadata.get(key)
            for key in ("device", "compute_capability", "driver", "toolkit", "compiler", "topology")
        },
        "counts": {key: counts.get(key, 0) for key in ("executed", "passed", "failed", "unsupported")},
        "duration_ms": elapsed_ms,
        "metrics": result.get("metrics", {}) if result else {},
        "artifact_checksums": artifact_checksums(result, output),
        "message": result.get("message") if result else text[-2000:].strip(),
    }


def run_probe(
    name: str, mutation: dict[str, str], env: dict[str, str], output: pathlib.Path
) -> dict[str, Any]:
    probe_env = dict(env)
    probe_env.update(mutation)
    probe_env["SONIUM_BACKEND"] = "cuda"
    log = output / "probes" / f"{name}.log"
    returncode, text, elapsed_ms = run_command(
        COMMAND + ["--bin", "qa-compute-operation-matrix"], probe_env, log, timeout=300
    )
    result = qa_result(text)
    # The required lane passes only when each deliberate fault is rejected.
    rejected = returncode != 0 and not (
        result and result.get("status") == "passed" and result.get("actual_backend") == "cuda"
    )
    return {
        "id": f"qa1.probe.{name}.v1",
        "fault": name,
        "expected": "required CUDA job exits non-zero",
        "status": "passed" if rejected else "failed",
        "returncode": returncode,
        "duration_ms": elapsed_ms,
        "requested_backend": result.get("requested_backend") if result else "cuda",
        "actual_backend": result.get("actual_backend") if result else None,
        "failure_stage": (result.get("runtime") or {}).get("failure_stage") if result else None,
        "message": result.get("message") if result else text[-2000:].strip(),
    }


def validate_source_lane(source: pathlib.Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the prior required shard without regenerating shared files."""

    errors: list[str] = []
    evidence_path = source / "accelerator-evidence.json"
    if not evidence_path.is_file():
        return {"status": "failed", "errors": [f"missing {evidence_path}"], "cases": []}
    try:
        shard = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "failed", "errors": [f"invalid source shard: {error}"], "cases": []}
    manifest_path = ROOT / "crates" / "sonium-qa" / "validation-manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cases = [
        case
        for case in manifest.get("case", [])
        if case.get("tier") == "accelerator-gate" and case.get("required_hardware") == "nvidia-cuda"
    ]
    expected_ids = [case["id"] for case in expected_cases]
    if shard.get("schema") != "sonium-qa-accelerator-evidence/v1":
        errors.append("source shard schema is not sonium-qa-accelerator-evidence/v1")
    if shard.get("hardware") != "nvidia-cuda" or shard.get("required_backend") != "cuda":
        errors.append("source shard is not labelled required nvidia-cuda/cuda")
    if shard.get("required_case_ids") != expected_ids:
        errors.append("source shard case coverage differs from the current manifest")
    current_manifest_digest = digest(manifest_path)
    if shard.get("manifest_sha256") != current_manifest_digest:
        errors.append(
            "source shard manifest checksum is stale: "
            f"published={shard.get('manifest_sha256')} current={current_manifest_digest}"
        )
    source_environment = shard.get("environment")
    if not isinstance(source_environment, dict):
        errors.append("source shard environment is missing")
    else:
        for key in ("device", "driver", "toolkit", "os", "compiler"):
            if not nonempty(source_environment.get(key)):
                errors.append(f"source shard environment.{key} is missing")
        for key, current_key in (("device", "device"), ("driver", "driver"), ("toolkit", "toolkit")):
            if nonempty(metadata.get(current_key)) and source_environment.get(key) != metadata[current_key]:
                errors.append(f"source shard {key} does not match live preflight metadata")
    coverage = shard.get("coverage") or {}
    if coverage.get("failed") != 0 or coverage.get("unsupported") != 0:
        errors.append("source shard contains failed or unsupported coverage")
    rows = shard.get("cases") if isinstance(shard.get("cases"), list) else []
    by_id = {row.get("case_id"): row for row in rows if isinstance(row, dict)}
    for case in expected_cases:
        row = by_id.get(case["id"])
        if row is None:
            errors.append(f"source shard is missing case {case['id']}")
            continue
        if row.get("status") != "passed" or row.get("requested_backend") != "cuda" or row.get("actual_backend") != "cuda":
            errors.append(f"source case {case['id']} is not a passing native CUDA result")
        counts = (row.get("runtime") or {}).get("counts") or {}
        if counts.get("executed", 0) < 1 or counts.get("passed", 0) < 1:
            errors.append(f"source case {case['id']} did not execute and pass")
    checksums = source / "checksums.sha256"
    if not checksums.is_file():
        errors.append("source shard checksum inventory is missing")
    else:
        declared: dict[str, str] = {}
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                value, name = line.split("  ", 1)
                declared[name] = value
        for name, value in declared.items():
            target = manifest_path if name == "validation-manifest.toml" else source / name
            if not target.is_file() or digest(target) != value:
                errors.append(f"source checksum mismatch: {name}")
    return {
        "status": "passed" if not errors else "blocked",
        "errors": errors,
        "source": str(source),
        "manifest_sha256": shard.get("manifest_sha256"),
        "current_manifest_sha256": current_manifest_digest,
        "coverage": coverage,
        "required_case_ids": expected_ids,
        "hardware": source_environment,
        "cases": rows,
    }


def write_checksums(output: pathlib.Path) -> None:
    rows: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{digest(path)}  {path.relative_to(output).as_posix()}\n")
    output.joinpath("checksums.sha256").write_text("".join(rows), encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "probes").mkdir(parents=True, exist_ok=True)
    metadata, preflight_errors = preflight()
    env = common_environment(metadata, OUTPUT)
    producers = [run_producer(case, env, OUTPUT, metadata) for case in PRODUCERS]
    probes = [
        run_probe(
            "missing-device",
            {"SONIUM_CUDA_DEVICE_ORDINAL": "2147483647"},
            env,
            OUTPUT,
        ),
        run_probe("forced-fallback", {"SONIUM_CUDA_FORCE_FALLBACK": "1"}, env, OUTPUT),
    ]
    source = validate_source_lane(SOURCE_LANE, metadata)
    producer_kinds = {case["kind"] for case in producers if case["status"] == "passed"}
    promotion_eligible = (
        not preflight_errors
        and source["status"] == "passed"
        and all(case["status"] == "passed" for case in producers)
        and all(probe["status"] == "passed" for probe in probes)
        and {"operation", "solver", "physical"} <= producer_kinds
    )
    report = {
        "schema": "sonium-qa-qa1-cuda/v1",
        "status": "passed" if promotion_eligible else "blocked",
        "promotion_eligible": promotion_eligible,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "required_backend": "cuda",
        "hardware_label": "nvidia-cuda",
        "scope": "native Linux/x86_64 NVIDIA CUDA; Windows and Arm64 are user-waived",
        "preflight": {"status": "passed" if not preflight_errors else "failed", "metadata": metadata, "errors": preflight_errors},
        "producers": producers,
        "probes": probes,
        "source_shard": source,
        "failure_policy": {
            "missing_device": "required job must exit non-zero",
            "initialization": "required_cuda_backend error is a required failure",
            "fallback": "SONIUM_CUDA_FORCE_FALLBACK=1 must never produce a native pass",
            "skip_or_unsupported": "required coverage with skipped/unsupported rows is not promotable",
            "provenance": "requested and actual backend must both be cuda",
        },
    }
    report_path = OUTPUT / "qa1-cuda-evidence.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_checksums(OUTPUT)
    print(json.dumps({"status": report["status"], "output": str(report_path), "preflight_errors": preflight_errors, "source_errors": source["errors"], "producers": producers, "probes": probes}, indent=2))
    return 0 if promotion_eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
