"""Strict validation for structured sonium-qa release evidence.

This module is intentionally dependency-free.  The Justfile runners import it
after a case has emitted its JSON record; the release-evidence bundler performs
the same artifact checks again while assembling the immutable bundle.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
from typing import Any


EVIDENCE_CLASSES = {
    "fault-injection",
    "numerical-parity",
    "workload-correctness",
    "performance",
}
MISSING_PROVENANCE = {"", "unknown", "unavailable", "unspecified", "n/a"}
FAILURE_STAGES = {
    "environment",
    "initialization",
    "allocation",
    "transfer",
    "synchronization",
    "launch",
    "execution",
    "validation",
    "artifact",
    "serialization",
}

RESULT_REQUIRED_FIELDS = {
    "schema_version",
    "case_id",
    "status",
    "backend_outcomes",
    "metrics",
    "thresholds",
    "threshold_directions",
    "elapsed_ms",
    "git_sha",
    "environment",
    "artifacts",
    "evidence",
    "runtime",
}
EVIDENCE_REQUIRED_FIELDS = {"timings_ms", "artifact_checksums"}
RUNTIME_REQUIRED_FIELDS = {"context_reuse", "counts", "counters", "memory", "hardware"}
COUNT_FIELDS = ("executed", "passed", "failed", "unsupported")
COUNTER_FIELDS = (
    "allocations",
    "h2d_transfers",
    "d2h_transfers",
    "d2d_transfers",
    "synchronizations",
    "launches",
)
MEMORY_FIELDS = ("peak_bytes", "live_bytes")
HARDWARE_FIELDS = (
    "device",
    "driver",
    "toolkit",
    "cpu_model",
    "microcode",
    "os",
    "compiler",
    "target_features",
    "runtime_dispatch",
    "thread_policy",
    "affinity_policy",
    "numa_policy",
    "deterministic_mode",
    "topology",
    "peak_flops",
    "memory_bandwidth",
)


def _missing(value: Any) -> bool:
    return not isinstance(value, str) or value.strip().lower() in MISSING_PROVENANCE


def _finite_map(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    for name, number in value.items():
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label} contains an empty name")
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
            errors.append(f"{label}.{name} must be finite")


def _safe_relative(path: Any) -> bool:
    if not isinstance(path, str) or not path.strip() or "\\" in path or ":" in path:
        return False
    candidate = pathlib.PurePosixPath(path)
    return not candidate.is_absolute() and not any(
        part in {"..", "."} for part in candidate.parts
    )


def _string_map(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    for name, item in value.items():
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label} contains an empty name")
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label}.{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{label} must be a non-negative integer")


def _sha256_checksum(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == len("sha256:") + 64
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _expected_backend(case: dict[str, Any]) -> str | None:
    hardware = case.get("required_hardware")
    if hardware == "nvidia-cuda":
        return "cuda"
    if hardware == "native-x86_64":
        return "cpu-native-x86_64"
    return None


def _artifact_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(
    case: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    artifact_root: pathlib.Path | None = None,
) -> list[str]:
    """Return actionable policy errors for one emitted result.

    Legacy non-accelerator cases retain their existing migration path.  Every
    accelerator-gate case is strict: it must provide the complete typed
    contract, use the requested native backend, and have verifiable artifacts.
    """

    errors: list[str] = []
    if not isinstance(result, dict):
        return ["structured QA result is not an object"]
    strict = case.get("tier") == "accelerator-gate" or bool(case.get("strict_result_contract"))

    if (
        not isinstance(result.get("schema_version"), int)
        or isinstance(result.get("schema_version"), bool)
        or result.get("schema_version") != 1
    ):
        errors.append("unsupported or missing schema_version")
    if result.get("case_id") != case.get("id"):
        errors.append("case_id does not match the manifest")
    if result.get("status") != "passed":
        errors.append(f"status is not passed: {result.get('status')!r}")

    if strict:
        missing_fields = sorted(RESULT_REQUIRED_FIELDS - result.keys())
        if missing_fields:
            errors.append(f"strict result is missing required fields: {', '.join(missing_fields)}")
        if not isinstance(result.get("case_id"), str) or not result.get("case_id", "").strip():
            errors.append("case_id must be a non-empty string")
        if not isinstance(result.get("git_sha"), str) or not result.get("git_sha", "").strip():
            errors.append("git_sha must be a non-empty string")
        _nonnegative_int(result.get("elapsed_ms"), "elapsed_ms", errors)
        _string_map(result.get("environment"), "environment", errors)

    metrics = result.get("metrics")
    thresholds = result.get("thresholds")
    directions = result.get("threshold_directions")
    if metrics is None:
        metrics = {}
    if thresholds is None:
        thresholds = {}
    if directions is None:
        directions = {}
    _finite_map(metrics, "metrics", errors)
    _finite_map(thresholds, "thresholds", errors)
    if not isinstance(directions, dict) or not isinstance(thresholds, dict) or set(directions) != set(thresholds):
        errors.append("every threshold must have exactly one explicit direction")
    elif any(direction not in {"minimum", "maximum"} for direction in directions.values()):
        errors.append("threshold directions must be minimum or maximum")
    if strict and case.get("thresholds_required", True) and not thresholds:
        errors.append("strict release case has no thresholds")
    if isinstance(metrics, dict) and isinstance(thresholds, dict) and isinstance(directions, dict):
        for name, threshold in thresholds.items():
            metric = metrics.get(name)
            direction = directions.get(name)
            if metric is None:
                errors.append(f"threshold {name!r} has no recorded metric")
            elif not _finite_number(metric) or not _finite_number(threshold):
                continue
            elif direction == "minimum" and metric < threshold:
                errors.append(f"threshold {name!r} minimum is not satisfied")
            elif direction == "maximum" and metric > threshold:
                errors.append(f"threshold {name!r} maximum is not satisfied")

    outcomes = result.get("backend_outcomes")
    if strict or outcomes is not None:
        if not isinstance(outcomes, list):
            errors.append("backend_outcomes must be an array")
            outcomes = []
        names: set[str] = set()
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                errors.append("backend outcome must be an object")
                continue
            backend_name = outcome.get("backend")
            status = outcome.get("status")
            reason = outcome.get("reason")
            if not isinstance(backend_name, str) or not backend_name.strip():
                errors.append("backend outcome name must be a non-empty string")
            elif backend_name in names:
                errors.append(f"backend outcome {backend_name!r} is duplicated")
            else:
                names.add(backend_name)
            if status not in {"passed", "failed", "skipped"}:
                errors.append("backend outcome status is invalid")
            elif status == "passed" and reason not in (None, ""):
                errors.append("passed backend outcome must not include a reason")
            elif status in {"failed", "skipped"} and _missing(reason):
                errors.append("failed or skipped backend outcome needs a reason")

    runtime = result.get("runtime")
    if strict or case.get("runtime_required"):
        if not isinstance(runtime, dict):
            errors.append("strict release case is missing typed runtime evidence")
            runtime = {}
        counts = runtime.get("counts")
        counters = runtime.get("counters")
        memory = runtime.get("memory")
        hardware = runtime.get("hardware")
        if not isinstance(counts, dict):
            errors.append("runtime.counts is missing")
            counts = {}
        if not isinstance(counters, dict):
            errors.append("runtime.counters is missing")
            counters = {}
        if not isinstance(memory, dict):
            errors.append("runtime.memory is missing")
            memory = {}
        if not isinstance(hardware, dict):
            errors.append("runtime.hardware is missing")
            hardware = {}
        for name in COUNT_FIELDS:
            if not isinstance(counts.get(name), int) or isinstance(counts.get(name), bool) or counts.get(name, -1) < 0:
                errors.append(f"runtime.counts.{name} is missing or not a non-negative integer")
        if all(isinstance(counts.get(name), int) and not isinstance(counts.get(name), bool) and counts.get(name, -1) >= 0 for name in COUNT_FIELDS):
            if any(counts[name] > counts["executed"] for name in ("passed", "failed", "unsupported")):
                errors.append("runtime counts exceed executed count")
            if counts["executed"] == 0 or counts["passed"] == 0:
                errors.append("strict release case did not execute and pass a case")
            if result.get("status") == "passed" and (
                counts["passed"] != counts["executed"]
                or counts["failed"] != 0
                or counts["unsupported"] != 0
            ):
                errors.append("passed strict release case has incomplete runtime counts")
        for name in COUNTER_FIELDS:
            if not isinstance(counters.get(name), int) or isinstance(counters.get(name), bool) or counters.get(name, -1) < 0:
                errors.append(f"runtime.counters.{name} is missing or not a non-negative integer")
        for name in MEMORY_FIELDS:
            if not isinstance(memory.get(name), int) or isinstance(memory.get(name), bool) or memory.get(name, -1) < 0:
                errors.append(f"runtime.memory.{name} is missing or not a non-negative integer")
        if isinstance(memory.get("peak_bytes"), int) and isinstance(memory.get("live_bytes"), int) and memory["live_bytes"] >= 0 and memory["peak_bytes"] >= 0 and memory["live_bytes"] > memory["peak_bytes"]:
            errors.append("runtime live memory exceeds peak memory")
        if runtime.get("failure_stage") not in (None, *FAILURE_STAGES):
            errors.append("runtime.failure_stage is not a known typed stage")
        if runtime.get("context_reuse") not in {"reused", "recreated", "not_applicable"}:
            errors.append("runtime.context_reuse is not a known value")
        if result.get("status") == "passed" and runtime.get("failure_stage") is not None:
            errors.append("passed result cannot report a failure stage")
        for name, value in hardware.items():
            if name not in HARDWARE_FIELDS or not isinstance(value, str) or not value.strip():
                errors.append(f"runtime.hardware.{name} must be a non-empty string")
        if strict:
            backend = _expected_backend(case)
            if backend == "cuda":
                required_hardware = ("device", "driver", "toolkit")
            elif backend == "cpu-native-x86_64":
                required_hardware = (
                    "cpu_model",
                    "microcode",
                    "os",
                    "compiler",
                    "target_features",
                    "runtime_dispatch",
                    "thread_policy",
                    "affinity_policy",
                    "numa_policy",
                    "deterministic_mode",
                )
            else:
                required_hardware = ("cpu_model", "os", "compiler")
            for name in required_hardware:
                if _missing(hardware.get(name)):
                    errors.append(f"runtime.hardware.{name} is missing")

    expected_backend = _expected_backend(case)
    if expected_backend is not None:
        if result.get("requested_backend") != expected_backend or result.get("actual_backend") != expected_backend:
            errors.append(f"result is not native {expected_backend}: requested/actual backend mismatch")
        outcomes = result.get("backend_outcomes")
        if not isinstance(outcomes, list) or not any(
            isinstance(outcome, dict)
            and outcome.get("backend") == expected_backend
            and outcome.get("status") == "passed"
            for outcome in outcomes
        ):
            errors.append(f"missing passing backend outcome for {expected_backend}")

    artifacts = result.get("artifacts")
    evidence = result.get("evidence")
    if artifacts is None:
        artifacts = []
    if evidence is None:
        evidence = {}
    if strict and not isinstance(evidence, dict):
        errors.append("strict result evidence must be an object")
    if not isinstance(evidence, dict):
        evidence = {}
    if strict:
        missing_evidence = sorted(EVIDENCE_REQUIRED_FIELDS - evidence.keys())
        if missing_evidence:
            errors.append(f"strict evidence is missing required fields: {', '.join(missing_evidence)}")
    checksums = evidence.get("artifact_checksums")
    provenance = evidence.get("artifact_provenance")
    if checksums is None:
        checksums = {}
    if provenance is None:
        provenance = {}
    if not isinstance(artifacts, list) or any(not isinstance(path, str) or not _safe_relative(path) for path in artifacts):
        errors.append("artifacts must contain safe relative paths")
        artifacts = []
    if len(set(artifacts)) != len(artifacts):
        errors.append("artifacts must not contain duplicate paths")
    if not isinstance(checksums, dict) or any(path not in artifacts or not isinstance(checksum, str) or _missing(checksum) for path, checksum in checksums.items()):
        errors.append("artifact checksums are missing or refer to undeclared artifacts")
    if not isinstance(checksums, dict):
        checksums = {}
    if not isinstance(provenance, dict):
        errors.append("artifact_provenance must be an object")
        provenance = {}
    if strict and case.get("artifact_provenance_required", True):
        if not artifacts:
            errors.append("strict release case has no artifacts")
        for path in artifacts:
            if path not in checksums:
                errors.append(f"artifact {path!r} has no checksum")
            item = provenance.get(path)
            if not isinstance(item, dict):
                errors.append(f"artifact {path!r} has no provenance")
                continue
            for name in ("source", "generation", "checksum", "version"):
                if _missing(item.get(name)):
                    errors.append(f"artifact {path!r} provenance.{name} is missing")
            if item.get("checksum") != checksums.get(path):
                errors.append(f"artifact {path!r} provenance checksum disagrees with artifact_checksums")
            if not _sha256_checksum(checksums.get(path)):
                errors.append(f"artifact {path!r} checksum is not a sha256 digest")
    if artifact_root is not None and strict:
        for path in artifacts:
            declared = checksums.get(path)
            source = artifact_root / pathlib.PurePosixPath(path)
            if not source.is_file():
                errors.append(f"declared artifact is missing: {path!r}")
                continue
            if _sha256_checksum(declared):
                actual = _artifact_digest(source)
                if actual != declared.removeprefix("sha256:"):
                    errors.append(f"stale checksum for artifact {path!r}")
            elif strict:
                errors.append(f"artifact {path!r} checksum is not verifiable")

    if strict:
        if not isinstance(evidence.get("timings_ms"), dict):
            errors.append("evidence.timings_ms must be an object")
        else:
            for name, value in evidence["timings_ms"].items():
                _nonnegative_int(value, f"evidence.timings_ms.{name}", errors)
        if not isinstance(evidence.get("artifact_checksums"), dict):
            errors.append("evidence.artifact_checksums must be an object")
        for name in ("physical_residual", "physical_reference_error"):
            if name in evidence and (
                not isinstance(evidence[name], (int, float))
                or isinstance(evidence[name], bool)
                or not math.isfinite(evidence[name])
            ):
                errors.append(f"evidence.{name} must be finite")
        if "memory_bytes" in evidence:
            _nonnegative_int(evidence["memory_bytes"], "evidence.memory_bytes", errors)

    if strict:
        evidence_class = case.get("evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            errors.append(f"unknown evidence class {evidence_class!r}")
        expected_reference_checksum = case.get("reference_checksum")
        if expected_reference_checksum and evidence.get("reference_checksum") != expected_reference_checksum:
            errors.append("result reference_checksum does not match the manifest")

    return errors
