#!/usr/bin/env python3
"""Strict checker for native single-device C4 scheduler contention evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import NoReturn


ARTIFACT_KEYS = {
    "schema",
    "artifact_kind",
    "fixture_version",
    "deterministic_fixture",
    "status",
    "target_os",
    "target_arch",
    "requested_backend",
    "actual_backend",
    "cuda_executed",
    "device_ordinal",
    "device_model",
    "compute_capability",
    "compiler",
    "host_submitter_count",
    "thread_policy",
    "scheduler_device",
    "budget_bytes",
    "budget_active_slots",
    "request_bytes",
    "request_active_slots",
    "over_budget_bytes",
    "workload",
    "workload_elements",
    "graph_replays",
    "graph_output_matches_reference",
    "contention_rejection",
    "over_budget_rejection",
    "during_contention",
    "after_first_refund",
    "after_retry",
    "replay_runtime",
    "publication_runtime",
    "high_water_bytes",
    "no_stranded_leases",
    "evidence_sha256",
    "unsupported_scope",
}
SNAPSHOT_KEYS = {
    "bytes_in_use",
    "active",
    "lost",
    "completed",
    "cancelled",
    "failed",
    "device_lost",
}
RUNTIME_KEYS = {
    "kernel_launches",
    "library_operations",
    "synchronization_calls",
    "completion_events",
    "completion_waits",
    "h2d_operations",
    "h2d_bytes",
    "d2h_operations",
    "d2h_bytes",
    "d2d_operations",
    "d2d_bytes",
}
ERROR_KEYS = {"kind", "requested_bytes", "requested_slots"}
BUDGET_BYTES = 4096
REQUEST_BYTES = 4096
OVER_BUDGET_BYTES = 4097
ACTIVE_SLOTS = 1
UNSUPPORTED_SCOPE = (
    "no multi-GPU placement, physical reset, Windows/Arm64, Nsight roofline, "
    "or broader application-flow claim"
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"C4 CUDA scheduler artifact rejected: {message}")


def integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        fail(f"{label} is not a non-negative integer")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown",
        "unavailable",
        "unspecified",
    }:
        fail(f"{label} is missing")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(artifact, dict):
        fail("artifact is not a JSON object")
    sidecar = Path(f"{path}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{sidecar} is missing or unreadable: {error}")
    expected = None
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and fields[1].lstrip("*") == path.name:
            expected = fields[0].lower()
            break
    if expected is None or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        fail(f"{sidecar} has no valid digest for {path.name}")
    actual = sha256(path)
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")
    return artifact


def check_snapshot(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != SNAPSHOT_KEYS:
        fail(f"{label} fields are incomplete")
    for key in SNAPSHOT_KEYS - {"lost"}:
        integer(value[key], f"{label}.{key}")
    if type(value["lost"]) is not bool:
        fail(f"{label}.lost is not boolean")
    return value


def check_runtime(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != RUNTIME_KEYS:
        fail(f"{label} fields are incomplete")
    for key in RUNTIME_KEYS:
        integer(value[key], f"{label}.{key}")
    return value


def check_error(value: object, label: str, requested_bytes: int) -> dict:
    if not isinstance(value, dict) or set(value) != ERROR_KEYS:
        fail(f"{label} fields are incomplete")
    if value["kind"] != "quota_exceeded":
        fail(f"{label} is not the typed quota error")
    if value["requested_bytes"] != requested_bytes or value["requested_slots"] != ACTIVE_SLOTS:
        fail(f"{label} request does not match its rejected work")
    return value


def evidence_payload(artifact: dict) -> list[object]:
    return [
        artifact["device_ordinal"],
        artifact["budget_bytes"],
        artifact["request_bytes"],
        artifact["over_budget_bytes"],
        artifact["contention_rejection"],
        artifact["over_budget_rejection"],
        artifact["during_contention"],
        artifact["after_first_refund"],
        artifact["after_retry"],
        artifact["replay_runtime"],
        artifact["publication_runtime"],
    ]


def check_artifact(artifact: dict) -> dict:
    if set(artifact) != ARTIFACT_KEYS:
        fail(
            "artifact fields differ: "
            f"missing={ARTIFACT_KEYS - set(artifact)}, extra={set(artifact) - ARTIFACT_KEYS}"
        )
    if (
        artifact["schema"] != "sonium-compute.c4.cuda-scheduler-contention/v1"
        or artifact["artifact_kind"] != "sonium-compute.c4.cuda-scheduler-contention"
        or artifact["fixture_version"] != 1
        or artifact["deterministic_fixture"] != "c4-cuda-scheduler-contention-v1"
    ):
        fail("artifact identity is invalid")
    if (
        artifact["status"] != "passed"
        or artifact["target_os"] != "linux"
        or artifact["target_arch"] != "x86_64"
        or artifact["requested_backend"] != "cuda"
        or artifact["actual_backend"] != "cuda"
        or artifact["cuda_executed"] is not True
    ):
        fail("artifact does not prove native CUDA execution")
    integer(artifact["device_ordinal"], "device_ordinal")
    if artifact["device_ordinal"] != 0:
        fail("fixture is not pinned to CUDA device ordinal 0")
    text(artifact["device_model"], "device_model")
    if artifact["compute_capability"] != "12.0":
        fail("fixture is not an SM120 run")
    text(artifact["compiler"], "compiler")
    if (
        artifact["host_submitter_count"] != 2
        or artifact["thread_policy"] != "two-host-submitters;single-cuda0-budget;barrier-held-lease"
        or artifact["scheduler_device"] != "cuda0"
    ):
        fail("host contention policy is not the pinned two-submitter contract")
    if (
        artifact["budget_bytes"] != BUDGET_BYTES
        or artifact["budget_active_slots"] != ACTIVE_SLOTS
        or artifact["request_bytes"] != REQUEST_BYTES
        or artifact["request_active_slots"] != ACTIVE_SLOTS
        or artifact["over_budget_bytes"] != OVER_BUDGET_BYTES
    ):
        fail("fixed scheduler budget/request fixture is invalid")
    text(artifact["workload"], "workload")
    if artifact["workload"] != "capture-safe native f64-to-f32 cast graph replay":
        fail("workload is not the pinned native graph fixture")
    if artifact["workload_elements"] != 5 or artifact["graph_replays"] != 2:
        fail("graph workload cardinality is invalid")
    if artifact["graph_output_matches_reference"] is not True:
        fail("native graph output parity is not proven")

    contention = check_error(artifact["contention_rejection"], "contention_rejection", REQUEST_BYTES)
    over_budget = check_error(artifact["over_budget_rejection"], "over_budget_rejection", OVER_BUDGET_BYTES)
    during = check_snapshot(artifact["during_contention"], "during_contention")
    after_first = check_snapshot(artifact["after_first_refund"], "after_first_refund")
    after_retry = check_snapshot(artifact["after_retry"], "after_retry")
    if (
        during["bytes_in_use"] != REQUEST_BYTES
        or during["active"] != ACTIVE_SLOTS
        or during["completed"] != 0
        or during["lost"]
        or after_first["bytes_in_use"] != 0
        or after_first["active"] != 0
        or after_first["completed"] != 1
        or after_retry["bytes_in_use"] != 0
        or after_retry["active"] != 0
        or after_retry["completed"] != 2
        or after_retry["cancelled"] != 0
        or after_retry["failed"] != 0
        or after_retry["device_lost"] != 0
        or after_retry["lost"]
    ):
        fail("scheduler snapshots do not prove admission/refund/retry")

    replay = check_runtime(artifact["replay_runtime"], "replay_runtime")
    publication = check_runtime(artifact["publication_runtime"], "publication_runtime")
    if (
        replay["kernel_launches"] != 2
        or replay["completion_events"] != 2
        or replay["completion_waits"] != 2
        or replay["h2d_operations"] != 0
        or replay["d2h_operations"] != 0
        or replay["d2d_operations"] != 0
        or publication["kernel_launches"] != 2
        or publication["completion_events"] != 2
        or publication["completion_waits"] != 2
        or publication["h2d_operations"] != 0
        or publication["d2h_operations"] != 2
        or publication["d2h_bytes"] < 2 * 5 * 4
        or publication["d2d_operations"] != 0
    ):
        fail("runtime counters do not prove two native graph replays")
    integer(artifact["high_water_bytes"], "high_water_bytes")
    if artifact["high_water_bytes"] != REQUEST_BYTES or artifact["no_stranded_leases"] is not True:
        fail("scheduler high-water or stranded-lease evidence is incomplete")
    text(artifact["unsupported_scope"], "unsupported_scope")
    if artifact["unsupported_scope"] != UNSUPPORTED_SCOPE:
        fail("unsupported scope is not explicit")
    expected_evidence = hashlib.sha256(
        json.dumps(evidence_payload(artifact), separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if artifact["evidence_sha256"] != expected_evidence:
        fail("evidence digest does not match measured rows")
    return {
        "device_ordinal": 0,
        "compute_capability": "12.0",
        "host_submitter_count": artifact["host_submitter_count"],
        "contention_rejection": contention,
        "over_budget_rejection": over_budget,
        "graph_replays": artifact["graph_replays"],
        "high_water_bytes": artifact["high_water_bytes"],
        "final_bytes_in_use": after_retry["bytes_in_use"],
        "evidence_sha256": artifact["evidence_sha256"],
    }


def main() -> None:
    if len(sys.argv) not in {2, 4} or (len(sys.argv) == 4 and sys.argv[2] != "--output"):
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [--output SUMMARY.json]")
    artifact_path = Path(sys.argv[1])
    artifact = load(artifact_path)
    evidence = check_artifact(artifact)
    summary = {
        "schema": "sonium-compute.c4.cuda-scheduler-contention/v1",
        "status": "passed",
        "target_os": "linux",
        "target_arch": "x86_64",
        "backend": "cuda",
        "cuda_executed": True,
        "artifact_sha256": sha256(artifact_path),
        "evidence": evidence,
        "unsupported_scope": UNSUPPORTED_SCOPE,
    }
    if len(sys.argv) == 4:
        output = Path(sys.argv[3])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
