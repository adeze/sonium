#!/usr/bin/env python3
"""Strict checker for host-local C4 MemoryPool contention evidence."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


EXPECTED_ARTIFACT_KEYS = {
    "schema",
    "artifact_kind",
    "fixture_version",
    "deterministic_fixture",
    "status",
    "target_os",
    "target_arch",
    "backend",
    "cuda_executed",
    "cpu_model",
    "compiler",
    "thread_count",
    "thread_policy",
    "cap_bytes",
    "anchor_bytes",
    "fragmentation_blocks_bytes",
    "contended_request_bytes",
    "worker_probes",
    "accepted_attempts",
    "rejected_attempts",
    "typed_quota_failures",
    "rejected_live_bytes_preserved",
    "anchor_contents_preserved",
    "retry_successes",
    "reuse_operations",
    "high_water_bytes",
    "baseline_live_bytes",
    "final_live_bytes",
    "corpus_sha256",
    "unsupported_scope",
}
EXPECTED_PROBE_KEYS = {
    "worker_id",
    "accepted_attempts",
    "rejected_attempts",
    "typed_quota_failures",
    "live_bytes_before_rejection",
    "live_bytes_after_rejection",
    "retry_successes",
    "reuse_operations",
    "retry_buffer_bytes",
    "destination_preserved",
}

EXPECTED_CUDA_ARTIFACT_KEYS = {
    "schema",
    "artifact_kind",
    "fixture_version",
    "deterministic_fixture",
    "status",
    "target_os",
    "target_arch",
    "backend",
    "cuda_executed",
    "device_ordinal",
    "total_device_memory_bytes",
    "compiler",
    "host_thread_count",
    "thread_policy",
    "cap_bytes",
    "anchor_bytes",
    "fragmentation_blocks_bytes",
    "request_bytes",
    "baseline_pool_live_bytes",
    "baseline_device_live_bytes",
    "probe",
    "rejected_live_bytes_preserved",
    "high_water_bytes",
    "backend_high_water_bytes",
    "retry_successes",
    "final_pool_live_bytes",
    "final_device_live_bytes",
    "restored_cap_bytes",
    "corpus_sha256",
    "unsupported_scope",
}
EXPECTED_CUDA_PROBE_KEYS = {
    "rejection_attempts",
    "typed_quota_failures",
    "pool_live_bytes_before_rejection",
    "pool_live_bytes_after_rejection",
    "device_live_bytes_before_rejection",
    "device_live_bytes_after_rejection",
    "retry_successes",
    "retry_buffer_bytes",
    "destination_preserved",
}

CAP_BYTES = 4096
ANCHOR_BYTES = 3584
FRAGMENTATION_BLOCKS = [256, 256]
REQUEST_BYTES = 1024
THREAD_COUNT = 4
REUSE_OPERATIONS_PER_THREAD = 4
UNSUPPORTED_SCOPE = "no CUDA execution, multi-GPU, external CI, energy, or vendor claim"
CUDA_UNSUPPORTED_SCOPE = "no multi-GPU, external CI, energy, or vendor claim"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"C4 memory artifact rejected: {message}")


def nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        fail(f"{label} is not a non-negative integer")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() in {
        "",
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
    return value


def corpus_digest(probes: list[dict]) -> str:
    payload = json.dumps(
        [CAP_BYTES, ANCHOR_BYTES, FRAGMENTATION_BLOCKS, REQUEST_BYTES, probes],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def check_artifact(artifact: dict) -> dict:
    if set(artifact) != EXPECTED_ARTIFACT_KEYS:
        fail(
            "artifact fields differ: "
            f"missing={EXPECTED_ARTIFACT_KEYS - set(artifact)}, "
            f"extra={set(artifact) - EXPECTED_ARTIFACT_KEYS}"
        )
    if (
        artifact["schema"] != 1
        or artifact["artifact_kind"] != "sonium-compute.c4.memory-quota-contention"
        or artifact["fixture_version"] != 1
        or artifact["deterministic_fixture"] != "c4-memory-quota-contention-v1"
    ):
        fail("artifact identity is invalid")
    if artifact["status"] != "passed" or artifact["target_os"] != "linux" or artifact["target_arch"] != "x86_64":
        fail("artifact status/target is invalid")
    if artifact["backend"] != "cpu-reference" or artifact["cuda_executed"] is not False:
        fail("artifact is not explicitly host-only")
    text(artifact["cpu_model"], "cpu_model")
    text(artifact["compiler"], "compiler")
    if artifact["thread_count"] != THREAD_COUNT or artifact["thread_policy"] != "four-host-threads;turn-gated-retry":
        fail("thread contention policy is not the pinned four-thread contract")
    if artifact["cap_bytes"] != CAP_BYTES or artifact["anchor_bytes"] != ANCHOR_BYTES:
        fail("fixed quota/anchor values are invalid")
    if artifact["fragmentation_blocks_bytes"] != FRAGMENTATION_BLOCKS or artifact["contended_request_bytes"] != REQUEST_BYTES:
        fail("fragmentation/request fixture is invalid")
    if artifact["unsupported_scope"] != UNSUPPORTED_SCOPE:
        fail("unsupported scope is not explicit")

    probes = artifact["worker_probes"]
    if not isinstance(probes, list) or len(probes) != THREAD_COUNT:
        fail("worker probe count is invalid")
    for worker_id, probe in enumerate(probes):
        if not isinstance(probe, dict) or set(probe) != EXPECTED_PROBE_KEYS:
            fail(f"worker {worker_id} fields are incomplete")
        if probe["worker_id"] != worker_id:
            fail(f"worker probe {worker_id} is unordered")
        for key in (
            "accepted_attempts",
            "rejected_attempts",
            "typed_quota_failures",
            "live_bytes_before_rejection",
            "live_bytes_after_rejection",
            "retry_successes",
            "reuse_operations",
            "retry_buffer_bytes",
        ):
            nonnegative(probe[key], f"worker {worker_id}.{key}")
        if (
            probe["accepted_attempts"] != 0
            or probe["rejected_attempts"] != 1
            or probe["typed_quota_failures"] != 1
            or probe["live_bytes_before_rejection"] != ANCHOR_BYTES
            or probe["live_bytes_after_rejection"] != ANCHOR_BYTES
            or probe["retry_successes"] != 1
            or probe["reuse_operations"] != REUSE_OPERATIONS_PER_THREAD
            or probe["retry_buffer_bytes"] != REQUEST_BYTES
            or probe["destination_preserved"] is not True
        ):
            fail(f"worker {worker_id} does not prove typed rejection/preservation/retry")

    count_fields = (
        "accepted_attempts",
        "rejected_attempts",
        "typed_quota_failures",
        "retry_successes",
        "reuse_operations",
        "high_water_bytes",
        "baseline_live_bytes",
        "final_live_bytes",
    )
    for key in count_fields:
        nonnegative(artifact[key], key)
    if (
        artifact["accepted_attempts"] != 0
        or artifact["rejected_attempts"] != THREAD_COUNT
        or artifact["typed_quota_failures"] != THREAD_COUNT
        or artifact["rejected_live_bytes_preserved"] is not True
        or artifact["anchor_contents_preserved"] is not True
        or artifact["retry_successes"] != THREAD_COUNT
        or artifact["reuse_operations"] != THREAD_COUNT * REUSE_OPERATIONS_PER_THREAD
        or artifact["high_water_bytes"] != CAP_BYTES
        or artifact["baseline_live_bytes"] != 0
        or artifact["final_live_bytes"] != 0
    ):
        fail("aggregate allocator accounting is incomplete")
    corpus = text(artifact["corpus_sha256"], "corpus_sha256").lower()
    if len(corpus) != 64 or any(c not in "0123456789abcdef" for c in corpus):
        fail("corpus digest is malformed")
    if corpus != corpus_digest(probes):
        fail("corpus digest does not match deterministic probe rows")
    return {
        "thread_count": THREAD_COUNT,
        "accepted_attempts": artifact["accepted_attempts"],
        "rejected_attempts": artifact["rejected_attempts"],
        "retry_successes": artifact["retry_successes"],
        "reuse_operations": artifact["reuse_operations"],
        "high_water_bytes": artifact["high_water_bytes"],
        "final_live_bytes": artifact["final_live_bytes"],
        "corpus_sha256": corpus,
    }


def check_cuda_artifact(artifact: dict) -> dict:
    if set(artifact) != EXPECTED_CUDA_ARTIFACT_KEYS:
        fail(
            "CUDA artifact fields differ: "
            f"missing={EXPECTED_CUDA_ARTIFACT_KEYS - set(artifact)}, "
            f"extra={set(artifact) - EXPECTED_CUDA_ARTIFACT_KEYS}"
        )
    if (
        artifact["schema"] != 1
        or artifact["artifact_kind"] != "sonium-compute.c4.cuda-memory-quota"
        or artifact["fixture_version"] != 1
        or artifact["deterministic_fixture"] != "c4-cuda-memory-quota-reuse-v1"
    ):
        fail("CUDA artifact identity is invalid")
    if artifact["status"] != "passed" or artifact["target_os"] != "linux" or artifact["target_arch"] != "x86_64":
        fail("CUDA artifact status/target is invalid")
    if artifact["backend"] != "cuda" or artifact["cuda_executed"] is not True:
        fail("CUDA artifact does not prove genuine CUDA execution")
    if artifact["device_ordinal"] != 0:
        fail("CUDA device ordinal is not the pinned single-device fixture")
    total_memory = nonnegative(artifact["total_device_memory_bytes"], "total_device_memory_bytes")
    text(artifact["compiler"], "compiler")
    if artifact["host_thread_count"] != 1 or artifact["thread_policy"] != "single-host-thread;cuda-context-affine":
        fail("CUDA host-thread/context-affinity policy is invalid")
    if artifact["cap_bytes"] != 16 * 1024 or artifact["anchor_bytes"] != 12 * 1024:
        fail("CUDA fixed quota/anchor values are invalid")
    if artifact["fragmentation_blocks_bytes"] != [2 * 1024, 2 * 1024] or artifact["request_bytes"] != 4 * 1024:
        fail("CUDA fragmentation/request fixture is invalid")
    cap = artifact["cap_bytes"]
    anchor = artifact["anchor_bytes"]
    request = artifact["request_bytes"]
    if total_memory < cap or artifact["restored_cap_bytes"] != total_memory:
        fail("CUDA total memory/cap restoration is invalid")
    baseline_pool = nonnegative(artifact["baseline_pool_live_bytes"], "baseline_pool_live_bytes")
    baseline_device = nonnegative(artifact["baseline_device_live_bytes"], "baseline_device_live_bytes")
    if baseline_pool != 0:
        fail("CUDA pool baseline must be zero")
    probe = artifact["probe"]
    if not isinstance(probe, dict) or set(probe) != EXPECTED_CUDA_PROBE_KEYS:
        fail("CUDA probe fields are incomplete")
    for key in (
        "rejection_attempts",
        "typed_quota_failures",
        "pool_live_bytes_before_rejection",
        "pool_live_bytes_after_rejection",
        "device_live_bytes_before_rejection",
        "device_live_bytes_after_rejection",
        "retry_successes",
        "retry_buffer_bytes",
    ):
        nonnegative(probe[key], f"probe.{key}")
    expected_pool_before = anchor + artifact["fragmentation_blocks_bytes"][1]
    if (
        probe["rejection_attempts"] != 1
        or probe["typed_quota_failures"] != 1
        or probe["pool_live_bytes_before_rejection"] != expected_pool_before
        or probe["pool_live_bytes_after_rejection"] != expected_pool_before
        or probe["device_live_bytes_before_rejection"] != baseline_device + expected_pool_before
        or probe["device_live_bytes_after_rejection"] != baseline_device + expected_pool_before
        or probe["retry_successes"] != 1
        or probe["retry_buffer_bytes"] != request
        or probe["destination_preserved"] is not True
    ):
        fail("CUDA probe does not prove typed rejection/preservation/retry")
    for key in (
        "high_water_bytes",
        "backend_high_water_bytes",
        "retry_successes",
        "final_pool_live_bytes",
        "final_device_live_bytes",
    ):
        nonnegative(artifact[key], key)
    if (
        artifact["rejected_live_bytes_preserved"] is not True
        or artifact["high_water_bytes"] != cap
        or artifact["backend_high_water_bytes"] < cap
        or artifact["retry_successes"] != 1
        or artifact["final_pool_live_bytes"] != baseline_pool
        or artifact["final_device_live_bytes"] != baseline_device
        or artifact["unsupported_scope"] != CUDA_UNSUPPORTED_SCOPE
    ):
        fail("CUDA aggregate allocator accounting is incomplete")
    corpus = text(artifact["corpus_sha256"], "corpus_sha256").lower()
    if len(corpus) != 64 or any(c not in "0123456789abcdef" for c in corpus):
        fail("CUDA corpus digest is malformed")
    payload = json.dumps(
        [0, cap, anchor, artifact["fragmentation_blocks_bytes"], request, probe],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if corpus != hashlib.sha256(payload).hexdigest():
        fail("CUDA corpus digest does not match deterministic probe")
    return {
        "device_ordinal": artifact["device_ordinal"],
        "total_device_memory_bytes": total_memory,
        "rejection_attempts": probe["rejection_attempts"],
        "retry_successes": artifact["retry_successes"],
        "high_water_bytes": artifact["high_water_bytes"],
        "final_device_live_bytes": artifact["final_device_live_bytes"],
        "corpus_sha256": corpus,
    }


def main() -> None:
    if len(sys.argv) not in {2, 4} or (len(sys.argv) == 4 and sys.argv[2] != "--output"):
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [--output SUMMARY.json]")
    artifact_path = Path(sys.argv[1])
    artifact = load(artifact_path)
    if artifact.get("artifact_kind") == "sonium-compute.c4.memory-quota-contention":
        result = check_artifact(artifact)
        backend = "cpu-reference"
        cuda_executed = False
        unsupported_scope = UNSUPPORTED_SCOPE
    elif artifact.get("artifact_kind") == "sonium-compute.c4.cuda-memory-quota":
        result = check_cuda_artifact(artifact)
        backend = "cuda"
        cuda_executed = True
        unsupported_scope = CUDA_UNSUPPORTED_SCOPE
    else:
        fail("unknown C4 memory artifact kind")
    summary = {
        "schema": "sonium-compute.c4.memory-quota-contention/v1",
        "status": "passed",
        "target_os": "linux",
        "target_arch": "x86_64",
        "backend": backend,
        "cuda_executed": cuda_executed,
        "artifact_sha256": sha256(artifact_path),
        "evidence": result,
        "unsupported_scope": unsupported_scope,
    }
    if len(sys.argv) == 4:
        output = Path(sys.argv[3])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
