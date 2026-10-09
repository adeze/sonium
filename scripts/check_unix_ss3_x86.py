#!/usr/bin/env python3
"""Strict checker for the portable/native SS3 NUMA policy artifact."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SS3 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
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
        (parts[0].lower() for row in rows if len(parts := row.split()) == 2 and parts[1].lstrip("*") == path.name),
        None,
    )
    if expected != digest(path):
        fail(f"checksum mismatch for {path.name}")
    return value


def check_partition_stats(stats: object, label: str) -> None:
    if not isinstance(stats, list) or not stats:
        fail(f"{label} has no partition statistics")
    previous_end = 0
    for index, entry in enumerate(stats):
        if not isinstance(entry, dict):
            fail(f"{label}[{index}] is not an object")
        for key in ("row_start", "num_rows", "nnz", "logical_cpu_count"):
            value = entry.get(key)
            if not isinstance(value, int) or value < 0:
                fail(f"{label}[{index}].{key} is invalid")
        if entry["row_start"] != previous_end or entry["num_rows"] <= 0:
            fail(f"{label} is not contiguous at partition {index}")
        previous_end = entry["row_start"] + entry["num_rows"]
        require_text(entry.get("first_touch_affinity"), f"{label}[{index}].first_touch_affinity")
        if not isinstance(entry.get("first_touch_affinity_applied"), bool):
            fail(f"{label}[{index}].first_touch_affinity_applied is invalid")


def main() -> None:
    if len(sys.argv) not in {2, 4}:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [--output SUMMARY.json]")
    artifact_path = Path(sys.argv[1])
    if len(sys.argv) == 4 and sys.argv[2] != "--output":
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json [--output SUMMARY.json]")
    artifact = load(artifact_path)
    if artifact.get("schema_version") != 1:
        fail("unsupported schema version")
    require_text(artifact.get("case_id"), "case_id")
    scope = require_text(artifact.get("evidence_scope"), "evidence_scope")
    if scope not in {"functional_policy_parity", "multi_socket_placement_candidate"}:
        fail("unknown evidence scope")

    topology = artifact.get("topology")
    if not isinstance(topology, dict):
        fail("topology is missing")
    require_text(topology.get("summary"), "topology.summary")
    for key in ("logical_cpus",):
        if not isinstance(topology.get(key), int) or topology[key] <= 0:
            fail(f"topology.{key} is invalid")
    sockets = topology.get("sockets")
    nodes = topology.get("numa_nodes")
    if not isinstance(sockets, list) or not sockets or not isinstance(nodes, list) or not nodes:
        fail("topology sockets/numa_nodes are incomplete")
    socket_count = len(sockets)
    node_count = len(nodes)
    if scope == "functional_policy_parity" and (socket_count >= 2 or node_count >= 2):
        fail("multi-socket topology must not be marked functional-only")
    if scope == "multi_socket_placement_candidate" and (socket_count < 2 or node_count < 2):
        fail("placement candidate lacks multi-socket topology")

    admission = artifact.get("admission")
    if not isinstance(admission, dict):
        fail("admission is missing")
    for key in ("requested_partitions", "admitted_partitions", "estimated_storage_bytes", "replicated_policy_storage_bytes"):
        if not isinstance(admission.get(key), int) or admission[key] <= 0:
            fail(f"admission.{key} is invalid")
    if admission["admitted_partitions"] > admission["requested_partitions"]:
        fail("admitted partitions exceed requested partitions")
    for key in ("memory_budget_admitted", "memory_budget_rejected", "policy_memory_budget_admitted", "policy_memory_budget_rejected"):
        if admission.get(key) is not True:
            fail(f"admission.{key} did not pass")

    partitioning = artifact.get("partitioning")
    if not isinstance(partitioning, dict):
        fail("partitioning is missing")
    require_text(partitioning.get("placement_status"), "partitioning.placement_status")
    if not isinstance(partitioning.get("placement_available"), bool):
        fail("partitioning.placement_available is invalid")
    check_partition_stats(partitioning.get("stats"), "partitioning.stats")
    finite(partitioning.get("row_imbalance"), "partitioning.row_imbalance")
    finite(partitioning.get("nnz_imbalance"), "partitioning.nnz_imbalance")

    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        fail("runtime is missing")
    matrix = runtime.get("matrix")
    if not isinstance(matrix, dict):
        fail("runtime.matrix is missing")
    for key in ("rows", "cols", "nnz"):
        if not isinstance(matrix.get(key), int) or matrix[key] <= 0:
            fail(f"runtime.matrix.{key} is invalid")
    for key in ("requested_partitions", "admitted_partitions", "iterations"):
        if not isinstance(runtime.get(key), int) or runtime[key] <= 0:
            fail(f"runtime.{key} is invalid")
    if runtime["admitted_partitions"] > runtime["requested_partitions"]:
        fail("runtime admitted partitions exceed requested partitions")
    rates = runtime.get("effective_gb_per_second")
    elapsed = runtime.get("elapsed_seconds")
    if not isinstance(rates, dict) or not isinstance(elapsed, dict):
        fail("runtime timing/bandwidth maps are missing")
    policies = {"shared_unpinned", "shared_pinned", "replicated_pinned"}
    if set(rates) != policies or set(elapsed) != policies:
        fail("runtime does not cover all three NUMA policies")
    for policy in policies:
        if finite(rates[policy], f"runtime.effective_gb_per_second.{policy}") <= 0:
            fail(f"runtime bandwidth for {policy} is not positive")
        if finite(elapsed[policy], f"runtime.elapsed_seconds.{policy}") <= 0:
            fail(f"runtime elapsed time for {policy} is not positive")
    parity = runtime.get("action_parity")
    if not isinstance(parity, dict) or set(parity) != policies:
        fail("runtime action parity does not cover all policies")
    for policy in policies:
        row = parity[policy]
        if not isinstance(row, dict):
            fail(f"runtime.action_parity.{policy} is invalid")
        for action in ("forward", "transpose", "hermitian"):
            if row.get(action) != "passed":
                fail(f"{policy}/{action} parity did not pass")
        if finite(row.get("max_abs_error"), f"{policy}.max_abs_error") > 1.0e-10:
            fail(f"{policy} parity exceeds 1e-10")
        if not isinstance(row.get("repetitions"), int) or row["repetitions"] < 2:
            fail(f"{policy} lacks repeated action evidence")
    require_text(runtime.get("locality_signal"), "runtime.locality_signal")
    check_partition_stats(runtime.get("partition_stats"), "runtime.partition_stats")
    if scope == "functional_policy_parity" and "unavailable" not in runtime["locality_signal"]:
        fail("UMA artifact must explicitly mark locality unavailable")

    summary = {
        "schema": "sonium-solvers-sparse.ss3.native-x86/v1",
        "status": "passed",
        "sha256": digest(artifact_path),
        "evidence_scope": scope,
        "socket_count": socket_count,
        "numa_node_count": node_count,
        "policies": sorted(policies),
        "action_parity": "forward/transpose/hermitian passed for all policies",
        "locality_signal": runtime["locality_signal"],
        "unsupported_scope": "genuine multi-socket locality/scaling is not claimed on UMA hosts",
    }
    if len(sys.argv) == 4:
        Path(sys.argv[3]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
