#!/usr/bin/env python3
"""Fail-closed checker for bounded PP2 CUDA lifecycle fault evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCHEMA = "sonium.propellers.cuda-native-adapter-failures.v1"
EXPECTED = {
    "allocation": ("allocation", "cuda_allocation", "reusable", "preserved"),
    "launch": ("launch", "cuda_launch", "synchronize_before_reuse", "unusable"),
    "synchronization": (
        "synchronization",
        "cuda_synchronize",
        "must_recreate",
        "unusable",
    ),
    "device_loss": ("device_loss", "cuda_device_loss", "must_recreate", "unusable"),
}


def fail(message: str) -> None:
    raise ValueError(f"PP2 CUDA lifecycle evidence rejected: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def positive(value: object, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{field} must be positive")
    return value


def nonnegative(value: object, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{field} must be non-negative")
    return value


def text(value: object, field: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{field} must be non-empty text")
    return value


def check_fresh_context(ledger: dict[str, object], field: str) -> None:
    require(isinstance(ledger, dict), f"{field} is missing")
    require(ledger.get("live_allocation_bytes") == 0, f"{field} leaked device bytes")
    require(ledger.get("live_staging_bytes") == 0, f"{field} leaked staging bytes")
    require(ledger.get("allocation_high_water_bytes") == 0, f"{field} retained allocation watermark")
    resources = ledger.get("driver_resources")
    require(isinstance(resources, dict), f"{field}.driver_resources is missing")
    positive(resources.get("live_contexts"), f"{field}.driver_resources.live_contexts")
    positive(resources.get("live_streams"), f"{field}.driver_resources.live_streams")


def check_runtime(ledger: dict[str, object], field: str) -> None:
    require(isinstance(ledger, dict), f"{field} is missing")
    positive(ledger.get("live_allocation_bytes_at_sample"), f"{field}.live_allocation_bytes_at_sample")
    positive(ledger.get("allocation_high_water_bytes"), f"{field}.allocation_high_water_bytes")
    nonnegative(ledger.get("allocation_operations"), f"{field}.allocation_operations")
    resources = ledger.get("driver_resources")
    require(isinstance(resources, dict), f"{field}.driver_resources is missing")
    positive(resources.get("live_contexts"), f"{field}.driver_resources.live_contexts")
    positive(resources.get("live_streams"), f"{field}.driver_resources.live_streams")


def check_lane(lane: dict[str, object]) -> None:
    name = text(lane.get("name"), "lane.name")
    require(lane.get("requested_backend") == "cuda", f"{name} requested backend is not cuda")
    output = lane.get("output")
    require(isinstance(output, dict), f"{name}.output is missing")
    require(output.get("unchanged") is True, f"{name} changed the prior output")
    require(output.get("published") is False, f"{name} published a failed result")

    if name == "cancellation":
        require(lane.get("status") == "unsupported", "cancellation must remain explicitly unsupported")
        require(lane.get("actual_backend") is None, "unsupported cancellation has an actual backend")
        require("unsupported" in text(lane.get("fallback_reason"), "cancellation.fallback_reason"), "cancellation fallback is not explicit")
        require(lane.get("failure") is None, "unsupported cancellation must not fabricate a failure")
        recovery = lane.get("recovery")
        require(isinstance(recovery, dict), "cancellation recovery record is missing")
        require(recovery.get("strategy") == "not_supported", "cancellation strategy was relabelled")
        text(lane.get("reason"), "cancellation.reason")
        return

    require(lane.get("actual_backend") == "cuda", f"{name} actual backend is not cuda")
    require(lane.get("fallback_reason") is None, f"{name} has a fallback reason")
    expected_kind, expected_operation, expected_context, expected_output = EXPECTED[name]
    require(lane.get("status") == "typed_rejected", f"{name} did not produce typed rejection")
    failure = lane.get("failure")
    require(isinstance(failure, dict), f"{name}.failure is missing")
    require(failure.get("kind") == expected_kind, f"{name} error kind is invalid")
    require(failure.get("operation") == expected_operation, f"{name} operation is invalid")
    require(failure.get("context_disposition") == expected_context, f"{name} recovery disposition is invalid")
    require(failure.get("output_disposition") == expected_output, f"{name} output disposition is invalid")
    text(failure.get("typed_debug"), f"{name}.failure.typed_debug")

    recovery = lane.get("recovery")
    require(isinstance(recovery, dict), f"{name}.recovery is missing")
    require(recovery.get("context_disposition") == expected_context, f"{name} recovery mismatch")
    require(recovery.get("recovery_status") in {"passed", "original_error_retained"}, f"{name} recovery was not checked")
    require(recovery.get("retry_status") in {"passed", "passed_on_fresh_context"}, f"{name} retry was not checked")

    ledger = lane.get("ledger")
    require(isinstance(ledger, dict), f"{name}.ledger is missing")
    before = ledger.get("before")
    after_failure = ledger.get("after_failure")
    check_runtime(before, f"{name}.ledger.before")
    check_runtime(after_failure, f"{name}.ledger.after_failure")
    require(
        after_failure["live_allocation_bytes_at_sample"] == before["live_allocation_bytes_at_sample"],
        f"{name} failure changed the resident allocation ledger",
    )
    check_fresh_context(ledger.get("fresh_context_after_drop"), f"{name}.ledger.fresh_context_after_drop")

    if name == "allocation":
        after_recovery = ledger.get("after_recovery")
        check_runtime(after_recovery, "allocation.ledger.after_recovery")
        require(
            after_recovery["live_allocation_bytes_at_sample"] > before["live_allocation_bytes_at_sample"],
            "allocation retry did not materialize the reusable workspace",
        )
    elif name == "launch":
        after_recovery = ledger.get("after_recovery")
        check_runtime(after_recovery, "launch.ledger.after_recovery")
        require(
            after_recovery["live_allocation_bytes_at_sample"] == before["live_allocation_bytes_at_sample"],
            "launch recovery changed resident allocation bytes",
        )
    else:
        check_fresh_context(ledger.get("fresh_context_before_retry"), f"{name}.ledger.fresh_context_before_retry")
        after_recovery = ledger.get("after_recovery")
        check_runtime(after_recovery, f"{name}.ledger.after_recovery")


def check(path: Path) -> dict[str, object]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read artifact: {error}")
    require(isinstance(artifact, dict), "artifact must be an object")
    require(artifact.get("schema") == SCHEMA, "schema is invalid")
    require(artifact.get("status") == "pass", "artifact did not pass")
    require(
        artifact.get("result_kind") == "native-production-connected-bem-panel-lifecycle-failures",
        "result kind is invalid",
    )
    require(str(artifact.get("scope", "")).startswith("native Linux x86_64; CUDA SM120"), "scope is too broad")
    backend = artifact.get("backend")
    require(isinstance(backend, dict), "backend provenance is missing")
    require(backend.get("requested") == backend.get("actual") == "cuda", "typed backend provenance is invalid")
    require(backend.get("fallback_reason") is None, "backend fallback was recorded")
    require(backend.get("precision") == "f64 / Complex64", "precision provenance is invalid")
    require(backend.get("fidelity") == "panel_geometry_surrogate", "fidelity provenance is invalid")
    text(backend.get("device_inventory"), "backend.device_inventory")

    lanes = artifact.get("lanes")
    require(isinstance(lanes, list), "lifecycle lanes are missing")
    require(all(isinstance(lane, dict) for lane in lanes), "lifecycle lane is not an object")
    require(
        {lane.get("name") for lane in lanes} == set(EXPECTED) | {"cancellation"},
        "lifecycle lanes are incomplete or duplicated",
    )
    for lane in lanes:
        check_lane(lane)

    unsupported = artifact.get("unsupported_or_unexercised")
    require(isinstance(unsupported, list) and unsupported, "unsupported scope is missing")
    joined = " ".join(str(item).lower() for item in unsupported)
    for term in ("host-side", "flow", "wake", "observer", "structural", "cancellation"):
        require(term in joined, f"unsupported scope does not retain {term} boundary")
    return {"schema": SCHEMA, "lanes": len(lanes), "status": "pass"}


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} ARTIFACT", file=sys.stderr)
        return 2
    try:
        result = check(Path(sys.argv[1]))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
