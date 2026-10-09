#!/usr/bin/env python3
"""Fail-closed checker for the bounded PP2 CUDA multipoint acoustic cell."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


SCHEMA = "sonium.propellers.cuda-native-acoustic-sweep.v1"
TOLERANCE = 1.0e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"PP2 CUDA acoustic sweep evidence rejected: {message}")


def finite(value: object, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} is not numeric")
    require(math.isfinite(value), f"{field} is not finite")
    return float(value)


def positive(value: object, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{field} is not positive")
    return value


def nonnegative(value: object, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{field} is not non-negative")
    return value


def complex_value(value: object, field: str) -> complex:
    require(isinstance(value, dict), f"{field} is missing")
    return complex(finite(value.get("re"), f"{field}.re"), finite(value.get("im"), f"{field}.im"))


def runtime(ledger: object, field: str, resident: bool) -> dict[str, object]:
    require(isinstance(ledger, dict), f"{field} is missing")
    if resident:
        positive(ledger.get("live_allocation_bytes_at_sample"), f"{field}.live_allocation_bytes_at_sample")
    else:
        nonnegative(ledger.get("live_allocation_bytes_at_sample"), f"{field}.live_allocation_bytes_at_sample")
    positive(ledger.get("allocation_high_water_bytes"), f"{field}.allocation_high_water_bytes")
    nonnegative(ledger.get("allocation_operations"), f"{field}.allocation_operations")
    nonnegative(ledger.get("kernel_launches"), f"{field}.kernel_launches")
    nonnegative(ledger.get("synchronization_calls"), f"{field}.synchronization_calls")
    transfer = ledger.get("transfers")
    require(isinstance(transfer, dict), f"{field}.transfers is missing")
    for name in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes"):
        nonnegative(transfer.get(name), f"{field}.transfers.{name}")
    resources = ledger.get("driver_resources")
    require(isinstance(resources, dict), f"{field}.driver_resources is missing")
    positive(resources.get("live_contexts"), f"{field}.driver_resources.live_contexts")
    positive(resources.get("live_streams"), f"{field}.driver_resources.live_streams")
    return ledger


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    try:
        artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        require(isinstance(artifact, dict), "artifact is not an object")
        require(artifact.get("schema") == SCHEMA, "schema is invalid")
        require(artifact.get("status") == "pass", "artifact did not pass")
        require(
            artifact.get("result_kind") == "native-production-connected-panel-acoustic-multipoint",
            "result kind is invalid",
        )
        require(str(artifact.get("scope", "")).startswith("native Linux x86_64; CUDA SM120"), "scope is too broad")

        backend = artifact.get("backend")
        require(isinstance(backend, dict), "backend provenance is missing")
        require(backend.get("requested") == backend.get("actual") == "cuda", "backend is not native CUDA")
        require(backend.get("fallback_reason") is None, "a fallback was recorded")
        require(backend.get("precision") == "f64 / Complex64", "precision is invalid")
        require(backend.get("fidelity") == "panel_geometry_surrogate", "fidelity is invalid")
        require(isinstance(backend.get("device_inventory"), str) and backend["device_inventory"].strip(), "inventory is missing")

        state = artifact.get("production_state")
        require(isinstance(state, dict), "production state is missing")
        panels = positive(state.get("panel_count"), "production_state.panel_count")
        require(panels == positive(state.get("pair_count"), "production_state.pair_count"), "panel/pair count mismatch")
        require("PropellerSource" in state.get("source", ""), "source is not production-connected")
        require("BladeMesh" in state.get("geometry", ""), "geometry is not production-connected")
        require("phase-zero" in state.get("phase_convention", ""), "acoustic phase convention is missing")

        multipoint = artifact.get("multipoint")
        require(isinstance(multipoint, dict), "multipoint result is missing")
        require(multipoint.get("batch") == "multipoint", "batch was relabelled")
        harmonics = multipoint.get("harmonics")
        require(isinstance(harmonics, list) and len(harmonics) == 3, "expected three harmonic points")
        require(multipoint.get("count") == len(harmonics), "harmonic count mismatch")
        require(multipoint.get("cpu_reference", "").startswith("independent f64"), "independent CPU oracle is missing")
        tolerance = finite(multipoint.get("parity_tolerance"), "multipoint.parity_tolerance")
        require(tolerance <= TOLERANCE, "parity tolerance is too loose")
        expected_harmonics = [1, 2, 3]
        for expected, harmonic in zip(expected_harmonics, harmonics):
            require(isinstance(harmonic, dict), "harmonic row is not an object")
            require(harmonic.get("harmonic") == expected, "harmonic sequence is not deterministic")
            finite(harmonic.get("frequency_hz"), "harmonic.frequency_hz")
            finite(harmonic.get("source_pressure_pa"), "harmonic.source_pressure_pa")
            cpu = complex_value(harmonic.get("cpu_panel_pressure"), "harmonic.cpu_panel_pressure")
            cuda = complex_value(harmonic.get("cuda_panel_pressure"), "harmonic.cuda_panel_pressure")
            error = finite(harmonic.get("max_relative_error"), "harmonic.max_relative_error")
            require(error <= tolerance, "CUDA/CPU panel acoustic parity exceeded tolerance")
            require(abs(cuda - cpu) <= tolerance * max(abs(cpu), 1.0), "acoustic state differs from CPU oracle")

        counters = runtime(artifact.get("counters"), "counters", resident=True)
        require(counters["kernel_launches"] == 3, "one native launch per harmonic was not recorded")
        require(counters["synchronization_calls"] == 3, "one synchronization per harmonic was not recorded")
        transfers = counters["transfers"]
        require(transfers["h2d_operations"] == transfers["h2d_bytes"] == 0, "warm sweep re-uploaded resident geometry")
        require(transfers["d2h_operations"] == 3 and transfers["d2h_bytes"] > 0, "harmonic publications were not copied back")
        initialization = runtime(artifact.get("initialization"), "initialization", resident=True)
        require(initialization["allocation_operations"] > 0, "initialization did not allocate resident buffers")
        require(initialization["transfers"]["h2d_operations"] > 0, "initialization did not upload geometry")
        positive(artifact.get("residency", {}).get("resident_bytes_at_sample"), "residency.resident_bytes_at_sample")

        malformed = artifact.get("malformed_input")
        require(isinstance(malformed, dict), "malformed-input evidence is missing")
        require(malformed.get("status") == "typed_rejected", "malformed input was not rejected")
        require(malformed.get("device_work") is False, "malformed input reached the device")
        require(malformed.get("output_unchanged") is True, "malformed input changed output")
        lanes = malformed.get("lanes")
        require(isinstance(lanes, list) and {lane.get("error_kind") for lane in lanes} == {"empty_harmonic_batch", "invalid_wave_number"}, "malformed lanes are incomplete")

        failure = artifact.get("failure_and_retry")
        require(isinstance(failure, dict), "failure/retry evidence is missing")
        require(failure.get("status") == "typed_rejected", "launch failure was not typed-rejected")
        error = failure.get("error")
        require(isinstance(error, dict), "typed launch error is missing")
        require(error.get("kind") == "launch", "failure kind is not launch")
        require(error.get("operation"), "launch operation is missing")
        require(error.get("context_disposition") == "synchronize_before_reuse", "launch recovery disposition is invalid")
        # The backend marks the in-flight device output unusable, while the
        # adapter contract separately guarantees that the prior host
        # publication remains unchanged and is not replaced by a failed row.
        require(error.get("output_disposition") == "unusable", "launch backend output disposition is invalid")
        require(failure.get("output") == {"unchanged": True, "published": False}, "failed sweep published output")
        recovery = failure.get("recovery")
        require(isinstance(recovery, dict), "recovery record is missing")
        require(recovery.get("strategy") == "recover_after_error_then_retry_same_session", "wrong retry strategy")
        require(recovery.get("retry_status") == "passed" and recovery.get("retry_points") == 2, "retry was not checked")
        before = runtime(failure.get("ledger", {}).get("before"), "failure.ledger.before", resident=True)
        after_failure = runtime(failure.get("ledger", {}).get("after_failure"), "failure.ledger.after_failure", resident=True)
        require(after_failure["live_allocation_bytes_at_sample"] == before["live_allocation_bytes_at_sample"], "failure changed resident bytes")
        after_retry = runtime(failure.get("ledger", {}).get("after_retry"), "failure.ledger.after_retry", resident=True)
        require(after_retry["allocation_operations"] == 0, "retry reallocated the resident workspace")
        require(after_retry["kernel_launches"] == after_retry["synchronization_calls"] == 2, "retry counters are incomplete")
        require(after_retry["transfers"]["h2d_operations"] == 0, "retry re-uploaded geometry")

        unsupported = artifact.get("unsupported_or_unexercised")
        require(isinstance(unsupported, list) and unsupported, "unsupported boundaries are missing")
        joined = " ".join(str(item).lower() for item in unsupported)
        for term in ("host-side", "observer", "wake", "transpose", "transient", "metal"):
            require(term in joined, f"unsupported boundary {term} was omitted")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps({"schema": SCHEMA, "harmonics": 3, "panel_count": panels, "status": "pass"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
