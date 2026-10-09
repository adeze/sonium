#!/usr/bin/env python3
"""Strict checker for the bounded production-connected PP2 CUDA cell."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def positive(value: int | float, name: str) -> None:
    assert isinstance(value, (int, float)) and value > 0, (name, value)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact = json.loads(Path(sys.argv[1]).read_text())
    assert artifact["schema"] == "sonium.propellers.cuda-native-adapter.v1"
    assert artifact["status"] == "pass"
    assert artifact["result_kind"] == "native-production-connected-bem-panel-pair"
    assert artifact["scope"].startswith("native Linux x86_64")

    backend = artifact["backend"]
    assert backend["requested"] == backend["actual"] == "cuda"
    assert backend["fallback_reason"] is None
    assert backend["precision"] == "f64 / Complex64"
    assert backend["fidelity"] == "panel_geometry_surrogate"
    assert backend["device_inventory"]

    source = artifact["source_model"]
    assert "host-side" in source["stage"]
    assert source["harmonic"] >= 1
    assert math.isfinite(source["wave_number"]) and source["wave_number"] > 0
    assert math.isfinite(source["harmonic_factor"])

    geometry = artifact["geometry"]
    positive(geometry["panel_count"], "panel_count")
    positive(geometry["pair_count"], "pair_count")
    assert geometry["panel_count"] == geometry["pair_count"]

    parity = artifact["parity"]
    assert parity["reference"] == "sonium_compute::bem_cpu::regular_pair_f64_batch"
    assert parity["sample_count"] == geometry["pair_count"] * 4
    assert math.isfinite(parity["max_relative_error"])
    assert parity["max_relative_error"] <= parity["tolerance"] <= 1.0e-10

    residency = artifact["residency"]
    assert "device buffer" in residency["panel_input"]
    assert "device buffer" in residency["pair_output"]
    assert "host-side" in residency["source_model_and_panelization"]
    assert "reusable session" in residency["steady_state_policy"]

    warm = artifact["warm_replay"]
    assert warm["replays"] == 2
    assert "harmonic" in warm["parameter_update"]
    positive(warm["resident_device_bytes_at_sample"], "warm resident bytes")
    initialization = warm["session_initialization"]
    positive(initialization["allocation_operations"], "session initialization allocations")
    positive(initialization["transfers"]["h2d_operations"], "session initialization H2D operations")
    first = warm["first_replay"]
    second = warm["second_replay"]
    positive(first["allocation_operations"], "first warm replay allocations")
    positive(first["kernel_launches"], "first warm replay launches")
    positive(first["synchronization_calls"], "first warm replay synchronizations")
    positive(first["transfers"]["d2h_operations"], "first warm replay D2H operations")
    assert second["allocation_operations"] == 0
    assert second["transfers"]["h2d_operations"] == 0
    assert second["transfers"]["h2d_bytes"] == 0
    assert second["transfers"]["d2h_operations"] == 1
    positive(second["transfers"]["d2h_bytes"], "second warm replay D2H bytes")
    assert second["kernel_launches"] == 1
    assert second["synchronization_calls"] == 1
    positive(second["live_allocation_bytes_at_sample"], "second warm resident bytes")
    assert second["driver_resources"]["live_contexts"] > 0
    assert second["driver_resources"]["live_streams"] > 0

    counters = artifact["counters"]
    for field in ("allocation_operations", "allocation_high_water_bytes", "kernel_launches", "synchronization_calls"):
        positive(counters[field], field)
    assert counters["live_allocation_bytes_at_sample"] == 0
    transfers = counters["transfers"]
    for field in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes"):
        positive(transfers[field], field)
    assert counters["driver_resources"]["live_contexts"] > 0
    assert counters["driver_resources"]["live_streams"] > 0

    unsupported = artifact["unsupported_or_unexercised"]
    assert isinstance(unsupported, list) and unsupported
    assert any("host-side" in item for item in unsupported)
    assert any("unsupported" in item or "unimplemented" in item for item in unsupported)

    print(json.dumps({"result_kind": artifact["result_kind"], "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
