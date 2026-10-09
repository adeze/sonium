#!/usr/bin/env python3
"""Strict checker for the SR2 native-CUDA breadth artifact."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    try:
        artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"SR2 breadth artifact error: {error}", file=sys.stderr)
        return 2

    assert artifact.get("schema_version") == 1
    assert artifact.get("artifact") == "sonium-sim-room.sr2-cuda-breadth"
    assert artifact.get("status") == "passed"
    contract = artifact["contract"]
    assert contract["same_fixture"] is True
    assert contract["source_count"] == 3
    assert contract["listener_count"] == 3
    assert contract["frequency_count"] == 4
    assert all(finite(value) and value > 0.0 for value in contract["frequencies_hz"])
    assert all(right > left for left, right in zip(contract["frequencies_hz"], contract["frequencies_hz"][1:]))
    for index, frequency in enumerate(contract["frequencies_hz"]):
        expected = math.exp(
            math.log(70.0) + (math.log(130.0) - math.log(70.0)) * index / 3.0
        )
        assert math.isclose(frequency, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert isinstance(contract["solver_tolerance"], (int, float)) and contract["solver_tolerance"] > 0.0
    assert isinstance(contract["precision"], str) and contract["precision"].startswith("f64-outer")

    backend = artifact["backend"]
    assert backend["requested"] == "cuda"
    assert backend["actual"] == "cuda"
    assert backend["fallback_status"] == "none"
    assert backend["host_architecture"] == "x86_64"

    resident = artifact["resident"]
    assert resident["state"].startswith("device CSR/RHS/solver state reused")
    assert resident["preconditioner_ages"] == [0, 1, 2, 3]
    assert resident["allocation_watermark_stable"] is True
    watermarks = resident["allocation_high_water_bytes_by_frequency"]
    assert len(watermarks) == 4
    assert all(isinstance(value, int) and value > 0 for value in watermarks)
    assert all(right >= left for left, right in zip(watermarks, watermarks[1:]))
    assert len(set(watermarks)) == 1
    assert resident["warm_frequency_count"] == 3
    assert isinstance(resident["warm_h2d_bytes"], int) and resident["warm_h2d_bytes"] > 0
    assert isinstance(resident["warm_d2h_bytes"], int) and resident["warm_d2h_bytes"] > 0
    assert resident["warm_transfers_are_explicit"] is True
    frequency_stats = resident["frequency_stats"]
    assert len(frequency_stats) == 4
    for index, stats in enumerate(frequency_stats):
        assert stats["preconditioner_age"] == index
        assert finite(stats["frequency_hz"]) and stats["frequency_hz"] == contract["frequencies_hz"][index]
        assert finite(stats["max_residual"]) and stats["max_residual"] >= 0.0
        assert stats["converged"] is True
        assert isinstance(stats["allocation_operations"], int) and stats["allocation_operations"] > 0
        assert stats["allocation_high_water_bytes"] == watermarks[index]
        assert isinstance(stats["synchronization_calls"], int) and stats["synchronization_calls"] > 0
        transfer = stats["transfer"]
        for key in ("h2d_operations", "h2d_bytes", "d2h_operations", "d2h_bytes", "d2d_operations", "d2d_bytes"):
            assert isinstance(transfer[key], int) and transfer[key] >= 0
        assert transfer["h2d_operations"] > 0 and transfer["h2d_bytes"] > 0
        assert transfer["d2h_operations"] > 0 and transfer["d2h_bytes"] > 0

    parity = artifact["parity"]
    assert parity["reference_backend"] == "cpu"
    assert parity["candidate_backend"] == "cuda"
    assert parity["sample_count"] == 3 * 3 * 4
    for metric, budget in (("rms_relative_error", "rms_relative"), ("max_relative_error", "max_relative")):
        assert finite(parity[metric]) and 0.0 <= parity[metric] <= parity["budgets"][budget]
    assert parity["budgets"] == {"rms_relative": 0.25, "max_relative": 0.25}

    derivative = artifact["derivative"]
    assert derivative["parameter"] == "source[0].amplitude"
    assert derivative["objective"] == "sum listener-frequency |pressure|² for source 0"
    assert derivative["step"] == 1.0e-3
    for key in (
        "cpu_finite_difference",
        "cpu_linear_source_oracle",
        "cuda_linear_source_derivative",
        "cpu_oracle_relative_error",
        "cuda_relative_error_to_cpu_oracle",
    ):
        assert finite(derivative[key])
    assert derivative["cpu_finite_difference"] > 0.0
    assert derivative["cpu_oracle_relative_error"] <= 5.0e-3
    assert derivative["cuda_relative_error_to_cpu_oracle"] <= 0.25
    assert "CPU centered difference" in derivative["oracle"]

    failure = artifact["failure_recovery"]
    assert failure["class"] == "launch"
    assert failure["deterministic"] is True
    assert failure["backend"] == "cuda"
    assert failure["stage"] == "launch"
    assert failure["operation"] == "cuda_launch"
    assert failure["context_recovery"] == "synchronize-before-reuse"
    assert failure["partial_result_published"] is False
    assert failure["resources_released_on_unwind"] is True
    assert "injected" in failure["detail"]
    assert failure["retry_mode"] == "fresh-context"
    assert failure["retry_backend"] == "cuda"
    assert failure["retry_fallback_status"] == "none"
    assert failure["retry_frequency_count"] == 4
    assert failure["retry_converged"] is True
    assert finite(failure["retry_parity_rms_relative_error"])
    assert finite(failure["retry_parity_max_relative_error"])
    assert failure["retry_parity_rms_relative_error"] <= 1.0e-8
    assert failure["retry_parity_max_relative_error"] <= 1.0e-6

    malformed = artifact["malformed_input"]
    assert malformed["status"] == "typed_rejected"
    assert malformed["publication"] == "none"
    assert "strictly increasing" in malformed["error"]
    assert artifact["external_physical_validation"].startswith("unavailable:")
    assert artifact["hermitian_operator_claim"].startswith("not_claimed:")
    print(json.dumps({"status": "passed", "frequency_count": 4, "parity_samples": 36, "derivative": "passed", "failure_recovery": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
