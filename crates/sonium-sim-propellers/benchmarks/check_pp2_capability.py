#!/usr/bin/env python3
"""Validate the complete propeller CUDA/Metal capability-boundary artifact."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact = json.loads(Path(sys.argv[1]).read_text())
    assert artifact["schema"] == "sonium.propellers.cuda-metal-capability.v1"
    assert artifact["status"] == "pass"
    assert artifact["result_kind"] == "capability-boundary-plus-native-cuda-cell"
    assert artifact["scope"].startswith("native Linux x86_64")
    assert artifact["host"]["os"] == "linux"
    assert artifact["host"]["target"].startswith("x86_64-")
    assert artifact["device_inventory"]

    matrix = artifact["matrix"]
    assert matrix["schema"] == "sonium.propeller.execution-evidence.v1"
    cells = matrix["cells"]
    assert len(cells) == 648
    requests = [
        (
            cell["request"]["backend"],
            cell["request"]["workload"],
            cell["request"]["precision"],
            cell["request"]["mesh_order"],
            cell["request"]["direction"],
            cell["request"]["batch"],
        )
        for cell in cells
    ]
    assert len(set(requests)) == 648
    backend_counts = Counter(request[0] for request in requests)
    assert backend_counts == {"cpu": 216, "cuda": 216, "metal": 216}
    device = [cell for cell in cells if cell["request"]["backend"] in {"cuda", "metal"}]
    assert len(device) == 432
    supported = [cell for cell in device if cell["supported"]]
    assert len(supported) == artifact["supported_device_cells"] == 2
    assert {cell["request"]["batch"] for cell in supported} == {"single", "multipoint"}
    assert all(cell["request"] | {"batch": "single"} == {
        "backend": "cuda",
        "workload": "blade_element_bem",
        "precision": "f64",
        "mesh_order": 1,
        "direction": "forward",
        "batch": "single",
    } for cell in supported)
    assert all(not cell["supported"] and cell["reason"] for cell in device if not cell["supported"])
    assert artifact["device_cells"] == 432
    assert artifact["typed_unsupported_cells"] == 430
    assert artifact["requests_with_no_actual_backend"] == 430
    assert artifact["requests_with_no_fallback"] == 430
    assert sum(
        cell["supported"] for cell in cells if cell["request"]["backend"] == "cpu"
    ) == 12

    execution = artifact["device_execution"]
    assert execution["status"] == "bounded-native-cell-separate-artifact"
    assert "Complex64" in execution["adapter"]
    assert execution["actual_backend"] is None
    assert execution["fallback_reason"] is None
    for field in ("allocations", "transfers", "synchronizations", "launches", "peak_live_bytes"):
        assert execution[field] == 0
    assert execution["warm_replays"].startswith("two bounded")
    for probe in artifact["fault_probes"]:
        assert probe["status"] == "not-applicable"
        assert probe["partial_publish"] is False
    print(json.dumps({"cells": len(cells), "device_cells": len(device), "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
