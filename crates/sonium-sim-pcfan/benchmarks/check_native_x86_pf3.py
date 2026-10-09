#!/usr/bin/env python3
"""Fail-closed schema/budget checker for the native PF3 product reference."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA = "sonium.pcfan.native-x86-pf3.v1"
ARTIFACT = "native-x86-pcfan-pf3-v1.json"
GRADIENT_BUDGET = 5.0e-3


def fail(message: str) -> None:
    raise SystemExit(f"PF3 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def summary(value: object, label: str) -> None:
    if not isinstance(value, dict) or value.get("finite") is not True:
        fail(f"{label} is missing or non-finite")
    count = value.get("count")
    if not isinstance(count, int) or count <= 0:
        fail(f"{label}.count is invalid")
    for key in ("min", "max", "mean", "l2"):
        finite(value.get(key), f"{label}.{key}")
    digest = value.get("sha256_le_f64")
    if not isinstance(digest, str) or len(digest) != 64:
        fail(f"{label} digest is missing")


def provenance(value: object, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} provenance is missing")
    if value.get("requested_backend") != "Cpu" or value.get("actual_backend") != "Cpu":
        fail(f"{label} backend provenance is not CPU/CPU")
    if value.get("execution_mode") != "Cpu":
        fail(f"{label} execution mode is not Cpu")
    environment = value.get("environment")
    if not isinstance(environment, dict):
        fail(f"{label} execution environment is missing")
    for key in (
        "target_arch",
        "target_os",
        "compiler",
        "cpu_features",
        "simd_dispatch",
        "linear_solver",
        "cpu_preconditioner",
    ):
        if not isinstance(environment.get(key), str) or not environment[key].strip():
            fail(f"{label}.environment.{key} is missing")
    if environment.get("target_arch") != "x86_64" or environment.get("target_os") != "linux":
        fail(f"{label} target is not native Linux x86_64")
    if not isinstance(environment.get("thread_count"), int) or environment["thread_count"] <= 0:
        fail(f"{label} thread count is invalid")


def check_steady(value: object) -> None:
    if not isinstance(value, dict):
        fail("steady record is missing")
    fields = value.get("fields")
    if not isinstance(fields, dict):
        fail("steady fields are missing")
    for key in ("nodes", "velocity_m_s", "pressure_pa"):
        summary(fields.get(key), f"steady.fields.{key}")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        fail("steady metrics are missing")
    for key in ("airflow_cfm", "pressure_drop_pa", "max_velocity_m_s", "reynolds_number", "residual"):
        finite(metrics.get(key), f"steady.metrics.{key}")
    if not isinstance(metrics.get("iterations"), int) or metrics["iterations"] < 1:
        fail("steady did not execute an iteration")
    objective = value.get("objective")
    if not isinstance(objective, dict) or objective.get("finite") is not True:
        fail("steady objective is not finite")
    if finite(objective.get("decomposition_error"), "steady objective decomposition") > 1.0e-10:
        fail("steady objective decomposition exceeds budget")
    provenance(value.get("provenance"), "steady")


def check_transient(value: object) -> None:
    if not isinstance(value, dict):
        fail("transient record is missing")
    steps = value.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        fail("transient trajectory has fewer than two steps")
    times: list[float] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            fail(f"transient step {index} is malformed")
        times.append(finite(step.get("time_s"), f"transient step {index}.time_s"))
        if index and times[-1] <= times[-2]:
            fail("transient times are not strictly increasing")
        summary(step.get("state"), f"transient step {index}.state")
        residual = finite(step.get("residual"), f"transient step {index}.residual")
        if residual < 0.0:
            fail(f"transient step {index} residual is negative")
        if not isinstance(step.get("ledger"), dict):
            fail(f"transient step {index} conservation ledger is missing")
        step_provenance = step.get("provenance")
        if not isinstance(step_provenance, dict):
            fail(f"transient step {index} provenance is missing")
        provenance(
            {
                "requested_backend": step.get("actual_backend"),
                "actual_backend": step.get("actual_backend"),
                "execution_mode": step_provenance.get("execution_mode"),
                "environment": step_provenance,
            },
            f"transient step {index}",
        )
    observation = value.get("observation")
    if not isinstance(observation, dict) or observation.get("converged") is not True:
        fail("transient observation did not converge")
    finite(observation.get("residual"), "transient observation residual")
    gradient = value.get("gradient")
    if not isinstance(gradient, dict) or gradient.get("converged") is not True:
        fail("transient adjoint did not converge")
    summary(gradient.get("design_gradient"), "transient gradient")
    finite(gradient.get("state_residual_norm"), "transient state residual")
    finite(gradient.get("adjoint_residual_norm"), "transient adjoint residual")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT.json")
    path = Path(sys.argv[1]).resolve()
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON: {error}")
    if artifact.get("schema") != SCHEMA:
        fail("schema/version mismatch")
    if artifact.get("status") != "pass":
        fail("native PF3 artifact is not a pass")
    if artifact.get("scope") != "native Linux x86_64 CPU/f64; Arm64 and Windows waived":
        fail("scope is not the authorized native Linux/x86_64 scope")
    if artifact.get("deterministic_mode") is not True:
        fail("deterministic mode was not enabled")

    host = artifact.get("host")
    if not isinstance(host, dict):
        fail("host provenance is missing")
    for key in ("target_arch", "target_os", "cpu_model", "rustc", "cargo_target_dir"):
        if not isinstance(host.get(key), str) or not host[key].strip():
            fail(f"host.{key} is missing")
    if host["target_arch"] != "x86_64" or host["target_os"] != "linux":
        fail("host is not native Linux x86_64")
    if ".target-unix-pf3" not in host["cargo_target_dir"]:
        fail("producer did not use the PF3 repository-local Cargo target")

    check_steady(artifact.get("steady"))
    check_transient(artifact.get("transient"))
    reduced = artifact.get("reduced_objective_gradient")
    if not isinstance(reduced, dict):
        fail("reduced objective/gradient evidence is missing")
    summary(reduced.get("gradient"), "reduced gradient")
    finite(reduced.get("residual_max_abs"), "reduced gradient residual")
    fd = reduced.get("finite_difference")
    if not isinstance(fd, dict):
        fail("reduced finite-difference evidence is missing")
    budget = finite(fd.get("budget_relative_max"), "finite-difference budget")
    maximum = finite(fd.get("maximum_relative_error"), "finite-difference maximum")
    if budget != GRADIENT_BUDGET or maximum > budget:
        fail("reduced adjoint finite-difference budget failed")
    component_errors = fd.get("component_errors")
    if not isinstance(component_errors, list) or not component_errors:
        fail("finite-difference component errors are missing")
    if any(finite(value, "finite-difference component error") > budget for value in component_errors):
        fail("a finite-difference component exceeds budget")

    cross = artifact.get("cross_architecture")
    if not isinstance(cross, dict) or cross.get("status") != "waived":
        fail("cross-architecture waiver is missing")
    reason = cross.get("reason", "")
    if not isinstance(reason, str) or "no Rosetta" not in reason or "synthetic" not in reason:
        fail("cross-architecture waiver does not reject fabricated evidence")

    checksum_path = path.with_name("checksums.sha256")
    if not checksum_path.is_file():
        fail("checksum sidecar is missing")
    declared = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            fail("malformed checksum sidecar")
        value, name = line.split("  ", 1)
        declared[name] = value
    if declared.get(ARTIFACT) != hashlib.sha256(path.read_bytes()).hexdigest():
        fail("artifact checksum is stale")
    print(json.dumps({"status": "pass", "artifact": str(path), "steps": len(artifact["transient"]["steps"]), "gradient_max_relative_error": maximum}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
