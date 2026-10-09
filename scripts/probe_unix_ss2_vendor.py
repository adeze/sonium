#!/usr/bin/env python3
"""Probe local sparse vendor candidates without manufacturing a speed baseline.

OpenBLAS is a dense BLAS implementation and PETSc's shared library alone does
not identify a pinned sparse MatMult configuration.  This probe records those
facts explicitly so SS2 can distinguish local kernel evidence from a vendor
comparison that has actually been supplied and configured.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path


SCHEMA = "sonium-solvers-sparse.ss2.vendor-probe/v1"


def command_output(command: list[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, "command unavailable"
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    text = (result.stdout or result.stderr).strip()
    return result.returncode == 0, text


def detected_libraries() -> dict[str, str]:
    ok, listing = command_output(["ldconfig", "-p"])
    if not ok:
        return {}
    result = {}
    for name in ("openblas", "petsc"):
        matches = [line.strip() for line in listing.splitlines() if name in line.lower()]
        if matches:
            result[name] = matches[0]
    return result


def candidate(
    name: str,
    detection: str,
    reason: str,
    *,
    sparse_spmv_api: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "detection": detection,
        "sparse_spmv_api": sparse_spmv_api,
        "pinned": False,
        "usable": False,
        "reason": reason,
    }


def probe() -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return {
            "schema": SCHEMA,
            "status": "unavailable",
            "target_os": platform.system().lower(),
            "target_arch": platform.machine(),
            "release_comparative_claims": False,
            "candidates": [
                candidate("openblas", "not_probed", "SS2 vendor probe requires native Linux x86_64"),
                candidate("petsc", "not_probed", "SS2 vendor probe requires native Linux x86_64"),
            ],
            "rejection_reasons": ["unsupported host for the Unix/x86 vendor probe"],
        }

    libraries = detected_libraries()
    openblas_detection = libraries.get("openblas", "not detected by ldconfig")
    petsc_detection = libraries.get("petsc", "not detected by ldconfig")
    candidates = [
        candidate(
            "openblas",
            openblas_detection,
            "OpenBLAS exposes dense BLAS kernels; no configured CSR/SpMV API is available for this fixture, so dgemv cannot serve as a sparse vendor reference.",
        ),
        candidate(
            "petsc",
            petsc_detection,
            "A PETSc shared library exposes sparse MatMult entry points, but it does not pin matrix format, options, thread affinity, or build provenance; no reproducible configured PETSc sparse run was supplied.",
            sparse_spmv_api=True,
        ),
    ]
    return {
        "schema": SCHEMA,
        "status": "unavailable",
        "target_os": "linux",
        "target_arch": "x86_64",
        "release_comparative_claims": False,
        "probe_contract": {
            "required_for_pass": [
                "configured sparse SpMV entry point",
                "pinned library/build/version",
                "identical matrix checksum and scalar/action corpus",
                "independent correctness gate before timing",
                "thread/affinity and cache-regime provenance",
            ],
            "speedup_publication": "withheld unless every required field is supplied",
        },
        "candidates": candidates,
        "rejection_reasons": [candidate_item["reason"] for candidate_item in candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
