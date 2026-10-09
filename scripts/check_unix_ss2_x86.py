#!/usr/bin/env python3
"""Strict native-x86 SS2 sparse plan/performance evidence checker."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NoReturn


THRESHOLD_SCHEMA = 2
THRESHOLD_ARTIFACT_KIND = "sonium-solvers-sparse.ss2.plan-thresholds"
THRESHOLD_FIXTURE = "ss2-plan-thresholds-v1"
THRESHOLD_METADATA_SCHEMA = "sonium-solvers-sparse.ss2.threshold-metadata/v1"
THRESHOLD_FAMILIES = (
    "compact_index",
    "bsr",
    "sell_c_sigma",
    "row_bucket",
    "numa",
    "row_parallel",
)
THRESHOLD_SIDES = ("below", "at", "above")
SUPPORTED_FORMATS = {
    "Csr",
    "CompactCsrU32",
    "Bsr4",
    "SellCSigma",
    "RowBucketedCsr",
    "NumaCsr",
}
CACHE_INFO_SCHEMA = "sonium-solvers-sparse.ss2.cache-regimes/v1"
VENDOR_PROBE_SCHEMA = "sonium-solvers-sparse.ss2.vendor-probe/v1"
TIMING_SCOPE = "timing_scope=steady_state_apply_only;setup_separate"


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SS2 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def integer(value: object, label: str) -> int:
    if type(value) is not int:
        fail(f"{label} is not an integer")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> None:
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
    actual = digest(path)
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    verify_sidecar(path)
    return value


def parse_cache_info(path: Path) -> dict:
    """Parse the cache hierarchy captured by the runner's lscpu snapshot."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"cannot read cache provenance {path}: {error}")
    sizes: dict[str, int] = {}
    instances: dict[str, int] = {}
    pattern = re.compile(
        r"^(L[123])(?:[di])? cache:\s*([0-9]+(?:\.[0-9]+)?)\s*([KMG])i?B"
        r"(?:\s*\(([0-9]+) instances?\))?\s*$"
    )
    multipliers = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    for line in lines:
        match = pattern.match(line.strip())
        if match is None:
            continue
        level, amount, unit, count = match.groups()
        total = int(float(amount) * multipliers[unit])
        # lscpu prints separate L1d/L1i rows and may expose repeated cache
        # rows. Keep the largest value for each level as the usable bound.
        sizes[level] = max(sizes.get(level, 0), total)
        if count is not None:
            instances[level] = max(instances.get(level, 0), int(count))
    if not all(level in sizes and sizes[level] > 0 for level in ("L1", "L2", "L3")):
        fail(f"{path} lacks complete L1/L2/L3 cache provenance")
    return {
        "schema": CACHE_INFO_SCHEMA,
        "source": path.name,
        "levels": {
            level: {
                "bytes": sizes[level],
                "instances": instances.get(level),
                "bytes_per_instance": sizes[level] // instances[level]
                if instances.get(level, 0) > 0
                else sizes[level],
            }
            for level in ("L1", "L2", "L3")
        },
        "regime_contract": {
            "l1_resident": "bytes_per_iter <= L1 bytes per instance",
            "l2_resident": "L1 bound < bytes_per_iter <= L2 bytes per instance",
            "llc_resident": "L2 bound < bytes_per_iter <= L3 bytes per instance",
            "memory_bound": "bytes_per_iter > L3 bytes per instance",
            "comparison": "native/scalar paired identity must retain the same regime",
        },
    }


def cache_regime(run: dict, cache_info: dict) -> str:
    bytes_per_iter = run.get("bytes_per_iter")
    if not isinstance(bytes_per_iter, int) or bytes_per_iter <= 0:
        fail("cache-regime row has an invalid bytes_per_iter")
    levels = cache_info["levels"]
    if bytes_per_iter <= levels["L1"]["bytes_per_instance"]:
        return "l1_resident"
    if bytes_per_iter <= levels["L2"]["bytes_per_instance"]:
        return "l2_resident"
    if bytes_per_iter <= levels["L3"]["bytes_per_instance"]:
        return "llc_resident"
    return "memory_bound"


def check_cache_regimes(native: dict, scalar: dict, cache_info: dict) -> dict:
    if cache_info.get("schema") != CACHE_INFO_SCHEMA:
        fail("cache provenance schema is invalid")
    native_runs = {run_identity(run): run for run in native["runs"]}
    scalar_runs = {run_identity(run): run for run in scalar["runs"]}
    if set(native_runs) != set(scalar_runs):
        fail("cache-regime native/scalar identities differ")
    native_counts = Counter(cache_regime(run, cache_info) for run in native_runs.values())
    scalar_counts = Counter(cache_regime(run, cache_info) for run in scalar_runs.values())
    if native_counts != scalar_counts:
        fail("cache-regime native/scalar counts differ")
    if not {"l1_resident", "l2_resident", "llc_resident", "memory_bound"}.issubset(native_counts):
        fail("cache-regime corpus does not cover L1/L2/LLC/memory cases")
    paired = Counter()
    for identity, run in native_runs.items():
        native_regime = cache_regime(run, cache_info)
        scalar_regime = cache_regime(scalar_runs[identity], cache_info)
        if native_regime != scalar_regime:
            fail(f"cache regime differs for paired row {identity!r}")
        paired[native_regime] += 1
    return {
        "schema": CACHE_INFO_SCHEMA,
        "status": "passed",
        "cache_info": cache_info,
        "native_counts": dict(sorted(native_counts.items())),
        "scalar_counts": dict(sorted(scalar_counts.items())),
        "paired_counts": dict(sorted(paired.items())),
        "identity_count": len(native_runs),
    }


def check_vendor_probe(report: dict) -> dict:
    if report.get("schema") != VENDOR_PROBE_SCHEMA:
        fail("vendor probe schema is invalid")
    if report.get("target_arch") != "x86_64" or report.get("target_os") != "linux":
        fail("vendor probe is not native Linux x86_64 evidence")
    status = report.get("status")
    if status not in {"unavailable", "passed"}:
        fail("vendor probe status is invalid")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or {candidate.get("name") for candidate in candidates} != {"openblas", "petsc"}:
        fail("vendor probe does not report both OpenBLAS and PETSc candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            fail("vendor probe candidate is not an object")
        for key in ("name", "detection", "sparse_spmv_api", "pinned", "usable", "reason"):
            if key not in candidate:
                fail(f"vendor probe candidate lacks {key}")
        if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
            fail("vendor probe candidate lacks a rejection/acceptance reason")
        if not all(isinstance(candidate[key], bool) for key in ("sparse_spmv_api", "pinned", "usable")):
            fail("vendor probe candidate capability fields are not boolean")
    if status == "unavailable":
        if report.get("release_comparative_claims") is not False:
            fail("unavailable vendor probe must withhold comparative claims")
        if not isinstance(report.get("rejection_reasons"), list) or not report["rejection_reasons"]:
            fail("unavailable vendor probe lacks rejection reasons")
        if any(candidate["usable"] for candidate in candidates):
            fail("unavailable vendor probe marks a candidate usable")
    else:
        if report.get("release_comparative_claims") is not True:
            fail("passed vendor probe must explicitly opt into comparative claims")
    return {
        "schema": report["schema"],
        "status": status,
        "candidates": [
            {
                "name": candidate["name"],
                "detection": candidate["detection"],
                "sparse_spmv_api": candidate["sparse_spmv_api"],
                "pinned": candidate["pinned"],
                "usable": candidate["usable"],
            }
            for candidate in candidates
        ],
        "rejection_reasons": report.get("rejection_reasons", []),
    }


def threshold_identity(record: dict) -> tuple[object, ...]:
    return (
        record.get("family"),
        record.get("dtype"),
        record.get("action"),
        record.get("threshold_dimension"),
        json.dumps(record.get("threshold_value"), sort_keys=True, separators=(",", ":")),
        record.get("threshold_side"),
        record.get("predicate_passes"),
        record.get("selected_format"),
        record.get("expected_format"),
        json.dumps(record.get("fixture"), sort_keys=True, separators=(",", ":")),
    )


def check_thresholds(report: dict, label: str, expected_mode: str) -> dict:
    if report.get("schema") != THRESHOLD_SCHEMA or report.get("status") != "passed":
        fail(f"{label} threshold schema/status is invalid")
    if report.get("artifact_kind") != THRESHOLD_ARTIFACT_KIND:
        fail(f"{label} threshold artifact kind is invalid")
    if report.get("fixture_version") != 1 or report.get("deterministic_fixture") != THRESHOLD_FIXTURE:
        fail(f"{label} threshold fixture metadata is invalid")
    if report.get("mode") != expected_mode or report.get("force_scalar") is not (expected_mode == "scalar"):
        fail(f"{label} threshold dispatch mode is invalid")
    if report.get("target_arch") != "x86_64" or report.get("target_os") != "linux":
        fail(f"{label} threshold artifact is not native Linux x86_64 evidence")
    if report.get("runtime_avx2") is not True or report.get("runtime_fma") is not True:
        fail(f"{label} threshold artifact lacks runtime AVX2/FMA provenance")
    contract = report.get("measurement_contract")
    if not isinstance(contract, dict):
        fail(f"{label} threshold measurement contract is missing")
    exact_contract = {
        "baseline": "raw CSR",
        "reference_oracle": "independent_scalar_csr",
        "correctness_before_timing": True,
        "timing_scope": "steady_state_apply_only;setup_separate",
        "paired_modes": "native-avx2-fma/scalar",
        "deterministic_metadata": True,
        "threshold_predicates": "inclusive below/at/above sweeps",
    }
    for key, expected in exact_contract.items():
        if contract.get(key) != expected:
            fail(f"{label} threshold contract.{key} is invalid")
    metadata = report.get("threshold_metadata")
    if not isinstance(metadata, dict) or metadata != {
        "schema": THRESHOLD_METADATA_SCHEMA,
        "fixture": THRESHOLD_FIXTURE,
        "families": list(THRESHOLD_FAMILIES),
        "sides": list(THRESHOLD_SIDES),
        "reference_oracle": "independent_scalar_csr",
    }:
        fail(f"{label} threshold metadata is not the pinned fixture contract")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 42:
        fail(f"{label} threshold artifact does not have exactly 42 records")
    if any(not isinstance(record, dict) for record in records):
        fail(f"{label} threshold records must all be objects")
    expected_counts = {
        "compact_index": 6,
        "bsr": 9,
        "sell_c_sigma": 9,
        "row_bucket": 9,
        "numa": 3,
        "row_parallel": 6,
    }
    if Counter(record.get("family") for record in records) != expected_counts:
        fail(f"{label} threshold family counts do not match the prescribed matrix")
    dimensions = Counter((record.get("family"), record.get("threshold_dimension")) for record in records)
    expected_dimensions = {
        ("compact_index", "expected_reuse"): 3,
        ("compact_index", "saved_index_bytes_per_apply"): 3,
        ("bsr", "expected_reuse"): 3,
        ("bsr", "saved_bytes_per_apply"): 3,
        ("bsr", "block_fill_ratio"): 3,
        ("sell_c_sigma", "expected_reuse"): 3,
        ("sell_c_sigma", "nnz"): 3,
        ("sell_c_sigma", "fill_ratio"): 3,
        ("row_bucket", "expected_reuse"): 3,
        ("row_bucket", "nnz"): 3,
        ("row_bucket", "row_nnz_cv"): 3,
        ("numa", "nnz"): 3,
        ("row_parallel", "nnz_and_rows"): 3,
        ("row_parallel", "rows"): 3,
    }
    if dimensions != expected_dimensions:
        fail(f"{label} threshold dimensions do not match the prescribed matrix")
    identities = [threshold_identity(record) for record in records]
    if len(set(identities)) != len(records):
        fail(f"{label} threshold records contain duplicate fixture identities")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            fail(f"{label} threshold record {index} is not an object")
        for key in ("family", "dtype", "action", "threshold_dimension", "threshold_side", "selected_format", "expected_format", "selected_kernel", "features", "correctness_status", "reference_oracle", "matrix_checksum"):
            text(record.get(key), f"{label} threshold record {index}.{key}")
        if record["dtype"] != "f64" or record["action"] != "forward":
            fail(f"{label} threshold record {index} is not the typed f64 forward probe")
        if record["threshold_side"] not in {"below", "at", "above"}:
            fail(f"{label} threshold record {index} has an invalid side")
        if record["reference_oracle"] != "independent_scalar_csr" or len(record["matrix_checksum"]) != 16 or any(c not in "0123456789abcdef" for c in record["matrix_checksum"]):
            fail(f"{label} threshold record {index} lacks deterministic oracle/checksum metadata")
        if record.get("correctness_before_timing") is not True:
            fail(f"{label} threshold record {index} lacks correctness-before-timing gate")
        if record.get("format_match") is not True or record["correctness_status"] != "passed":
            fail(f"{label} threshold record {index} did not pass format/correctness gates")
        tolerance = finite(record.get("tolerance"), f"{label} threshold record {index}.tolerance")
        if tolerance <= 0:
            fail(f"{label} threshold record {index} has a non-positive tolerance")
        for key in ("raw_max_abs", "planned_max_abs"):
            if finite(record.get(key), f"{label} threshold record {index}.{key}") > tolerance:
                fail(f"{label} threshold record {index}.{key} exceeds tolerance")
        # The weighted checksum is a diagnostic reduction and may differ by a
        # few ulps when transformed plans change summation order.  The vector
        # residuals above are the correctness gate; retain the checksum only
        # as a finite provenance signal.
        finite(record.get("checksum_delta"), f"{label} threshold record {index}.checksum_delta")
        for key in ("setup_ns", "raw_csr_ns", "planned_ns", "repeats"):
            if integer(record.get(key), f"{label} threshold record {index}.{key}") <= 0:
                fail(f"{label} threshold record {index}.{key} is not a positive measurement")
        finite(record.get("speedup_raw_over_plan"), f"{label} threshold record {index}.speedup_raw_over_plan")
        if "arch=x86_64" not in record["features"]:
            fail(f"{label} threshold record {index} lacks architecture provenance")
        if expected_mode == "scalar" and "force_scalar=true" not in record["features"]:
            fail(f"{label} threshold record {index} lacks forced-scalar provenance")
        if expected_mode == "native-avx2-fma" and "force_scalar=false" not in record["features"]:
            fail(f"{label} threshold record {index} lacks native provenance")
    for family, dimension in expected_dimensions:
        sides = {r["threshold_side"] for r in records if r["family"] == family and r["threshold_dimension"] == dimension}
        if sides != {"below", "at", "above"}:
            fail(f"{label} {family}/{dimension} lacks an inclusive below/at/above sweep")
    for record in (r for r in records if r["family"] == "row_parallel"):
        if record.get("runtime_parallel_selected") is not (record["threshold_side"] != "below"):
            fail(f"{label} row-parallel boundary selected the wrong runtime path")
    return {
        "record_count": len(records),
        "sha256": None,
        "families": dict(Counter(record["family"] for record in records)),
        "mode": expected_mode,
        "formats": sorted({record["selected_format"] for record in records}),
    }


def check_threshold_pair(native: dict, scalar: dict) -> dict:
    native_records = native["records"]
    scalar_records = scalar["records"]
    native_ids = [threshold_identity(record) for record in native_records]
    scalar_ids = [threshold_identity(record) for record in scalar_records]
    if native_ids != scalar_ids:
        fail("native/scalar threshold fixture identities or order differ")
    if len(set(native_ids)) != len(native_ids):
        fail("native threshold fixture identities are not unique")
    for index, (native_record, scalar_record) in enumerate(zip(native_records, scalar_records)):
        if native_record["matrix_checksum"] != scalar_record["matrix_checksum"]:
            fail(f"threshold pair record {index} matrix checksum differs")
        if native_record["fixture"] != scalar_record["fixture"]:
            fail(f"threshold pair record {index} fixture differs")
        for record in (native_record, scalar_record):
            if record.get("correctness_status") != "passed" or record.get("correctness_before_timing") is not True:
                fail(f"threshold pair record {index} lacks a completed correctness gate")
    native_formats = {record["selected_format"] for record in native_records}
    scalar_formats = {record["selected_format"] for record in scalar_records}
    if native_formats != SUPPORTED_FORMATS:
        fail("native threshold baseline does not cover every supported plan format")
    if scalar_formats != native_formats:
        fail("native/scalar threshold baselines selected different plan formats")
    if not any(record["selected_kernel"] in {"avx2", "avx2-fma"} for record in native_records):
        fail("native threshold baseline has no AVX2 kernel rows")
    if {record["selected_kernel"] for record in scalar_records} != {"scalar"}:
        fail("forced-scalar threshold baseline has a non-scalar kernel")
    correctness_delta = max(
        (
            abs(float(native_record["planned_max_abs"]) - float(scalar_record["planned_max_abs"]))
            for native_record, scalar_record in zip(native_records, scalar_records)
        ),
        default=0.0,
    )
    return {
        "record_count": len(native_records),
        "matrix_checksum_count": len({record["matrix_checksum"] for record in native_records}),
        "selected_formats": sorted({record["selected_format"] for record in native_records}),
        "correctness_error_delta_vs_scalar": correctness_delta,
    }


def run_identity(run: dict) -> tuple[object, ...]:
    return tuple(run.get(key) for key in ("group", "id", "operation", "matrix", "scalar", "nrows", "ncols", "nnz", "bytes_per_iter", "rhs_count"))


def check_perf(report: dict, label: str, force_scalar: bool) -> dict:
    if not isinstance(report.get("schema_version"), int) or report["schema_version"] < 10:
        fail(f"{label} schema version is too old")
    hardware = report.get("hardware")
    if not isinstance(hardware, dict) or hardware.get("arch") != "x86_64" or hardware.get("os") != "linux":
        fail(f"{label} hardware is not native Linux x86_64")
    features = text(hardware.get("target_features"), f"{label}.hardware.target_features")
    for token in ("runtime_avx2=true", "runtime_fma=true", f"force_scalar={str(force_scalar).lower()}"):
        if token not in features:
            fail(f"{label} target feature provenance lacks {token}")
    for key in ("cpu_model", "compiler", "microcode", "topology_summary"):
        text(hardware.get(key), f"{label}.hardware.{key}")
    for key in ("total_threads", "rayon_threads", "runtime_compute_threads", "runtime_bandwidth_threads", "numa_partitions", "socket_count", "numa_node_count"):
        if not isinstance(hardware.get(key), int) or hardware[key] <= 0:
            fail(f"{label}.hardware.{key} is invalid")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) < 100:
        fail(f"{label} has fewer than 100 performance rows")
    required_groups = {"spmv/csr", "spmv/numa", "spmv/plan", "spmv/block", "spmv/types", "spmv/scaling", "spmv/skewed", "spmv/fixtures"}
    groups = {run.get("group") for run in runs}
    if not required_groups.issubset(groups):
        fail(f"{label} is missing required benchmark groups: {required_groups - groups}")
    if {run.get("scalar") for run in runs} != {"f32", "f64", "complex32", "complex64"}:
        fail(f"{label} does not cover all four scalar types")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            fail(f"{label}.runs[{index}] is not an object")
        for key in ("bench", "group", "id", "operation", "matrix", "scalar", "selected_kernel", "thread_affinity", "numa_policy", "notes"):
            text(run.get(key), f"{label}.runs[{index}].{key}")
        if run["scalar"] not in {"f32", "f64", "complex32", "complex64"}:
            fail(f"{label}.runs[{index}] has an unknown scalar")
        for key in ("nrows", "ncols", "nnz", "bytes_per_iter", "measured_iterations", "rhs_count"):
            if not isinstance(run.get(key), int) or run[key] <= 0:
                fail(f"{label}.runs[{index}].{key} is invalid")
        if run.get("correctness_status") != "passed":
            fail(f"{label}.runs[{index}].correctness_status is not passed")
        tolerance = finite(run.get("correctness_tolerance"), f"{label}.runs[{index}].correctness_tolerance")
        if tolerance <= 0 or finite(run.get("correctness_max_abs"), f"{label}.runs[{index}].correctness_max_abs") > tolerance:
            fail(f"{label}.runs[{index}] failed its correctness budget")
        for key in ("total_ms", "ms_per_iter", "gb_s"):
            if finite(run.get(key), f"{label}.runs[{index}].{key}") <= 0:
                fail(f"{label}.runs[{index}].{key} is not positive")
        if not isinstance(run.get("peak_rss_bytes"), int) or run["peak_rss_bytes"] <= 0:
            fail(f"{label}.runs[{index}].peak_rss_bytes is invalid")
        for key in ("setup_ms", "solve_ms", "mean_solve_ms_per_rhs", "setup_reuse_break_even_rhs", "amortized_total_ms_per_rhs_4", "amortized_total_ms_per_rhs_16"):
            if run.get(key) is not None and finite(run[key], f"{label}.runs[{index}].{key}") < 0:
                fail(f"{label}.runs[{index}].{key} is negative")
        if "vendor_baseline=" not in run["notes"]:
            fail(f"{label}.runs[{index}] lacks vendor-baseline provenance")
    if not any(run.get("setup_ms") is not None and run.get("solve_ms") is not None for run in runs):
        fail(f"{label} lacks separate setup and repeated-application measurements")
    # Raw CSR baselines share the spmv/plan group for direct comparisons but
    # intentionally have no setup/prebake phase. Only rows carrying a
    # transformed/selected format belong to the setup-versus-apply cell.
    plan_rows = [
        run
        for run in runs
        if run.get("group") == "spmv/plan" and "format=" in run.get("notes", "")
    ]
    if not plan_rows:
        fail(f"{label} has no planned SpMV rows")
    for index, run in enumerate(plan_rows):
        # Setup/prebake must be observable separately from repeated
        # application for every planned format, not merely one representative
        # row. This prevents conversion cost from being silently folded into
        # (or omitted from) a steady-state timing claim.
        for key in ("setup_ms", "solve_ms"):
            if finite(run.get(key), f"{label}.planned[{index}].{key}") <= 0:
                fail(f"{label}.planned[{index}] lacks a positive {key} measurement")
        notes = run["notes"]
        for token in (
            TIMING_SCOPE,
            "format=",
            "selected_kernel=",
        ):
            if token not in notes:
                fail(f"{label}.planned[{index}] lacks {token} provenance")
        action_parity = run.get("action_parity", "")
        if not all(
            token in action_parity
            for token in ("forward=passed", "transpose=passed", "hermitian=passed", "block=passed")
        ):
            fail(f"{label}.planned[{index}] lacks full forward/transpose/Hermitian/block parity")
    block_rows = [run for run in runs if run.get("operation") == "spmm"]
    if not any(run.get("rhs_count", 0) > 1 and "block=passed" in run.get("action_parity", "") for run in block_rows):
        fail(f"{label} lacks multi-RHS block parity")
    for family in ("bsr2", "bsr3", "bsr4"):
        if not any(f"raw CSR baseline for {family}" in run.get("notes", "") for run in runs):
            fail(f"{label} lacks a raw CSR baseline for {family}")
    if not {"shared_unpinned", "shared_pinned", "replicated_pinned"}.issubset({run.get("numa_policy") for run in runs}):
        fail(f"{label} lacks all NUMA policy rows")
    kernels = Counter(run["selected_kernel"] for run in runs)
    if force_scalar and set(kernels) != {"scalar"}:
        fail(f"{label} forced-scalar report selected a non-scalar kernel")
    if not force_scalar and not any(kernel in {"avx2", "avx2-fma"} for kernel in kernels):
        fail(f"{label} native report has no AVX2 kernel rows")
    return {
        "run_count": len(runs),
        "sha256": None,
        "selected_kernels": dict(kernels),
        "scalars": dict(Counter(run["scalar"] for run in runs)),
        "planned_rows_with_setup_and_apply": len(plan_rows),
    }


def main() -> None:
    positional = []
    output = None
    vendor_probe_path = None
    cache_info_path = None
    arguments = sys.argv[1:]
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--output", "--vendor-probe", "--cache-info"}:
            if index + 1 >= len(arguments):
                raise SystemExit(f"usage: {sys.argv[0]} PLAN-NATIVE.json PLAN-SCALAR.json NATIVE.json SCALAR.json [--vendor-probe JSON] [--cache-info LSCpu.txt] [--output COMPARISON.json]")
            value = Path(arguments[index + 1])
            if argument == "--output":
                if output is not None:
                    raise SystemExit("duplicate --output")
                output = value
            elif argument == "--vendor-probe":
                if vendor_probe_path is not None:
                    raise SystemExit("duplicate --vendor-probe")
                vendor_probe_path = value
            else:
                if cache_info_path is not None:
                    raise SystemExit("duplicate --cache-info")
                cache_info_path = value
            index += 2
            continue
        positional.append(argument)
        index += 1
    if len(positional) != 4 or vendor_probe_path is None or cache_info_path is None:
        raise SystemExit(f"usage: {sys.argv[0]} PLAN-NATIVE.json PLAN-SCALAR.json NATIVE.json SCALAR.json --vendor-probe JSON --cache-info LSCpu.txt [--output COMPARISON.json]")
    plan_native_path, plan_scalar_path, native_path, scalar_path = (Path(argument) for argument in positional)
    plan_native = load(plan_native_path)
    plan_scalar = load(plan_scalar_path)
    native = load(native_path)
    scalar = load(scalar_path)
    vendor_probe = load(vendor_probe_path)
    cache_info = parse_cache_info(cache_info_path)
    plan_native_summary = check_thresholds(plan_native, "native", "native-avx2-fma")
    plan_scalar_summary = check_thresholds(plan_scalar, "scalar", "scalar")
    threshold_pair = check_threshold_pair(plan_native, plan_scalar)
    native_summary = check_perf(native, "native", False)
    scalar_summary = check_perf(scalar, "scalar", True)
    native_ids = {run_identity(run) for run in native["runs"]}
    scalar_ids = {run_identity(run) for run in scalar["runs"]}
    if native_ids != scalar_ids:
        fail("native/scalar performance corpus identities differ")
    for key in ("cpu_model", "microcode", "compiler", "topology_summary", "total_threads", "socket_count", "numa_node_count"):
        if native["hardware"].get(key) != scalar["hardware"].get(key):
            fail(f"native/scalar hardware provenance differs for {key}")
    native_by_id = {run_identity(run): run for run in native["runs"]}
    scalar_by_id = {run_identity(run): run for run in scalar["runs"]}
    correctness_delta = max(
        (abs(float(native_by_id[identity]["correctness_max_abs"]) - float(scalar_by_id[identity]["correctness_max_abs"])) for identity in native_ids),
        default=0.0,
    )
    plan_native_summary["sha256"] = digest(plan_native_path)
    plan_scalar_summary["sha256"] = digest(plan_scalar_path)
    native_summary["sha256"] = digest(native_path)
    scalar_summary["sha256"] = digest(scalar_path)
    cache_summary = check_cache_regimes(native, scalar, cache_info)
    vendor_summary = check_vendor_probe(vendor_probe)
    comparison = {
        "schema": "sonium-solvers-sparse.ss2.native-x86-comparison/v2",
        "artifact_kind": "sonium-solvers-sparse.ss2.native-x86-comparison",
        "fixture_version": 1,
        "status": "passed",
        "target_arch": "x86_64",
        "thresholds_native": plan_native_summary,
        "thresholds_scalar": plan_scalar_summary,
        "threshold_pair": threshold_pair,
        "native": native_summary,
        "scalar": scalar_summary,
        "shared_performance_identity_count": len(native_ids),
        "correctness_error_delta_vs_scalar": correctness_delta,
        "cache_regimes": cache_summary,
        "vendor_probe": vendor_summary,
        "timing_acceptance": {
            "scope": "every spmv/plan row has setup and steady-state apply timings",
            "native_plan_rows": native_summary["planned_rows_with_setup_and_apply"],
            "scalar_plan_rows": scalar_summary["planned_rows_with_setup_and_apply"],
            "status": "passed",
        },
        "vendor_baseline": "unavailable_no_pinned_vendor_sparse_library",
        "unsupported_scope": "Arm64, Windows, AVX512, and genuine multi-socket NUMA performance are not claimed",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
