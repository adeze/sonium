#!/usr/bin/env python3
"""Strict checker for the SS1 native-x86 sparse action matrix.

The Rust integration test owns numerical correctness against its independent
scalar oracle.  This checker owns the release-evidence contract: every
dispatch mode must be identifiable, complete, destination-preserving failures
must be present, and all mode artifacts must describe the same matrix corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import NoReturn


SCHEMA = 2
ARTIFACT_KIND = "sonium-solvers-sparse.ss1.native-x86"
FIXTURE_VERSION = 1
ACTION_SET = ("accumulation", "block", "forward", "hermitian", "transpose")
CASE_ACTION_ORDER = ("forward", "accumulation", "transpose", "hermitian", "block")
FORMATS = (
    "Csr",
    "CompactCsrU32",
    "RowBucketedCsr",
    "SellCSigma",
    "NumaCsr",
    "Bsr2",
    "Bsr3",
    "Bsr4",
)
DTYPES = ("f32", "f64", "complex32", "complex64")
CASE_CAPABILITIES = (
    "accumulation",
    "block",
    "forward",
    "hermitian",
    "independent-scalar-oracle",
    "transpose",
    "zero-rhs",
)
DECLARED_CAPABILITIES = (
    "independent-scalar-oracle",
    "typed-errors",
    "destination-preserving-failures",
    "zero-rhs",
)
ERROR_FIELDS = (
    "forward_max_abs",
    "accumulation_max_abs",
    "transpose_max_abs",
    "hermitian_max_abs",
    "block_max_abs",
)
TOP_LEVEL_KEYS = {
    "adversarial_fixtures",
    "artifact_kind",
    "avx2",
    "build_target_features",
    "capability_contract",
    "case_count",
    "cases",
    "checked_failures",
    "compile_avx2",
    "compile_fma",
    "corpus_sha256",
    "coverage",
    "disable_fma",
    "fixture_version",
    "fma",
    "force_scalar",
    "matrix_checksum",
    "matrix_cols",
    "matrix_condition_proxy",
    "matrix_nnz",
    "matrix_rows",
    "negative_probes",
    "oracle",
    "schema",
    "selected_kernels",
    "status",
    "target_arch",
    "dispatch_features",
}
CASE_KEYS = {
    "accumulation_max_abs",
    "actions",
    "block_max_abs",
    "capabilities",
    "dtype",
    "features",
    "format",
    "forward_max_abs",
    "hermitian_max_abs",
    "kernel",
    "matrix_checksum",
    "matrix_cols",
    "matrix_condition_proxy",
    "matrix_nnz",
    "matrix_rows",
    "norm_tolerance",
    "oracle",
    "transpose_max_abs",
    "ulp_budget",
}
ADVERSARIAL_KEYS = {
    "actions",
    "dtype",
    "fixture",
    "matrix_checksum",
    "matrix_cols",
    "matrix_condition_proxy",
    "matrix_nnz",
    "matrix_rows",
    "max_abs",
    "plan_cases",
    "tolerance",
}
ADVERSARIAL_SKEWED_KEYS = ADVERSARIAL_KEYS - {"plan_cases"}
FAILURE_KEYS = {"action", "destination_preserved", "error_kind", "fixture", "operation"}
EXPECTED_FAILURES = [
    ("malformed_csr_metadata", "forward_into", "OperationFailed", "CSR validation"),
    ("non_finite_input", "forward_into", "OperationFailed", "CSR SpMV"),
    ("generated_non_finite_output", "forward_into", "OperationFailed", "CSR SpMV"),
    ("generated_non_finite_output", "transpose_into", "OperationFailed", "CSR transpose SpMV"),
    ("generated_non_finite_output", "hermitian_into", "OperationFailed", "CSR Hermitian SpMV"),
    *[
        (f"malformed_{layout}_destination", "block", "DimensionMismatch", "dimension validation")
        for layout in FORMATS
    ],
]


def fail(message: str) -> NoReturn:
    raise SystemExit(f"SS1 artifact rejected: {message}")


def finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        fail(f"{label} is not finite")
    return float(value)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksum(path: Path) -> None:
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
    if expected is None or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        fail(f"{sidecar} has no valid digest for {path.name}")
    actual = checksum(path)
    if actual != expected:
        fail(f"checksum mismatch for {path.name}: {actual} != {expected}")


def load(path: Path) -> dict:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(artifact, dict):
        fail(f"{path} is not a JSON object")
    verify_checksum(path)
    return artifact


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} is missing")
    return value


def exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        fail(f"{label} keys are {actual!r}, expected {sorted(expected)!r}")


def hex_digest(value: object, label: str, length: int) -> str:
    digest = text(value, label).lower()
    if len(digest) != length or any(character not in "0123456789abcdef" for character in digest):
        fail(f"{label} is not a {length * 4}-bit hexadecimal digest")
    return digest


def case_signature(case: dict) -> list[object]:
    return [
        case["dtype"],
        case["format"],
        case["matrix_rows"],
        case["matrix_cols"],
        case["matrix_nnz"],
        case["matrix_checksum"],
        case["actions"],
        case["capabilities"],
    ]


def adversarial_signature(fixture: dict) -> list[object]:
    plans = fixture.get("plan_cases")
    return [
        fixture["fixture"],
        fixture["dtype"],
        fixture["matrix_rows"],
        fixture["matrix_cols"],
        fixture["matrix_nnz"],
        fixture["matrix_checksum"],
        fixture["actions"],
        [case_signature(plan) for plan in plans] if plans is not None else None,
    ]


def failure_signature(failure: dict) -> list[object]:
    return [
        failure["fixture"],
        failure["action"],
        failure["error_kind"],
        failure["operation"],
        failure["destination_preserved"],
    ]


def corpus_sha256(artifact: dict) -> str:
    canonical = {
        "cases": [case_signature(case) for case in artifact["cases"]],
        "adversarial_fixtures": [
            adversarial_signature(fixture) for fixture in artifact["adversarial_fixtures"]
        ],
        "checked_failures": [failure_signature(failure) for failure in artifact["checked_failures"]],
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_case_kernel(dtype: str, layout: str, selected: dict[str, str]) -> str:
    if layout in {"RowBucketedCsr", "Bsr2", "Bsr3", "Bsr4"}:
        return "scalar"
    if layout == "NumaCsr" and dtype != "f64":
        return "scalar"
    return selected.get(dtype, "__invalid__")


def structural_cases(artifact: dict) -> list[tuple[object, ...]]:
    return sorted(
        (
            case["dtype"],
            case["format"],
            case["matrix_rows"],
            case["matrix_cols"],
            case["matrix_nnz"],
            case["matrix_checksum"],
            case["matrix_condition_proxy"],
            case["norm_tolerance"],
            case["ulp_budget"],
            tuple(case["actions"]),
            tuple(case["capabilities"]),
        )
        for case in artifact["cases"]
    )


def structural_adversarial(artifact: dict) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for fixture in artifact["adversarial_fixtures"]:
        plan_rows = fixture.get("plan_cases")
        plan_signature = None
        if plan_rows is not None:
            plan_signature = tuple(
                sorted(
                    (
                        plan["dtype"],
                        plan["format"],
                        plan["matrix_rows"],
                        plan["matrix_cols"],
                        plan["matrix_nnz"],
                        plan["matrix_checksum"],
                        tuple(plan["actions"]),
                        tuple(plan["capabilities"]),
                    )
                    for plan in plan_rows
                )
            )
        rows.append(
            (
                fixture["fixture"],
                fixture["dtype"],
                fixture["matrix_rows"],
                fixture["matrix_cols"],
                fixture["matrix_nnz"],
                fixture["matrix_checksum"],
                tuple(fixture["actions"]),
                plan_signature,
            )
        )
    return sorted(rows)


def structural_failures(artifact: dict) -> list[tuple[object, ...]]:
    return sorted(
        (
            failure["fixture"],
            failure["action"],
            failure["error_kind"],
            failure["operation"],
            failure["destination_preserved"],
        )
        for failure in artifact["checked_failures"]
    )


def check_case(
    case: object,
    label: str,
    expected_kernel: str,
    expected_features: str,
    allow_null_condition: bool = False,
) -> None:
    exact_keys(case, CASE_KEYS, label)
    if case["dtype"] not in DTYPES or case["format"] not in FORMATS:
        fail(f"{label} has an unsupported scalar/layout pair")
    if case["kernel"] != expected_kernel:
        fail(f"{label}.kernel is {case['kernel']!r}, expected {expected_kernel!r}")
    if case["features"] != expected_features:
        fail(f"{label}.features is inconsistent with the artifact dispatch")
    if case["oracle"] != "independent_scalar_csr":
        fail(f"{label} does not name the independent oracle")
    if case["actions"] != list(CASE_ACTION_ORDER):
        fail(f"{label}.actions is not the complete ordered operation set")
    if case["capabilities"] != list(CASE_CAPABILITIES):
        fail(f"{label}.capabilities is not the complete capability set")
    for key in ("matrix_rows", "matrix_cols", "matrix_nnz", "ulp_budget"):
        if type(case[key]) is not int or case[key] < 0:
            fail(f"{label}.{key} is invalid")
    hex_digest(case["matrix_checksum"], f"{label}.matrix_checksum", 16)
    if case["matrix_condition_proxy"] is None and allow_null_condition:
        pass
    else:
        condition = finite(case["matrix_condition_proxy"], f"{label}.matrix_condition_proxy")
        if condition <= 0:
            fail(f"{label}.matrix_condition_proxy is not positive")
    tolerance = finite(case["norm_tolerance"], f"{label}.norm_tolerance")
    if tolerance <= 0:
        fail(f"{label}.norm_tolerance is not positive")
    for key in ERROR_FIELDS:
        measured = finite(case[key], f"{label}.{key}")
        if measured < 0 or measured > tolerance:
            fail(f"{label}.{key} is outside its norm tolerance")


def check_artifact(
    artifact: dict,
    label: str,
    expected_status: str,
    expected_kernels: dict[str, str],
    force_scalar: bool,
    disable_fma: bool,
    require_compiled_simd: bool,
) -> None:
    exact_keys(artifact, TOP_LEVEL_KEYS, label)
    if artifact["schema"] != SCHEMA or type(artifact["schema"]) is not int:
        fail(f"{label} schema is not {SCHEMA}")
    if artifact["artifact_kind"] != ARTIFACT_KIND or artifact["fixture_version"] != FIXTURE_VERSION:
        fail(f"{label} artifact identity/version is invalid")
    if artifact["status"] != expected_status:
        fail(f"{label} status is {artifact['status']!r}, expected {expected_status!r}")
    if artifact["target_arch"] != "x86_64" or artifact["avx2"] is not True or artifact["fma"] is not True:
        fail(f"{label} did not run on an AVX2/FMA-capable x86 host")
    if type(artifact["compile_avx2"]) is not bool or type(artifact["compile_fma"]) is not bool:
        fail(f"{label} compile feature provenance is not boolean")
    if artifact["force_scalar"] is not force_scalar or artifact["disable_fma"] is not disable_fma:
        fail(f"{label} dispatch override provenance is inconsistent")
    build = artifact["build_target_features"]
    exact_keys(build, {"avx2", "fma"}, f"{label}.build_target_features")
    if any(type(build[key]) is not bool for key in ("avx2", "fma")):
        fail(f"{label}.build_target_features is not boolean")
    if require_compiled_simd and build != {"avx2": True, "fma": True}:
        fail(f"{label} lacks explicit +avx2,+fma build provenance")
    selected = artifact["selected_kernels"]
    exact_keys(selected, set(DTYPES), f"{label}.selected_kernels")
    if selected != expected_kernels:
        fail(f"{label} selected kernel map is {selected!r}, expected {expected_kernels!r}")
    expected_features = (
        "arch=x86_64;avx2=true;fma=true;"
        f"avx2_fma={str(not force_scalar and not disable_fma).lower()};"
        f"disable_fma={str(disable_fma).lower()};force_scalar={str(force_scalar).lower()}"
    )
    if artifact["dispatch_features"] != expected_features:
        fail(f"{label}.dispatch_features is not the exact declared AVX2/FMA contract")
    if artifact["oracle"] != "independent_scalar_csr":
        fail(f"{label}.oracle is not independent_scalar_csr")

    coverage = artifact["coverage"]
    exact_keys(coverage, {"scalar_types", "layouts", "operations", "capabilities"}, f"{label}.coverage")
    if coverage["scalar_types"] != list(DTYPES) or coverage["layouts"] != list(FORMATS):
        fail(f"{label}.coverage scalar/layout axes are incomplete or reordered")
    if coverage["operations"] != list(ACTION_SET) or coverage["capabilities"] != list(DECLARED_CAPABILITIES):
        fail(f"{label}.coverage operation/capability axes are incomplete or reordered")
    contract = artifact["capability_contract"]
    exact_keys(contract, {"operations", "zero_rhs", "checked_failures", "destination"}, f"{label}.capability_contract")
    exact_keys(contract["operations"], set(ACTION_SET), f"{label}.capability_contract.operations")
    if contract["operations"] != {operation: "exact" for operation in ACTION_SET}:
        fail(f"{label}.capability_contract does not mark every operation exact")
    if contract["zero_rhs"] != "accepted" or contract["checked_failures"] != "typed_solver_error" or contract["destination"] != "preserved":
        fail(f"{label}.capability_contract is incomplete")

    cases = artifact["cases"]
    if artifact["case_count"] != 32 or not isinstance(cases, list) or len(cases) != 32:
        fail(f"{label} does not contain exactly 32 action cases")
    case_counts = Counter()
    for index, case in enumerate(cases):
        check_case(case, f"{label}.cases[{index}]", expected_case_kernel(case.get("dtype", ""), case.get("format", ""), expected_kernels), expected_features)
        case_counts[(case["dtype"], case["format"])] += 1
    if any(case_counts[(dtype, layout)] != 1 for dtype in DTYPES for layout in FORMATS):
        fail(f"{label} does not have exactly one case for every scalar/layout pair")

    adversarial = artifact["adversarial_fixtures"]
    if not isinstance(adversarial, list) or len(adversarial) != 8:
        fail(f"{label} does not contain two adversarial fixtures for every scalar")
    fixture_counts = Counter()
    for fixture_index, fixture in enumerate(adversarial):
        name = fixture.get("fixture") if isinstance(fixture, dict) else None
        fixture_label = f"{label}.adversarial_fixtures[{fixture_index}]"
        exact_keys(fixture, ADVERSARIAL_KEYS if name == "zero_matrix" else ADVERSARIAL_SKEWED_KEYS, fixture_label)
        dtype = text(fixture["dtype"], f"{fixture_label}.dtype")
        if dtype not in DTYPES or name not in {"zero_matrix", "skewed_rows"}:
            fail(f"{fixture_label} has an unknown scalar/fixture")
        fixture_counts[(dtype, name)] += 1
        expected_actions = ["forward", "transpose", "hermitian"] if name == "zero_matrix" else ["forward"]
        if fixture["actions"] != expected_actions:
            fail(f"{fixture_label}.actions are incomplete")
        for key in ("matrix_rows", "matrix_cols", "matrix_nnz"):
            if type(fixture[key]) is not int or fixture[key] < 0:
                fail(f"{fixture_label}.{key} is invalid")
        hex_digest(fixture["matrix_checksum"], f"{fixture_label}.matrix_checksum", 16)
        tolerance = finite(fixture["tolerance"], f"{fixture_label}.tolerance")
        maximum = finite(fixture["max_abs"], f"{fixture_label}.max_abs")
        if tolerance <= 0 or maximum < 0 or maximum > tolerance:
            fail(f"{fixture_label} error is outside tolerance")
        if name == "zero_matrix":
            if fixture["matrix_condition_proxy"] is not None:
                fail(f"{fixture_label}.matrix_condition_proxy must be null")
            plans = fixture["plan_cases"]
            if not isinstance(plans, list) or len(plans) != 10:
                fail(f"{fixture_label}.plan_cases is incomplete")
            plan_counts = Counter()
            for plan_index, plan in enumerate(plans):
                check_case(plan, f"{fixture_label}.plan_cases[{plan_index}]", expected_case_kernel(plan.get("dtype", ""), plan.get("format", ""), expected_kernels), expected_features, plan.get("matrix_nnz") == 0)
                plan_counts[plan["format"]] += 1
            if any(plan_counts[layout] != 2 for layout in FORMATS[:5]):
                fail(f"{fixture_label}.plan_cases does not cover every non-BSR layout twice")
        else:
            condition = finite(fixture["matrix_condition_proxy"], f"{fixture_label}.matrix_condition_proxy")
            if condition <= 0:
                fail(f"{fixture_label}.matrix_condition_proxy is not positive")
    if any(fixture_counts[(dtype, name)] != 1 for dtype in DTYPES for name in ("zero_matrix", "skewed_rows")):
        fail(f"{label} does not have exactly one zero and skewed fixture for every scalar")

    failures = artifact["checked_failures"]
    if not isinstance(failures, list) or len(failures) != len(EXPECTED_FAILURES):
        fail(f"{label} does not contain the complete negative-probe set")
    actual_failures = []
    for index, failure in enumerate(failures):
        failure_label = f"{label}.checked_failures[{index}]"
        exact_keys(failure, FAILURE_KEYS, failure_label)
        actual = tuple(text(failure[key], f"{failure_label}.{key}") for key in ("fixture", "action", "error_kind", "operation"))
        if failure["destination_preserved"] is not True:
            fail(f"{failure_label} did not preserve its destination")
        actual_failures.append(actual)
    if actual_failures != EXPECTED_FAILURES:
        fail(f"{label}.checked_failures are not the exact typed negative probes")
    probes = artifact["negative_probes"]
    exact_keys(probes, {"schema", "count", "probe_ids", "rejection", "destination"}, f"{label}.negative_probes")
    expected_probe_ids = [f"{fixture}:{action}" for fixture, action, _, _ in EXPECTED_FAILURES]
    if probes != {
        "schema": "sonium-solvers-sparse.ss1.negative-probes/v1",
        "count": len(EXPECTED_FAILURES),
        "probe_ids": expected_probe_ids,
        "rejection": "typed_solver_error",
        "destination": "preserved",
    }:
        fail(f"{label}.negative_probes manifest is incomplete")

    for key in ("matrix_rows", "matrix_cols", "matrix_nnz"):
        if type(artifact[key]) is not int or artifact[key] < 0:
            fail(f"{label}.{key} is invalid")
    hex_digest(artifact["matrix_checksum"], f"{label}.matrix_checksum", 16)
    finite(artifact["matrix_condition_proxy"], f"{label}.matrix_condition_proxy")
    declared_corpus = hex_digest(artifact["corpus_sha256"], f"{label}.corpus_sha256", 64)
    if declared_corpus != corpus_sha256(artifact):
        fail(f"{label}.corpus_sha256 does not match the canonical corpus")


def run_negative_probes(artifact: dict, expected_kernels: dict[str, str]) -> None:
    """Ensure the checker rejects incomplete or tampered acceptance evidence."""

    expected_features = artifact["dispatch_features"]
    probes: list[tuple[str, object]] = [
        (
            "missing-scalar",
            lambda candidate: candidate["coverage"]["scalar_types"].pop(),
        ),
        (
            "unknown-layout",
            lambda candidate: candidate["coverage"]["layouts"].append("UnknownLayout"),
        ),
        (
            "missing-operation",
            lambda candidate: candidate["coverage"]["operations"].pop(),
        ),
        (
            "unknown-capability",
            lambda candidate: candidate["coverage"]["capabilities"].append("unclaimed-capability"),
        ),
        (
            "missing-case-operation",
            lambda candidate: candidate["cases"][0]["actions"].pop(),
        ),
        (
            "corpus-tamper",
            lambda candidate: candidate["cases"][0].__setitem__("matrix_checksum", "0000000000000000"),
        ),
        (
            "missing-negative-probe",
            lambda candidate: candidate["negative_probes"]["probe_ids"].pop(),
        ),
    ]
    for name, mutate in probes:
        candidate = deepcopy(artifact)
        mutate(candidate)
        try:
            check_artifact(candidate, f"negative/{name}", "passed", expected_kernels, False, False, False)
        except SystemExit:
            continue
        fail(f"negative probe {name} was accepted")
    print(f"SS1 negative probes: passed ({len(probes)} strict rejection probes)")


def main() -> None:
    arguments = sys.argv[1:]
    negative_probes = "--negative-probes" in arguments
    arguments = [argument for argument in arguments if argument != "--negative-probes"]
    if len(arguments) not in {4, 6}:
        raise SystemExit(
            f"usage: {sys.argv[0]} NATIVE.json SCALAR.json AVX2.json FMA.json "
            "[--output COMPARISON.json] [--negative-probes]"
        )
    paths = [Path(argument) for argument in arguments[:4]]
    output = None
    if len(arguments) == 6:
        if arguments[4] != "--output":
            raise SystemExit(f"usage: {sys.argv[0]} NATIVE.json SCALAR.json AVX2.json FMA.json [--output COMPARISON.json] [--negative-probes]")
        output = Path(arguments[5])
    labels = ("native", "scalar", "avx2", "fma")
    artifacts = [load(path) for path in paths]
    expected = (
        ("passed", {"f32": "avx2-fma", "f64": "avx2-fma", "complex32": "scalar", "complex64": "avx2-fma"}, False, False, False),
        ("scalar_baseline", {dtype: "scalar" for dtype in DTYPES}, True, False, False),
        ("passed", {"f32": "avx2", "f64": "avx2", "complex32": "scalar", "complex64": "avx2"}, False, True, False),
        ("passed", {"f32": "avx2-fma", "f64": "avx2-fma", "complex32": "scalar", "complex64": "avx2-fma"}, False, False, True),
    )
    for artifact, label, (status, kernels, force, disable, compiled) in zip(artifacts, labels, expected):
        check_artifact(artifact, label, status, kernels, force, disable, compiled)
    if negative_probes:
        run_negative_probes(artifacts[0], expected[0][1])
    baseline = artifacts[1]
    for artifact, label in zip(artifacts, labels):
        for key, extractor in (
            ("cases", structural_cases),
            ("adversarial_fixtures", structural_adversarial),
            ("checked_failures", structural_failures),
        ):
            if extractor(artifact) != extractor(baseline):
                fail(f"{label}/{labels[1]} {key} corpus differs")
        if artifact["corpus_sha256"] != baseline["corpus_sha256"]:
            fail(f"{label}/{labels[1]} canonical corpus checksum differs")
    cross_mode_errors: dict[str, float] = {}
    for artifact, label in zip(artifacts, labels):
        maximum = 0.0
        for actual, reference in zip(artifact["cases"], baseline["cases"]):
            for key in ERROR_FIELDS:
                maximum = max(maximum, abs(float(actual[key]) - float(reference[key])))
        cross_mode_errors[label] = maximum
        allowed = max(float(case["norm_tolerance"]) for case in artifact["cases"]) * 2.0
        if maximum > allowed:
            fail(f"{label}/scalar error-envelope drift {maximum:.6e} exceeds {allowed:.6e}")
    comparison = {
        "schema": "sonium-solvers-sparse.ss1.native-x86-comparison/v2",
        "artifact_kind": "sonium-solvers-sparse.ss1.native-x86-comparison",
        "fixture_version": FIXTURE_VERSION,
        "status": "passed",
        "target_arch": "x86_64",
        "corpus": {"case_count": 32, "adversarial_fixture_count": 8, "checked_failure_count": len(baseline["checked_failures"])},
        "modes": {
            label: {
                "artifact": path.name,
                "sha256": checksum(path),
                "status": artifact["status"],
                "selected_kernels": artifact["selected_kernels"],
                "max_error_envelope_delta_vs_scalar": cross_mode_errors[label],
            }
            for label, path, artifact in zip(labels, paths, artifacts)
        },
        "cross_mode_error_envelope_budget": max(float(case["norm_tolerance"]) for case in baseline["cases"]) * 2.0,
        "corpus_sha256": baseline["corpus_sha256"],
        "negative_probes": {"status": "passed" if negative_probes else "not-run", "count": 7},
        "oracle": "independent_scalar_csr",
        "unsupported_hardware": "represented_by_test_status; no non-AVX2 execution is claimed on this host",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
