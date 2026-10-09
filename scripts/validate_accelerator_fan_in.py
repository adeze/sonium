"""Validate the immutable CUDA/native-x86 QA shard fan-in."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tomllib

from qa_result_policy import validate_result


EXPECTED_LANES = {
    "nvidia-cuda": "cuda",
    "native-x86_64": "cpu-native-x86_64",
}
PLATFORM_EVIDENCE = {
    "avx512": ("target/sonium-avx512-evidence/provenance.json", "native-x86_64-avx512", "x86_64"),
    "windows": ("target/sonium-windows-x86-evidence/provenance.json", "portable-x86_64-windows", "x86_64"),
}
PLACEHOLDER_VALUES = {"", "unknown", "unavailable", "unspecified", "n/a"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def validate_checksum_inventory(root: pathlib.Path, lane: pathlib.Path) -> list[str]:
    """Verify every immutable shard file against its published checksum list."""
    checksum_path = lane / "checksums.sha256"
    errors: list[str] = []
    try:
        lines = checksum_path.read_text().splitlines()
    except OSError as error:
        return [f"{lane.name}: cannot read checksum inventory: {error}"]
    declared: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            errors.append(f"{lane.name}: malformed checksum line {line_number}")
            continue
        digest, name = parts[0], parts[1]
        path = pathlib.PurePosixPath(name)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or "\\" in name
            or ":" in name
            or path.is_absolute()
            or any(part in {".", ".."} for part in path.parts)
        ):
            errors.append(f"{lane.name}: unsafe or invalid checksum entry {name!r}")
            continue
        if name in declared:
            errors.append(f"{lane.name}: duplicate checksum entry {name!r}")
        declared[name] = digest

    expected: dict[str, pathlib.Path] = {
        "validation-manifest.toml": root / "crates" / "sonium-qa" / "validation-manifest.toml"
    }
    for path in lane.rglob("*"):
        if path.is_file() and path.name != "checksums.sha256":
            expected[str(path.relative_to(lane).as_posix())] = path
    if set(declared) != set(expected):
        missing = sorted(set(expected) - set(declared))
        extra = sorted(set(declared) - set(expected))
        if missing:
            errors.append(f"{lane.name}: checksum inventory is missing {missing}")
        if extra:
            errors.append(f"{lane.name}: checksum inventory has undeclared files {extra}")
    for name, path in expected.items():
        digest = declared.get(name)
        if digest is None:
            continue
        if not path.is_file():
            errors.append(f"{lane.name}: checksum target is missing {name!r}")
            continue
        if sha256(path) != digest:
            errors.append(f"{lane.name}: stale checksum for {name!r}")
    return errors


def validate_shard_environment(shard: dict, hardware: str) -> list[str]:
    environment = shard.get("environment")
    if not isinstance(environment, dict):
        return [f"{hardware}: shard environment is missing"]
    if hardware == "nvidia-cuda":
        required = (
            "device",
            "driver",
            "toolkit",
            "peak_flops",
            "memory_bandwidth",
            "os",
            "compiler",
        )
    else:
        required = (
            "cpu_model",
            "microcode",
            "os",
            "compiler",
            "target_features",
            "runtime_dispatch",
            "thread_policy",
            "affinity_policy",
            "numa_policy",
            "deterministic_mode",
        )
    return [
        f"{hardware}: shard environment {name!r} is missing"
        for name in required
        if not isinstance(environment.get(name), str)
        or environment[name].strip().lower() in PLACEHOLDER_VALUES
    ]


def validate_platform_evidence(root: pathlib.Path, label: str, path: str, hardware: str, target_arch: str) -> list[str]:
    try:
        evidence = load_json(root / path)
    except ValueError as error:
        return [f"{label}: {error}"]
    errors = []
    if evidence.get("schema") != "sonium-qa-platform-evidence/v1":
        errors.append(f"{label}: unsupported platform evidence schema")
    if evidence.get("hardware") != hardware:
        errors.append(f"{label}: platform hardware identity mismatch")
    if evidence.get("target_arch") != target_arch:
        errors.append(f"{label}: platform target architecture mismatch")
    required = (
        "cpu_model",
        "microcode",
        "os",
        "compiler",
        "target_features",
        "runtime_dispatch",
        "thread_policy",
        "affinity_policy",
        "numa_policy",
        "deterministic_mode",
    )
    for name in required:
        value = evidence.get(name)
        if not isinstance(value, str) or value.strip().lower() in PLACEHOLDER_VALUES:
            errors.append(f"{label}: platform evidence {name!r} is missing")
    return errors


def validate_lane(
    root: pathlib.Path,
    manifest: dict,
    hardware: str,
    expected_backend: str,
) -> list[str]:
    lane = root / "target" / "sonium-qa-accelerators" / hardware
    errors: list[str] = []
    try:
        shard = load_json(lane / "accelerator-evidence.json")
    except ValueError as error:
        return [str(error)]
    cases = [
        case
        for case in manifest.get("case", [])
        if case.get("tier") == "accelerator-gate"
        and case.get("required_hardware") == hardware
    ]
    expected_ids = [case["id"] for case in cases]
    if shard.get("schema") != "sonium-qa-accelerator-evidence/v1":
        errors.append(f"{hardware}: unsupported shard schema")
    if shard.get("hardware") != hardware:
        errors.append(f"{hardware}: shard hardware identity mismatch")
    if shard.get("required_backend") != expected_backend:
        errors.append(f"{hardware}: shard backend identity mismatch")
    if shard.get("required_case_ids") != expected_ids:
        errors.append(f"{hardware}: shard case coverage does not match manifest")
    if shard.get("manifest_sha256") != sha256(
        root / "crates" / "sonium-qa" / "validation-manifest.toml"
    ):
        errors.append(f"{hardware}: shard manifest checksum is stale")
    errors.extend(validate_shard_environment(shard, hardware))
    if not isinstance(shard.get("failures"), list):
        errors.append(f"{hardware}: shard failure list is missing")
    elif shard["failures"]:
        errors.append(f"{hardware}: shard reports failures")
    errors.extend(validate_checksum_inventory(root, lane))

    results = shard.get("cases")
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results) or [item.get("case_id") for item in results] != expected_ids:
        errors.append(f"{hardware}: shard result rows do not cover cases exactly once")
        results = []
    counts = shard.get("coverage")
    if not isinstance(counts, dict):
        errors.append(f"{hardware}: shard coverage is missing")
    else:
        valid_counts = not any(
            not isinstance(counts.get(name), int) or counts.get(name, -1) < 0
            for name in ("executed", "passed", "failed", "unsupported")
        )
        if not valid_counts:
            errors.append(f"{hardware}: shard coverage contains invalid counts")
        elif counts["executed"] < len(cases) or counts["passed"] < len(cases):
            errors.append(f"{hardware}: shard did not execute and pass every case")
        if counts.get("failed") != 0 or counts.get("unsupported") != 0:
            errors.append(f"{hardware}: shard has failed or unsupported coverage")

    observed_counts = {name: 0 for name in ("executed", "passed", "failed", "unsupported")}
    for result in results:
        runtime_counts = (result.get("runtime") or {}).get("counts", {})
        if isinstance(runtime_counts, dict):
            for name in observed_counts:
                if isinstance(runtime_counts.get(name), int):
                    observed_counts[name] += runtime_counts[name]
    if isinstance(counts, dict) and any(counts.get(name) != observed_counts[name] for name in observed_counts):
        errors.append(f"{hardware}: shard coverage does not match result runtime counts")

    for case, result in zip(cases, results):
        errors.extend(
            f"{hardware}/{case['id']}: {message}"
            for message in validate_result(
                case,
                result,
                artifact_root=lane / "artifacts",
            )
        )
    return errors


def main(root: pathlib.Path) -> int:
    manifest_path = root / "crates" / "sonium-qa" / "validation-manifest.toml"
    try:
        manifest = tomllib.loads(manifest_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"cannot read validation manifest: {error}", file=sys.stderr)
        return 1
    errors = []
    for hardware, backend in EXPECTED_LANES.items():
        errors.extend(validate_lane(root, manifest, hardware, backend))
    for label, (path, hardware, target_arch) in PLATFORM_EVIDENCE.items():
        errors.extend(validate_platform_evidence(root, label, path, hardware, target_arch))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("SUCCESS: accelerator shard fan-in covers CUDA and native-x86 evidence")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    raise SystemExit(main(pathlib.Path.cwd()))
