#!/usr/bin/env python3
"""Strict checker for the SS1 sparse AddressSanitizer bundle.

The normal SS1 checker owns the action-matrix contract. This wrapper adds the
sanitizer release-bundle contract: schema-2 evidence, explicit unaccepted
provenance, and checksums covering every payload and metadata file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import check_unix_ss1_x86 as ss1


MANIFEST_SCHEMA = "sonium-solvers-sparse.ss1.native-x86-sanitizer/v1"
MANIFEST_KIND = "sonium-solvers-sparse.ss1.native-x86-sanitizer-bundle"
PAYLOAD = (
    "native.json",
    "native.json.sha256",
    "native-sanitizer.log",
    "rustc-nightly-vv.txt",
    "cargo-nightly-version.txt",
    "lscpu.txt",
    "environment.txt",
)
MANIFEST_KEYS = {
    "schema",
    "artifact_kind",
    "run_kind",
    "acceptance_status",
    "sanitizer",
    "toolchain",
    "target_arch",
    "evidence",
    "files",
    "checksums",
}
EVIDENCE_KEYS = {"path", "sha256", "schema", "artifact_kind", "corpus_sha256"}
FILE_KEYS = {"path", "sha256", "bytes"}


def fail(message: str) -> None:
    raise SystemExit(f"SS1 sanitizer artifact rejected: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")


def exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        fail(f"{label} keys are {actual!r}, expected {sorted(expected)!r}")


def safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        fail(f"{label} is not a relative path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        fail(f"{label} escapes the bundle")
    return value


def verify_sha_sidecar(root: Path, path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    try:
        rows = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{sidecar.relative_to(root)} is missing or unreadable: {error}")
    matches = [
        row.split()
        for row in rows
        if len(row.split()) == 2 and row.split()[1].lstrip("*") == path.name
    ]
    if len(matches) != 1 or len(matches[0][0]) != 64:
        fail(f"{sidecar.relative_to(root)} does not contain one valid digest")
    if matches[0][0].lower() != digest(path):
        fail(f"checksum mismatch for {path.relative_to(root)}")


def verify_checksum_manifest(root: Path, path: Path, expected: set[str]) -> None:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{path.name} is missing or unreadable: {error}")
    seen: dict[str, str] = {}
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            fail(f"{path.name} contains a malformed row")
        checksum, name = fields
        name = name.lstrip("*")
        name = safe_relative_path(name, f"{path.name} entry")
        if len(checksum) != 64 or any(character not in "0123456789abcdefABCDEF" for character in checksum):
            fail(f"{path.name} contains an invalid digest")
        if name in seen:
            fail(f"{path.name} contains duplicate entry {name}")
        seen[name] = checksum.lower()
    if set(seen) != expected:
        fail(f"{path.name} entries are {sorted(seen)!r}, expected {sorted(expected)!r}")
    for name, expected_digest in seen.items():
        candidate = root / name
        if not candidate.is_file() or candidate.is_symlink():
            fail(f"{name} is not a regular bundle file")
        if digest(candidate) != expected_digest:
            fail(f"aggregate checksum mismatch for {name}")


def check_manifest(manifest_path: Path, negative_probes: bool) -> None:
    root = manifest_path.parent.resolve()
    manifest = load_json(manifest_path)
    exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["artifact_kind"] != MANIFEST_KIND:
        fail("manifest identity/version is invalid")
    if manifest["run_kind"] not in {"local", "ci"}:
        fail("run_kind must be local or ci")
    if manifest["acceptance_status"] != "not_accepted":
        fail("sanitizer artifacts must remain explicitly unaccepted")
    if manifest["sanitizer"] != "address" or manifest["toolchain"] != "nightly":
        fail("sanitizer/toolchain provenance is incomplete")
    if manifest["target_arch"] != "x86_64":
        fail("bundle is not native x86_64 evidence")

    evidence = manifest["evidence"]
    exact_keys(evidence, EVIDENCE_KEYS, "manifest.evidence")
    evidence_path = safe_relative_path(evidence["path"], "manifest.evidence.path")
    if evidence_path != "native.json" or evidence["schema"] != ss1.SCHEMA or evidence["artifact_kind"] != ss1.ARTIFACT_KIND:
        fail("manifest evidence identity is invalid")
    ss1.hex_digest(evidence["sha256"], "manifest.evidence.sha256", 64)
    ss1.hex_digest(evidence["corpus_sha256"], "manifest.evidence.corpus_sha256", 64)

    files = manifest["files"]
    if not isinstance(files, list) or len(files) != len(PAYLOAD):
        fail("manifest.files does not contain the complete payload")
    file_map: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(files):
        exact_keys(entry, FILE_KEYS, f"manifest.files[{index}]")
        name = safe_relative_path(entry["path"], f"manifest.files[{index}].path")
        if name in file_map:
            fail(f"manifest.files contains duplicate {name}")
        if name not in PAYLOAD:
            fail(f"manifest.files contains unexpected {name}")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            fail(f"manifest.files[{index}].bytes is invalid")
        ss1.hex_digest(entry["sha256"], f"manifest.files[{index}].sha256", 64)
        candidate = root / name
        if not candidate.is_file() or candidate.is_symlink():
            fail(f"payload file {name} is missing or not regular")
        if candidate.stat().st_size != entry["bytes"] or digest(candidate) != entry["sha256"]:
            fail(f"manifest digest/size mismatch for {name}")
        file_map[name] = entry
    if tuple(sorted(file_map)) != tuple(sorted(PAYLOAD)):
        fail("manifest payload set is incomplete")

    checksums = manifest["checksums"]
    exact_keys(checksums, {"path", "entries"}, "manifest.checksums")
    checksum_name = safe_relative_path(checksums["path"], "manifest.checksums.path")
    if checksum_name != "checksums.sha256":
        fail("manifest checksum path is invalid")
    checksum_path = root / checksum_name
    if not checksum_path.is_file() or checksum_path.is_symlink():
        fail("manifest checksum file is missing or not regular")
    if checksums["entries"] != len(PAYLOAD) + 2:
        fail("manifest checksum entry count is invalid")

    artifact_path = root / evidence_path
    if digest(artifact_path) != evidence["sha256"]:
        fail("manifest evidence digest does not match native.json")
    artifact = ss1.load(artifact_path)
    ss1.check_artifact(
        artifact,
        "sanitizer/native",
        "passed",
        {"f32": "avx2-fma", "f64": "avx2-fma", "complex32": "scalar", "complex64": "avx2-fma"},
        False,
        False,
        False,
    )
    if negative_probes:
        ss1.run_negative_probes(
            artifact,
            {"f32": "avx2-fma", "f64": "avx2-fma", "complex32": "scalar", "complex64": "avx2-fma"},
        )
    if artifact["corpus_sha256"] != evidence["corpus_sha256"]:
        fail("manifest corpus digest does not match native.json")
    verify_sha_sidecar(root, artifact_path)
    verify_sha_sidecar(root, manifest_path)
    aggregate_expected = set(PAYLOAD) | {manifest_path.name, f"{manifest_path.name}.sha256"}
    verify_checksum_manifest(root, checksum_path, aggregate_expected)
    regular_files = {path.name for path in root.iterdir() if path.is_file()}
    if regular_files != aggregate_expected | {checksum_path.name}:
        fail(f"bundle contains unexpected regular files: {sorted(regular_files - aggregate_expected)!r}")
    print(
        f"SS1 sanitizer bundle: passed; run_kind={manifest['run_kind']}; "
        f"acceptance_status={manifest['acceptance_status']}; files={len(aggregate_expected)}"
    )


def main() -> None:
    if len(sys.argv) not in {2, 3} or (len(sys.argv) == 3 and sys.argv[2] != "--negative-probes"):
        raise SystemExit(f"usage: {sys.argv[0]} MANIFEST.json [--negative-probes]")
    check_manifest(Path(sys.argv[1]), len(sys.argv) == 3)


if __name__ == "__main__":
    main()
