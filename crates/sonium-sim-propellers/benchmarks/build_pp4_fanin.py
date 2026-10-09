#!/usr/bin/env python3
"""Assemble the host-applicable PP4 propeller release fan-in.

The three producer lanes retain their own schemas and checkers.  This
producer only joins already-produced, checksum-bound trees; it never turns a
missing or failed lane into a pass and never promotes the unresolved PP2/PP3
full-fidelity gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "sonium.propellers.pp4-release-fanin.v1"
SCOPE = "native Linux x86_64 CPU/f64; Windows/Arm64 waived"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required PP4 shard artifact: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid PP4 shard JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"PP4 shard JSON is not an object: {path}")
    return value


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if not source.is_dir():
        raise SystemExit(f"missing required PP4 shard directory: {source}")
    shutil.copytree(source, destination)


def collect_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not safe_relative(relative):
            raise SystemExit(f"unsafe fan-in path: {relative}")
        result[relative] = sha256(path)
    return result


def write_manifest(output: Path, artifact: Path, checker_summary: Path | None) -> None:
    entries = collect_files(output)
    entries.pop(artifact.relative_to(output).as_posix(), None)
    entries.pop("checksums.sha256", None)
    if checker_summary is not None:
        relative = checker_summary.relative_to(output).as_posix()
        if not checker_summary.is_file():
            raise SystemExit(f"checker summary requested but missing: {checker_summary}")
        entries[relative] = sha256(checker_summary)
    entries[artifact.relative_to(output).as_posix()] = sha256(artifact)
    (output / "checksums.sha256").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(entries.items()))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="fresh PP4 fan-in evidence directory")
    parser.add_argument("--qa", type=Path, required=True, help="PP4 QA shard directory")
    parser.add_argument("--release", type=Path, required=True, help="PP4 release shard directory")
    parser.add_argument("--gallery", type=Path, required=True, help="PP4 replay shard directory")
    parser.add_argument(
        "--checker-summary",
        type=Path,
        help="optional finalized strict-checker JSON to include in checksums.sha256",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shards = output / "shards"
    if shards.exists():
        shutil.rmtree(shards)
    shards.mkdir()

    qa = read_object(args.qa.resolve() / "qa-propeller-fan-in.json")
    release = read_object(args.release.resolve() / "native-x86-pp4-release-v1.json")
    gallery = read_object(args.gallery.resolve() / "native-x86-pp4-gallery-replay-v1.json")
    expected_schemas = {
        "qa": ("sonium.propellers.pp4-qa-fan-in.v1", "gated"),
        "release": ("sonium.propellers.native-x86-pp4.v1", "gated"),
        "gallery": ("sonium.propellers.pp4-gallery-replay-evidence.v1", "pass"),
    }
    for lane, document in (("qa", qa), ("release", release), ("gallery", gallery)):
        schema, status = expected_schemas[lane]
        if document.get("schema") != schema or document.get("status") != status:
            raise SystemExit(
                f"{lane} shard is not a passing/gated producer result: "
                f"schema={document.get('schema')!r} status={document.get('status')!r}"
            )
    if qa.get("release_join", {}).get("status") != "blocked":
        raise SystemExit("QA shard unexpectedly claims an unblocked release join")
    if len(qa.get("registered_case_ids", [])) != 29 or qa.get("coverage", {}).get("passed") != 29:
        raise SystemExit("QA shard does not contain all 29 passing propeller cases")
    if len(gallery.get("files", [])) != 14 or gallery.get("deterministic") is not True:
        raise SystemExit("gallery shard does not contain the deterministic 14-file replay")

    copy_tree(args.qa.resolve(), shards / "qa")
    copy_tree(args.release.resolve(), shards / "release")
    copy_tree(args.gallery.resolve(), shards / "gallery-replay")

    artifact = output / "pp4-release-fanin-v1.json"
    record = {
        "schema": SCHEMA,
        "status": "gated",
        "scope": SCOPE,
        "claim_class": "host-applicable PP4 shard fan-in; no external/full-fidelity promotion",
        "promotion_eligible": False,
        "inventory": {
            "source": "crates/sonium-qa/validation-manifest.toml",
            "manifest_sha256": qa.get("manifest_sha256"),
            "registered_case_count": qa.get("registered_case_count"),
            "registered_case_ids": qa.get("registered_case_ids"),
            "executed": qa.get("coverage", {}).get("executed"),
            "passed": qa.get("coverage", {}).get("passed"),
            "unsupported": qa.get("coverage", {}).get("unsupported"),
        },
        "shards": [
            {
                "id": "qa",
                "path": "shards/qa/qa-propeller-fan-in.json",
                "schema": qa["schema"],
                "status": qa["status"],
                "checker": "crates/sonium-sim-propellers/benchmarks/check_pp4_qa.py",
            },
            {
                "id": "release",
                "path": "shards/release/native-x86-pp4-release-v1.json",
                "schema": release["schema"],
                "status": release["status"],
                "checker": "crates/sonium-sim-propellers/benchmarks/check_pp4_release.py",
            },
            {
                "id": "gallery-replay",
                "path": "shards/gallery-replay/native-x86-pp4-gallery-replay-v1.json",
                "schema": gallery["schema"],
                "status": gallery["status"],
                "checker": "crates/sonium-sim-propellers/benchmarks/check_pp4_gallery_replay.py",
            },
        ],
        "final_release_gate": {
            "status": "blocked",
            "official_recipe": "just qa-release-critical",
            "reason": "PP2 and PP3 remain partial: external/full-fidelity promotion and accepted physical validation are unavailable; this local join must not claim a release pass",
            "blocked_lanes": [
                "PP2 external/full-fidelity CUDA propeller validation",
                "PP3 external/full-fidelity resolved-flow/aeroacoustic validation",
            ],
            "physical_claim": "not promoted",
        },
        "artifact_checksums_sha256": {},
    }
    artifact.write_text(json.dumps(record, indent=2) + "\n")
    checksums = collect_files(output)
    checksums.pop(artifact.relative_to(output).as_posix(), None)
    checksums.pop("checksums.sha256", None)
    if args.checker_summary is not None:
        checksums.pop(args.checker_summary.resolve().relative_to(output).as_posix(), None)
    record["artifact_checksums_sha256"] = checksums
    artifact.write_text(json.dumps(record, indent=2) + "\n")
    write_manifest(output, artifact, args.checker_summary.resolve() if args.checker_summary else None)
    print(
        json.dumps(
            {
                "status": record["status"],
                "promotion_eligible": record["promotion_eligible"],
                "registered": record["inventory"]["registered_case_count"],
                "shard_count": len(record["shards"]),
                "manifest_entries": len((output / "checksums.sha256").read_text().splitlines()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
