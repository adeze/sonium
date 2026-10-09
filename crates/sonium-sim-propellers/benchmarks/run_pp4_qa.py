#!/usr/bin/env python3
"""Run and archive every propeller case registered in sonium-qa's manifest.

This is a propeller-owned shard. It reads the QA manifest but never edits it;
the workspace-wide ``just qa-release-critical`` aggregate remains owned by
sonium-qa and is represented as a blocked final join in the output record.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    manifest_path = root / "crates/sonium-qa/validation-manifest.toml"
    output = Path(
        os.environ.get(
            "SONIUM_PP4_QA_DIR", root / "target" / "sonium-unix-pp4" / "qa"
        )
    )
    cases_dir = output / "cases"
    logs_dir = output / "logs"
    artifacts_dir = output / "artifacts"
    for path in (cases_dir, logs_dir, artifacts_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    manifest = tomllib.loads(manifest_path.read_text())
    cases = [
        case
        for case in manifest["case"]
        if case.get("owner") == "sim-propellers"
        and case.get("bin", "").startswith("qa-propeller-")
    ]
    if not cases:
        raise SystemExit("no registered propeller QA cases found")
    env = os.environ.copy()
    env["SONIUM_QA_RESULT_DIR"] = str(cases_dir)
    env["SONIUM_OUTPUT_DIR"] = str(artifacts_dir)
    env.setdefault("RAYON_NUM_THREADS", "1")
    env.setdefault("SONIUM_QA_DETERMINISTIC_MODE", "1")
    env.setdefault("SONIUM_QA_THREAD_POLICY", "RAYON_NUM_THREADS=1")
    results = []
    failures = []
    for case in cases:
        features = ",".join(case.get("features", []))
        command = [
            "just",
            "_qa-run-features",
            case["bin"],
            str(case["timeout_seconds"]),
            features,
            case["id"],
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (logs_dir / f"{case['id']}.log").write_text(completed.stdout)
        result_path = cases_dir / f"{case['id']}.json"
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            result = {
                "case_id": case["id"],
                "status": "failed",
                "message": f"missing or malformed structured result: {error}",
            }
        results.append(result)
        if completed.returncode != 0 or result.get("status") != "passed":
            failures.append(
                {
                    "id": case["id"],
                    "returncode": completed.returncode,
                    "status": result.get("status"),
                }
            )
        print(completed.stdout, end="")

    case_results = {result.get("case_id"): result for result in results}
    checksums = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"qa-propeller-fan-in.json", "checksums.sha256"}
    }
    record = {
        "schema": "sonium.propellers.pp4-qa-fan-in.v1",
        "status": "failed" if failures else "gated",
        "scope": "native Linux x86_64 CPU/f64; Windows/Arm64 waived",
        "manifest_sha256": sha256(manifest_path),
        "registered_case_count": len(cases),
        "registered_case_ids": [case["id"] for case in cases],
        "coverage": {
            "executed": len(results),
            "passed": sum(result.get("status") == "passed" for result in results),
            "failed": len(failures),
            "unsupported": sum(result.get("status") == "unsupported" for result in results),
        },
        "case_results": results,
        "failures": failures,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "os": platform.system(),
            "kernel": platform.release(),
            "compiler": subprocess.run(
                ["rustc", "--version"], text=True, stdout=subprocess.PIPE, check=False
            ).stdout.strip(),
            "target": f"{platform.machine()}-{sys.platform}",
            "thread_policy": env["SONIUM_QA_THREAD_POLICY"],
            "deterministic_mode": env["SONIUM_QA_DETERMINISTIC_MODE"],
        },
        "release_join": {
            "status": "blocked",
            "owner": "sonium-qa",
            "official_recipe": "just qa-release-critical",
            "reason": "propeller shard is archived independently; PP2--PP3 remain partial and the workspace aggregate must not claim a release pass",
        },
        "artifact_checksums_sha256": checksums,
    }
    evidence = output / "qa-propeller-fan-in.json"
    evidence.write_text(json.dumps(record, indent=2) + "\n")
    checksums[str(evidence.relative_to(output))] = sha256(evidence)
    (output / "checksums.sha256").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(checksums.items()))
    )
    print(
        json.dumps(
            {
                "registered": len(cases),
                "passed": record["coverage"]["passed"],
                "failed": len(failures),
                "status": record["status"],
            }
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
