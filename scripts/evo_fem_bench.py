#!/usr/bin/env python3
"""Evo benchmark wrapper for sonium-sim-room FEM solver.

Runs the Rust benchmark example and reports the score to evo.
"""

import json
import subprocess
import sys
import os

def main():
    target = os.environ.get("EVO_WORKTREE", ".")
    cmd = [
        "/Users/pierre/.cargo/bin/cargo",
        "run",
        "--release",
        "--manifest-path",
        os.path.join(target, "Cargo.toml"),
        "-p", "sonium-sim-room",
        "--example", "evo_fem_bench",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Benchmark failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON output from the last line
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                print(json.dumps(data))
                return
            except json.JSONDecodeError:
                continue

    print(f"Could not find JSON output in:\n{result.stdout}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
