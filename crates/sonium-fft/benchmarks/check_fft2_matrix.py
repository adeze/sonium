#!/usr/bin/env python3
"""Validate the required-device execution matrix emitted by the FFT2 runner.

The benchmark rows carry the detailed correctness and warm-resource gates;
this small companion check validates the aggregate accounting so an
unavailable, unsupported, or host-staged row cannot silently disappear from a
required CUDA result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_fft2_matrix.py ARTIFACT.json", file=sys.stderr)
        return 2
    try:
        artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FFT2 matrix error: {error}", file=sys.stderr)
        return 2

    matrix = artifact.get("matrix")
    rows = artifact.get("rows")
    unsupported_cells = artifact.get("unsupported_cells")
    if not isinstance(matrix, dict) or not isinstance(rows, list):
        print("FFT2 matrix error: matrix and rows are required", file=sys.stderr)
        return 2
    if not isinstance(unsupported_cells, list) or not all(
        isinstance(cell, str) and cell for cell in unsupported_cells
    ):
        print("FFT2 matrix error: unsupported_cells must be a non-empty string list", file=sys.stderr)
        return 2
    if matrix.get("fail_closed") is not True:
        print("FFT2 matrix error: fail_closed marker is missing", file=sys.stderr)
        return 2

    counts = {
        "rows_total": len(rows),
        "executed": sum(row.get("status") in ("passed", "failed") for row in rows),
        "passed": sum(row.get("status") == "passed" for row in rows),
        "failed": sum(row.get("status") == "failed" for row in rows),
        "skipped": sum(row.get("status") == "unavailable" for row in rows),
        "unsupported_rows": sum(row.get("status") == "unsupported" for row in rows),
        "unsupported_cells": len(unsupported_cells),
    }
    for key, expected in counts.items():
        if matrix.get(key) != expected:
            print(
                f"FFT2 matrix error: {key}={matrix.get(key)!r}, expected {expected}",
                file=sys.stderr,
            )
            return 2
    if matrix.get("required_cuda") is not True:
        print("FFT2 matrix error: required_cuda must be true", file=sys.stderr)
        return 2
    cancellation = artifact.get("cancellation")
    if not isinstance(cancellation, dict):
        print("FFT2 matrix error: cancellation evidence is missing", file=sys.stderr)
        return 2
    required_cancellation = (
        "regular_c32",
        "regular_c64",
        "strided_c64",
        "pre_cancel_rejected",
        "post_submit_cancelled",
        "completion_fence_drained",
        "plan_reused",
    )
    if cancellation.get("status") != "passed" or any(
        cancellation.get(field) is not True for field in required_cancellation
    ):
        print("FFT2 matrix error: cancellation cleanup/reuse gate failed", file=sys.stderr)
        return 1
    if artifact.get("status") != "passed" or counts["failed"] or counts["skipped"]:
        print(
            "FFT2 matrix error: required CUDA run did not pass all executed rows "
            f"(status={artifact.get('status')!r}, counts={counts})",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "status": "passed",
                "required_cuda": True,
                **counts,
                "unsupported_cells": unsupported_cells,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
