#!/usr/bin/env python3
"""Check that every dated next-review document has one grouped index entry."""

from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "reviews"
INDEX = REVIEWS / "next-index.md"
ENTRY = re.compile(r"^- `(?P<name>next-[^`]+\.md)`$", re.MULTILINE)


def main() -> int:
    expected = {
        path.name
        for path in REVIEWS.glob("next-*.md")
        if path.name != INDEX.name
    }
    indexed = [match.group("name") for match in ENTRY.finditer(INDEX.read_text())]
    counts = {name: indexed.count(name) for name in set(indexed)}
    missing = sorted(expected - set(indexed))
    unknown = sorted(set(indexed) - expected)
    duplicates = sorted(name for name, count in counts.items() if count != 1)

    if not (missing or unknown or duplicates):
        return 0

    if missing:
        print("review index is missing: " + ", ".join(missing), file=sys.stderr)
    if unknown:
        print("review index references unknown files: " + ", ".join(unknown), file=sys.stderr)
    if duplicates:
        print("review index entries must appear once: " + ", ".join(duplicates), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
