#!/usr/bin/env python3
"""Minimal dependency-free XEM v1 reader used by the compatibility gate.

This is intentionally independent from serde and the Rust implementation.  It
checks the stable envelope and the fields a consumer must understand before
handing the document to an application-specific decoder.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys


KNOWN_ROOT = {
    "schema_version",
    "room",
    "sources",
    "listening_positions",
    "frequencies",
    "boundaries",
    "solver",
    "visualization",
    "metadata",
    "conventions",
}


def read(path: pathlib.Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("XEM room document must be an object")
    unknown = set(document) - KNOWN_ROOT
    if unknown:
        raise ValueError(f"unknown XEM root fields: {sorted(unknown)}")
    if document.get("schema_version") != 1:
        raise ValueError("compatibility reader supports schema_version 1 only")
    room = document.get("room")
    if not isinstance(room, dict) or room.get("type") not in {"rectangular", "lshaped"}:
        raise ValueError("room type is missing or unsupported")
    sources = document.get("sources")
    positions = document.get("listening_positions")
    if not isinstance(sources, list) or not sources:
        raise ValueError("at least one source is required")
    if not isinstance(positions, list) or not positions:
        raise ValueError("at least one listening position is required")
    frequencies = document.get("frequencies")
    if not isinstance(frequencies, dict):
        raise ValueError("frequency configuration is missing")
    minimum = frequencies.get("min_freq")
    maximum = frequencies.get("max_freq")
    points = frequencies.get("num_points")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (minimum, maximum)):
        raise ValueError("frequency endpoints must be finite numbers")
    if minimum <= 0 or maximum <= 0 or minimum > maximum:
        raise ValueError("frequency endpoints must be positive and ordered")
    if not isinstance(points, int) or points < 1 or (points > 1 and minimum == maximum):
        raise ValueError("frequency grid is degenerate")
    return document


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: read_room_config.py <room-config.json>")
    parsed = read(pathlib.Path(sys.argv[1]))
    print(json.dumps({"schema_version": parsed["schema_version"], "sources": len(parsed["sources"]), "positions": len(parsed["listening_positions"])}, sort_keys=True))
