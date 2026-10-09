#!/usr/bin/env python3
"""Dependency-free reader for the canonical XEM multiphysics v1 envelope."""

from __future__ import annotations

import json
import pathlib
import sys


KNOWN_ROOT = {
    "schema_version",
    "id",
    "conventions",
    "domains",
    "regions",
    "fields",
    "boundary_conditions",
    "interface_conditions",
    "material_laws",
    "couplings",
    "design_variables",
    "objectives",
    "constraints",
    "provenance",
}


def read(path: pathlib.Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("XEM multiphysics document must be an object")
    unknown = set(document) - KNOWN_ROOT
    if unknown:
        raise ValueError(f"unknown XEM multiphysics root fields: {sorted(unknown)}")
    if document.get("schema_version") != 1:
        raise ValueError("compatibility reader supports schema_version 1 only")
    if not isinstance(document.get("id"), str) or not document["id"].strip():
        raise ValueError("multiphysics problem id is required")
    for name in KNOWN_ROOT - {"schema_version", "id", "conventions", "provenance"}:
        if not isinstance(document.get(name), list):
            raise ValueError(f"{name} must be an array")
    if not isinstance(document.get("conventions"), dict):
        raise ValueError("conventions must be an object")
    if not isinstance(document.get("provenance"), dict):
        raise ValueError("provenance must be an object")
    return document


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: read_multiphysics_problem.py <problem.json>")
    parsed = read(pathlib.Path(sys.argv[1]))
    print(
        json.dumps(
            {
                "schema_version": parsed["schema_version"],
                "id": parsed["id"],
                "domains": len(parsed["domains"]),
                "fields": len(parsed["fields"]),
            },
            sort_keys=True,
        )
    )
