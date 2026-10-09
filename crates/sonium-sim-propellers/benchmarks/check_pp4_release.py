#!/usr/bin/env python3
"""Strict checker for the native PP4 gallery/release evidence bundle."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path, PurePosixPath


FIGURES = {
    "source_spectrum.csv",
    "blade_load_distribution.csv",
    "optimized-blade-loading.png",
    "axial-wake-proxy.png",
    "spectrum.svg",
    "thrust-versus-operating-point.svg",
    "torque-versus-operating-point.svg",
    "blade-load-distribution.svg",
    "conventional-toroidal-comparison.svg",
    "rotation.gif",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def check_png(path: Path) -> None:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1920, 1080)
    assert len(data) > 100


def check_csv(path: Path, header: str, rows: int) -> None:
    lines = path.read_text().splitlines()
    assert lines and lines[0] == header
    assert len(lines) == rows + 1
    for line in lines[1:]:
        values = line.split(",")
        assert values
        for value in values:
            number = float(value)
            assert math.isfinite(number)


def check_cli_gallery(root: Path) -> None:
    gallery = root / "cli-gallery"
    assert gallery.is_dir()
    manifest = json.loads((gallery / "gallery.json").read_text())
    assert manifest["schema_version"] == 1
    assert (manifest["width_px"], manifest["height_px"]) == (1920, 1080)
    assert {figure["filename"] for figure in manifest["figures"]} == FIGURES
    assert all("fidelity=PanelGeometrySurrogate" in figure["caption"] for figure in manifest["figures"])
    assert "Gallery manifest written to" in (root / "cli.log").read_text()
    for figure in manifest["figures"]:
        path = gallery / figure["filename"]
        if figure["kind"] == "png":
            check_png(path)
        elif figure["kind"] == "svg":
            svg = path.read_text()
            assert '<svg ' in svg and 'width="1920"' in svg and 'height="1080"' in svg
        elif figure["kind"] == "gif":
            data = path.read_bytes()
            assert data[:6] == b"GIF89a"
            assert struct.unpack("<HH", data[6:10]) == (320, 180)
        elif figure["kind"] == "csv":
            lines = path.read_text().splitlines()
            assert len(lines) > 1
            assert all(math.isfinite(float(value)) for line in lines[1:] for value in line.split(","))
        else:
            raise AssertionError(figure["kind"])


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ARTIFACT")
    artifact_path = Path(sys.argv[1]).resolve()
    root = artifact_path.parent
    artifact = json.loads(artifact_path.read_text())

    assert artifact["schema"] == "sonium.propellers.native-x86-pp4.v1"
    assert artifact["status"] == "gated"
    assert artifact["scope"] == "native Linux x86_64 CPU/f64; Windows/Arm64 waived"
    assert artifact["deterministic_mode"] is True
    assert artifact["thread_policy"].startswith("RAYON_NUM_THREADS=")
    assert artifact["final_release_gate"].startswith("blocked until PP1")
    capability = artifact["capability"]
    assert capability["renderer"] == "sonium_xem::io::GalleryRenderer"
    assert capability["renderer_status"] == "available"
    assert capability["locale"]
    assert "locale-independent" in capability["locale_policy"]
    assert capability["pre_publication_hook"] == "SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION"
    assert capability["renderer_fault_hook"] == "SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE"
    assert capability["fault_directory"] == ".target-unix-pp4-faults"
    assert "/tmp" not in capability["fault_directory"]
    check_cli_gallery(root)
    host = artifact["host"]
    for key in ("compiler", "os", "kernel", "cpu_model", "microcode", "target", "target_features"):
        assert isinstance(host[key], str) and host[key]
    execution = artifact["execution"]
    assert execution["requested_backend"] == "cpu"
    assert execution["actual_backend"] == "cpu"
    assert execution["precision"] == "f64"
    assert execution["fidelity"] == "panel_geometry_surrogate"
    assert len(execution["input_checksum_sha256"]) == 64
    assert len(execution["output_checksum_sha256"]) == 64

    gallery = artifact["gallery"]
    gallery_path = Path(gallery["path"])
    assert safe_relative(gallery["path"])
    gallery_root = root / gallery_path
    assert gallery["manifest_schema"] == "sonium-xem-gallery/v1"
    assert gallery["figure_count"] == 10
    assert gallery["png_count"] == 2
    assert gallery["svg_count"] == 5
    assert gallery["gif_dimensions"] == [320, 180]
    assert gallery["gif_frames"] == 12
    assert gallery["spectrum_rows"] >= 2
    assert gallery["panel_load_rows"] >= 1
    assert gallery["vtu_files"] == 2
    assert digest(gallery_root / "gallery.json") == gallery["manifest_sha256"]
    assert set(gallery["artifact_checksums_sha256"]) >= {
        "gallery.json",
        "execution_evidence.json",
        "optimized_blade.vtu",
        "baseline_blade.vtu",
    }
    for name, expected in gallery["artifact_checksums_sha256"].items():
        assert safe_relative(name)
        assert digest(gallery_root / name) == expected

    manifest = json.loads((gallery_root / "gallery.json").read_text())
    assert manifest["schema_version"] == 1
    assert (manifest["width_px"], manifest["height_px"]) == (1920, 1080)
    assert {figure["filename"] for figure in manifest["figures"]} == FIGURES
    assert all("fidelity=PanelGeometrySurrogate" in figure["caption"] for figure in manifest["figures"])
    for figure in manifest["figures"]:
        path = gallery_root / figure["filename"]
        if figure["kind"] == "png":
            check_png(path)
        elif figure["kind"] == "svg":
            svg = path.read_text()
            assert '<svg ' in svg and 'width="1920"' in svg and 'height="1080"' in svg
        elif figure["kind"] == "gif":
            data = path.read_bytes()
            assert data[:6] == b"GIF89a"
            assert struct.unpack("<HH", data[6:10]) == (320, 180)
        elif figure["kind"] == "csv":
            if figure["filename"] == "source_spectrum.csv":
                check_csv(path, "frequency_hz,pressure_pa,baseline_pressure_pa", gallery["spectrum_rows"])
            else:
                check_csv(path, "radius_fraction,loading_proxy_m2", gallery["panel_load_rows"])
        else:
            raise AssertionError(figure["kind"])

    probes = artifact["failure_probes"]
    assert len(probes) >= 8
    required_probe_fields = {
        "id",
        "status",
        "typed_error",
        "partial_publish",
        "exit_code",
        "sentinel_preserved",
        "retry_succeeded",
        "published_artifact_count",
    }
    assert all(required_probe_fields <= set(probe) for probe in probes)
    assert all(
        probe["status"] == "typed_rejected"
        and probe["partial_publish"] is False
        and probe["published_artifact_count"] == 0
        for probe in probes
    )
    fault_probes = {
        probe["id"]: probe
        for probe in probes
        if probe["id"]
        in {
            "child-process.pre-publication-interruption",
            "child-process.renderer-unavailable",
        }
    }
    assert set(fault_probes) == {
        "child-process.pre-publication-interruption",
        "child-process.renderer-unavailable",
    }
    for probe in fault_probes.values():
        assert isinstance(probe["exit_code"], int) and not isinstance(probe["exit_code"], bool)
        assert probe["exit_code"] != 0
        assert "ErrorKind::" in probe["typed_error"]
        assert probe["sentinel_preserved"] is True
        assert probe["retry_succeeded"] is True
        assert probe["published_artifact_count"] == 0
    qa_join = artifact["qa_join"]
    assert qa_join["status"] == "blocked"
    assert len(qa_join["registered_propeller_cases"]) == 29
    assert all(case.startswith("propeller.") for case in qa_join["registered_propeller_cases"])
    assert "PP1--PP3 remain partial" in qa_join["reason"]

    checksums = {}
    for line in (root / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert len(expected) == 64 and safe_relative(name)
        assert name not in checksums
        checksums[name] = expected
    assert "native-x86-pp4-release-v1.json" in checksums
    assert all(digest(root / name) == expected for name, expected in checksums.items())
    print(json.dumps({"gallery_figures": 10, "failure_probes": len(probes), "qa_join": "blocked", "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
