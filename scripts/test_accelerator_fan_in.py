import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_accelerator_fan_in import validate_lane


CASE = {
    "id": "qa.fan-in-fixture.v1",
    "tier": "accelerator-gate",
    "required_hardware": "nvidia-cuda",
    "evidence_class": "numerical-parity",
    "thresholds_required": True,
    "runtime_required": True,
    "hardware_provenance_required": True,
    "artifact_provenance_required": True,
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_fixture(root: Path, nested_count: int = 1) -> tuple[Path, Path]:
    manifest_path = root / "crates" / "sonium-qa" / "validation-manifest.toml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("# deterministic fan-in fixture\n")
    lane = root / "target" / "sonium-qa-accelerators" / "nvidia-cuda"
    case_dir = lane / "cases"
    artifact_dir = lane / "artifacts"
    case_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    payload = b"deterministic fan-in artifact\n"
    (artifact_dir / "report.json").write_bytes(payload)
    checksum = f"sha256:{digest(payload)}"
    result = {
        "schema_version": 1,
        "case_id": CASE["id"],
        "status": "passed",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "backend_outcomes": [{"backend": "cuda", "status": "passed"}],
        "metrics": {"error": 0.0},
        "thresholds": {"error": 1.0},
        "threshold_directions": {"error": "maximum"},
        "elapsed_ms": 1,
        "git_sha": "fixture",
        "environment": {},
        "artifacts": ["report.json"],
        "evidence": {
            "reference_checksum": "fixture-reference-v1",
            "timings_ms": {},
            "artifact_checksums": {"report.json": checksum},
            "artifact_provenance": {
                "report.json": {
                    "source": "fixture",
                    "generation": "deterministic",
                    "checksum": checksum,
                    "version": "v1",
                }
            },
        },
        "runtime": {
            "context_reuse": "reused",
            "counts": {"executed": nested_count, "passed": nested_count, "failed": 0, "unsupported": 0},
            "counters": {
                "allocations": 1,
                "h2d_transfers": 1,
                "d2h_transfers": 1,
                "d2d_transfers": 0,
                "synchronizations": 1,
                "launches": 1,
            },
            "memory": {"peak_bytes": 8, "live_bytes": 4},
            "hardware": {
                "device": "fixture-device",
                "driver": "fixture-driver",
                "toolkit": "fixture-toolkit",
                "peak_flops": "1000000000000",
                "memory_bandwidth": "100000000000",
            },
        },
    }
    (case_dir / f"{CASE['id']}.json").write_text(json.dumps(result) + "\n")
    shard = {
        "schema": "sonium-qa-accelerator-evidence/v1",
        "hardware": "nvidia-cuda",
        "required_backend": "cuda",
        "required_case_ids": [CASE["id"]],
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "coverage": {"executed": nested_count, "passed": nested_count, "failed": 0, "unsupported": 0},
        "cases": [result],
        "failures": [],
        "environment": {
            "device": "fixture-device",
            "driver": "fixture-driver",
            "toolkit": "fixture-toolkit",
            "peak_flops": "1000000000000",
            "memory_bandwidth": "100000000000",
            "os": "fixture-os",
            "compiler": "fixture-compiler",
        },
    }
    (lane / "accelerator-evidence.json").write_text(json.dumps(shard) + "\n")
    checksum_files = {
        "accelerator-evidence.json": lane / "accelerator-evidence.json",
        "validation-manifest.toml": manifest_path,
        f"cases/{CASE['id']}.json": case_dir / f"{CASE['id']}.json",
        "artifacts/report.json": artifact_dir / "report.json",
    }
    (lane / "checksums.sha256").write_text(
        "".join(f"{digest(path.read_bytes())}  {name}\n" for name, path in checksum_files.items())
    )
    manifest = {"case": [{**CASE, "reference_checksum": "fixture-reference-v1"}]}
    return root, manifest


class AcceleratorFanInTest(unittest.TestCase):
    def test_complete_shard_and_checksum_inventory_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = build_fixture(Path(directory))
            self.assertEqual(validate_lane(root, manifest, "nvidia-cuda", "cuda"), [])

    def test_stale_shard_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = build_fixture(Path(directory))
            artifact = root / "target" / "sonium-qa-accelerators" / "nvidia-cuda" / "artifacts" / "report.json"
            artifact.write_bytes(b"tampered\n")
            errors = validate_lane(root, manifest, "nvidia-cuda", "cuda")
            self.assertTrue(any("stale checksum" in error for error in errors), errors)

    def test_nested_runtime_counts_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = build_fixture(Path(directory), nested_count=3)
            self.assertEqual(validate_lane(root, manifest, "nvidia-cuda", "cuda"), [])


if __name__ == "__main__":
    unittest.main()
