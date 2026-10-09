import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from qa_result_policy import validate_result


CASE = {
    "id": "qa.policy-test.v1",
    "tier": "accelerator-gate",
    "required_hardware": "nvidia-cuda",
    "evidence_class": "numerical-parity",
    "thresholds_required": True,
    "runtime_required": True,
    "hardware_provenance_required": True,
    "artifact_provenance_required": True,
}


def valid_result(artifact_checksum: str) -> dict:
    return {
        "schema_version": 1,
        "case_id": CASE["id"],
        "status": "passed",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "backend_outcomes": [{"backend": "cuda", "status": "passed"}],
        "metrics": {"residual": 0.0},
        "thresholds": {"residual": 1.0},
        "threshold_directions": {"residual": "maximum"},
        "elapsed_ms": 1,
        "git_sha": "test",
        "environment": {},
        "artifacts": ["report.json"],
        "evidence": {
            "timings_ms": {},
            "artifact_checksums": {"report.json": artifact_checksum},
            "artifact_provenance": {
                "report.json": {
                    "source": "test-fixture",
                    "generation": "deterministic",
                    "checksum": artifact_checksum,
                    "version": "v1",
                }
            },
        },
        "runtime": {
            "context_reuse": "reused",
            "counts": {"executed": 1, "passed": 1, "failed": 0, "unsupported": 0},
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
                "device": "NVIDIA test device",
                "driver": "test-driver",
                "toolkit": "test-toolkit",
            },
        },
    }


class QaResultPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        payload = b"deterministic QA artifact\n"
        (self.root / "report.json").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.result = valid_result(f"sha256:{digest}")

    def tearDown(self):
        self.tempdir.cleanup()

    def assert_rejected(self, result):
        self.assertTrue(
            validate_result(CASE, result, artifact_root=self.root),
            msg="invalid QA result was accepted",
        )

    def test_valid_result_is_accepted(self):
        self.assertEqual(validate_result(CASE, self.result, artifact_root=self.root), [])

    def test_missing_runtime_is_rejected(self):
        result = copy.deepcopy(self.result)
        result.pop("runtime")
        self.assert_rejected(result)

    def test_skipped_as_passed_is_rejected(self):
        result = copy.deepcopy(self.result)
        result["status"] = "skipped"
        self.assert_rejected(result)

    def test_fallback_and_wrong_hardware_are_rejected(self):
        result = copy.deepcopy(self.result)
        result["actual_backend"] = "cpu"
        self.assert_rejected(result)
        result = copy.deepcopy(self.result)
        result["runtime"]["hardware"]["device"] = "unknown"
        self.assert_rejected(result)

    def test_stale_checksum_is_rejected(self):
        result = copy.deepcopy(self.result)
        result["evidence"]["artifact_checksums"]["report.json"] = "sha256:stale"
        result["evidence"]["artifact_provenance"]["report.json"]["checksum"] = "sha256:stale"
        self.assert_rejected(result)

    def test_nonfinite_and_threshold_free_results_are_rejected(self):
        result = copy.deepcopy(self.result)
        result["metrics"]["residual"] = float("nan")
        self.assert_rejected(result)
        result = copy.deepcopy(self.result)
        result["thresholds"] = {}
        result["threshold_directions"] = {}
        self.assert_rejected(result)

    def test_unsatisfied_threshold_and_nonfinite_physical_evidence_are_rejected(self):
        result = copy.deepcopy(self.result)
        result["metrics"]["residual"] = 2.0
        self.assert_rejected(result)
        result = copy.deepcopy(self.result)
        result["evidence"]["physical_residual"] = float("inf")
        self.assert_rejected(result)

    def test_missing_schema_fields_and_negative_runtime_values_are_rejected(self):
        result = copy.deepcopy(self.result)
        result.pop("git_sha")
        self.assert_rejected(result)
        result = copy.deepcopy(self.result)
        result["runtime"]["counters"]["launches"] = -1
        self.assert_rejected(result)


if __name__ == "__main__":
    unittest.main()
