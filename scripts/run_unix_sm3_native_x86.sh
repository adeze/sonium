#!/usr/bin/env bash
# Run and check the complete native Linux/x86_64 SM3 modeling evidence lane.
# The ignored tests are explicit because the bounded sector and FEM--BEM
# replay is intentionally too expensive for the default unit-test gate.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "SM3 evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || {
    echo "SM3 evidence requires native x86_64" >&2
    exit 2
}

target_dir="${CARGO_TARGET_DIR:-$repo_root/.target-unix-sm3-observables}"
output_root="${SONIUM_OUTPUT_DIR:-$repo_root/target/sonium-unix-sm3-observables}"
mkdir -p "$output_root"

run_test() {
    local test_name="$1"
    local log_name="$2"
    local ignored="$3"
    if [[ "$ignored" == yes ]]; then
        SONIUM_OUTPUT_DIR="$output_root" \
            RAYON_NUM_THREADS=1 \
            CARGO_TARGET_DIR="$target_dir" \
            cargo test --release -p sonium-sim-speaker-modeling \
            --test native_x86_evidence "$test_name" -- \
            --ignored --nocapture --test-threads=1 2>&1 | tee "$output_root/$log_name"
    else
        SONIUM_OUTPUT_DIR="$output_root" \
            RAYON_NUM_THREADS=1 \
            CARGO_TARGET_DIR="$target_dir" \
            cargo test --release -p sonium-sim-speaker-modeling \
            --test native_x86_evidence "$test_name" -- \
            --nocapture --test-threads=1 2>&1 | tee "$output_root/$log_name"
    fi
}

run_test native_x86_driver_artifact_smoke_evidence_v1 driver.log no
run_test native_x86_axisymmetric_coupled_replay_evidence_v1 axisymmetric.log no
run_test native_x86_sector_mode_replay_evidence_v1 sector.log yes
run_test native_x86_modeling_replay_evidence_v1 modeling.log yes

artifact="$output_root/native-x86-sm3-v1.json"
test -s "$artifact" || { echo "SM3 producer did not emit $artifact" >&2; exit 1; }
(cd "$output_root" && sha256sum "$(basename "$artifact")" > checksums.sha256)
python3 "$repo_root/crates/sonium-sim-speaker-modeling/benchmarks/check_native_x86_sm3.py" \
    "$artifact"
(cd "$output_root" && sha256sum -c checksums.sha256 >/dev/null)
echo "SUCCESS: SM3 native x86_64 artifact and checksum written to $output_root"
