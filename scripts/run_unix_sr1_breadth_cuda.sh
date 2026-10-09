#!/usr/bin/env bash
# Native SM120 SR1 room-FEM resident-state breadth and recovery evidence.
# Every build, runtime, temporary, and evidence path is repository-local.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
test "$(uname -s)" = Linux || { echo "SR1 breadth evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SR1 breadth evidence requires x86_64" >&2; exit 2; }

output="${SONIUM_OUTPUT_DIR:-$repo_root/.target-sr1-breadth-fresh/evidence}"
target="${CARGO_TARGET_DIR:-$repo_root/.target-sr1-breadth-fresh/cargo}"
tmp_dir="$output/.tmp"
runtime_dir="$output/.runtime"
mkdir -p "$output" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"

resident_artifact="$output/f64-resident-reuse-4-frequency.json"
resident_log="$output/f64-resident-reuse-4-frequency-test.log"
resident_check="$output/f64-resident-reuse-4-frequency-check.json"
recovery_artifact="$output/cuda-resident-recovery.json"
recovery_log="$output/cuda-resident-recovery-test.log"
recovery_check="$output/cuda-resident-recovery-check.json"

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_RESIDENT_ARTIFACT="$resident_artifact" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_fem_four_frequency_resident_state_reuses_preconditioner \
    -- --nocapture --test-threads=1 >"$resident_log" 2>&1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$resident_artifact" >"$resident_check"

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_FREQUENCY=2 \
    SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_ARTIFACT="$recovery_artifact" \
    SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_SUCCESS_ARTIFACT="$output/should-not-publish.json" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_resident_mid_frequency_cancellation_retries_after_warm_state \
    -- --nocapture --test-threads=1 >"$recovery_log" 2>&1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$recovery_artifact" >"$recovery_check"

(cd "$output" && \
    sha256sum "$resident_artifact" >"$(basename "$resident_artifact").sha256" && \
    sha256sum "$recovery_artifact" >"$(basename "$recovery_artifact").sha256" && \
    sha256sum -c "$(basename "$resident_artifact").sha256" && \
    sha256sum -c "$(basename "$recovery_artifact").sha256")

{
    printf 'runner=run_unix_sr1_breadth_cuda.sh\n'
    printf 'host_architecture=%s\n' "$(uname -m)"
    printf 'kernel=%s\n' "$(uname -srv)"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'rustc=%s\n' "$(rustc --version)"
} >"$output/runner-metadata.txt"

(cd "$output" && sha256sum \
    "$(basename "$resident_artifact")" "$(basename "$resident_artifact").sha256" \
    "$(basename "$resident_log")" "$(basename "$resident_check")" \
    "$(basename "$recovery_artifact")" "$(basename "$recovery_artifact").sha256" \
    "$(basename "$recovery_log")" "$(basename "$recovery_check")" \
    runner-metadata.txt > checksums.sha256 && sha256sum -c checksums.sha256)

echo "SR1 breadth CUDA evidence passed; artifacts and checksums: $output"
