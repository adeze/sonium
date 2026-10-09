#!/usr/bin/env bash
# Stable SR2 room CPU/CUDA parity runner.
#
# Keep every Cargo target, test temporary, runtime, and evidence path inside
# the repository.  This matters on managed runners where /run/user and /tmp
# may be unavailable or shared with unrelated review lanes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/.target-sr2-fresh/artifact"
target_dir="$repo_root/.target-sr2-fresh/cargo"
tmp_dir="$repo_root/.target-sr2-fresh/tmp"
runtime_dir="$repo_root/.target-sr2-fresh/runtime"

test "$(uname -s)" = Linux || {
    echo "SR2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SR2 CUDA evidence requires x86_64" >&2
    exit 2
}

mkdir -p "$output_dir" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"
export CARGO_TARGET_DIR="$target_dir"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR2_ARTIFACT="$output_dir/cuda-physical-parity.json" \
    SONIUM_SIM_ROOM_SR2_DIAGNOSTICS="$output_dir/physical-ledger-diagnostics.json" \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_physical_parity_matrix -- \
    --nocapture --test-threads=1
SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR2_BLOCK_ARTIFACT="$output_dir/cuda-block-rhs.json" \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_block_rhs_parity_matrix -- \
    --nocapture --test-threads=1
SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_single_source_parity_matrix -- \
    --nocapture --test-threads=1
cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_malformed_input_fails_before_device_setup -- \
    --nocapture --test-threads=1
cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_cancellation_is_typed_and_transactional -- \
    --nocapture --test-threads=1

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR2_FAILURE_POINT=solver-nonconvergence \
    SONIUM_SIM_ROOM_SR2_FAILURE_ARTIFACT="$output_dir/cuda-solver-nonconvergence.json" \
    SONIUM_SIM_ROOM_SR2_FAILURE_RETRY_ARTIFACT="$output_dir/cuda-solver-nonconvergence-retry.json" \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_solver_nonconvergence_is_typed_transactional_and_retryable -- \
    --nocapture --test-threads=1
SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR2_FAILURE_POINT=device-loss \
    SONIUM_SIM_ROOM_SR2_FAILURE_ARTIFACT="$output_dir/cuda-device-loss.json" \
    SONIUM_SIM_ROOM_SR2_FAILURE_RETRY_ARTIFACT="$output_dir/cuda-device-loss-retry.json" \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_sr2 cuda_sr2_device_loss_is_typed_transactional_and_recreated -- \
    --nocapture --test-threads=1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr2_cuda.py" \
    "$output_dir/cuda-physical-parity.json" \
    "$output_dir/cuda-solver-nonconvergence.json" \
    "$output_dir/cuda-device-loss.json" \
    > "$output_dir/cuda-physical-parity-check.json"
python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr2_cuda.py" \
    "$output_dir/cuda-block-rhs.json" \
    "$output_dir/cuda-solver-nonconvergence.json" \
    "$output_dir/cuda-device-loss.json" \
    > "$output_dir/cuda-block-rhs-check.json"

(cd "$output_dir" && \
    sha256sum cuda-physical-parity.json \
        physical-ledger-diagnostics.json \
        cuda-block-rhs.json \
        cuda-solver-nonconvergence.json \
        cuda-device-loss.json \
        cuda-solver-nonconvergence-retry.json \
        cuda-device-loss-retry.json \
        cuda-physical-parity-check.json \
        cuda-block-rhs-check.json > checksums.sha256 && \
    sha256sum -c checksums.sha256)
