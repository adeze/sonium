#!/usr/bin/env bash
# Stable SR1 production room-FEM CUDA evidence on native Linux/x86_64.
# Keep all Cargo, temporary, runtime, and evidence paths inside the repository
# so repeated QA runs do not depend on host-global temporary state.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
test "$(uname -s)" = Linux || { echo "SR1 CUDA evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SR1 CUDA evidence requires x86_64" >&2; exit 2; }

output="${SONIUM_OUTPUT_DIR:-$repo_root/.target-sr1-fresh/artifact}"
target="${CARGO_TARGET_DIR:-$repo_root/.target-sr1-fresh/cargo}"
tmp_dir="$output/.tmp"
runtime_dir="$output/.runtime"
mkdir -p "$output" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_ARTIFACT="$output/f64-outer.json" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_fem_matches_cpu_backend_with_checked_parity_budget \
    -- --nocapture --test-threads=1 \
    >"$output/f64-outer-test.log" 2>&1

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_ARTIFACT="$output/f32-block-rhs.json" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_fem_block_rhs_publishes_physical_ledgers \
    -- --nocapture --test-threads=1 \
    >"$output/f32-block-rhs-test.log" 2>&1

SONIUM_SIM_ROOM_REQUIRE_CUDA=1 CARGO_TARGET_DIR="$target" \
    cargo test --release -p sonium-sim-room --features cuda \
    --test backend_parity_runtime \
    cuda_cancellation_rejects_before_device_setup \
    -- --nocapture --test-threads=1 \
    >"$output/pre-setup-cancellation-test.log" 2>&1

# The injection is consumed after frequency zero's device solve and before
# diagnostics/output publication. The test then reacquires CUDA after Drop
# unwinding and writes a separate typed cancellation artifact.
SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_CANCEL_FREQUENCY=0 \
    SONIUM_SIM_ROOM_SR1_CANCEL_ARTIFACT="$output/cuda-mid-frequency-cancellation.json" \
    SONIUM_SIM_ROOM_SR1_CANCEL_SUCCESS_ARTIFACT="$output/should-not-publish.json" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_mid_frequency_cancellation_is_typed_and_does_not_publish_partial_result \
    -- --nocapture --test-threads=1 \
    >"$output/mid-frequency-cancellation-test.log" 2>&1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$output/f64-outer.json" >"$output/f64-outer-check.json"
python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$output/f32-block-rhs.json" >"$output/f32-block-rhs-check.json"
python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$output/cuda-mid-frequency-cancellation.json" \
    >"$output/cuda-mid-frequency-cancellation-check.json"

# Native launch failure: the room runner arms the checked CUDA seam before
# allocating its FEM work buffers. The failure must not publish output, while
# a recreated backend must complete the same production fixture.
SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_FAILURE_POINT=launch \
    SONIUM_SIM_ROOM_SR1_FAILURE_ARTIFACT="$output/cuda-launch-failure.json" \
    SONIUM_SIM_ROOM_SR1_FAILURE_RETRY_ARTIFACT="$output/cuda-launch-retry.json" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_injected_launch_failure_is_typed_unwound_and_retryable \
    -- --nocapture --test-threads=1 \
    >"$output/cuda-launch-failure-test.log" 2>&1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$output/cuda-launch-failure.json" >"$output/cuda-launch-failure-check.json"

# Native terminal device loss: unlike launch failure, this disposition must
# recreate the context. The same no-partial-publication and retry checks apply.
SONIUM_SIM_ROOM_REQUIRE_CUDA=1 \
    SONIUM_SIM_ROOM_SR1_FAILURE_POINT=device-loss \
    SONIUM_SIM_ROOM_SR1_FAILURE_ARTIFACT="$output/cuda-device-loss.json" \
    SONIUM_SIM_ROOM_SR1_FAILURE_RETRY_ARTIFACT="$output/cuda-device-loss-retry.json" \
    CARGO_TARGET_DIR="$target" cargo test --release -p sonium-sim-room \
    --features cuda --test backend_parity_runtime \
    cuda_injected_device_loss_requires_recreated_context_and_retries \
    -- --nocapture --test-threads=1 \
    >"$output/cuda-device-loss-test.log" 2>&1

python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_sr1_cuda.py" \
    "$output/cuda-device-loss.json" >"$output/cuda-device-loss-check.json"

{
    printf 'runner=run_unix_sr1_cuda.sh\n'
    printf 'host_architecture=%s\n' "$(uname -m)"
    printf 'kernel=%s\n' "$(uname -srv)"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'rustc=%s\n' "$(rustc --version)"
} >"$output/runner-metadata.txt"

(
    cd "$output"
    sha256sum \
        f64-outer.json f64-outer-check.json f64-outer-test.log \
        f32-block-rhs.json f32-block-rhs-check.json f32-block-rhs-test.log \
        pre-setup-cancellation-test.log \
        cuda-mid-frequency-cancellation.json \
        cuda-mid-frequency-cancellation-check.json \
        mid-frequency-cancellation-test.log \
        cuda-launch-failure.json cuda-launch-failure-check.json \
        cuda-launch-failure-test.log cuda-launch-retry.json \
        cuda-device-loss.json cuda-device-loss-check.json \
        cuda-device-loss-test.log cuda-device-loss-retry.json \
        runner-metadata.txt \
        > checksums.sha256
)
echo "SR1 CUDA evidence passed; artifacts and checksums: $output"
