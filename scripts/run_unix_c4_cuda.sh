#!/usr/bin/env bash
# Focused C4 scheduling/graph evidence runner.  It keeps the scheduler race
# corpus and the real CUDA completion tests behind one stable command.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-c4}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-c4}"

test "$(uname -s)" = Linux || {
    echo "C4 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C4 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

# Preserve the exact focused run as a local evidence artifact for the review.
# The runner remains the single invocation point for this unit.
exec > >(tee "$output_dir/c4-run.log") 2>&1

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# The scheduler unit tests provide deterministic and randomized race coverage
# for admission, completion, cancellation, device loss, and exactly-once
# refund.  These tests are backend-independent and run in the same release
# target as the hardware cases below.
cargo test --release -p sonium-compute --features cuda --lib scheduler::tests -- \
    --nocapture --test-threads=1

# Checked graph capture/replay and the scheduler lease bridge use the real
# CUDA event completion boundary.  The final case injects device loss at that
# asynchronous boundary and verifies refund plus explicit restoration.
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_replays_stable_native_kernel_segment -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_replays_device_resident_krylov_update -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_replays_bem_fem_device_resident_sequence -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_submission_holds_and_refunds_scheduler_lease -- \
    --exact --nocapture --test-threads=1
# `spawn` must keep the scheduler lease alive until the native CUDA event
# completes; the same captured graph is then submitted again to prove retry
# and steady-state residency through the asynchronous bridge.
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_submission_spawn_replays_and_refunds_scheduler_lease -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_submission_device_loss_refunds_and_blocks_readmission -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_graph_submission_launch_failure_refunds_and_reuses_lease -- \
    --exact --nocapture --test-threads=1

# Keep the typed CUDA completion/resource source checks in the same evidence
# run; they ensure the event token retains failures and the runtime counters
# observe the completion boundary.
cargo test --release -p sonium-compute --features cuda --lib \
    cuda::cublas::tests::cuda_checked_completion_paths_preserve_failure_contracts -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --lib \
    cuda::cublas::tests::cuda_runtime_stats_since_reports_per_workload_deltas -- \
    --exact --nocapture --test-threads=1
