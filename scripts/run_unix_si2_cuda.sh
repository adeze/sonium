#!/usr/bin/env bash
# SI2 iterative CUDA capability/failure evidence runner.
#
# This is the one stable command used for SI2 validation.  The producer is
# required-device and therefore exits non-zero when CUDA is unavailable; that
# result is retained as an explicit unsupported artifact, never treated as a
# CPU pass.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-si2}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-si2}"

test "$(uname -s)" = Linux || {
    echo "SI2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SI2 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

overall=0
run_phase() {
    local name="$1"
    shift
    echo "=== SI2 phase: $name ==="
    "$@" 2>&1 | tee "$output_dir/${name}.log"
    local status=${PIPESTATUS[0]}
    printf '%s\t%s\n' "$name" "$status" >> "$output_dir/phase-status.tsv"
    if [ "$status" -ne 0 ]; then
        overall=1
    fi
}

run_phase backend-unit-tests \
    cargo test -p sonium-solvers-iterative --lib backend -- --nocapture --test-threads=1
run_phase iterative-check \
    cargo check -p sonium-solvers-iterative
run_phase facade-check \
    cargo check -p sonium-solvers
run_phase producer-check \
    cargo check -p sonium-qa --features 'qa-binaries cuda' --bin qa-solvers-iterative-cuda-failure-matrix
run_phase required-cuda-matrix \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
        --bin qa-solvers-iterative-cuda-failure-matrix

if [ -f "$output_dir/solvers_iterative_cuda_failure_matrix.json" ]; then
    sha256sum "$output_dir/solvers_iterative_cuda_failure_matrix.json" \
        > "$output_dir/checksums.sha256"
fi

if [ "$overall" -eq 0 ]; then
    echo "SUCCESS: SI2 evidence written to $output_dir"
else
    echo "SI2 completed with one or more explicit phase failures; inspect $output_dir" >&2
fi
exit "$overall"
