#!/usr/bin/env bash
# C1 native CUDA failure/resource-contract evidence runner.
#
# The producer is required-device and emits a deterministic 16-cell capability
# matrix plus 14 typed failure rows.  This runner binds those rows to the
# focused source tests, a native QA_RESULT, strict schema/resource checking,
# and one complete repo-local checksum bundle.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-c1-next-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-c1-next-fresh/artifact}"
tmp_dir="$output_dir/.tmp"
runtime_dir="$output_dir/.runtime"
artifact="$output_dir/solvers_iterative_cuda_failure_matrix.json"
checker="$repo_root/crates/sonium-compute/benchmarks/check_c1_cuda.py"
phase_status="$output_dir/phase-status.tsv"

test "$(uname -s)" = Linux || {
    echo "C1 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C1 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"
: > "$phase_status"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# These fields are runner inputs, not claims inferred from a host CPU.  A
# rerun on another validated NVIDIA host can override them explicitly.
export SONIUM_QA_CUDA_DEVICE_NAME="${SONIUM_QA_CUDA_DEVICE_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
export SONIUM_QA_CUDA_DRIVER_VERSION="${SONIUM_QA_CUDA_DRIVER_VERSION:-610.57.04}"
export SONIUM_QA_CUDA_TOOLKIT_VERSION="${SONIUM_QA_CUDA_TOOLKIT_VERSION:-CUDA UMD 13.3}"
export SONIUM_QA_CUDA_TOPOLOGY="${SONIUM_QA_CUDA_TOPOLOGY:-GPU0 PCIe Gen5 x16}"
export SONIUM_QA_CUDA_PEAK_FLOPS="${SONIUM_QA_CUDA_PEAK_FLOPS:-125000000000000}"
export SONIUM_QA_CUDA_MEMORY_BANDWIDTH="${SONIUM_QA_CUDA_MEMORY_BANDWIDTH:-1792000000000}"
export SONIUM_QA_OS="${SONIUM_QA_OS:-linux-x86_64}"
export SONIUM_QA_TARGET_FEATURES="${SONIUM_QA_TARGET_FEATURES:-cuda}"
export SONIUM_QA_RUNTIME_DISPATCH="${SONIUM_QA_RUNTIME_DISPATCH:-cuda-native}"
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-rayon-default}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

overall=0
run_phase() {
    local name="$1"
    shift
    echo "=== C1 phase: $name ==="
    set +e
    "$@" 2>&1 | tee "$output_dir/${name}.log"
    local status=${PIPESTATUS[0]}
    set -e
    printf '%s\t%s\n' "$name" "$status" >> "$phase_status"
    if [ "$status" -ne 0 ]; then
        overall=1
        echo "C1 phase $name failed with status $status" >&2
    fi
}

run_phase compute-check \
    cargo check -p sonium-compute --features cuda

run_phase cuda-source-contract \
    bash -c 'cargo test --release -p sonium-compute --features cuda --test parity cuda_failure_contract -- --nocapture --test-threads=1 && cargo test --release -p sonium-compute --features cuda --lib cuda_ -- --nocapture --test-threads=1'

run_phase producer-check \
    cargo check -p sonium-qa --features 'qa-binaries cuda' --bin qa-solvers-iterative-cuda-failure-matrix

run_phase native-failure-matrix \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
        --bin qa-solvers-iterative-cuda-failure-matrix

if [ -f "$artifact" ]; then
    (cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
fi
run_phase strict-check \
    python3 "$checker" "$artifact" "$output_dir/native-failure-matrix.log" "$phase_status"

if [ -f "$output_dir/strict-check.log" ]; then
    cp "$output_dir/strict-check.log" "$output_dir/c1_checker_summary.json"
fi

if [ "$overall" -ne 0 ]; then
    echo "C1 completed with one or more failed phases; no passing bundle was emitted" >&2
    exit "$overall"
fi

manifest_files=(
    compute-check.log
    cuda-source-contract.log
    producer-check.log
    native-failure-matrix.log
    strict-check.log
    c1_checker_summary.json
    phase-status.tsv
    "$(basename "$artifact")"
    "$(basename "$artifact").sha256"
)
(cd "$output_dir" && sha256sum "${manifest_files[@]}" > checksums.sha256 && sha256sum -c checksums.sha256)

echo "SUCCESS: C1 native CUDA failure/resource evidence written to $output_dir"
