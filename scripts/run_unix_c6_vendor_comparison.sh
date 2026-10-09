#!/usr/bin/env bash
# Reproducible native-CUDA C6 shape-matched vendor comparison.
#
# This lane compares the production fused multi-dot CUDA kernel with repeated
# cuBLAS GEMV reductions over the same real-f64 basis/vector shape. It is
# intentionally separate from the broader C6 BEM/FEM runner so its timing and
# transfer contract cannot be confused with the accounting-only release rows.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-c6-next}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-c6-vendor-comparison}"
checker="$repo_root/crates/sonium-compute/benchmarks/check_c6_vendor_comparison.py"

test "$(uname -s)" = Linux || {
    echo "C6 CUDA vendor comparison requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C6 CUDA vendor comparison requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SONIUM_QA_CUDA_DEVICE_NAME="${SONIUM_QA_CUDA_DEVICE_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
export SONIUM_QA_CUDA_DRIVER_VERSION="${SONIUM_QA_CUDA_DRIVER_VERSION:-610.57.04}"
export SONIUM_QA_CUDA_TOOLKIT_VERSION="${SONIUM_QA_CUDA_TOOLKIT_VERSION:-CUDA UMD 13.3}"
export SONIUM_QA_CUDA_TOPOLOGY="${SONIUM_QA_CUDA_TOPOLOGY:-GPU0 PCIe Gen5 x16}"
export SONIUM_QA_CUDA_PEAK_FLOPS="${SONIUM_QA_CUDA_PEAK_FLOPS:-125000000000000}"
export SONIUM_QA_CUDA_MEMORY_BANDWIDTH="${SONIUM_QA_CUDA_MEMORY_BANDWIDTH:-1792000000000}"
export SONIUM_QA_OS="${SONIUM_QA_OS:-linux-x86_64}"
export SONIUM_QA_TARGET_FEATURES="${SONIUM_QA_TARGET_FEATURES:-cuda}"
export SONIUM_QA_RUNTIME_DISPATCH="${SONIUM_QA_RUNTIME_DISPATCH:-cuda-native}"
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-single-cuda-context}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-compute-cuda-vendor-comparison
python3 "$checker" "$output_dir/cuda_vendor_comparison.json" \
    | tee "$output_dir/c6_vendor_comparison_checker_summary.json"
sha256sum \
    "$output_dir/cuda_vendor_comparison.json" \
    "$output_dir/c6_vendor_comparison_checker_summary.json" \
    > "$output_dir/checksums.sha256"
echo "SUCCESS: C6 CUDA vendor-comparison artifact and checksums written to $output_dir"
