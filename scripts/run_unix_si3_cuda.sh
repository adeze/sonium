#!/usr/bin/env bash
# Reproducible native-CUDA SI3 iterative-throughput evidence.
# The producer requires CUDA itself; the checker rejects missing provenance,
# unsupported capability cells, failed residual/parity gates, or zero device
# work before a checksum bundle is published.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-si3}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-si3}"

test "$(uname -s)" = Linux || {
    echo "SI3 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SI3 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# These defaults identify the validated single-SM120 lane.  A different
# validated runner may override them, and the override is preserved in the
# artifact instead of being silently inferred.
export SONIUM_QA_CUDA_DEVICE_NAME="${SONIUM_QA_CUDA_DEVICE_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
export SONIUM_QA_CUDA_DRIVER_VERSION="${SONIUM_QA_CUDA_DRIVER_VERSION:-610.57.04}"
export SONIUM_QA_CUDA_TOOLKIT_VERSION="${SONIUM_QA_CUDA_TOOLKIT_VERSION:-CUDA UMD 13.3}"
export SONIUM_QA_CUDA_TOPOLOGY="${SONIUM_QA_CUDA_TOPOLOGY:-GPU0 PCIe Gen5 x16}"
export SONIUM_QA_OS="${SONIUM_QA_OS:-linux-x86_64}"
export SONIUM_QA_TARGET_FEATURES="${SONIUM_QA_TARGET_FEATURES:-cuda}"
export SONIUM_QA_RUNTIME_DISPATCH="${SONIUM_QA_RUNTIME_DISPATCH:-cuda-native}"
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-rayon-default}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-solvers-iterative-cuda-throughput \
    | tee "$output_dir/si3-run.log"

python3 "$repo_root/crates/sonium-qa/benchmarks/check_si3_cuda.py" \
    "$output_dir/solvers_iterative_cuda_throughput.json" \
    --output "$output_dir/si3-throughput-check-v1.json"

(cd "$output_dir" && sha256sum \
    solvers_iterative_cuda_throughput.json \
    si3-throughput-check-v1.json \
    si3-run.log > checksums.sha256 && sha256sum -c checksums.sha256)

echo "SUCCESS: SI3 CUDA throughput artifacts and checksums written to $output_dir"
