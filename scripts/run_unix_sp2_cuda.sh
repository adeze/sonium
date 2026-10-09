#!/usr/bin/env bash
# Reproducible native-CUDA SP2 setup/runtime diagnostics evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-sp2}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-sp2}"

test "$(uname -s)" = Linux || { echo "SP2 CUDA evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SP2 CUDA evidence requires x86_64" >&2; exit 2; }
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
export SONIUM_QA_OS="${SONIUM_QA_OS:-linux-x86_64}"
export SONIUM_QA_TARGET_FEATURES="${SONIUM_QA_TARGET_FEATURES:-cuda}"
export SONIUM_QA_RUNTIME_DISPATCH="${SONIUM_QA_RUNTIME_DISPATCH:-cuda-native}"
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-rayon-default}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-solvers-preconditioner-cuda \
    | tee "$output_dir/sp2-native-run.log"
cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-solvers-preconditioner-failure-cuda \
    | tee "$output_dir/sp2-failure-run.log"

python3 "$repo_root/crates/sonium-qa/benchmarks/check_sp2_cuda.py" \
    "$output_dir/solvers_preconditioner_cuda.json" \
    "$output_dir/solvers_preconditioner_failure_cuda.json" \
    --output "$output_dir/sp2-diagnostics-check-v1.json"

(cd "$output_dir" && sha256sum \
    solvers_preconditioner_cuda.json \
    solvers_preconditioner_failure_cuda.json \
    sp2-diagnostics-check-v1.json \
    sp2-native-run.log \
    sp2-failure-run.log > checksums.sha256 && sha256sum -c checksums.sha256)

echo "SUCCESS: SP2 CUDA diagnostics artifacts and checksums written to $output_dir"
