#!/usr/bin/env bash
# FEM2 evidence runner: required native CUDA device-solver, constraint-transfer,
# true-residual, and warm-residency contract.  Unsupported FEM combinations
# remain explicit in the producer capability matrix and are never CPU-fallback
# results.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-$repo_root/.target-fem2-next-fresh/cargo}"
output_dir="${SONIUM_OUTPUT_DIR:-$repo_root/.target-fem2-next-fresh/artifact}"
tmp_dir="$output_dir/.tmp"
runtime_dir="$output_dir/.runtime"

test "$(uname -s)" = Linux || {
    echo "FEM2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "FEM2 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"
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
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-rayon-default}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem-device-solver \
    2>&1 | tee "$output_dir/qa_result.log"

# The producer writes its own artifact checksum for the strict checker.  A
# separate forced-fallback invocation proves required CUDA admission fails
# closed; it must not overwrite the successful artifact.
(cd "$output_dir" && sha256sum fem_device_solver.json > fem_device_solver.json.sha256)
set +e
SONIUM_CUDA_FORCE_FALLBACK=1 \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem-device-solver \
    > "$output_dir/fallback_probe.log" 2>&1
fallback_status=$?
set -e
test "$fallback_status" -ne 0 || {
    echo "FEM2 fallback probe unexpectedly succeeded" >&2
    exit 1
}
grep -q "CUDA fallback fault injection is active" "$output_dir/fallback_probe.log"
grep -q "FAILED" "$output_dir/fallback_probe.log"

python3 "$repo_root/crates/sonium-qa/benchmarks/check_fem2_cuda.py" \
    "$output_dir/fem_device_solver.json" "$output_dir/fallback_probe.log" \
    > "$output_dir/fem2_checker_summary.json"

(cd "$output_dir" && \
    sha256sum fem_device_solver.json fem_device_solver.json.sha256 qa_result.log \
        fallback_probe.log fem2_checker_summary.json > checksums.sha256 && \
    sha256sum -c checksums.sha256)

echo "SUCCESS: FEM2 CUDA evidence written to $output_dir"
