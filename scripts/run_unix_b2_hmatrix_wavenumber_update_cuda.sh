#!/usr/bin/env bash
# Reproducible native-SM120 B2 resident production H-matrix frequency update.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="$repo_root/.target-b2-breadth-fresh"
output_dir="$target_dir/evidence"
runtime_dir="$target_dir/runtime"
export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="$runtime_dir"
export TMP="$runtime_dir"
export TEMP="$runtime_dir"
mkdir -p "$output_dir" "$runtime_dir"

test "$(uname -s)" = Linux || {
    echo "B2 H-matrix update evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B2 H-matrix update evidence requires x86_64" >&2
    exit 2
}

export SONIUM_QA_CUDA_DEVICE_NAME="${SONIUM_QA_CUDA_DEVICE_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
export SONIUM_QA_CUDA_DRIVER_VERSION="${SONIUM_QA_CUDA_DRIVER_VERSION:-610.57.04}"
export SONIUM_QA_CUDA_TOOLKIT_VERSION="${SONIUM_QA_CUDA_TOOLKIT_VERSION:-CUDA UMD 13.3}"
export SONIUM_QA_CUDA_TOPOLOGY="${SONIUM_QA_CUDA_TOPOLOGY:-GPU0 PCIe Gen5 x16}"
export SONIUM_QA_CUDA_PEAK_FLOPS="${SONIUM_QA_CUDA_PEAK_FLOPS:-native-sm120}"
export SONIUM_QA_CUDA_MEMORY_BANDWIDTH="${SONIUM_QA_CUDA_MEMORY_BANDWIDTH:-native-sm120}"
export SONIUM_QA_OS="${SONIUM_QA_OS:-linux-x86_64}"
export SONIUM_QA_TARGET_FEATURES="${SONIUM_QA_TARGET_FEATURES:-cuda}"
export SONIUM_QA_RUNTIME_DISPATCH="${SONIUM_QA_RUNTIME_DISPATCH:-cuda-native}"
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-rayon-default}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

cargo build --release -p sonium-qa --features "qa-binaries,cuda" \
    --bin qa-bem-cuda-hmatrix-wavenumber-update
producer="$target_dir/release/qa-bem-cuda-hmatrix-wavenumber-update"
test -x "$producer" || {
    echo "B2 H-matrix update producer is missing: $producer" >&2
    exit 2
}

artifact="$output_dir/bem_cuda_hmatrix_wavenumber_update_report.json"
log="$output_dir/qa_result.log"
checker_log="$output_dir/checker.log"
"$producer" | tee "$log"
test -s "$artifact"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-qa/benchmarks/check_b2_hmatrix_wavenumber_update_cuda.py" \
    "$artifact" "$log" | tee "$checker_log"
python3 -m py_compile \
    "$repo_root/crates/sonium-qa/benchmarks/check_b2_hmatrix_wavenumber_update_cuda.py"

# The manifest is deliberately the final write.  Replay it immediately after
# generation so the published bundle cannot contain a stale QA log digest.
(cd "$output_dir" && sha256sum \
    "$(basename "$artifact")" "$(basename "$artifact").sha256" \
    "$(basename "$log")" "$(basename "$checker_log")" > checksums.sha256 \
    && sha256sum -c checksums.sha256)
echo "SUCCESS: B2 resident H-matrix wavenumber-update evidence written to $output_dir"
