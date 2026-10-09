#!/usr/bin/env bash
# Bounded native-NVIDIA B3 producer.  This delegates formulation and physical
# gates to the existing QA producer; no result is synthesized by this wrapper.
set -euo pipefail

target_dir="${CARGO_TARGET_DIR:-.target-unix-b3}"
output_dir="${SONIUM_OUTPUT_DIR:-target/sonium-unix-b3}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

test "$(uname -s)" = Linux || {
    echo "B3 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B3 CUDA evidence requires x86_64" >&2
    exit 2
}

# These are the validated native-SM120 host values used by the other CUDA
# evidence lanes.  Callers may override each field for a rerun on another
# host; the producer records the resulting values in its artifact.  Keeping
# the fields mandatory prevents a raw CUDA pass from losing its hardware
# identity before it reaches the checker.
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

producer_root="$target_dir"
case "$producer_root" in
    /*) ;;
    *) producer_root="$repo_root/${producer_root#./}" ;;
esac
producer="${SONIUM_B3_PRODUCER:-$producer_root/debug/qa-bem-cuda-matrix}"
test -x "$producer" || {
    echo "B3 prebuilt producer is missing or not executable: $producer" >&2
    exit 2
}
"$producer"

artifact="$output_dir/bem_cuda_matrix_report.json"
(cd "$output_dir" && \
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-qa/benchmarks/check_b3_cuda.py" "$artifact"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" > checksums.sha256 && sha256sum -c checksums.sha256)
echo "SUCCESS: B3 CUDA artifact, strict check, and checksums written to $output_dir"
