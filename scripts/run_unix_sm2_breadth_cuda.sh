#!/usr/bin/env bash
# Focused SM2 breadth evidence for the production axisymmetric response replay.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-sm2-breadth-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${target_dir}/artifact}"
checker="$repo_root/crates/sonium-sim-speaker-modeling/benchmarks/check_sm2_breadth_cuda.py"

test "$(uname -s)" = Linux || {
    echo "SM2 breadth CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SM2 breadth CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$target_dir" "$output_dir/.tmp" "$output_dir/.runtime"
export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export TMPDIR="$output_dir/.tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_RUNTIME_DIR="$output_dir/.runtime"
export SONIUM_BACKEND=cuda
export SONIUM_REQUIRE_BACKEND=cuda
export SONIUM_REQUIRE_CUDA=1
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

artifact="$output_dir/speaker_modeling_sm2_breadth.json"
log="$output_dir/sm2_breadth_cuda.log"
check="$output_dir/sm2_breadth_cuda-check.txt"
manifest="$output_dir/sm2_breadth_cuda-manifest.sha256"
rm -f "$artifact" "$artifact.sha256" "$log" "$check" "$manifest"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-speaker-modeling-sm2-breadth-cuda >"$log" 2>&1
test -s "$artifact" || {
    echo "SM2 breadth producer did not publish $artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$artifact")" >"$(basename "$artifact").sha256")
python3 "$checker" "$artifact" "$log" >"$check"
(cd "$output_dir" && {
    sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" \
        "$(basename "$log")" "$(basename "$check")" >"$(basename "$manifest")"
    sha256sum -c "$(basename "$manifest")"
})
echo "SM2 breadth CUDA evidence passed: $output_dir"
