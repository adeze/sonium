#!/usr/bin/env bash
# Focused C2 breadth evidence for the resident Complex32 BEM pair ABI.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-c2-breadth-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${target_dir}/evidence}"

test "$(uname -s)" = Linux || {
    echo "C2 breadth CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C2 breadth CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_BACKEND=cuda
export SONIUM_REQUIRE_BACKEND=cuda
export SONIUM_REQUIRE_CUDA=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# These are release-lane provenance values, overridable for a rerun on an
# equivalent NVIDIA host.  The checker rejects missing/placeholder values.
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

log="$output_dir/c2-c32-resident.log"
cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-compute-c2-breadth >"$log" 2>&1

artifact="$output_dir/c2-c32-resident.json"
test -s "$artifact" || {
    echo "C2 breadth producer did not write $artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$artifact")" >"$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-qa/benchmarks/check_c2_breadth_cuda.py" \
    "$artifact" "$log" >"$output_dir/c2-c32-resident-check.txt"

# Keep a content-addressed release record for every output of this focused
# invocation.  The checker above remains the semantic gate; this manifest
# catches stale logs/check output and is itself covered by its sidecar.
(cd "$output_dir" && {
    sha256sum c2-c32-resident.json c2-c32-resident.json.sha256 \
        c2-c32-resident.log c2-c32-resident-check.txt >c2-c32-resident-manifest.sha256
    sha256sum -c c2-c32-resident-manifest.sha256
})

echo "C2 breadth CUDA evidence passed: $output_dir"
