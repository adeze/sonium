#!/usr/bin/env bash
# Reproducible native-SM120 C6 breadth evidence for the production BEM
# quadrature adapter.  Timing is measured locally; roofline and energy are
# intentionally not claimed without profiler/counter evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-c6-breadth-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-c6-breadth-fresh/artifact}"
checker="${repo_root}/crates/sonium-compute/benchmarks/check_c6_breadth_cuda.py"
artifact="${output_dir}/cuda_c6_breadth_bem_quadrature.json"
summary="${output_dir}/cuda_c6_breadth_bem_quadrature-check.txt"
log="${output_dir}/c6_breadth_cuda.log"
manifest="${output_dir}/c6_breadth_cuda-manifest.sha256"

test "$(uname -s)" = Linux || {
    echo "C6 breadth CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C6 breadth CUDA evidence requires x86_64" >&2
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
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-compute-c6-breadth-cuda 2>&1 | tee "$log"
python3 "$checker" "$artifact" | tee "$summary"
sha256sum "$artifact" > "${artifact}.sha256"
sha256sum "$artifact" "${artifact}.sha256" "$log" "$summary" > "$manifest"
(cd "$output_dir" && sha256sum -c "$(basename "$manifest")")
echo "SUCCESS: C6 breadth CUDA evidence written to $output_dir"
