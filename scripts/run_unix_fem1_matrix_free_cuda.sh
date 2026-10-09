#!/usr/bin/env bash
# FEM1 bounded native CUDA matrix-free P1/Tri3 action evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-fem1-next-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${target_dir}/evidence}"
artifact="${output_dir}/fem1_matrix_free_cuda.json"

test "$(uname -s)" = Linux || { echo "FEM1 requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "FEM1 requires x86_64" >&2; exit 2; }
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export SONIUM_CUDA_DEVICE_ORDINAL="${SONIUM_CUDA_DEVICE_ORDINAL:-0}"
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

cargo check -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem1-matrix-free-cuda
cargo build --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem1-matrix-free-cuda
producer="$target_dir/release/qa-fem1-matrix-free-cuda"
test -x "$producer"
"$producer"

(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-fem/benchmarks/check_fem1_matrix_free_cuda.py" "$artifact"
(cd "$output_dir" && sha256sum -c "$(basename "$artifact").sha256")
sha256sum "$artifact" "$artifact.sha256" > "$output_dir/checksums.sha256"
echo "SUCCESS: FEM1 matrix-free CUDA evidence written to $output_dir"
