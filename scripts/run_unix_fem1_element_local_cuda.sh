#!/usr/bin/env bash
# FEM1 bounded native CUDA local-element parity evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="$repo_root/.target-unix-fem1-next"
output_dir="$repo_root/target/sonium-unix-fem1-next"
artifact="$output_dir/fem1_element_local_cuda.json"

test "$(uname -s)" = Linux || { echo "FEM1 requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "FEM1 requires x86_64" >&2; exit 2; }
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_CUDA_DEVICE_ORDINAL="${SONIUM_CUDA_DEVICE_ORDINAL:-0}"
export SONIUM_QA_CUDA_DEVICE_NAME="${SONIUM_QA_CUDA_DEVICE_NAME:-NVIDIA RTX PRO 6000 Blackwell Workstation Edition}"
export SONIUM_QA_CUDA_DRIVER_VERSION="${SONIUM_QA_CUDA_DRIVER_VERSION:-610.57.04}"
export SONIUM_QA_CUDA_TOOLKIT_VERSION="${SONIUM_QA_CUDA_TOOLKIT_VERSION:-CUDA UMD 13.3}"
export SONIUM_QA_CUDA_TOPOLOGY="${SONIUM_QA_CUDA_TOPOLOGY:-GPU0 PCIe Gen5 x16}"
export SONIUM_QA_OS="${SONIUM_QA_OS:-linux-x86_64}"
export SONIUM_QA_TARGET_FEATURES="${SONIUM_QA_TARGET_FEATURES:-cuda}"
export SONIUM_QA_RUNTIME_DISPATCH="${SONIUM_QA_RUNTIME_DISPATCH:-cuda-native}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cargo check -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem1-element-local-cuda
cargo build --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem1-element-local-cuda
producer="$target_dir/release/qa-fem1-element-local-cuda"
test -x "$producer"
"$producer"

python3 "$repo_root/crates/sonium-fem/benchmarks/check_fem1_element_local_cuda.py" "$artifact"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > checksums.sha256 && sha256sum -c checksums.sha256)
echo "SUCCESS: FEM1 element-local CUDA parity evidence written to $output_dir"
