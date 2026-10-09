#!/usr/bin/env bash
# Native CUDA PF1 breadth evidence: accepted production BDF state/Jacobian
# replay with resident CSR actions and fail-closed admission checks.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-pf1-breadth-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-pf1-breadth-fresh/evidence}"
producer="${SONIUM_PF1_BREADTH_PRODUCER:-${target_dir}/release/qa-pcfan-bdf-cuda-breadth}"
checker="${repo_root}/crates/sonium-sim-pcfan/benchmarks/check_pf1_breadth_cuda.py"

test "$(uname -s)" = Linux || { echo "PF1 breadth requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PF1 breadth requires x86_64" >&2; exit 2; }
test -x "$producer" || {
    echo "PF1 breadth producer is missing or not executable: $producer" >&2
    echo "Build with CARGO_TARGET_DIR=.target-pf1-breadth-fresh cargo build --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-pcfan-bdf-cuda-breadth" >&2
    exit 2
}
test -f "$checker" || { echo "PF1 breadth checker is missing: $checker" >&2; exit 2; }

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

log="$output_dir/qa_result.log"
"$producer" 2>&1 | tee "$log"
artifact="$output_dir/pcfan_bdf_cuda_breadth.json"
test -f "$artifact" || { echo "PF1 breadth producer did not publish $artifact" >&2; exit 1; }
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$checker" "$artifact"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" qa_result.log > checksums.sha256 && sha256sum -c checksums.sha256)
echo "SUCCESS: PF1 accepted-BDF CUDA breadth artifact, strict check, and checksums written to $output_dir"
