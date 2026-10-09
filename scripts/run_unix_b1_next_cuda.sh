#!/usr/bin/env bash
# Reproducible native-SM120 evidence for the bounded B1 transmission cell.
# Every build/runtime scratch file and artifact stays below the repository
# target directory; this runner never falls back to a system temporary path.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-b1-next-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-b1-next-fresh/evidence}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_dir" in /*) ;; *) output_dir="$repo_root/$output_dir" ;; esac
runtime_dir="$target_dir/tmp"

test "$(uname -s)" = Linux || {
    echo "B1 next CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B1 next CUDA evidence requires x86_64" >&2
    exit 2
}

mkdir -p "$output_dir" "$runtime_dir"
export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="$runtime_dir"
export TMP="$runtime_dir"
export TEMP="$runtime_dir"
export XDG_RUNTIME_DIR="$runtime_dir"
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

rustc -Vv > "$output_dir/rustc-vv.txt"
cargo -V > "$output_dir/cargo-version.txt"
lscpu > "$output_dir/lscpu.txt"
artifact="$output_dir/bem_cuda_pair_field_report.json"
log="$output_dir/b1-next-cuda-run.log"

cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
  --bin qa-bem-cuda-pair-field > "$log" 2>&1

(
    cd "$output_dir"
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256"
    sha256sum -c "$(basename "$artifact").sha256"
)
python3 "$repo_root/crates/sonium-qa/benchmarks/check_b1_cuda.py" "$artifact"
(
    cd "$output_dir"
    sha256sum \
      rustc-vv.txt cargo-version.txt lscpu.txt \
      "$(basename "$artifact")" "$(basename "$artifact").sha256" \
      "$(basename "$log")" > checksums.sha256
    sha256sum -c checksums.sha256
)

echo "B1 next native CUDA transmission evidence: passed; bundle=$output_dir"
