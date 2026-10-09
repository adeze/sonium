#!/usr/bin/env bash
# Reproducible native-SM120 evidence for the bounded B1 Quad4/real breadth cell.
# Cargo state, runtime scratch, and every evidence file stay below this repo.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="$repo_root/.target-b1-quad-real-fresh"
output_dir="$target_dir/evidence"
runtime_dir="$target_dir/runtime"

test "$(uname -s)" = Linux || {
    echo "B1 Quad4 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B1 Quad4 CUDA evidence requires x86_64" >&2
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
artifact="$output_dir/bem_cuda_quad4_real_report.json"
log="$output_dir/b1-quad4-real-cuda-run.log"

cargo build --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-bem-cuda-quad4-real
producer="$target_dir/release/qa-bem-cuda-quad4-real"
test -x "$producer"
"$producer" > "$log" 2>&1
test -s "$artifact"

(
    cd "$output_dir"
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256"
    sha256sum -c "$(basename "$artifact").sha256"
)
python3 "$repo_root/crates/sonium-qa/benchmarks/check_b1_quad4_real_cuda.py" "$artifact"
python3 -m py_compile "$repo_root/crates/sonium-qa/benchmarks/check_b1_quad4_real_cuda.py"

# Write the final manifest only after the producer and checker have stopped
# changing evidence, then replay it immediately.
(
    cd "$output_dir"
    sha256sum \
      rustc-vv.txt cargo-version.txt lscpu.txt \
      "$(basename "$artifact")" "$(basename "$artifact").sha256" \
      "$(basename "$log")" > checksums.sha256
    sha256sum -c checksums.sha256
)

echo "B1 Quad4/real native CUDA evidence passed: $output_dir"
