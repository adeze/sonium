#!/usr/bin/env bash
# Reproducible native-CUDA C6 workload evidence.  The two production QA
# binaries emit versioned raw BEM/FEM forward+adjoint artifacts; the checked
# summarizer then rejects missing provenance, thresholds, counters, or device
# baselines before a summary is written.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-c6}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-c6}"
release_artifact="$repo_root/crates/sonium-compute/benchmarks/artifacts/cuda-c6-release-v1.json"
release_checker="$repo_root/crates/sonium-compute/benchmarks/check_c6_release.py"

test "$(uname -s)" = Linux || {
    echo "C6 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C6 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# These are the measured/provisioned values used by the existing live
# RTX PRO 6000 Blackwell evidence.  Callers may override them when rerunning
# the same producer on a different validated CUDA host; the raw artifact then
# records the override instead of silently inheriting this host's identity.
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

# Both binaries require a native CUDA context themselves.  A missing device,
# an execution error, or an artifact-write error exits non-zero here.
cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-bem-cuda-matrix
cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-compute-fem-adjoint

python3 "$repo_root/crates/sonium-compute/benchmarks/summarize_cuda_evidence.py" \
    "$output_dir/bem_cuda_matrix_report.json" \
    | tee "$output_dir/bem_cuda_matrix_summary.json"
python3 "$repo_root/crates/sonium-compute/benchmarks/summarize_cuda_evidence.py" \
    "$output_dir/compute_fem_cuda_adjoint.json" \
    | tee "$output_dir/compute_fem_cuda_adjoint_summary.json"
python3 "$release_checker" \
    "$release_artifact" \
    "$output_dir/bem_cuda_matrix_report.json" \
    "$output_dir/compute_fem_cuda_adjoint.json" \
    | tee "$output_dir/c6_release_checker_summary.json"

sha256sum \
    "$output_dir/bem_cuda_matrix_report.json" \
    "$output_dir/bem_cuda_matrix_summary.json" \
    "$output_dir/compute_fem_cuda_adjoint.json" \
    "$output_dir/compute_fem_cuda_adjoint_summary.json" \
    "$output_dir/c6_release_checker_summary.json" \
    > "$output_dir/checksums.sha256"

echo "SUCCESS: C6 CUDA raw artifacts, summaries, release check, and checksums written to $output_dir"
