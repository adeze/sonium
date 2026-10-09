#!/usr/bin/env bash
# Reproducible native-CUDA evidence for the sonium-bem B1 review unit.
# Keep all build output and the producer artifact in the repository workspace.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-.target-unix-b1}"
output_dir="${SONIUM_OUTPUT_DIR:-target/sonium-unix-b1}"
export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$output_dir"

test "$(uname -s)" = Linux || {
    echo "B1 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B1 CUDA evidence requires x86_64" >&2
    exit 2
}

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

cargo test -p sonium-compute --features cuda --test parity \
  cuda_bem_regular_pair_c64_matches_cpu_and_records_device_work \
  -- --nocapture --test-threads=1
cargo test -p sonium-compute --features cuda --test parity \
  cuda_field_residency::field_launch_failure_preserves_output_and_releases_workspace \
  -- --exact --nocapture --test-threads=1
cargo test -p sonium-compute --features cuda --test parity \
  cuda_quadrature_residency::complex64_quadrature_pair_operator_covers_bulk_singular_and_edge_channels \
  -- --exact --nocapture --test-threads=1
cargo test -p sonium-compute --features cuda --test parity \
  cuda_quadrature_residency::regular_pair_launch_failure_preserves_output_and_releases_workspace \
  -- --exact --nocapture --test-threads=1
cargo test -p sonium-bem --features cuda --lib \
  checked_cuda_field_matches_cpu_reference \
  -- --nocapture --test-threads=1
cargo test -p sonium-bem --features cuda --lib \
  checked_cuda_field_rejects_quad4_and_on_surface_cells_explicitly \
  -- --nocapture --test-threads=1
cargo test -p sonium-bem --features cuda --lib \
  cuda_tbem_launch_failure_is_typed_and_releases_transient_buffers \
  -- --nocapture --test-threads=1

SONIUM_OUTPUT_DIR="$output_dir" \
  cargo run -p sonium-qa --features 'qa-binaries cuda' --bin qa-bem-cuda-matrix

# The dedicated B1 producer records the scalar, pair-channel, quadrature,
# field-residency, admission, and transactional-failure rows.  It is separate
# from the B3 physical producer so adding B1 coverage cannot weaken or relabel
# the formulation-wide physical matrix.
cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-bem-cuda-pair-field

artifact="$output_dir/bem_cuda_pair_field_report.json"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-qa/benchmarks/check_b1_cuda.py" "$artifact"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" > checksums.sha256 && sha256sum -c checksums.sha256)
echo "SUCCESS: B1 CUDA pair/field artifact, strict check, and checksums written to $output_dir"
