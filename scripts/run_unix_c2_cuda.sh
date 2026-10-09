#!/usr/bin/env bash
# Focused C2 CUDA operation-portfolio evidence runner.
# The operation matrix is the producer of the machine-readable contract
# result; the parity cases below cover native-only portfolio seams that are
# intentionally outside the generic Backend matrix (BEM/FMM and Krylov).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-c2-views}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-c2}"

test "$(uname -s)" = Linux || {
    echo "C2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C2 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_BACKEND=cuda
export SONIUM_REQUIRE_BACKEND=cuda
export SONIUM_REQUIRE_CUDA=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Keep the strict specialized checker tied to a concrete native-SM120
# provenance record.  Every value is overridable for a rerun on another
# NVIDIA host; a missing value must not silently become an "unknown" pass.
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

# The checked operation matrix covers f32/f64/Complex32/Complex64, padded
# dense layouts, casts, gather, reductions, CSR SpMV/SpMM/transpose, local
# contractions, strided views, malformed inputs, overflow, and reference
# comparisons.  Keep its output as the C2 evidence record.
cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-compute-operation-matrix \
    | tee "$output_dir/compute-operation-matrix.log"

# The ABI-specific BEM/FMM producer publishes per-cell residency counters for
# the specialised descriptor paths exercised below.
cargo run --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-compute-specialized-cuda-matrix \
    | tee "$output_dir/compute-specialized-cuda-matrix.log"

# Check the specialized producer before running the longer focused parity
# suite.  The checker requires a content sidecar so a copied/stale artifact
# cannot masquerade as the result of this invocation.
checker="$repo_root/crates/sonium-qa/benchmarks/check_c2_cuda.py"
test -f "$checker" || {
    echo "C2 specialized strict checker is missing: $checker" >&2
    exit 2
}
(cd "$output_dir" && \
    sha256sum c2-specialized-residency.json > c2-specialized-residency.json.sha256)
python3 "$checker" \
    "$output_dir/c2-specialized-residency.json" \
    "$output_dir/compute-specialized-cuda-matrix.log"

# Native-only seams not represented by the generic operation matrix.
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_spmm_prebaked_colmajor_matches_repeated_spmv -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_csr_transpose_matches_cpu_reference -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    complex::cuda_device_precision_casts_roundtrip -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    complex::cuda_gather4_weighted_complex32_matches_cpu -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    complex::cuda_gather4_weighted_complex64_matches_cpu -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_scalar_gmres_workspace_has_constant_steady_state_counters -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_block_gmres_workspace_has_constant_steady_state_counters -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    bem_regular_pair_parity::cuda_bem_regular_pair_c64_matches_cpu_and_records_device_work -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    bem_regular_pair_parity::cuda_bem_regular_pair_c64_strided_views_match_cpu_without_staging -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_fmm_residency::complex64_fmm_primitives_stay_on_device_between_upload_and_readback -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_fmm_residency::complex64_fmm_strided_views_match_cpu_and_keep_residency -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_fmm_residency::complex64_mlfmm_resident_forward_transpose_and_hermitian_parity -- \
    --exact --nocapture --test-threads=1
cargo test --release -p sonium-compute --features cuda --test parity \
    cuda_fmm_residency::complex64_hmatrix_forward_transpose_and_hermitian_reuse_device_blocks -- \
    --exact --nocapture --test-threads=1

# Keep the generic matrix's machine-readable QA_RESULT log reproducible and
# bind the focused specialized parity suite to the same evidence bundle.
sha256sum "$output_dir/compute-operation-matrix.log" \
    "$output_dir/compute-specialized-cuda-matrix.log" \
    "$output_dir/c2-specialized-residency.json" \
    "$output_dir/c2-specialized-residency.json.sha256" \
    > "$output_dir/checksums.sha256"
