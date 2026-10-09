#!/usr/bin/env bash
# FEM1 evidence runner: CPU element/geometry contracts plus the supported
# native-CUDA H1/Tri3/P1 assembled CSR action.  Unsupported FEM cells are
# reported by the producer's capability matrix; this runner never substitutes
# a CPU result for required CUDA execution.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-fem1}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-fem1}"

test "$(uname -s)" = Linux || {
    echo "FEM1 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "FEM1 CUDA evidence requires x86_64" >&2
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
export SONIUM_QA_THREAD_POLICY="${SONIUM_QA_THREAD_POLICY:-rayon-default}"
export SONIUM_QA_AFFINITY_POLICY="${SONIUM_QA_AFFINITY_POLICY:-runner-default}"
export SONIUM_QA_NUMA_POLICY="${SONIUM_QA_NUMA_POLICY:-runner-default}"
export SONIUM_QA_DETERMINISTIC_MODE="${SONIUM_QA_DETERMINISTIC_MODE:-enabled}"

# Independent CPU element/geometry contracts cover curved geometry and
# malformed topology before the required CUDA producer below.
cargo test --release -p sonium-fem --test curved_geometry_contract -- --nocapture
cargo test --release -p sonium-fem --test checked_cross_space_assembly -- --nocapture
cargo test --release -p sonium-fem --test h1_curved_p2_transfer -- --nocapture

# The supported accelerator cell is production Helmholtz assembly uploaded to
# native CUDA CSR, with forward/Hermitian actions and f64/Complex64 parity and
# residency counters. Execute the already-built release producer so the
# evidence command is stable and cannot be confused with a partial rebuild.
producer="${SONIUM_FEM1_PRODUCER:-$repo_root/.target-unix-fem1/release/qa-compute-fem-adjoint}"
test -x "$producer" || {
    echo "FEM1 release producer is missing or not executable: $producer" >&2
    exit 2
}
"$producer"

artifact="$output_dir/compute_fem_cuda_adjoint.json"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-fem/benchmarks/check_fem1_cuda.py" "$artifact"

python3 "$repo_root/crates/sonium-compute/benchmarks/summarize_cuda_evidence.py" \
    "$artifact" \
    | tee "$output_dir/compute_fem_cuda_adjoint_summary.json"
sha256sum \
    "$artifact" \
    "$artifact.sha256" \
    "$output_dir/compute_fem_cuda_adjoint_summary.json" \
    > "$output_dir/checksums.sha256"

echo "SUCCESS: FEM1 CPU/CUDA evidence written to $output_dir"
