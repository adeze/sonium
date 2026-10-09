#!/usr/bin/env bash
# FEM3 evidence runner: deterministic CPU physics references plus the bounded
# native-CUDA acoustic, structural, and H(curl) Maxwell action cells.  This
# runner deliberately does not turn a CPU physics result into CUDA parity:
# CFD, general EM, coupled, and device-gradient paths remain explicit gaps.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-fem3}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-fem3}"

test "$(uname -s)" = Linux || {
    echo "FEM3 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "FEM3 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
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

status_file="$output_dir/phase-status.tsv"
: > "$status_file"

run_phase() {
    local name="$1"
    shift
    local log="$output_dir/${name}.log"
    local code=0
    echo "== FEM3 phase: $name =="
    set +e
    "$@" 2>&1 | tee "$log"
    code="${PIPESTATUS[0]}"
    set -e
    printf '%s\t%s\n' "$name" "$code" >> "$status_file"
    if [ "$code" -ne 0 ]; then
        echo "FEM3 phase '$name' failed; no parity is inferred from it" >&2
        return "$code"
    fi
}

# Independent CPU references.  These are intentionally reported as CPU
# evidence; none is substituted for the required device execution below.
run_phase acoustics_cpu \
    cargo run --release -p sonium-qa --features qa-binaries --bin qa-fem-analytical-validation
run_phase structural_cpu \
    cargo run --release -p sonium-qa --features qa-binaries --bin qa-fem-elasticity-validation
run_phase cfd_mms_cpu \
    cargo run --release -p sonium-qa --features qa-binaries --bin qa-fem-ns-mms
run_phase em_cpu \
    cargo run --release -p sonium-qa --features qa-binaries --bin qa-fem-maxwell-validation
run_phase coupled_assembly_cpu \
    cargo run --release -p sonium-qa --features qa-binaries --bin qa-fem-physics-assembly-integration
run_phase gradient_taylor_cpu \
    cargo run --release -p sonium-qa --features qa-binaries --bin qa-solvers-differentiable-taylor-replay

# Required native CUDA evidence.  required_cuda_backend() makes an absent or
# unusable device an environment failure; it is never recorded as parity.
run_phase acoustics_cuda_adjoint \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-compute-fem-adjoint

# Bounded structural parity: production 3-D vector-H1/P1 tetrahedral
# ElasticityForm assembly plus resident CUDA forward/transpose/Hermitian CSR
# actions.  This is an action cell only; no CUDA quadrature or full structural
# solve is inferred from it.
run_phase structural_cuda_action \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-compute-fem-structural-cuda

# Bounded Maxwell parity: production 2-D H(curl)/Tri3/P1 MaxwellForm mass
# assembly plus resident CUDA Complex64 forward/transpose/Hermitian CSR
# actions.  This is an explicit complex mass action cell; CUDA quadrature,
# general Maxwell assembly, and device gradients remain unimplemented.
run_phase maxwell_cuda_action \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-compute-fem-maxwell-cuda

structural_artifact="$output_dir/compute_fem_structural_cuda.json"
test -s "$structural_artifact" || {
    echo "FEM3 structural CUDA artifact is missing: $structural_artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$structural_artifact")" > "$(basename "$structural_artifact").sha256")
python3 "$repo_root/crates/sonium-fem/benchmarks/check_fem3_structural_cuda.py" "$structural_artifact"

maxwell_artifact="$output_dir/compute_fem_maxwell_cuda.json"
test -s "$maxwell_artifact" || {
    echo "FEM3 Maxwell CUDA artifact is missing: $maxwell_artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$maxwell_artifact")" > "$(basename "$maxwell_artifact").sha256")
python3 "$repo_root/crates/sonium-fem/benchmarks/check_fem3_maxwell_cuda.py" "$maxwell_artifact"

find "$output_dir" -maxdepth 1 -type f ! -name checksums.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output_dir/checksums.sha256"

echo "SUCCESS: FEM3 phase logs and native-CUDA evidence written to $output_dir"
