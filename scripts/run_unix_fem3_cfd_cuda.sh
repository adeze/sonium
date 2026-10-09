#!/usr/bin/env bash
# FEM3 bounded native-CUDA Taylor--Hood Stokes action evidence.
# Production mixed assembly and quadrature remain host-owned; this runner
# requires native CUDA only for the resident monolithic CSR actions.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-fem3-cfd-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-fem3-cfd-fresh/artifact}"
producer="${SONIUM_FEM3_STOKES_PRODUCER:-${target_dir}/release/qa-fem3-stokes-cuda}"
checker="${repo_root}/crates/sonium-fem/benchmarks/check_fem3_stokes_cuda.py"

test "$(uname -s)" = Linux || { echo "FEM3 Stokes requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "FEM3 Stokes requires x86_64" >&2; exit 2; }
test -f "$checker" || { echo "FEM3 Stokes strict checker is missing" >&2; exit 2; }
mkdir -p "$output_dir" "$output_dir/.tmp" "$output_dir/.runtime"
export TMPDIR="$output_dir/.tmp"
export TMP="$output_dir/.tmp"
export TEMP="$output_dir/.tmp"
export XDG_RUNTIME_DIR="$output_dir/.runtime"
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

status_file="$output_dir/phase-status.tsv"
: > "$status_file"

run_phase() {
    local name="$1"
    shift
    local log="$output_dir/${name}.log"
    local code=0
    echo "== FEM3 Stokes phase: $name =="
    set +e
    "$@" 2>&1 | tee "$log"
    code="${PIPESTATUS[0]}"
    set -e
    printf '%s\t%s\n' "$name" "$code" >> "$status_file"
    if [ "$code" -ne 0 ]; then
        echo "FEM3 Stokes phase '$name' failed" >&2
        return "$code"
    fi
}

run_phase focused_fem_tests \
    cargo test --release -p sonium-fem --lib backend -- --nocapture
run_phase build_stokes_producer \
    cargo build --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem3-stokes-cuda
test -x "$producer" || { echo "FEM3 Stokes producer missing: $producer" >&2; exit 2; }
run_phase native_stokes_cuda \
    "$producer"

artifact="$output_dir/compute_fem_stokes_cuda.json"
test -s "$artifact" || { echo "FEM3 Stokes artifact missing: $artifact" >&2; exit 1; }
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
run_phase strict_checker \
    python3 "$checker" "$artifact"

: > "$output_dir/checksums.sha256"
find "$output_dir" -maxdepth 1 -type f ! -name checksums.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output_dir/checksums.sha256"
(cd "$output_dir" && sha256sum -c checksums.sha256 >/dev/null)
echo "SUCCESS: FEM3 Stokes native CUDA evidence written to $output_dir"
