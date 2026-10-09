#!/usr/bin/env bash
# FEM1 bounded native CUDA H1/Tri3/P2 assembled action evidence.
# Host FEM assembly is explicit; this runner never relabels CPU assembly as
# device execution and requires native CUDA for the resident CSR actions.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-fem1-breadth-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-fem1-breadth-fresh/artifact}"
producer="${SONIUM_FEM1_P2_PRODUCER:-${target_dir}/release/qa-fem1-p2-cuda}"
checker="${repo_root}/crates/sonium-fem/benchmarks/check_fem1_p2_cuda.py"

test "$(uname -s)" = Linux || { echo "FEM1 P2 requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "FEM1 P2 requires x86_64" >&2; exit 2; }
test -f "$checker" || { echo "FEM1 P2 strict checker is missing" >&2; exit 2; }
mkdir -p "$output_dir" "$output_dir/.tmp" "$output_dir/.runtime"
rm -f "$output_dir/checksums.sha256"
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

run_phase() {
    local name="$1"
    shift
    local log="$output_dir/${name}.log"
    local code=0
    echo "== FEM1 P2 phase: $name =="
    set +e
    "$@" 2>&1 | tee "$log"
    code="${PIPESTATUS[0]}"
    set -e
    printf '%s\t%s\n' "$name" "$code" >> "$status_file"
    if [ "$code" -ne 0 ]; then
        echo "FEM1 P2 phase '$name' failed" >&2
        return "$code"
    fi
}

status_file="$output_dir/phase-status.tsv"
: > "$status_file"
run_phase admission_tests \
    cargo test --release -p sonium-fem --test fem_cuda_admission -- --nocapture
run_phase build_p2_producer \
    cargo build --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-fem1-p2-cuda
test -x "$producer" || { echo "FEM1 P2 producer missing: $producer" >&2; exit 2; }
run_phase native_p2_cuda \
    "$producer"

artifact="$output_dir/fem1_p2_cuda.json"
test -f "$artifact" || { echo "FEM1 P2 artifact missing: $artifact" >&2; exit 1; }
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
run_phase strict_checker \
    python3 "$checker" "$artifact"

find "$output_dir" -maxdepth 1 -type f ! -name checksums.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output_dir/checksums.sha256"
(cd "$output_dir" && sha256sum -c checksums.sha256 >/dev/null)
echo "SUCCESS: FEM1 P2 native CUDA evidence written to $output_dir"
