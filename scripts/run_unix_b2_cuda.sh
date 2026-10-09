#!/usr/bin/env bash
# Reproducible native-CUDA evidence for the sonium-bem B2 review unit.
# Tree construction/compression remain an explicit host-shared setup phase;
# all actions below use the resident CUDA representations.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-.target-unix-b2}"
output_dir="${SONIUM_OUTPUT_DIR:-target/sonium-unix-b2}"
export CARGO_TARGET_DIR="$target_dir"
export SONIUM_OUTPUT_DIR="$output_dir"
export SONIUM_REQUIRE_CUDA=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$output_dir"

test "$(uname -s)" = Linux || {
    echo "B2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B2 CUDA evidence requires x86_64" >&2
    exit 2
}

# Match the validated native-SM120 provenance used by the other CUDA lanes.
# Every value remains overridable for a new host and is serialized in the
# QA_RESULT log, so a direct prebuilt run cannot silently lose its identity.
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

# This lane intentionally executes the already-built release producer.  The
# focused source tests remain available through the normal Cargo test command,
# while this release evidence path avoids rebuilding the workspace under a
# shared Cargo lock.
producer="${SONIUM_B2_PRODUCER:-$repo_root/target/release/qa-bem-cuda-hmatrix-fmm}"
test -x "$producer" || {
    echo "B2 prebuilt producer is missing or not executable: $producer" >&2
    exit 2
}

"$producer" | tee "$output_dir/qa_result.log"

artifact="$output_dir/bem_cuda_hmatrix_fmm_report.json"
(cd "$output_dir" && \
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$repo_root/crates/sonium-qa/benchmarks/check_b2_cuda.py" "$artifact" "$output_dir/qa_result.log"
(cd "$output_dir" && sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" qa_result.log > checksums.sha256 && sha256sum -c checksums.sha256)
echo "SUCCESS: B2 CUDA artifact, strict check, and checksums written to $output_dir"
