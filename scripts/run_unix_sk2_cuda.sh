#!/usr/bin/env bash
# SK2 evidence runner: required native CUDA whole-speaker, typed-driver, and
# failure-contract producers. Every build, temporary, runtime, log, and
# artifact path is repo-local; missing hardware or fallback is a hard failure.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-sk2}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-sk2}"
whole="$output_dir/whole"
typed="$output_dir/typed"
failure="$output_dir/failure"
tmp_dir="$output_dir/.tmp"
runtime_dir="$output_dir/.runtime"
checker="$repo_root/crates/sonium-qa/benchmarks/check_speaker_cuda_sk2.py"

test "$(uname -s)" = Linux || {
    echo "SK2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SK2 CUDA evidence requires native x86_64" >&2
    exit 2
}
test -f "$checker" || {
    echo "SK2 strict checker is missing: $checker" >&2
    exit 2
}
mkdir -p "$whole" "$typed" "$failure" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"

export CARGO_TARGET_DIR="$target_dir"
export SONIUM_REQUIRE_CUDA=1
export SONIUM_REQUIRE_BACKEND=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"
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

run_producer() {
    local name="$1"
    local out="$2"
    local binary="$3"
    echo "== SK2 producer: $name =="
    SONIUM_OUTPUT_DIR="$out" \
        cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin "$binary" \
        2>&1 | tee "$out/qa_result.log"
}

run_producer whole "$whole" qa-speaker-cuda-workload
run_producer typed "$typed" qa-speaker-typed-cuda-workload
run_producer failure "$failure" qa-speaker-cuda-failure-contract

# Required CUDA must fail closed before writing a product artifact when the
# fallback hook is armed. Keep the probe in its own directory so it cannot
# overwrite the successful whole-speaker evidence.
fallback_dir="$whole/fallback"
mkdir -p "$fallback_dir"
set +e
SONIUM_OUTPUT_DIR="$fallback_dir" SONIUM_CUDA_FORCE_FALLBACK=1 \
    cargo run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-speaker-cuda-workload \
    > "$whole/fallback_probe.log" 2>&1
fallback_status=$?
set -e
test "$fallback_status" -ne 0 || {
    echo "SK2 required-CUDA fallback probe unexpectedly succeeded" >&2
    exit 1
}
grep -q "CUDA fallback fault injection is active" "$whole/fallback_probe.log"
grep -q "FAILED" "$whole/fallback_probe.log"
test -z "$(find "$fallback_dir" -maxdepth 1 -type f -name '*.json' -print -quit)"

for directory in "$whole" "$typed" "$failure"; do
    (cd "$directory" && sha256sum -- *.json *.log > checksums.sha256 && sha256sum -c checksums.sha256 >/dev/null)
done
python3 "$checker" "$output_dir" > "$output_dir/sk2_checker_summary.json"
(cd "$output_dir" && sha256sum sk2_checker_summary.json > sk2_checker_summary.json.sha256 && sha256sum -c sk2_checker_summary.json.sha256 >/dev/null)

echo "SUCCESS: SK2 CUDA evidence written to $output_dir"
