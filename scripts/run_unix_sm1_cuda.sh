#!/usr/bin/env bash
# SM1 evidence runner: CPU contract checks plus bounded native-CUDA typed
# speaker-driver and production axisymmetric frequency-step paths. High-level
# FEM assembly remains host-owned; CPU output is never relabelled as CUDA.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-sm1-breadth-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-sm1-breadth-fresh/artifact}"
producer="${SONIUM_SM1_PRODUCER:-${target_dir}/release/qa-speaker-driver-cuda-reference}"
failure_producer="${SONIUM_SM1_FAILURE_PRODUCER:-${target_dir}/release/qa-speaker-modeling-cuda-failure-contract}"
production_producer="${SONIUM_SM1_PRODUCTION_PRODUCER:-${target_dir}/release/qa-speaker-modeling-cuda-production-frequency}"
checker="${repo_root}/crates/sonium-sim-speaker-modeling/benchmarks/check_sm1_cuda.py"
failure_checker="${repo_root}/crates/sonium-sim-speaker-modeling/benchmarks/check_sm1_cuda_failure.py"
production_checker="${repo_root}/crates/sonium-sim-speaker-modeling/benchmarks/check_sm1_cuda_production_frequency.py"

test "$(uname -s)" = Linux || {
    echo "SM1 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SM1 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir"
mkdir -p "$output_dir/.tmp" "$output_dir/.runtime"
# Remove only stale outputs owned by this runner; the dedicated fresh bundle
# must not retain a partial log from an interrupted broad test attempt.
rm -f "$output_dir/modeling_cpu.log" "$output_dir/checksums.sha256"
export TMPDIR="$output_dir/.tmp"
export TMP="$output_dir/.tmp"
export TEMP="$output_dir/.tmp"
export XDG_RUNTIME_DIR="$output_dir/.runtime"
test -f "$checker" && test -f "$failure_checker" && test -f "$production_checker" || {
    echo "SM1 strict checker is missing: $checker" >&2
    exit 2
}

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

run_phase() {
    local name="$1"
    shift
    local log="$output_dir/${name}.log"
    local code=0
    echo "== SM1 phase: $name =="
    set +e
    "$@" 2>&1 | tee "$log"
    code="${PIPESTATUS[0]}"
    set -e
    printf '%s\t%s\n' "$name" "$code" >> "$status_file"
    if [ "$code" -ne 0 ]; then
        echo "SM1 phase '$name' failed; no CUDA parity is inferred from it" >&2
        return "$code"
    fi
}

status_file="$output_dir/phase-status.tsv"
: > "$status_file"

# Build both production producers in the declared repo-local target tree.
run_phase build_cuda_producers \
    cargo build --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-speaker-driver-cuda-reference \
    --bin qa-speaker-modeling-cuda-failure-contract \
    --bin qa-speaker-modeling-cuda-production-frequency

test -x "$producer" && test -x "$failure_producer" && test -x "$production_producer" || {
    echo "SM1 CUDA producers were not built under $target_dir" >&2
    exit 2
}

# Keep this release runner bounded: the full modeling library contains
# multi-minute resolved convergence studies. The SM1 admission/failure test is
# the focused CPU-side contract needed before the native CUDA producers.
run_phase modeling_cpu_contract \
    cargo test --release -p sonium-sim-speaker-modeling --features cuda --lib \
    cuda_failure_contract_is_typed_and_retries_without_partial_publication \
    -- --nocapture --test-threads=1

# The bounded native CUDA path is required: an absent device is an environment
# failure and cannot be recorded as a CPU-backed CUDA pass.
run_phase typed_driver_cuda \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    "$producer"

# The modeling-specific failure lane arms the native checked CUDA seam after
# resident setup and emits launch/device-loss records only after a fresh
# context retry has passed the independent CPU/f64 parity gates.
run_phase typed_failure_recovery_cuda \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    "$failure_producer"

run_phase production_frequency_cuda \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    "$production_producer"

artifact="$output_dir/speaker_driver_cuda_reference.json"
test -f "$artifact" || {
    echo "SM1 CUDA producer did not publish $artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
python3 "$checker" "$artifact" "$output_dir/typed_driver_cuda.log"

launch_artifact="$output_dir/speaker_modeling_cuda_launch_failure.json"
device_loss_artifact="$output_dir/speaker_modeling_cuda_device_loss_failure.json"
test -f "$launch_artifact" && test -f "$device_loss_artifact" || {
    echo "SM1 failure producer did not publish both typed failure artifacts" >&2
    exit 1
}
(cd "$output_dir" && sha256sum \
    "$(basename "$launch_artifact")" > "$(basename "$launch_artifact").sha256"
    sha256sum "$(basename "$device_loss_artifact")" > "$(basename "$device_loss_artifact").sha256"
    sha256sum "$(basename "$launch_artifact")" "$(basename "$device_loss_artifact")" \
        > failure-artifacts.sha256)
python3 "$failure_checker" \
    "$launch_artifact" "$device_loss_artifact" \
    "$output_dir/typed_failure_recovery_cuda.log"

production_artifact="$output_dir/speaker_modeling_cuda_production_frequency.json"
test -f "$production_artifact" || {
    echo "SM1 production-frequency producer did not publish $production_artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$production_artifact")" > "$(basename "$production_artifact").sha256")
python3 "$production_checker" \
    "$production_artifact" "$output_dir/production_frequency_cuda.log"

rm -f "$output_dir/checksums.sha256"
find "$output_dir" -maxdepth 1 -type f ! -name checksums.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output_dir/checksums.sha256"
sha256sum -c "$output_dir/checksums.sha256" >/dev/null

echo "SUCCESS: SM1 phase logs and bounded CUDA evidence written to $output_dir"
