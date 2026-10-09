#!/usr/bin/env bash
# SM2 evidence runner: deterministic CPU/f64 artifact contracts plus the
# required live CUDA multi-frequency/derivative/failure replay. A CPU fallback
# is never relabelled as a CUDA candidate artifact.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-sm2-fresh/cargo}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-sm2-fresh/artifact}"
producer="${SONIUM_SM2_PRODUCER:-${repo_root}/.target-sm2-fresh/cargo/release/qa-speaker-driver-sm2-cuda-reference}"
checker="${repo_root}/crates/sonium-sim-speaker-modeling/benchmarks/check_sm2_cuda.py"

test "$(uname -s)" = Linux || {
    echo "SM2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SM2 CUDA evidence requires x86_64" >&2
    exit 2
}
mkdir -p "$output_dir/.tmp" "$output_dir/.runtime"
export TMPDIR="$output_dir/.tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_RUNTIME_DIR="$output_dir/.runtime"
rm -f "$output_dir/checksums.sha256" \
    "$output_dir/speaker_driver_sm2_cuda_reference.json" \
    "$output_dir/speaker_driver_sm2_cuda_reference.json.sha256" \
    "$output_dir/speaker_driver_sm2_cuda_launch_failure.json" \
    "$output_dir/speaker_driver_sm2_cuda_launch_failure.json.sha256" \
    "$output_dir/speaker_driver_sm2_cuda_device_loss_failure.json" \
    "$output_dir/speaker_driver_sm2_cuda_device_loss_failure.json.sha256" \
    "$output_dir/qa_result.log" "$output_dir/phase-status.tsv"
mkdir -p "$target_dir" "$output_dir"
test -f "$checker" || {
    echo "SM2 strict checker is missing: $checker" >&2
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

status_file="$output_dir/phase-status.tsv"
: > "$status_file"

run_phase() {
    local name="$1"
    shift
    local log="$output_dir/${name}.log"
    local code=0
    echo "== SM2 phase: $name =="
    set +e
    "$@" 2>&1 | tee "$log"
    code="${PIPESTATUS[0]}"
    set -e
    printf '%s\t%s\n' "$name" "$code" >> "$status_file"
    if [ "$code" -ne 0 ]; then
        echo "SM2 phase '$name' failed; no candidate artifact is promoted" >&2
        return "$code"
    fi
}

# Build the production producer in the fresh repo-local target. This keeps
# the proof independent from any stale default target artifacts.
run_phase build_sm2_producer \
    cargo build --release -p sonium-qa --features 'qa-binaries cuda' \
    --bin qa-speaker-driver-sm2-cuda-reference
test -x "$producer" || {
    echo "SM2 producer build did not create executable: $producer" >&2
    exit 1
}

# CPU/f64 golden construction and artifact tamper/replay rejection tests.
run_phase driver_model_cpu_contracts \
    cargo test --release -p sonium-sim-speaker-modeling --lib reference_artifact -- --nocapture

# Required native CUDA replay. The producer emits the 15-point resident
# frequency sweep, independent CPU derivative and coupled-power observables,
# and the two typed failure/recreated-context records.
run_phase sm2_cuda_replay \
    env SONIUM_REQUIRE_CUDA=1 SONIUM_REQUIRE_BACKEND=cuda \
    "$producer"

artifact="$output_dir/speaker_driver_sm2_cuda_reference.json"
launch_failure="$output_dir/speaker_driver_sm2_cuda_launch_failure.json"
device_loss_failure="$output_dir/speaker_driver_sm2_cuda_device_loss_failure.json"
qa_log="$output_dir/sm2_cuda_replay.log"
test -f "$artifact" || {
    echo "SM2 CUDA producer did not publish $artifact" >&2
    exit 1
}
(cd "$output_dir" && sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256")
(cd "$output_dir" && sha256sum "$(basename "$launch_failure")" > "$(basename "$launch_failure").sha256")
(cd "$output_dir" && sha256sum "$(basename "$device_loss_failure")" > "$(basename "$device_loss_failure").sha256")
python3 "$checker" \
    "$artifact" \
    "$launch_failure" \
    "$device_loss_failure" \
    "$qa_log" \
    "$status_file"

rm -f "$output_dir/checksums.sha256"
find "$output_dir" -maxdepth 1 -type f ! -name checksums.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output_dir/checksums.sha256"
sha256sum -c "$output_dir/checksums.sha256" >/dev/null

echo "SUCCESS: SM2 artifact/tamper logs and CUDA replay evidence written to $output_dir"
