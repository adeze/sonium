#!/usr/bin/env bash
# PP2 bounded CUDA panel-session driver-shim lifecycle evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
test "$(uname -s)" = Linux || { echo "PP2 lifecycle evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP2 lifecycle evidence requires x86_64" >&2; exit 2; }

output="${SONIUM_PP2_LIFECYCLE_OUTPUT_DIR:-$repo_root/.target-pp2-lifecycle-evidence}"
target_dir="${CARGO_TARGET_DIR:-$repo_root/.target-pp2-fresh}"
checker="$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp2_cuda_lifecycle.py"
mkdir -p "$output/runtime"

export TMPDIR="$output/runtime"
export TMP="$output/runtime"
export TEMP="$output/runtime"
export XDG_RUNTIME_DIR="$output/runtime"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SONIUM_PP2_REQUIRE_CUDA=1
export SONIUM_PP2_DEVICE_INVENTORY_FILE="$output/gpu-inventory.csv"
export SONIUM_PP2_CUDA_LIFECYCLE_ARTIFACT="$output/cuda-native-adapter-failures-v1.json"
export CARGO_TARGET_DIR="$target_dir"

nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader \
    > "$output/gpu-inventory.csv"

cargo test --release -p sonium-sim-propellers --features cuda \
    --test pp2_cuda_adapter pp2_panel_session_driver_shim_lifecycle_evidence \
    -- --nocapture --test-threads=1 2>&1 | tee "$output/lifecycle-test.log"
python3 "$checker" "$output/cuda-native-adapter-failures-v1.json" \
    | tee "$output/lifecycle-check.json"

{
    echo "host=$(uname -a)"
    echo "target_dir=$target_dir"
    echo "output=$output"
    echo "runner=run_unix_pp2_lifecycle_cuda.sh"
} > "$output/metadata.txt"
(
    cd "$output"
    sha256sum \
        gpu-inventory.csv \
        cuda-native-adapter-failures-v1.json \
        lifecycle-test.log lifecycle-check.json metadata.txt \
        > checksums.sha256
    sha256sum -c checksums.sha256
)
echo "SUCCESS: PP2 CUDA lifecycle artifact and checksums written to $output"
