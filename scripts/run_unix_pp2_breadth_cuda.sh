#!/usr/bin/env bash
# PP2 bounded CUDA multipoint production panel-acoustic evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
test "$(uname -s)" = Linux || { echo "PP2 breadth evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP2 breadth evidence requires x86_64" >&2; exit 2; }

target_dir="${CARGO_TARGET_DIR:-$repo_root/.target-pp2-breadth-fresh}"
output="${SONIUM_PP2_BREADTH_OUTPUT_DIR:-$target_dir/evidence}"
checker="$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp2_cuda_acoustic_sweep.py"
capability_checker="$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp2_capability.py"
mkdir -p "$output/runtime"

export TMPDIR="$output/runtime"
export TMP="$output/runtime"
export TEMP="$output/runtime"
export XDG_RUNTIME_DIR="$output/runtime"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SONIUM_PP2_REQUIRE_CUDA=1
export SONIUM_PP2_DEVICE_INVENTORY_FILE="$output/gpu-inventory.csv"
export SONIUM_OUTPUT_DIR="$output"
export SONIUM_PP2_CUDA_ADAPTER_ARTIFACT="$output/cuda-native-adapter-v1.json"
export SONIUM_PP2_CUDA_ACOUSTIC_SWEEP_ARTIFACT="$output/cuda-native-acoustic-sweep-v1.json"
export CARGO_TARGET_DIR="$target_dir"

nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader \
    > "$output/gpu-inventory.csv"

cargo test --release -p sonium-sim-propellers --features cuda \
    --test pp2_cuda_metal_capability pp2_cuda_metal_matrix_is_complete_and_fail_closed \
    -- --nocapture --test-threads=1 2>&1 | tee "$output/capability-test.log"
python3 "$capability_checker" "$output/cuda-metal-capability-v1.json" \
    | tee "$output/capability-check.json"

cargo test --release -p sonium-sim-propellers --features cuda \
    --test pp2_cuda_adapter 2>&1 | tee "$output/adapter-test.log"
python3 "$checker" "$output/cuda-native-acoustic-sweep-v1.json" \
    | tee "$output/acoustic-sweep-check.json"

{
    echo "host=$(uname -a)"
    echo "target_dir=$target_dir"
    echo "output=$output"
    echo "runner=run_unix_pp2_breadth_cuda.sh"
    echo "cell=cuda/f64/Complex64/blade_element_bem/order1/forward/multipoint"
} > "$output/metadata.txt"
(
    cd "$output"
    sha256sum \
        gpu-inventory.csv \
        cuda-metal-capability-v1.json cuda-native-adapter-v1.json \
        cuda-native-acoustic-sweep-v1.json \
        capability-test.log adapter-test.log capability-check.json \
        acoustic-sweep-check.json metadata.txt \
        > checksums.sha256
    sha256sum -c checksums.sha256
)
echo "SUCCESS: PP2 CUDA multipoint acoustic artifact and checksums written to $output"
