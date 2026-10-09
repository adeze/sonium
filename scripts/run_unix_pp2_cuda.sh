#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
test "$(uname -s)" = Linux || { echo "PP2 evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP2 evidence requires x86_64" >&2; exit 2; }

output="${SONIUM_PP2_OUTPUT_DIR:-$repo_root/target/sonium-unix-pp2}"
target_dir="${CARGO_TARGET_DIR:-$repo_root/.target-unix-pp2-warm}"
mkdir -p "$output" "$output/runtime"
export TMPDIR="$output/runtime"
export TMP="$output/runtime"
export TEMP="$output/runtime"
export XDG_RUNTIME_DIR="$output/runtime"

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader > "$output/gpu-inventory.csv"

common_env=(
    "SONIUM_PP2_DEVICE_INVENTORY_FILE=$output/gpu-inventory.csv"
    "SONIUM_OUTPUT_DIR=$output"
    "CARGO_TARGET_DIR=$target_dir"
)

env "${common_env[@]}" cargo test --release -p sonium-sim-propellers \
    --features cuda --test pp2_cuda_metal_capability -- --nocapture --test-threads=1 \
    2>&1 | tee "$output/capability-test.log"
python3 crates/sonium-sim-propellers/benchmarks/check_pp2_capability.py \
    "$output/cuda-metal-capability-v1.json" | tee "$output/capability-check.json"

env "${common_env[@]}" SONIUM_PP2_REQUIRE_CUDA=1 \
    SONIUM_PP2_CUDA_ADAPTER_ARTIFACT="$output/cuda-native-adapter-v1.json" \
    cargo test --release -p sonium-sim-propellers --features cuda \
    --test pp2_cuda_adapter -- --nocapture --test-threads=1 \
    2>&1 | tee "$output/adapter-test.log"
python3 crates/sonium-sim-propellers/benchmarks/check_pp2_cuda_adapter.py \
    "$output/cuda-native-adapter-v1.json" | tee "$output/adapter-check.json"

{
    echo "host=$(uname -a)"
    echo "target_dir=$target_dir"
    echo "output=$output"
    echo "runner=run_unix_pp2_cuda.sh"
} > "$output/metadata.txt"
(
    cd "$output"
    sha256sum \
        gpu-inventory.csv \
        cuda-metal-capability-v1.json \
        cuda-native-adapter-v1.json \
        capability-test.log adapter-test.log capability-check.json adapter-check.json metadata.txt \
        > checksums.sha256
    sha256sum -c checksums.sha256
)
