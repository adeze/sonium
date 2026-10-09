#!/usr/bin/env bash
# Run the complete native Linux/x86_64 PP1 producer and all-target sweep.
# The two evidence artifacts are intentionally separate so the checker can
# compare one worker with a configured Rayon run byte-for-byte.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PP1 evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP1 evidence requires x86_64" >&2; exit 2; }

target_dir="${CARGO_TARGET_DIR:-$repo_root/.target-unix-pp1}"
output_root="${SONIUM_OUTPUT_DIR:-$repo_root/target/sonium-unix-pp1}"
mkdir -p "$output_root/single" "$output_root/configured"

run_producer() {
    local workers="$1"
    local label="$2"
    SONIUM_PROPELLER_DETERMINISTIC=1 \
        RAYON_NUM_THREADS="$workers" \
        SONIUM_OUTPUT_DIR="$output_root/$label" \
        CARGO_TARGET_DIR="$target_dir" \
        cargo test --release -p sonium-sim-propellers \
            --test native_x86_pp1_evidence -- --nocapture --test-threads=1
    (cd "$output_root/$label" && sha256sum native-x86-pp1-v1.json > checksums.sha256)
}

run_producer 1 single
run_producer 2 configured

python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_native_x86_pp1.py" \
    "$output_root/single/native-x86-pp1-v1.json" \
    "$output_root/configured/native-x86-pp1-v1.json"

# Exercise every native library, binary, integration-test, example, and
# producer target in the crate under the same deterministic CPU policy.
SONIUM_PROPELLER_DETERMINISTIC=1 \
    RAYON_NUM_THREADS=1 \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release -p sonium-sim-propellers --all-targets -- --test-threads=1

# A representative invalid CLI request must be rejected, not silently
# clamped or published as an apparently valid product result.
if SONIUM_PROPELLER_DETERMINISTIC=1 \
    RAYON_NUM_THREADS=1 \
    CARGO_TARGET_DIR="$target_dir" \
    cargo run --release -p sonium-sim-propellers --bin propeller-3d-opt -- --rpm 1000 \
    > "$output_root/invalid-cli.log" 2>&1; then
    echo "invalid propeller CLI input unexpectedly succeeded" >&2
    exit 1
fi

