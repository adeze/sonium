#!/usr/bin/env bash
# Produce and gate the bounded PP3 observer-distance analytic ladder.
# All Cargo state and generated evidence remain repository-local.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PP3 observer-distance evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP3 observer-distance evidence requires x86_64" >&2; exit 2; }

target_dir="$repo_root/.target-unix-pp3-next"
output_root="$repo_root/target/sonium-unix-pp3-next"
artifact="$output_root/native-x86-pp3-observer-distance-v1.json"
mkdir -p "$output_root"

SONIUM_OUTPUT_DIR="$output_root" \
    RAYON_NUM_THREADS=1 \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release -p sonium-sim-propellers \
    --test pp3_observer_distance_evidence -- --nocapture --test-threads=1

(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" > checksums.sha256
    sha256sum -c checksums.sha256
)
python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp3_observer_distance.py" \
    "$artifact"
