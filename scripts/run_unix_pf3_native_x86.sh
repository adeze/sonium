#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PF3 evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PF3 evidence requires native x86_64" >&2; exit 2; }

output="${SONIUM_OUTPUT_DIR:-$repo_root/target/sonium-unix-pf3}"
target="${CARGO_TARGET_DIR:-$repo_root/.target-unix-pf3}"
mkdir -p "$output"

SONIUM_PF3_DETERMINISTIC=1 \
    SONIUM_PF3_OUTPUT_DIR="$output" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}" \
    CARGO_TARGET_DIR="$target" \
    cargo test --release -p sonium-sim-pcfan --test native_x86_pf3_evidence \
    -- --nocapture --test-threads=1

python3 "$repo_root/crates/sonium-sim-pcfan/benchmarks/check_native_x86_pf3.py" \
    "$output/native-x86-pcfan-pf3-v1.json"
