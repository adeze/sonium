#!/usr/bin/env bash
# Produce and gate the bounded PP4 production-gallery replay evidence.
# Cargo state and all generated gallery/evidence files remain repository-local.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PP4 gallery replay requires native Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP4 gallery replay requires native x86_64" >&2; exit 2; }

target_dir="$repo_root/.target-pp4-next-fresh"
output_root="$target_dir/evidence"
artifact="$output_root/native-x86-pp4-gallery-replay-v1.json"
mkdir -p "$output_root"

SONIUM_OUTPUT_DIR="$output_root" \
    RAYON_NUM_THREADS=1 \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release -p sonium-sim-propellers \
    --test pp4_gallery_replay_evidence -- --nocapture --test-threads=1

(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" > checksums.sha256
    sha256sum -c checksums.sha256
)
python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp4_gallery_replay.py" \
    "$artifact"
