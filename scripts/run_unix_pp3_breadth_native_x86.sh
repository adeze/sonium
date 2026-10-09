#!/usr/bin/env bash
# Produce and gate the bounded PP3 reduced-order ducted convergence ladder.
# All Cargo state and evidence stay in the repository-local target directory.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PP3 ducted evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP3 ducted evidence requires x86_64" >&2; exit 2; }

target_dir="$repo_root/.target-pp3-breadth-fresh"
output_root="$target_dir/evidence"
artifact="$output_root/native-x86-pp3-ducted-convergence-v1.json"
sidecar="$artifact.sha256"
mkdir -p "$output_root"

SONIUM_OUTPUT_DIR="$output_root" \
    RAYON_NUM_THREADS=1 \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release -p sonium-sim-propellers \
    --test pp3_ducted_convergence_evidence -- --nocapture --test-threads=1

python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp3_ducted_convergence.py" \
    "$artifact"

(cd "$output_root" && sha256sum "$(basename "$artifact")" > "$(basename "$sidecar")")
(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" "$(basename "$sidecar")" > checksums.sha256
    sha256sum -c checksums.sha256
)
python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp3_ducted_convergence.py" \
    "$artifact"
