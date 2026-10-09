#!/usr/bin/env bash
# Produce and gate the bounded PP3 body-fitted FEM mesh-convergence ladder.
# Cargo state and generated evidence stay in the repository-local target dir.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PP3 body-fitted evidence requires native Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP3 body-fitted evidence requires native x86_64" >&2; exit 2; }

target_dir="$repo_root/.target-pp3-next-fresh"
output_root="$target_dir/evidence"
artifact="$output_root/native-x86-pp3-body-fitted-convergence-v1.json"
mkdir -p "$output_root"

SONIUM_OUTPUT_DIR="$output_root" \
    RAYON_NUM_THREADS=1 \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release -p sonium-sim-propellers \
    --test pp3_body_fitted_convergence_evidence -- --nocapture --test-threads=1

(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" > checksums.sha256
    sha256sum -c checksums.sha256
)
python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp3_body_fitted_convergence.py" \
    "$artifact"
