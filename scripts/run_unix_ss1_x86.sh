#!/usr/bin/env bash
# Reproducible native Linux/x86 sparse action and dispatch evidence for SS1.
# Four release runs make scalar, AVX2, AVX2+FMA, and runtime-native dispatch
# observable. The checker validates the independent-oracle matrix before a
# checksummed bundle is published.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-ss1-closure}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-ss1}"

test "$(uname -s)" = Linux || {
    echo "SS1 native sparse evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "SS1 native sparse evidence requires x86_64" >&2
    exit 2
}
grep -qw avx2 /proc/cpuinfo || {
    echo "SS1 native sparse evidence requires an AVX2 host; no unsupported run is claimed" >&2
    exit 2
}
grep -qw fma /proc/cpuinfo || {
    echo "SS1 native sparse evidence requires an FMA host; no unsupported run is claimed" >&2
    exit 2
}

mkdir -p "$output_root"

run_matrix() {
    local label="$1"
    local evidence="$output_root/${label}.json"
    shift
    env "$@" \
        SONIUM_SPARSE_EVIDENCE_OUT="$evidence" \
        CARGO_TARGET_DIR="$target_dir" \
        cargo test --release -p sonium-solvers-sparse --test native_x86_sparse_evidence -- \
        --nocapture --test-threads=1 | tee "$output_root/${label}.log"
    (cd "$output_root" && sha256sum "${label}.json" > "${label}.json.sha256" && sha256sum -c "${label}.json.sha256")
}

run_matrix native RUSTFLAGS= SONIUM_SPARSE_FORCE_SCALAR=0 SONIUM_SPARSE_DISABLE_FMA=0
run_matrix scalar RUSTFLAGS= SONIUM_SPARSE_FORCE_SCALAR=1 SONIUM_SPARSE_DISABLE_FMA=0
run_matrix avx2 RUSTFLAGS= SONIUM_SPARSE_FORCE_SCALAR=0 SONIUM_SPARSE_DISABLE_FMA=1
run_matrix fma RUSTFLAGS=-C\ target-feature=+avx2,+fma SONIUM_SPARSE_FORCE_SCALAR=0 SONIUM_SPARSE_DISABLE_FMA=0

comparison="$output_root/native-vs-scalar-comparison-v2.json"
python3 "$repo_root/scripts/check_unix_ss1_x86.py" \
    "$output_root/native.json" \
    "$output_root/scalar.json" \
    "$output_root/avx2.json" \
    "$output_root/fma.json" \
    --output "$comparison" \
    --negative-probes

sha256sum "$comparison" > "${comparison}.sha256"
sha256sum -c "${comparison}.sha256"

(cd "$output_root" && sha256sum \
    native.json native.json.sha256 \
    scalar.json scalar.json.sha256 \
    avx2.json avx2.json.sha256 \
    fma.json fma.json.sha256 \
    "$(basename "$comparison")" "$(basename "$comparison").sha256" > checksums.sha256 && \
    sha256sum -c checksums.sha256)

echo "SS1 native x86 sparse evidence: passed; bundle=$output_root"
