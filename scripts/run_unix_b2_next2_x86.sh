#!/usr/bin/env bash
# Reproducible native-x86 B2 physical H/FMM CPU-reference evidence.
# H-matrix field parity is checked strictly; the fixed production MLFMM
# expansion is recorded under an explicit low-ka approximation budget.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-b2-next2}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-b2-next2}"
physical_output="${output_root}/physical"
artifact="${physical_output}/sonium_bem_native_x86_b2_physical.json"
checksum="${artifact}.sha256"
check="${output_root}/native-x86-b2-physical-check-v1.json"

test "$(uname -s)" = Linux || {
    echo "B2 physical evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B2 physical evidence requires x86_64" >&2
    exit 2
}

mkdir -p "$physical_output"
rm -f "$artifact" "$checksum" "$check" "$output_root/checksums.sha256"

SONIUM_B2_PHYSICAL_REQUIRE_NATIVE_X86=1 \
SONIUM_OUTPUT_DIR="$physical_output" \
SONIUM_QA_THREAD_POLICY=single \
SONIUM_QA_AFFINITY_POLICY=unbound \
SONIUM_QA_NUMA_POLICY=default \
SONIUM_QA_DETERMINISTIC_MODE=enabled \
CARGO_TARGET_DIR="$target_dir" \
cargo test --release -p sonium-bem --test native_x86_b2_physical_evidence -- \
    --nocapture --test-threads=1

test -s "$artifact"
(cd "$physical_output" && sha256sum "$(basename "$artifact")" > "$(basename "$checksum")" && sha256sum -c "$(basename "$checksum")")
python3 "$repo_root/scripts/check_unix_b2_next2_x86.py" \
    "$artifact" --checksum "$checksum" --output "$check"

(cd "$output_root" && sha256sum \
    "physical/$(basename "$artifact")" \
    "physical/$(basename "$checksum")" \
    "$(basename "$check")" > checksums.sha256 && sha256sum -c checksums.sha256)
