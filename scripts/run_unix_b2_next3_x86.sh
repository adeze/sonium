#!/usr/bin/env bash
# Reproducible native-x86 B2 MLFMM frequency/ka limitation-map evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-b2-next3}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-b2-next3}"
frequency_output="${output_root}/frequency"
artifact="${frequency_output}/sonium_bem_native_x86_b2_frequency.json"
checksum="${artifact}.sha256"
check="${output_root}/native-x86-b2-frequency-check-v1.json"

test "$(uname -s)" = Linux || {
    echo "B2 frequency evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B2 frequency evidence requires x86_64" >&2
    exit 2
}

mkdir -p "$frequency_output"
rm -f "$artifact" "$checksum" "$check" "$output_root/checksums.sha256"

SONIUM_B2_FREQUENCY_REQUIRE_NATIVE_X86=1 \
SONIUM_OUTPUT_DIR="$frequency_output" \
SONIUM_QA_THREAD_POLICY=single \
SONIUM_QA_AFFINITY_POLICY=unbound \
SONIUM_QA_NUMA_POLICY=default \
SONIUM_QA_DETERMINISTIC_MODE=enabled \
CARGO_TARGET_DIR="$target_dir" \
cargo test --release -p sonium-bem --test native_x86_b2_frequency_evidence -- \
    --nocapture --test-threads=1

test -s "$artifact"
(cd "$frequency_output" && sha256sum "$(basename "$artifact")" > "$(basename "$checksum")" && sha256sum -c "$(basename "$checksum")")
python3 "$repo_root/scripts/check_unix_b2_next3_x86.py" \
    "$artifact" --checksum "$checksum" --output "$check"

(cd "$output_root" && sha256sum \
    "frequency/$(basename "$artifact")" \
    "frequency/$(basename "$checksum")" \
    "$(basename "$check")" > checksums.sha256 && sha256sum -c checksums.sha256)
