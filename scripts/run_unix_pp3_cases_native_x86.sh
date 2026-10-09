#!/usr/bin/env bash
# Produce and gate the bounded PP3 production multi-rotor/off-design packet.
# Cargo and temporary files remain inside the repository-local target tree.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "PP3 multi-rotor evidence requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "PP3 multi-rotor evidence requires x86_64" >&2; exit 2; }

target_dir="$repo_root/.target-pp3-cases-fresh"
output_root="$target_dir/evidence"
runtime_root="$target_dir/runtime"
artifact="$output_root/native-x86-pp3-multirotor-offdesign-v1.json"
checker="$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp3_multirotor_offdesign.py"
mkdir -p "$output_root" "$runtime_root"

SONIUM_PP3_MULTI_ROTOR_ARTIFACT="$artifact" \
RAYON_NUM_THREADS=1 \
TMPDIR="$runtime_root" TMP="$runtime_root" TEMP="$runtime_root" \
XDG_RUNTIME_DIR="$runtime_root" \
CARGO_TARGET_DIR="$target_dir" \
cargo test --release -p sonium-sim-propellers \
    --test pp3_multirotor_offdesign_evidence -- --nocapture --test-threads=1 \
    2>&1 | tee "$output_root/multirotor-test.log"

python3 "$checker" "$artifact" 2>&1 | tee "$output_root/checker.log"
python3 -m py_compile "$checker"

(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256"
    sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" \
        "multirotor-test.log" "checker.log" > checksums.sha256
    sha256sum -c checksums.sha256
)

python3 "$checker" "$artifact" 2>&1 | tee "$output_root/checker-replay.log"
(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" "$(basename "$artifact").sha256" \
        "multirotor-test.log" "checker.log" "checker-replay.log" > checksums.sha256
    sha256sum -c checksums.sha256
)
