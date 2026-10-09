#!/usr/bin/env bash
# Native single-device C4 scheduler contention evidence on SM120.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-c4-contention-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-c4-contention-fresh/evidence}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_dir" in /*) ;; *) output_dir="$repo_root/$output_dir" ;; esac
runtime_dir="$target_dir/tmp"

test "$(uname -s)" = Linux || {
    echo "C4 native scheduler contention requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C4 native scheduler contention requires x86_64" >&2
    exit 2
}
command -v nvidia-smi >/dev/null || {
    echo "C4 native scheduler contention requires nvidia-smi" >&2
    exit 2
}

mkdir -p "$output_dir" "$runtime_dir"
rustc -Vv > "$output_dir/rustc-vv.txt"
cargo -V > "$output_dir/cargo-version.txt"
nvidia-smi --id=0 --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits \
  > "$output_dir/device-inventory.txt"

artifact="$output_dir/c4-cuda-scheduler-contention.json"
summary="$output_dir/c4-cuda-scheduler-contention-v1.json"
rm -f "$artifact" "$artifact.sha256" "$summary" "$summary.sha256" \
  "$output_dir/c4-cuda-scheduler-contention-run.log" "$output_dir/checksums.sha256"

TMPDIR="$runtime_dir" TMP="$runtime_dir" TEMP="$runtime_dir" XDG_RUNTIME_DIR="$runtime_dir" \
SONIUM_REQUIRE_CUDA=1 SONIUM_OUTPUT_DIR="$output_dir" CARGO_TARGET_DIR="$target_dir" \
  cargo test --release -p sonium-compute --features cuda \
    --test c4_cuda_scheduler_contention_evidence -- \
    --nocapture --test-threads=1 > "$output_dir/c4-cuda-scheduler-contention-run.log" 2>&1

(
    cd "$output_dir"
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256"
    sha256sum -c "$(basename "$artifact").sha256"
)
python3 "$repo_root/crates/sonium-compute/benchmarks/check_c4_cuda_scheduler_contention.py" \
  "$artifact" --output "$summary"
sha256sum "$summary" > "$summary.sha256"
sha256sum -c "$summary.sha256"

(
    cd "$output_dir"
    sha256sum \
      rustc-vv.txt cargo-version.txt device-inventory.txt \
      "$(basename "$artifact")" "$(basename "$artifact").sha256" \
      "$(basename "$summary")" "$(basename "$summary").sha256" \
      c4-cuda-scheduler-contention-run.log > checksums.sha256
    sha256sum -c checksums.sha256
)

echo "C4 native CUDA scheduler contention evidence: passed; bundle=$output_dir"
