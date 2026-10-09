#!/usr/bin/env bash
# Stable native-CUDA FFT2 validation.  All build, runtime, and evidence
# scratch paths are repository-local so cancellation tests cannot leak into a
# shared system temporary directory.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/target/sonium-unix-fft2"
target_dir="$repo_root/.target-unix-fft2"
tmp_dir="$output_dir/.tmp"
runtime_dir="$output_dir/.runtime"

test "$(uname -s)" = Linux || {
    echo "FFT2 CUDA evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "FFT2 CUDA evidence requires x86_64" >&2
    exit 2
}

mkdir -p "$output_dir" "$tmp_dir" "$runtime_dir"
chmod 700 "$runtime_dir"
export CARGO_TARGET_DIR="$target_dir"
export TMPDIR="$tmp_dir"
export TMP="$tmp_dir"
export TEMP="$tmp_dir"
export XDG_RUNTIME_DIR="$runtime_dir"

SONIUM_REQUIRE_CUDA=1 cargo test -p sonium-fft --features cuda --lib 'cuda_' -- \
    --nocapture --test-threads=1
SONIUM_REQUIRE_CUDA=1 SONIUM_EVIDENCE_OUTPUT="$output_dir/fft_cuda_benchmark.json" \
    cargo run --release -p sonium-fft --features cuda --bin fft_cuda_benchmark
python3 "$repo_root/crates/sonium-fft/benchmarks/summarize_cuda_evidence.py" \
    "$output_dir/fft_cuda_benchmark.json" > "$output_dir/fft_cuda_summary.json"
python3 "$repo_root/crates/sonium-fft/benchmarks/check_fft2_matrix.py" \
    "$output_dir/fft_cuda_benchmark.json" > "$output_dir/fft2_matrix_summary.json"

(
    cd "$output_dir"
    sha256sum fft_cuda_benchmark.json fft_cuda_summary.json \
        fft2_matrix_summary.json > checksums.sha256
    sha256sum -c checksums.sha256
)
