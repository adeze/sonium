#!/usr/bin/env bash
# Produce and gate bounded native x86 FastMath process-CPU-time evidence.
# All Cargo state and generated evidence remain repository-local.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(uname -s)" = Linux || { echo "F3 CPU-time evidence requires native Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "F3 CPU-time evidence requires native x86_64" >&2; exit 2; }
grep -qw avx2 /proc/cpuinfo || { echo "F3 CPU-time evidence requires AVX2" >&2; exit 2; }
grep -qw fma /proc/cpuinfo || { echo "F3 CPU-time evidence requires FMA" >&2; exit 2; }

target_dir="$repo_root/.target-f3-next-fresh"
output_root="$target_dir/evidence"
artifact="$output_root/native-x86-f3-cpu-time-v1.json"
binary="$target_dir/release/fastmath_benchmark"
mkdir -p "$output_root"

CARGO_TARGET_DIR="$target_dir" cargo build --release -p sonium-fastmath --bin fastmath_benchmark
python3 "$repo_root/crates/sonium-fastmath/benchmarks/produce_f3_cpu_time.py" \
    "$binary" "$artifact"

(
    cd "$output_root"
    sha256sum "$(basename "$artifact")" > checksums.sha256
    sha256sum -c checksums.sha256
)
python3 "$repo_root/crates/sonium-fastmath/benchmarks/check_f3_cpu_time.py" \
    "$artifact"
