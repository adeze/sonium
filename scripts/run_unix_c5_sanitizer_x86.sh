#!/usr/bin/env bash
# Reproducible native Linux x86_64 C5 AddressSanitizer/negative-path evidence.
# This lane is CPU-only: it intentionally does not enable CUDA or run a GPU
# workload.  The parity and negative-path artifacts are checked independently
# and then tied together by a checksummed comparison summary.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-c5-sanitizer}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-c5-sanitizer}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_root" in /*) ;; *) output_root="$repo_root/$output_root" ;; esac

test "$(uname -s)" = Linux || { echo "C5 sanitizer requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "C5 sanitizer requires x86_64" >&2; exit 2; }
grep -qw avx2 /proc/cpuinfo || { echo "C5 sanitizer requires AVX2; no unsupported result is claimed" >&2; exit 2; }
grep -qw fma /proc/cpuinfo || { echo "C5 sanitizer requires FMA; no unsupported result is claimed" >&2; exit 2; }
test "$(rustc +nightly -vV | sed -n 's/^host: //p')" = x86_64-unknown-linux-gnu

mkdir -p "$output_root/parity" "$output_root/negative"
rm -f "$output_root/parity/sonium_native_x86_evidence.json" \
  "$output_root/parity/sonium_native_x86_evidence.json.sha256" \
  "$output_root/negative/native-x86-sanitizer-negative.json" \
  "$output_root/negative/native-x86-sanitizer-negative.json.sha256" \
  "$output_root/native-x86-sanitizer-comparison-v1.json" \
  "$output_root/native-x86-sanitizer-comparison-v1.json.sha256" \
  "$output_root/checksums.sha256"

rustc +nightly -Vv > "$output_root/rustc-nightly-vv.txt"
cargo +nightly -V > "$output_root/cargo-nightly-version.txt"
lscpu > "$output_root/lscpu.txt"

common_env=(
  "ASAN_OPTIONS=detect_leaks=0"
  "LSAN_OPTIONS=detect_leaks=0"
  "RUSTFLAGS=-Zsanitizer=address"
  "RUSTC_WRAPPER=$repo_root/scripts/sonium-sanitizer-rustc-wrapper"
  "SONIUM_REQUIRE_AVX2_FMA=1"
  "SONIUM_C5_SANITIZER=address"
  "SONIUM_C5_THREAD_POLICY=single"
  "SONIUM_C5_AFFINITY_POLICY=unbound"
  "SONIUM_C5_NUMA_POLICY=default"
  "SONIUM_QA_THREAD_POLICY=single"
  "SONIUM_QA_AFFINITY_POLICY=unbound"
  "SONIUM_QA_NUMA_POLICY=default"
  "SONIUM_QA_DETERMINISTIC_MODE=enabled"
  "CARGO_TARGET_DIR=$target_dir"
)

env "${common_env[@]}" SONIUM_OUTPUT_DIR="$output_root/parity" \
  cargo +nightly test --release -p sonium-compute --test native_x86_evidence \
  -- --nocapture --test-threads=1 > "$output_root/parity.log" 2>&1

(cd "$output_root/parity" && sha256sum sonium_native_x86_evidence.json > sonium_native_x86_evidence.json.sha256 && sha256sum -c sonium_native_x86_evidence.json.sha256)

env "${common_env[@]}" SONIUM_OUTPUT_DIR="$output_root/negative" \
  cargo +nightly test --release -p sonium-compute --test native_x86_sanitizer_evidence \
  -- --ignored --nocapture --test-threads=1 > "$output_root/negative.log" 2>&1

(cd "$output_root/negative" && sha256sum native-x86-sanitizer-negative.json > native-x86-sanitizer-negative.json.sha256 && sha256sum -c native-x86-sanitizer-negative.json.sha256)

comparison="$output_root/native-x86-sanitizer-comparison-v1.json"
python3 "$repo_root/crates/sonium-compute/benchmarks/check_native_x86_c5_sanitizer.py" \
  "$output_root/parity/sonium_native_x86_evidence.json" \
  "$output_root/negative/native-x86-sanitizer-negative.json" \
  --output "$comparison"
sha256sum "$comparison" > "$comparison.sha256"
sha256sum -c "$comparison.sha256"

(cd "$output_root" && sha256sum \
  parity/sonium_native_x86_evidence.json parity/sonium_native_x86_evidence.json.sha256 \
  negative/native-x86-sanitizer-negative.json negative/native-x86-sanitizer-negative.json.sha256 \
  native-x86-sanitizer-comparison-v1.json native-x86-sanitizer-comparison-v1.json.sha256 \
  parity.log negative.log rustc-nightly-vv.txt cargo-nightly-version.txt lscpu.txt > checksums.sha256 && \
  sha256sum -c checksums.sha256)

echo "C5 native x86 sanitizer/negative evidence: passed; bundle=$output_root"
