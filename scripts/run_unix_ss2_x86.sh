#!/usr/bin/env bash
# Reproducible native Linux/x86 SS2 plan-selection and SpMV evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-ss2-baseline}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-ss2-baseline}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_root" in /*) ;; *) output_root="$repo_root/$output_root" ;; esac

test "$(uname -s)" = Linux || { echo "SS2 requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SS2 requires x86_64" >&2; exit 2; }
grep -qw avx2 /proc/cpuinfo || { echo "SS2 requires AVX2; no unsupported result is claimed" >&2; exit 2; }
grep -qw fma /proc/cpuinfo || { echo "SS2 requires FMA; no unsupported result is claimed" >&2; exit 2; }

mkdir -p "$output_root"
rm -f "$output_root/plan-thresholds-native.json" "$output_root/plan-thresholds-native.json.sha256" \
    "$output_root/plan-thresholds-scalar.json" "$output_root/plan-thresholds-scalar.json.sha256" \
    "$output_root/spmv-types-native.json" "$output_root/spmv-types-native.json.sha256" \
    "$output_root/spmv-types-scalar.json" "$output_root/spmv-types-scalar.json.sha256" \
    "$output_root/vendor-baseline-probe.json" "$output_root/vendor-baseline-probe.json.sha256" \
    "$output_root/native-vs-scalar-comparison-v2.json" "$output_root/checksums.sha256"

rustc -Vv > "$output_root/rustc-vv.txt"
cargo -V > "$output_root/cargo-version.txt"
lscpu > "$output_root/lscpu.txt"

RUSTFLAGS= \
SONIUM_SPARSE_FORCE_SCALAR=0 \
SONIUM_SPARSE_PLAN_EVIDENCE_OUT="$output_root/plan-thresholds-native.json" \
CARGO_TARGET_DIR="$target_dir" \
cargo bench -p sonium-solvers-sparse --bench sparse_x86 -- --noplot > "$output_root/plan-thresholds-native.log" 2>&1

RUSTFLAGS= \
SONIUM_SPARSE_FORCE_SCALAR=1 \
SONIUM_SPARSE_PLAN_EVIDENCE_OUT="$output_root/plan-thresholds-scalar.json" \
CARGO_TARGET_DIR="$target_dir" \
cargo bench -p sonium-solvers-sparse --bench sparse_x86 -- --noplot > "$output_root/plan-thresholds-scalar.log" 2>&1

RUSTFLAGS= \
SONIUM_SPARSE_FORCE_SCALAR=0 \
SONIUM_SOLVERS_PERF_JSON="$output_root/spmv-types-native.json" \
SONIUM_SS2_FAST_BENCH=1 \
CARGO_TARGET_DIR="$target_dir" \
cargo bench -p sonium-solvers --features cli --bench spmv_bench -- \
    --noplot > "$output_root/spmv-types-native.log" 2>&1

SONIUM_SPARSE_FORCE_SCALAR=1 \
RUSTFLAGS= \
SONIUM_SOLVERS_PERF_JSON="$output_root/spmv-types-scalar.json" \
SONIUM_SS2_FAST_BENCH=1 \
CARGO_TARGET_DIR="$target_dir" \
cargo bench -p sonium-solvers --features cli --bench spmv_bench -- \
    --noplot > "$output_root/spmv-types-scalar.log" 2>&1

python3 "$repo_root/scripts/probe_unix_ss2_vendor.py" \
    --output "$output_root/vendor-baseline-probe.json" \
    > "$output_root/vendor-baseline-probe.log"

(cd "$output_root" && sha256sum plan-thresholds-native.json > plan-thresholds-native.json.sha256 && \
    sha256sum plan-thresholds-scalar.json > plan-thresholds-scalar.json.sha256 && \
    sha256sum spmv-types-native.json > spmv-types-native.json.sha256 && \
    sha256sum spmv-types-scalar.json > spmv-types-scalar.json.sha256 && \
    sha256sum vendor-baseline-probe.json > vendor-baseline-probe.json.sha256 && \
    sha256sum -c plan-thresholds-native.json.sha256 plan-thresholds-scalar.json.sha256 \
      spmv-types-native.json.sha256 spmv-types-scalar.json.sha256 vendor-baseline-probe.json.sha256)

comparison="$output_root/native-vs-scalar-comparison-v2.json"
python3 "$repo_root/scripts/check_unix_ss2_x86.py" \
    "$output_root/plan-thresholds-native.json" \
    "$output_root/plan-thresholds-scalar.json" \
    "$output_root/spmv-types-native.json" \
    "$output_root/spmv-types-scalar.json" \
    --vendor-probe "$output_root/vendor-baseline-probe.json" \
    --cache-info "$output_root/lscpu.txt" \
    --output "$comparison"

(cd "$output_root" && sha256sum \
    plan-thresholds-native.json plan-thresholds-native.json.sha256 plan-thresholds-native.log \
    plan-thresholds-scalar.json plan-thresholds-scalar.json.sha256 plan-thresholds-scalar.log \
    spmv-types-native.json spmv-types-native.json.sha256 spmv-types-native.log \
    spmv-types-scalar.json spmv-types-scalar.json.sha256 spmv-types-scalar.log \
    vendor-baseline-probe.json vendor-baseline-probe.json.sha256 vendor-baseline-probe.log \
    native-vs-scalar-comparison-v2.json rustc-vv.txt cargo-version.txt lscpu.txt > checksums.sha256 && \
    sha256sum -c checksums.sha256)

echo "SS2 native x86 sparse evidence: passed; bundle=$output_root"
