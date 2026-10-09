#!/usr/bin/env bash
# Reproducible native-x86 BEM/SIMD evidence for Unix review unit B4.
# The two invocations exercise the same public kernels with native dispatch
# and forced scalar dispatch; the final check compares their serialized
# observables and provenance rather than relying on human-readable output.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-b4-physical}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-b4-physical}"
native_output="${output_root}/native"
scalar_output="${output_root}/scalar"
quad4_output="${output_root}/quad4"

test "$(uname -s)" = Linux || {
    echo "B4 native reference requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B4 native reference requires x86_64" >&2
    exit 2
}
grep -qw avx2 /proc/cpuinfo || {
    echo "B4 native reference requires AVX2" >&2
    exit 2
}
grep -qw fma /proc/cpuinfo || {
    echo "B4 native reference requires FMA" >&2
    exit 2
}

mkdir -p "$native_output" "$scalar_output" "$quad4_output"
rm -f \
    "$native_output/sonium_native_x86_evidence.json" \
    "$native_output/sonium_native_x86_evidence.json.sha256" \
    "$scalar_output/sonium_native_x86_evidence.json" \
    "$scalar_output/sonium_native_x86_evidence.json.sha256" \
    "$quad4_output/sonium_bem_native_x86_b4_quad.json" \
    "$quad4_output/sonium_bem_native_x86_b4_quad.json.sha256" \
    "$output_root/native-vs-scalar-comparison-v1.json" \
    "$output_root/checksums.sha256"

SONIUM_REQUIRE_AVX2_FMA=1 \
SONIUM_OUTPUT_DIR="$native_output" \
SONIUM_QA_THREAD_POLICY=single \
SONIUM_QA_AFFINITY_POLICY=unbound \
SONIUM_QA_NUMA_POLICY=default \
SONIUM_QA_DETERMINISTIC_MODE=enabled \
CARGO_TARGET_DIR="$target_dir" \
cargo test --release -p sonium-compute --test native_x86_evidence -- \
    --nocapture --test-threads=1

SONIUM_COMPUTE_FORCE_SCALAR=1 \
SONIUM_OUTPUT_DIR="$scalar_output" \
SONIUM_QA_THREAD_POLICY=single \
SONIUM_QA_AFFINITY_POLICY=unbound \
SONIUM_QA_NUMA_POLICY=default \
SONIUM_QA_DETERMINISTIC_MODE=enabled \
CARGO_TARGET_DIR="$target_dir" \
cargo test --release -p sonium-compute --test native_x86_evidence -- \
    --nocapture --test-threads=1

(cd "$native_output" && sha256sum sonium_native_x86_evidence.json > sonium_native_x86_evidence.json.sha256 && sha256sum -c sonium_native_x86_evidence.json.sha256)
(cd "$scalar_output" && sha256sum sonium_native_x86_evidence.json > sonium_native_x86_evidence.json.sha256 && sha256sum -c sonium_native_x86_evidence.json.sha256)

SONIUM_B4_QUAD4_REQUIRE_NATIVE_X86=1 \
SONIUM_BEM_QUAD4_FORCE_SCALAR=1 \
SONIUM_OUTPUT_DIR="$quad4_output" \
SONIUM_QA_THREAD_POLICY=single \
SONIUM_QA_AFFINITY_POLICY=unbound \
SONIUM_QA_NUMA_POLICY=default \
SONIUM_QA_DETERMINISTIC_MODE=enabled \
CARGO_TARGET_DIR="$target_dir" \
cargo test --release -p sonium-bem --test native_x86_b4_quad_evidence -- \
    --nocapture --test-threads=1

(cd "$quad4_output" && sha256sum sonium_bem_native_x86_b4_quad.json > sonium_bem_native_x86_b4_quad.json.sha256 && sha256sum -c sonium_bem_native_x86_b4_quad.json.sha256)
comparison="$output_root/native-vs-scalar-comparison-v1.json"
python3 "$repo_root/crates/sonium-compute/benchmarks/check_native_x86_b4.py" \
    "$native_output/sonium_native_x86_evidence.json" \
    "$scalar_output/sonium_native_x86_evidence.json" \
    --quad4 "$quad4_output/sonium_bem_native_x86_b4_quad.json" \
    --output "$comparison"
(cd "$output_root" && sha256sum \
    native/sonium_native_x86_evidence.json \
    native/sonium_native_x86_evidence.json.sha256 \
    scalar/sonium_native_x86_evidence.json \
    scalar/sonium_native_x86_evidence.json.sha256 \
    quad4/sonium_bem_native_x86_b4_quad.json \
    quad4/sonium_bem_native_x86_b4_quad.json.sha256 \
    native-vs-scalar-comparison-v1.json > checksums.sha256 && \
    sha256sum -c checksums.sha256)
