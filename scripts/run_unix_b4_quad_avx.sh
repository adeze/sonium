#!/usr/bin/env bash
# Reproducible native-x86 AVX2/FMA parity evidence for the production Quad4
# separated regular Galerkin cell.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-b4-quad-avx-fresh}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/.target-b4-quad-avx-fresh/evidence}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_dir" in /*) ;; *) output_dir="$repo_root/$output_dir" ;; esac
runtime_dir="$target_dir/tmp"

test "$(uname -s)" = Linux || {
    echo "B4 Quad4 AVX evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "B4 Quad4 AVX evidence requires x86_64" >&2
    exit 2
}
grep -qw avx2 /proc/cpuinfo || {
    echo "B4 Quad4 AVX evidence requires AVX2" >&2
    exit 2
}
grep -qw fma /proc/cpuinfo || {
    echo "B4 Quad4 AVX evidence requires FMA" >&2
    exit 2
}

mkdir -p "$output_dir" "$runtime_dir"
rustc -Vv > "$output_dir/rustc-vv.txt"
cargo -V > "$output_dir/cargo-version.txt"
lscpu > "$output_dir/lscpu.txt"

artifact="$output_dir/sonium_bem_native_x86_b4_quad_avx.json"
summary="$output_dir/sonium_bem_native_x86_b4_quad_avx-v1.json"
log="$output_dir/b4-quad-avx-run.log"
rm -f "$artifact" "$artifact.sha256" "$summary" "$summary.sha256" "$log" \
  "$output_dir/checksums.sha256"

TMPDIR="$runtime_dir" TMP="$runtime_dir" TEMP="$runtime_dir" XDG_RUNTIME_DIR="$runtime_dir" \
SONIUM_OUTPUT_DIR="$output_dir" CARGO_TARGET_DIR="$target_dir" \
  cargo test --release -p sonium-bem --test native_x86_b4_quad_avx_evidence -- \
    --nocapture --test-threads=1 > "$log" 2>&1

(
    cd "$output_dir"
    sha256sum "$(basename "$artifact")" > "$(basename "$artifact").sha256"
    sha256sum -c "$(basename "$artifact").sha256"
)
python3 "$repo_root/crates/sonium-compute/benchmarks/check_native_x86_b4_quad_avx.py" \
  "$artifact" --output "$summary"
sha256sum "$summary" > "$summary.sha256"
sha256sum -c "$summary.sha256"

(
    cd "$output_dir"
    sha256sum \
      rustc-vv.txt cargo-version.txt lscpu.txt \
      "$(basename "$artifact")" "$(basename "$artifact").sha256" \
      "$(basename "$summary")" "$(basename "$summary").sha256" \
      "$(basename "$log")" > checksums.sha256
    sha256sum -c checksums.sha256
)

echo "B4 native Quad4 AVX2/FMA evidence: passed; bundle=$output_dir"
