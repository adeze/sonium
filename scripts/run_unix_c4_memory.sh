#!/usr/bin/env bash
# Deterministic C4 MemoryPool quota/contention evidence.
#
# The default `cpu` mode is host-only. Set SONIUM_C4_MEMORY_MODE=cuda to run
# the genuine single-device CUDA producer; that mode is intentionally separate
# so the caller can serialize it with other GPU work. `both` runs both lanes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-c4-memory}"
output_dir="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-c4-memory}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_dir" in /*) ;; *) output_dir="$repo_root/$output_dir" ;; esac
mode="${SONIUM_C4_MEMORY_MODE:-cpu}"
case "$mode" in
    cpu|cuda|both) ;;
    *)
        echo "SONIUM_C4_MEMORY_MODE must be cpu, cuda, or both" >&2
        exit 2
        ;;
esac

test "$(uname -s)" = Linux || {
    echo "C4 memory evidence requires Linux" >&2
    exit 2
}
test "$(uname -m)" = x86_64 || {
    echo "C4 memory evidence requires x86_64" >&2
    exit 2
}

mkdir -p "$output_dir"
rustc -Vv > "$output_dir/rustc-vv.txt"
cargo -V > "$output_dir/cargo-version.txt"
lscpu > "$output_dir/lscpu.txt"

if [[ "$mode" == cpu || "$mode" == both ]]; then
    rm -f "$output_dir/c4-memory-quota-contention.json" \
      "$output_dir/c4-memory-quota-contention.json.sha256" \
      "$output_dir/c4-memory-quota-contention-v1.json" \
      "$output_dir/c4-memory-quota-contention-v1.json.sha256" \
      "$output_dir/c4-memory-run.log"
    SONIUM_OUTPUT_DIR="$output_dir" \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release --no-default-features --features cpu \
      -p sonium-compute --test memory_quota_evidence -- \
      --nocapture --test-threads=1 > "$output_dir/c4-memory-run.log" 2>&1
    (
        cd "$output_dir"
        sha256sum c4-memory-quota-contention.json > c4-memory-quota-contention.json.sha256
        sha256sum -c c4-memory-quota-contention.json.sha256
    )
    summary="$output_dir/c4-memory-quota-contention-v1.json"
    python3 "$repo_root/crates/sonium-compute/benchmarks/check_c4_memory_quota.py" \
      "$output_dir/c4-memory-quota-contention.json" --output "$summary"
    sha256sum "$summary" > "$summary.sha256"
    sha256sum -c "$summary.sha256"
fi

if [[ "$mode" == cuda || "$mode" == both ]]; then
    cuda_output="$output_dir/cuda"
    mkdir -p "$cuda_output"
    rm -f "$cuda_output/c4-cuda-memory-quota.json" \
      "$cuda_output/c4-cuda-memory-quota.json.sha256" \
      "$cuda_output/c4-cuda-memory-quota-v1.json" \
      "$cuda_output/c4-cuda-memory-quota-v1.json.sha256" \
      "$cuda_output/c4-cuda-memory-run.log"
    SONIUM_REQUIRE_CUDA=1 \
    SONIUM_OUTPUT_DIR="$cuda_output" \
    CARGO_TARGET_DIR="$target_dir" \
    cargo test --release -p sonium-compute --features cuda \
      --test cuda_memory_quota_evidence -- \
      --nocapture --test-threads=1 > "$cuda_output/c4-cuda-memory-run.log" 2>&1
    (
        cd "$cuda_output"
        sha256sum c4-cuda-memory-quota.json > c4-cuda-memory-quota.json.sha256
        sha256sum -c c4-cuda-memory-quota.json.sha256
    )
    cuda_summary="$cuda_output/c4-cuda-memory-quota-v1.json"
    python3 "$repo_root/crates/sonium-compute/benchmarks/check_c4_memory_quota.py" \
      "$cuda_output/c4-cuda-memory-quota.json" --output "$cuda_summary"
    sha256sum "$cuda_summary" > "$cuda_summary.sha256"
    sha256sum -c "$cuda_summary.sha256"
fi

(cd "$output_dir" && {
    files=(rustc-vv.txt cargo-version.txt lscpu.txt)
    if [[ "$mode" == cpu || "$mode" == both ]]; then
        files+=(
          c4-memory-quota-contention.json c4-memory-quota-contention.json.sha256
          c4-memory-quota-contention-v1.json c4-memory-quota-contention-v1.json.sha256
          c4-memory-run.log
        )
    fi
    if [[ "$mode" == cuda || "$mode" == both ]]; then
        files+=(
          cuda/c4-cuda-memory-quota.json cuda/c4-cuda-memory-quota.json.sha256
          cuda/c4-cuda-memory-quota-v1.json cuda/c4-cuda-memory-quota-v1.json.sha256
          cuda/c4-cuda-memory-run.log
        )
    fi
    sha256sum "${files[@]}" > checksums.sha256
    sha256sum -c checksums.sha256
})

echo "C4 memory quota evidence: mode=$mode passed; bundle=$output_dir"
