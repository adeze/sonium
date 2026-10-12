#!/usr/bin/env bash
# Reproducible native Linux SS3 NUMA policy/admission evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${CARGO_TARGET_DIR:-${repo_root}/.target-unix-ss3}"
output_root="${SONIUM_OUTPUT_DIR:-${repo_root}/target/sonium-unix-ss3}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
case "$output_root" in /*) ;; *) output_root="$repo_root/$output_root" ;; esac

test "$(uname -s)" = Linux || { echo "SS3 requires Linux" >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo "SS3 requires x86_64" >&2; exit 2; }
mkdir -p "$output_root"
rm -f "$output_root/solvers_numa_partitioning_report.json" \
  "$output_root/solvers_numa_partitioning_report.json.sha256" \
  "$output_root/ss3-summary.json" "$output_root/checksums.sha256"

rustc -Vv > "$output_root/rustc-vv.txt"
cargo -V > "$output_root/cargo-version.txt"
lscpu > "$output_root/lscpu.txt"
if command -v numactl >/dev/null 2>&1; then numactl --hardware > "$output_root/numactl-hardware.txt"; else echo "numactl unavailable" > "$output_root/numactl-hardware.txt"; fi
if command -v numastat >/dev/null 2>&1; then numastat -m > "$output_root/numastat-memory.txt"; else echo "numastat unavailable" > "$output_root/numastat-memory.txt"; fi

SONIUM_OUTPUT_DIR="$output_root" CARGO_TARGET_DIR="$target_dir" \
  cargo run --release -p sonium-qa --features qa-binaries --bin qa-solvers-numa-partitioning-report \
  > "$output_root/producer.log" 2>&1

(cd "$output_root" && sha256sum solvers_numa_partitioning_report.json > solvers_numa_partitioning_report.json.sha256)
python3 "$repo_root/scripts/check_unix_ss3_x86.py" \
  "$output_root/solvers_numa_partitioning_report.json" \
  --output "$output_root/ss3-summary.json"
(cd "$output_root" && sha256sum \
  solvers_numa_partitioning_report.json solvers_numa_partitioning_report.json.sha256 \
  ss3-summary.json rustc-vv.txt cargo-version.txt lscpu.txt numactl-hardware.txt \
  numastat-memory.txt producer.log > checksums.sha256 && sha256sum -c checksums.sha256)
echo "SS3 native x86 NUMA policy evidence: passed; bundle=$output_root"
