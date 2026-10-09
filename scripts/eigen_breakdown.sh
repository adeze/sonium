#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
output="${1:-${repo_root}/crates/sonium-solvers/EIGEN_BREAKDOWN.md}"
cargo_bin="${CARGO:-cargo}"
grid="${EIGEN_BREAKDOWN_GRID:-60}"
modes="${EIGEN_BREAKDOWN_MODES:-2,6,12}"
paddings="${EIGEN_BREAKDOWN_PADDINGS:-0,4}"
shifts="${EIGEN_BREAKDOWN_SHIFTS:-0.5,4.0,7.0}"

cd "${repo_root}"
"${cargo_bin}" run --release -p sonium-solvers \
    --features cli \
    --bin eigen-breakdown -- \
    --output "${output}" \
    --grid "${grid}" \
    --modes "${modes}" \
    --paddings "${paddings}" \
    --shifts "${shifts}"
