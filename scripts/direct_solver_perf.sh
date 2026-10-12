#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
output="${1:-${repo_root}/perf/PERF.md}"
cargo_bin="${CARGO:-cargo}"
dense_sizes="${DIRECT_PERF_DENSE_SIZES:-64,128,256,512}"
sparse_grids="${DIRECT_PERF_SPARSE_GRIDS:-8,12,16,20,28}"
repeats="${DIRECT_PERF_REPEATS:-3}"

cd "${repo_root}"
"${cargo_bin}" run --release -p sonium-solvers \
    --features native-direct-perf \
    --bin direct-solver-perf -- \
    --output "${output}" \
    --dense-sizes "${dense_sizes}" \
    --sparse-grids "${sparse_grids}" \
    --repeats "${repeats}"
