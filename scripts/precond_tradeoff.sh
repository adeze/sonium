#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
output="${1:-${repo_root}/crates/sonium-solvers/PRECOND_TRADEOFF.md}"
cargo_bin="${CARGO:-cargo}"
repeats="${PRECOND_TRADEOFF_REPEATS:-3}"
matrices="${PRECOND_TRADEOFF_MATRICES:-all}"
preconds="${PRECOND_TRADEOFF_PRECONDS:-all}"

cd "${repo_root}"
"${cargo_bin}" run --release -p sonium-solvers \
    --features cli \
    --bin precond-tradeoff -- \
    --output "${output}" \
    --repeats "${repeats}" \
    --matrices "${matrices}" \
    --precond "${preconds}"
