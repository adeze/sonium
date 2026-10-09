#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: scripts/iterative_solver_perf.sh [OUTPUT.json] [sonium-benchmark options...]

Runs the release-mode iterative solver benchmark matrix. The first positional
argument selects the JSON report path; remaining arguments are passed directly
to sonium-benchmark.

Environment:
  ITERATIVE_PERF_SIZES                 DOF sizes (default: 1000,10000,100000)
  ITERATIVE_PERF_FILTER                solver/preconditioner/matrix filter
  ITERATIVE_PERF_RHS_COUNT             solves per setup (default: 1)
  ITERATIVE_PERF_QUICK                 set to 1 for the smallest size only
  ITERATIVE_PERF_VERBOSE               set to 0 to hide extra progress details
  ITERATIVE_PERF_PETSC                 optional petsc_reference binary
  ITERATIVE_PERF_PLATFORM_NAME         optional platform label
  ITERATIVE_PERF_REGRESSION_BASELINE   optional previous JSON report
  ITERATIVE_PERF_MAX_TOTAL_SLOWDOWN    fraction (default: 0.10)
  ITERATIVE_PERF_MAX_SOLVE_SLOWDOWN    fraction (default: 0.10)
  ITERATIVE_PERF_MAX_ITERATION_REGRESSION
                                      fraction (default: 0.10)

Filter examples:
  gmres//               all GMRES cases
  cg/amg/poisson        AMG-CG on Poisson matrices
  '*/*/laplacian_3d'    all 3D Laplacian cases
EOF
    exit 0
fi

output="${1:-${repo_root}/ITERATIVE_PERF.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

cargo_bin="${CARGO:-cargo}"
sizes="${ITERATIVE_PERF_SIZES:-1000,10000,100000}"
filter="${ITERATIVE_PERF_FILTER:-}"
rhs_count="${ITERATIVE_PERF_RHS_COUNT:-1}"
quick="${ITERATIVE_PERF_QUICK:-0}"
verbose="${ITERATIVE_PERF_VERBOSE:-1}"
petsc="${ITERATIVE_PERF_PETSC:-}"
platform_name="${ITERATIVE_PERF_PLATFORM_NAME:-}"
regression_baseline="${ITERATIVE_PERF_REGRESSION_BASELINE:-}"
max_total_slowdown="${ITERATIVE_PERF_MAX_TOTAL_SLOWDOWN:-}"
max_solve_slowdown="${ITERATIVE_PERF_MAX_SOLVE_SLOWDOWN:-}"
max_iteration_regression="${ITERATIVE_PERF_MAX_ITERATION_REGRESSION:-}"

benchmark_args=(
    --output "${output}"
    --sizes "${sizes}"
    --rhs-count "${rhs_count}"
)

if [[ -n "${filter}" ]]; then
    benchmark_args+=(--filter "${filter}")
fi
if [[ "${quick}" == "1" ]]; then
    benchmark_args+=(--quick)
fi
if [[ "${verbose}" != "0" ]]; then
    benchmark_args+=(--verbose)
fi
if [[ -n "${petsc}" ]]; then
    benchmark_args+=(--petsc "${petsc}")
fi
if [[ -n "${platform_name}" ]]; then
    benchmark_args+=(--platform-name "${platform_name}")
fi
if [[ -n "${regression_baseline}" ]]; then
    benchmark_args+=(--regression-baseline "${regression_baseline}")
fi
if [[ -n "${max_total_slowdown}" ]]; then
    benchmark_args+=(--max-total-slowdown "${max_total_slowdown}")
fi
if [[ -n "${max_solve_slowdown}" ]]; then
    benchmark_args+=(--max-solve-slowdown "${max_solve_slowdown}")
fi
if [[ -n "${max_iteration_regression}" ]]; then
    benchmark_args+=(--max-iteration-regression "${max_iteration_regression}")
fi

cd "${repo_root}"
"${cargo_bin}" run --release -p sonium-solvers \
    --features cli \
    --bin sonium-benchmark -- \
    "${benchmark_args[@]}" \
    "$@"
