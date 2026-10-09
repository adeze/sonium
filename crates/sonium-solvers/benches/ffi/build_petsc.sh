#!/usr/bin/env bash
# Build the PETSc reference solver for benchmarking.
#
# Prerequisites:
#   - PETSc 3.20+ installed
#   - export PETSC_DIR=/path/to/petsc
#   - export PETSC_ARCH=arch-name (e.g., arch-darwin-c-opt)
#
# For HYPRE AMG support, PETSc must be configured with:
#   ./configure --download-hypre
#
# Usage:
#   cd crates/sonium-solvers/benches/ffi
#   ./build_petsc.sh

set -euo pipefail

if [ -z "${PETSC_DIR:-}" ]; then
    echo "Error: PETSC_DIR is not set."
    echo "  export PETSC_DIR=/path/to/petsc"
    exit 1
fi

if [ -z "${PETSC_ARCH:-}" ]; then
    echo "Error: PETSC_ARCH is not set."
    echo "  export PETSC_ARCH=arch-name (e.g., arch-darwin-c-opt)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PETSC_CONF="${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf"

if [ ! -f "${PETSC_CONF}/petscvariables" ]; then
    echo "Error: Cannot find ${PETSC_CONF}/petscvariables"
    echo "  Is PETSC_DIR/PETSC_ARCH correct?"
    exit 1
fi

# Homebrew and several package managers emit `petscvariables` as a Makefile
# fragment (sometimes one long line), which cannot be sourced by a shell.
# Prefer the pkg-config contract and retain the older shell-variable fallback
# for PETSc builds that still export a shell-compatible configuration.
PKG_CONFIG_PATH="${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PKG_CONFIG_PATH
PKG_CONFIG_BIN="${PKG_CONFIG:-pkg-config}"
if { [ -x "${PKG_CONFIG_BIN}" ] || command -v "${PKG_CONFIG_BIN}" >/dev/null 2>&1; } \
    && "${PKG_CONFIG_BIN}" --exists PETSc; then
    CC="${PCC:-mpicc}"
    CFLAGS="-O2 $("${PKG_CONFIG_BIN}" --cflags PETSc)"
    LDFLAGS="$("${PKG_CONFIG_BIN}" --libs PETSc)"
else
    # shellcheck disable=SC1090
    source "${PETSC_CONF}/petscvariables"
    CC="${PCC:-mpicc}"
    CFLAGS="-O2 ${PETSC_CC_INCLUDES}"
    LDFLAGS="${PETSC_LIB}"
fi

echo "Building petsc_reference..."
echo "  CC:     ${CC}"
echo "  PETSC:  ${PETSC_DIR}/${PETSC_ARCH}"

${CC} ${CFLAGS} -o "${SCRIPT_DIR}/petsc_reference" "${SCRIPT_DIR}/petsc_reference.c" ${LDFLAGS}

echo "Built: ${SCRIPT_DIR}/petsc_reference"
echo ""
echo "Usage:"
echo "  ./petsc_reference matrix.mtx [solver] [precond] [tol] [max_iter]"
echo ""
echo "Examples:"
echo "  ./petsc_reference test.mtx gmres none 1e-6 500"
echo "  ./petsc_reference test.mtx gmres ilu 1e-6 500"
echo "  ./petsc_reference test.mtx fgmres amg 1e-6 500"
echo "  ./petsc_reference test.mtx cg jacobi 1e-6 500"
