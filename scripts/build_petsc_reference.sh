#!/usr/bin/env bash
# Build the PETSc KSP reference binary used by bench-petsc-run.
#
# Linux (Debian/Ubuntu): downloads libpetsc dev headers with apt (no sudo,
# extracted to $HOME/petsc-dev) and compiles against the system libpetsc.
# Output: target/petsc_reference
#
# macOS: install PETSc with Hypre support, export PETSC_DIR/PETSC_ARCH,
# then run crates/sonium-solvers/benches/ffi/build_petsc.sh directly.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
out="$repo_root/target/petsc_reference"

if [ "$(uname -s)" != "Linux" ]; then
    echo "Linux-only script. On macOS, set PETSC_DIR/PETSC_ARCH and run"
    echo "crates/sonium-solvers/benches/ffi/build_petsc.sh instead."
    exit 2
fi

dev_root="$HOME/petsc-dev"
inc="$dev_root/root/usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-real/include"
if [ ! -f "$inc/petscksp.h" ]; then
    echo "Downloading PETSc dev headers to $dev_root ..."
    mkdir -p "$dev_root"
    (cd "$dev_root" && apt-get download libpetsc-real3.19-dev libpetsc3.19-dev-common)
    for deb in "$dev_root"/*.deb; do
        dpkg-deb -x "$deb" "$dev_root/root"
    done
fi

mkdir -p "$repo_root/target"
mpicc -O2 -o "$out" \
    "$repo_root/crates/sonium-solvers/benches/ffi/petsc_reference.c" \
    -I"$inc" -L/usr/lib/x86_64-linux-gnu -lpetsc \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu
echo "built $out"
"$out" 2>&1 | head -n 1 || true
