#!/usr/bin/env -S cargo +nightly -Zscript
//! Compare sonium-solvers against PETSc on identical matrices.
//!
//! This is a standalone script that:
//! 1. Generates test matrices and writes them as MatrixMarket files
//! 2. Runs our solvers and records timing as JSON
//! 3. Optionally runs PETSc (if petsc_reference binary exists) on the same matrices
//! 4. Prints a comparison table
//!
//! Usage:
//!   cargo run --release --example compare_petsc -p sonium-solvers
//!
//! Or run from the benches/ffi directory after building.

// This file is meant to be copied to examples/ or run manually.
// It is kept here for reference alongside the PETSc C harness.

fn main() {
    eprintln!("This is a reference script. See the actual runner below.");
    eprintln!("Copy to examples/compare_petsc.rs or use the shell runner.");
}
