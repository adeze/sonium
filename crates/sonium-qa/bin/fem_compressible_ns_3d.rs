//! Compressible Navier-Stokes uniform-state smoke gate (3D).
//!
//! Thin wrapper over [`sonium_qa::fem_physics_3d::run_compressible_case`]. Scope
//! is deliberately limited to uniform-flow invariants: free-stream residual on
//! triangles/tetrahedra, one SSPRK3 step that preserves positive density and
//! pressure, and the Sutherland transport reference value. It solves no
//! nonuniform viscous flow and measures no spatial or temporal accuracy.
//!
//! Do not cite this binary as compressible-flow accuracy evidence; that
//! requires a nonuniform manufactured or benchmark solution with
//! spatial/time convergence, which this gate does not run.
fn main() {
    sonium_qa::fem_physics_3d::run_compressible_case();
}
