//! Mixed biharmonic operator assembly-contract check (3D smoke gate).
//!
//! Thin wrapper over [`sonium_qa::fem_physics_3d::run_biharmonic_case`]. Scope is
//! deliberately limited to the mixed P1 `q = -Laplacian(u)` split assembly
//! contract: nonzero 2D/3D operator blocks, the forbidden lower-left block
//! stays empty, and total DOFs equal twice the node count. It solves no plate
//! problem and measures no discretization accuracy.
//!
//! Plate accuracy with a manufactured solution and refinement lives in
//! `qa-fem-biharmonic-mms-3d`; do not cite this binary as biharmonic accuracy
//! evidence.
fn main() {
    sonium_qa::fem_physics_3d::run_biharmonic_case();
}
