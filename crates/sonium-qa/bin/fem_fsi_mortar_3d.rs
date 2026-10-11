//! Nonmatching FSI mortar operator contract check (3D smoke gate).
//!
//! Thin wrapper over [`sonium_qa::fem_physics_3d::run_fsi_case`]. Scope is
//! deliberately limited to the dual-P1 mortar interface contract: a constant
//! trace patch residual, the exact transpose virtual-work identity, and the
//! rebuild trigger on a perturbed interface. It integrates no coupled
//! fluid/solid benchmark and measures no FSI solution accuracy.
//!
//! Do not cite this binary as coupled-FSI accuracy evidence; that requires a
//! dynamic fluid/solid benchmark comparison, which this gate does not run.
fn main() {
    sonium_qa::fem_physics_3d::run_fsi_case();
}
