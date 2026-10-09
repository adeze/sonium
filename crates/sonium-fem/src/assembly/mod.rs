//! Finite element matrix assembly
//!
//! # Generic infrastructure (physics-agnostic)
//!
//! - [`assemble_bilinear`] / [`assemble_linear`]: generic assembly from any
//!   [`BilinearForm`](crate::forms::BilinearForm) / [`LinearForm`](crate::forms::LinearForm)
//! - [`assemble_boundary_mass_generic`]: boundary surface mass matrix
//! - [`PrebakedAssembler`]: O(nnz) re-assembly for frequency sweeps
//! - [`TripletMatrix`]: COO sparse matrix accumulation format
//!
//! # Legacy Helmholtz-specific types (deprecated)
//!
//! The following types are retained for backward compatibility. New code
//! should use the generic infrastructure + [`physics::acoustics::pressure`](crate::physics::acoustics::pressure)
//! assembler bridge instead.
//!
//! - `HelmholtzMatrix`, `HelmholtzProblem` → use `PrebakedAssembler<2>` + `AcousticsAssemblerExt`
//! - `StiffnessMatrix`, `MassMatrix` → use `TripletMatrix<f64>`
//! - `assemble_stiffness`, `assemble_mass` → use `assemble_bilinear(&LaplacianForm, ...)` / `assemble_bilinear(&MassForm, ...)`
//! - `assemble_boundary_mass` → use `assemble_boundary_mass_generic`

// --- Generic assembly (physics-agnostic) ---
pub mod dg;
pub mod generic;
pub mod mixed;
pub mod prebaked;
pub mod region;
pub mod triplet;

pub use generic::*;
pub use mixed::*;
pub use prebaked::*;
pub use region::*;
pub use triplet::TripletMatrix;

// --- Legacy (Helmholtz-specific, retained for backward compatibility) ---
mod assembler;
mod helmholtz;
mod mass;
mod stiffness;

// Re-export legacy types at crate level for backward compatibility.
// These will be removed in a future version.
pub use assembler::*;
pub use helmholtz::*;
pub use mass::*;
pub use stiffness::*;
