//! Piezoelectric transducer physics
//!
//! Coupled electromechanical formulation for piezoelectric materials
//! (PZT ceramics, PVDF films, balanced armature drivers, MEMS).
//!
//! The constitutive law in stress-charge form:
//! ```text
//! T = cᴱ·S − eᵗ·E       (stress = stiffness×strain − piezo×field)
//! D = e·S  + εˢ·E        (charge = piezo×strain + permittivity×field)
//! ```
//!
//! This produces a symmetric 2×2 block FEM system coupling displacement u
//! and electric potential φ.

pub mod forms;
pub mod materials;

pub use forms::*;
pub use materials::*;
