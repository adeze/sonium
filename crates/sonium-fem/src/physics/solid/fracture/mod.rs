//! Phase-field fracture and scalar damage forms.
//!
//! # Regularization notes (Phase 3 lane 3D plan item)
//!
//! Mesh-size dependence by model, as implemented:
//!
//! - [`fail_tab`] carries the OpenCourant element-size regularization:
//!   the `fct_IDel` curve scales the failure strain (`fel` in eq
//!   952) inside the triaxiality window (953), so calibrated
//!   decks reproduce the vendor size effect. Triaxiality ends
//!   clamp through the frozen `Funct1D` reader (OpenCourant default).
//! - [`wierzbicki`] and the [`ductile`] Johnson-Cook/GTN hooks
//!   have no length scale: they are local damage models, so the
//!   localization width follows the mesh and refinement changes
//!   the post-peak response. Any nonlocal or gradient
//!   regularization for these paths is future work.
//! - [`deletion`] policies are topological (`Ifail_so`,
//!   `Ifail_sh`, `P_thickfail` ratio); eroded elements move
//!   their plastic/internal energy to the eroded ledger column,
//!   so erosion never leaks energy (total = active + eroded).
//!   Exponential softening (`Fad_exp`) stays deferred.

mod assembler;
pub mod cohesive;
pub mod coupled;
pub mod deletion;
pub mod ductile;
pub mod fail_tab;
mod forms;
mod materials;
mod split;
pub mod wierzbicki;

pub use assembler::*;
pub use forms::*;
pub use materials::*;
pub use split::*;
