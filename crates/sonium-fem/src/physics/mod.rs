//! Physics-specific FEM implementations
//!
//! Each sub-module implements the generic [`forms`](crate::forms) traits
//! for a specific physics domain. Physics modules are **orthogonal** to the
//! assembly, boundary, and solver infrastructure: adding a new physics
//! domain requires zero changes to the infrastructure layers.
//!
//! # Architecture
//!
//! ```text
//! Layer 0: Geometry    mesh/, basis/, quadrature/
//! Layer 1: Forms       forms/traits.rs  (BilinearForm<T>, LinearForm<T>)
//! Layer 2: Assembly    assembly/generic.rs, prebaked.rs  (physics-agnostic)
//! Layer 3: BCs         boundary/generic.rs  (FemSystem<T>)
//! Layer 4: Solver      solver/  (iterative/direct solvers)
//! Layer 5: Physics     physics/<group>/*/forms.rs  (THIS LAYER)
//! ```
//!
//! # Adding a new physics module
//!
//! 1. **Define forms** — Implement [`BilinearForm<T>`](crate::forms::BilinearForm)
//!    and/or [`LinearForm<T>`](crate::forms::LinearForm) for your weak form.
//!    The form struct stores material parameters; `integrate_point()` computes
//!    the element-level contribution at a quadrature point. This is the only
//!    required step.
//!
//! 2. **Define materials** — Structs holding physical properties (Young's modulus,
//!    permittivity, conductivity, etc.). Embedded in the form struct.
//!
//! 3. **(Optional) Assembler bridge** — For efficient frequency sweeps or
//!    multi-component assembly, provide a `build_*_assembler()` function that:
//!    - Calls `assembly::assemble_bilinear()` with your forms
//!    - Packages results into `PrebakedAssembler<N>`
//!    - Provides an extension trait with domain-specific assembly methods
//!
//!    See `acoustics/assembler.rs` for the canonical example.
//!
//! # Example: minimal physics module
//!
//! ```ignore
//! // physics/solid/my_domain/forms.rs
//! use crate::forms::{BilinearForm, LocalMatrix};
//! use crate::spaces::BasisTabulation;
//!
//! pub struct MyStiffnessForm {
//!     pub coefficient: f64,
//! }
//!
//! impl BilinearForm<f64> for MyStiffnessForm {
//!     fn integrate_point(
//!         &self, test: &BasisTabulation, trial: &BasisTabulation,
//!         weight: f64, local_matrix: &mut LocalMatrix<f64>,
//!     ) {
//!         let w = weight * test.det_j;
//!         for i in 0..test.n_dofs {
//!             for j in 0..trial.n_dofs {
//!                 let grad_dot = (0..test.spatial_dim)
//!                     .map(|d| test.grad(i, d) * trial.grad(j, d))
//!                     .sum::<f64>();
//!                 local_matrix.add(i, j, self.coefficient * grad_dot * w);
//!             }
//!         }
//!     }
//! }
//!
//! // Usage:
//! let form = MyStiffnessForm { coefficient: 1.5 };
//! let matrix = assembly::assemble_bilinear(&form, &space, &space, &mesh, 3);
//! ```

pub mod acoustics;
pub mod coupled;
pub mod electromagnetics;
pub mod fluid;
pub mod solid;
pub mod thermal;
