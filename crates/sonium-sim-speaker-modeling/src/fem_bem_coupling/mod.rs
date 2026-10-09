//! 3D interior-FEM/exterior-BEM coupling for cabinets and ports.
//!
//! The exterior BEM problem is condensed to a Dirichlet-to-Neumann (DtN)
//! map on the aperture elements, then the interior 3D Helmholtz FEM is
//! solved with that nonlocal boundary operator. This mirrors the C1
//! condensation pattern of [`crate::resolved_coupling`] with a distributed
//! (per-aperture-element) interface instead of a scalar piston load.
//!
//! # Phasor convention
//!
//! All three ingredients share one convention, so **no conjugation is
//! needed inside the coupling**:
//!
//! - `sonium-bem` TBEM uses `harmonic_factor = +1` (Green's function
//!   `exp(+ikr)`), prescribed outward normal velocity enters as `+iωρ vₙ`;
//! - `sonium-wave` analytical solutions use the same `exp(+ikr)` convention
//!   ("the outward normal velocity satisfies ∂p/∂n = iωρ₀v₀",
//!   `solutions_3d.rs`), which is what the BEM Mie/scattering tests
//!   validate against;
//! - `sonium-fem` acoustics is validated against `sonium-wave` by
//!   `qa-fem-acoustics-sphere-scattering` with the Sommerfeld condition
//!   `∂p/∂n = ik p` on outgoing waves — again the `exp(+ikr)` convention,
//!   with Neumann data `h = iωρ vₙ`.
//!
//! Note that `sonium-accuracy/crates/sonium-qa/src/analytical.rs` electroacoustics helpers
//! (baffled piston, pulsating-sphere impedance) use the conjugate
//! (`exp(−ikr)`) phasor convention: compare against them by conjugating
//! one side (magnitudes are unaffected).
//!
//! # Coupling formulation
//!
//! Aperture element velocities `v_j` are piecewise constant (BEM constant
//! elements align 1:1 with FEM aperture boundary faces). The DtN map
//! `D(ω)` maps element velocities to element-average pressures,
//! `p̄ = D v`. The interior weak form with `∂p/∂n = iωρ vₙ` on the aperture
//! gives the condensed system
//!
//! ```text
//! ( A_fem − iωρ · B · D⁻¹ · N · Bᵀ ) p = f
//! ```
//!
//! with `B_{i,j} = ∫_{face_j} φ_i dΓ` (P1: `area_j / 3` per face vertex)
//! and `N = diag(1/area_j)` mapping nodal pressures to element averages.
//! The dense block touches only aperture DOFs and is materialized into the
//! CSR system; the combined matrix is solved with the existing
//! FGMRES+ILU (`solve_complex`). Passivity requires `Re(v* D v) ≥ 0`,
//! checked by the sign-validation test in [`dtn`].

pub mod coupled;
pub mod dtn;
pub mod fixtures;
pub mod mesh_bridge;
pub mod monolithic;
pub mod rom;

pub use coupled::{
    CoupledSolution, CouplingDrive, CouplingError, DriverVelocityProfile, FemBemCoupling,
};
pub use dtn::{DtnBuilder, DtnError, DtnFactors, DtnMap};
pub use fixtures::{
    APERTURE_MARKER, CouplingFixture, CylinderFixtureDiscretization, DRIVER_MARKER, FixtureError,
    WALL_MARKER, cylindrical_box_with_circular_aperture, cylindrical_box_with_external_port,
};
pub use mesh_bridge::{
    BemInterfaceMesh, MeshBridgeError, PRODUCT_COUPLING_MESH_EVIDENCE_SCHEMA_VERSION,
    ProductCouplingMesh, ProductCouplingMeshError, ProductCouplingMeshEvidence,
    ProductCouplingMeshEvidenceError, ProductInterfaceTrace, build_exterior_bem_mesh,
    build_product_coupling_mesh,
};
pub use monolithic::{
    CompressedMonolithicFemBemOperator, CompressedMonolithicFieldSplitPreconditioner,
    CompressedMonolithicProductOperator, CompressedMonolithicProductSolution,
    CompressedMonolithicProductSweepPoint, CompressedMonolithicSolution,
    DEFAULT_MONOLITHIC_MAX_DOFS, MonolithicCoupledSolution, MonolithicCouplingError,
    MonolithicFemBemCoupling, MonolithicProductCouplingError, MonolithicSealedProductCoupling,
    MonolithicSealedProductSolution,
};
pub use rom::{
    FemBemCouplingRom, RomAdaptiveSolution, RomCoupledSolution, RomCouplingError, RomFidelity,
    RomPromotionDiagnostics, RomPromotionPolicy, RomPromotionReason, RomPromotionReport,
    RomPromotionValidator,
};
