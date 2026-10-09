//! Modal reduced-order fast path for the coupled FEM–BEM sweep.
//!
//! # Basis: rigid-wall modes + Craig–Bampton attachment modes
//!
//! The interior FEM block of the coupled system
//! `(A_fem − iωρ·B·D⁻¹·N·Bᵀ) p = f` (see [`super::coupled`]) is reduced with
//! an augmented basis `Φ̂ = [Φ | A]`:
//!
//! - `Φ`: the rigid-wall modal basis of
//!   [`crate::acoustic_rom::AcousticRom`],
//! - `A`: one **attachment mode** per aperture face — the shifted-static
//!   response `(K + σ_c M)⁻¹ b_j` of the face's unit Neumann data `b_j`,
//!   minus its modal projection, M-orthonormalized
//!   ([`crate::acoustic_rom::AcousticRom::attachment_modes`]).
//!
//! The attachment modes are essential, not optional polish: the coupled
//! solution below the first cavity mode is quasi-static in the box with a
//! pressure drop along the port tube, and that aperture boundary layer
//! converges at a slow 1/n rate in rigid-wall modes (measured: ~40 % field
//! error with 16 modes on the QA port fixture — wrong answer, not slow
//! convergence). Putting the static aperture-flux responses in the trial
//! space makes the quasi-static limit exact by construction.
//!
//! # Reduced system
//!
//! With `p = Φ̂ q̂` and the real projected aperture incidence
//! `Ĉ = Bᵀ Φ̂`, the reduced coupled system is
//!
//! ```text
//! ( Φ̂ᵀ K Φ̂ − k² I − iωρ · Ĉᵀ · D⁻¹ · N · Ĉ ) q̂ = Φ̂ᵀ f
//! ```
//!
//! (`Φ̂` is M-orthonormal). The stiffness block `Φ̂ᵀ K Φ̂` is precomputed by
//! matvecs at build time; per frequency the `D⁻¹` term and an
//! `(m + a) × (m + a)` dense LU remain. Aperture element velocities are
//! recovered exactly as in the full solve: `v = D⁻¹ · p̄` from the element
//! averages of the reconstructed pressure.
//!
//! # Static tail correction
//!
//! [`FemBemCouplingRom::with_static_correction`] precomputes the two-term
//! static correction for the bound drive's (ω-independent) RHS pattern —
//! two shifted-static solves — with the in-basis part subtracted against the
//! *augmented* basis: `z0 = w0 − Φ̂(K̂K + σI)⁻¹ Φ̂ᵀ f`,
//! `z1 = w1 − Φ̂(K̂K + σI)⁻¹ Φ̂ᵀ M w0`, so nothing is double-counted. It
//! fixes the modal tail at the *driven* face; the aperture layer is already
//! in the basis via the attachment modes.
//!
//! # What the ROM does and does not accelerate
//!
//! The per-frequency BEM cost (TBEM assembly + LU + the `m_ap` DtN
//! back-solves in [`super::dtn::DtnBuilder::factor`]) is *unchanged* — the
//! ROM only removes the interior FEM assembly and solve. For coarse
//! apertures the DtN condensation dominates, so the coupled-ROM speedup is
//! smaller than the sealed-cavity one; it grows with interior mesh size.
//! Build cost: one eigensolve + `m_ap + 2` shifted-static solves.

use super::coupled::{CoupledSolution, CouplingDrive, CouplingError, FemBemCoupling};
use super::dtn::DtnError;
use crate::acoustic_rom::{AcousticRom, AcousticRomConfig, RomError};
use crate::conventions::ArtifactProvenance;
use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_solvers::LinearSolveConfig;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use std::collections::HashMap;

/// Errors raised by the coupled reduced-order path.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RomCouplingError {
    #[error("modal basis: {0}")]
    Rom(#[from] RomError),
    #[error("coupling: {0}")]
    Coupling(#[from] CouplingError),
    #[error("DtN condensation failed: {0}")]
    Dtn(#[from] DtnError),
    #[error("ROM promotion policy is invalid: {0}")]
    InvalidPromotionPolicy(&'static str),
    #[error("ROM promotion frequency grid must be finite, positive, and strictly increasing")]
    InvalidPromotionFrequencyGrid,
    #[error("ROM promotion diagnostics are invalid: {0}")]
    InvalidPromotionDiagnostics(&'static str),
    #[error(
        "ROM promotion gate {reason:?} at {frequency_hz:.3} Hz (residual {residual:.3e}, limit {limit:.3e}); full-order promotion budget {budget} is exhausted"
    )]
    PromotionBudgetExceeded {
        frequency_hz: f64,
        residual: f64,
        limit: f64,
        budget: usize,
        reason: RomPromotionReason,
    },
}

/// Solution of the coupled ROM at one frequency.
#[derive(Debug, Clone)]
pub struct RomCoupledSolution {
    /// The coupled solution itself (pressure + aperture velocity).
    /// `solver_iterations` is 0 — the reduced system is solved by dense LU.
    pub solution: CoupledSolution,
    /// Number of retained interior modes.
    pub num_modes: usize,
    /// Number of aperture attachment modes.
    pub num_attachment_modes: usize,
    /// Full-space relative residual `‖A_coupled p − f‖/‖f‖` when requested:
    /// the exact opt-in spot-check (assembles the full coupled operator).
    pub relative_residual: Option<f64>,
}

/// Fidelity selected for one point in an adaptive ROM sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RomFidelity {
    /// The reduced solution passed its configured residual gate.
    Reduced,
    /// The reduced solution exceeded its gate and was replaced by a full solve.
    FullOrderPromoted,
}

/// Typed agreement evidence supplied by a coupled high-fidelity validator.
///
/// `rank_agreement` is normalized to `[0, 1]`, where one means that the
/// reduced and reference active subspaces agree. The other two values are
/// non-negative relative errors/violations. Keeping these quantities typed at
/// the ROM boundary makes promotion decisions explainable without coupling
/// this module to a particular product objective or constraint set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RomPromotionDiagnostics {
    pub rank_agreement: f64,
    pub gradient_relative_error: f64,
    pub maximum_constraint_violation: f64,
}

impl RomPromotionDiagnostics {
    fn validate(self) -> Result<(), RomCouplingError> {
        if !self.rank_agreement.is_finite() || !(0.0..=1.0).contains(&self.rank_agreement) {
            return Err(RomCouplingError::InvalidPromotionDiagnostics(
                "rank_agreement must be finite and in [0, 1]",
            ));
        }
        if !self.gradient_relative_error.is_finite() || self.gradient_relative_error < 0.0 {
            return Err(RomCouplingError::InvalidPromotionDiagnostics(
                "gradient_relative_error must be finite and non-negative",
            ));
        }
        if !self.maximum_constraint_violation.is_finite() || self.maximum_constraint_violation < 0.0
        {
            return Err(RomCouplingError::InvalidPromotionDiagnostics(
                "maximum_constraint_violation must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Why a reduced point was replaced by a full-order solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RomPromotionReason {
    ResidualExceeded,
    RankAgreementBelowThreshold,
    GradientErrorExceeded,
    ConstraintViolationExceeded,
    DiagnosticsMissing,
}

/// Callback boundary for product-specific ROM validation.
///
/// The validator may compare the reduced state with a reference state or
/// evaluate product objectives/constraints. Returning `None` means that no
/// agreement evidence was available for this point; when any agreement gate
/// is enabled, that absence is conservatively treated as a promotion trigger.
pub trait RomPromotionValidator {
    fn validate(
        &mut self,
        frequency_hz: f64,
        reduced: &RomCoupledSolution,
    ) -> Result<Option<RomPromotionDiagnostics>, RomCouplingError>;
}

impl<F> RomPromotionValidator for F
where
    F: FnMut(f64, &RomCoupledSolution) -> Result<Option<RomPromotionDiagnostics>, RomCouplingError>,
{
    fn validate(
        &mut self,
        frequency_hz: f64,
        reduced: &RomCoupledSolution,
    ) -> Result<Option<RomPromotionDiagnostics>, RomCouplingError> {
        self(frequency_hz, reduced)
    }
}

struct NoopRomPromotionValidator;

impl RomPromotionValidator for NoopRomPromotionValidator {
    fn validate(
        &mut self,
        _frequency_hz: f64,
        _reduced: &RomCoupledSolution,
    ) -> Result<Option<RomPromotionDiagnostics>, RomCouplingError> {
        Ok(None)
    }
}

/// One point returned by [`FemBemCouplingRom::solve_sweep_with_promotion`].
#[derive(Debug, Clone)]
pub struct RomAdaptiveSolution {
    pub solution: CoupledSolution,
    pub fidelity: RomFidelity,
    /// Exact full-space residual measured before any promotion.
    pub reduced_relative_residual: Option<f64>,
    pub promotion_diagnostics: Option<RomPromotionDiagnostics>,
    pub promotion_reason: Option<RomPromotionReason>,
    pub num_modes: usize,
    pub num_attachment_modes: usize,
}

/// Configuration for residual-gated ROM promotion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RomPromotionPolicy {
    /// Check every `n`th frequency. Use one for a gate on every point.
    pub residual_check_every: usize,
    /// Run the validator every `n`th frequency. Use one for a gate on every
    /// point.
    pub agreement_check_every: usize,
    /// Maximum accepted full-space relative residual for the reduced solve.
    pub max_relative_residual: f64,
    /// Minimum accepted reduced/reference active-subspace agreement.
    pub minimum_rank_agreement: Option<f64>,
    /// Maximum accepted relative gradient disagreement.
    pub max_gradient_relative_error: Option<f64>,
    /// Maximum accepted product-constraint violation.
    pub max_constraint_violation: Option<f64>,
    /// Maximum number of reduced points that may be promoted to full order.
    pub max_full_order_solves: usize,
}

impl Default for RomPromotionPolicy {
    fn default() -> Self {
        Self {
            residual_check_every: 1,
            agreement_check_every: 1,
            max_relative_residual: 1.0e-2,
            minimum_rank_agreement: None,
            max_gradient_relative_error: None,
            max_constraint_violation: None,
            max_full_order_solves: 8,
        }
    }
}

impl RomPromotionPolicy {
    fn validate(self) -> Result<(), RomCouplingError> {
        if self.residual_check_every == 0 {
            return Err(RomCouplingError::InvalidPromotionPolicy(
                "residual_check_every must be positive",
            ));
        }
        if self.agreement_check_every == 0 {
            return Err(RomCouplingError::InvalidPromotionPolicy(
                "agreement_check_every must be positive",
            ));
        }
        if !self.max_relative_residual.is_finite() || self.max_relative_residual < 0.0 {
            return Err(RomCouplingError::InvalidPromotionPolicy(
                "max_relative_residual must be finite and non-negative",
            ));
        }
        if let Some(value) = self.minimum_rank_agreement
            && (!value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(RomCouplingError::InvalidPromotionPolicy(
                "minimum_rank_agreement must be finite and in [0, 1]",
            ));
        }
        if let Some(value) = self.max_gradient_relative_error
            && (!value.is_finite() || value < 0.0)
        {
            return Err(RomCouplingError::InvalidPromotionPolicy(
                "max_gradient_relative_error must be finite and non-negative",
            ));
        }
        if let Some(value) = self.max_constraint_violation
            && (!value.is_finite() || value < 0.0)
        {
            return Err(RomCouplingError::InvalidPromotionPolicy(
                "max_constraint_violation must be finite and non-negative",
            ));
        }
        Ok(())
    }

    fn agreement_gate_enabled(self) -> bool {
        self.minimum_rank_agreement.is_some()
            || self.max_gradient_relative_error.is_some()
            || self.max_constraint_violation.is_some()
    }

    fn agreement_reason(
        self,
        diagnostics: Option<RomPromotionDiagnostics>,
    ) -> Result<Option<RomPromotionReason>, RomCouplingError> {
        let Some(diagnostics) = diagnostics else {
            return Ok(self
                .agreement_gate_enabled()
                .then_some(RomPromotionReason::DiagnosticsMissing));
        };
        diagnostics.validate()?;
        if self
            .minimum_rank_agreement
            .is_some_and(|limit| diagnostics.rank_agreement < limit)
        {
            return Ok(Some(RomPromotionReason::RankAgreementBelowThreshold));
        }
        if self
            .max_gradient_relative_error
            .is_some_and(|limit| diagnostics.gradient_relative_error > limit)
        {
            return Ok(Some(RomPromotionReason::GradientErrorExceeded));
        }
        if self
            .max_constraint_violation
            .is_some_and(|limit| diagnostics.maximum_constraint_violation > limit)
        {
            return Ok(Some(RomPromotionReason::ConstraintViolationExceeded));
        }
        Ok(None)
    }
}

/// Summary of an adaptive ROM sweep, including its promotion cost.
#[derive(Debug, Clone)]
pub struct RomPromotionReport {
    pub points: Vec<RomAdaptiveSolution>,
    pub promoted_count: usize,
    pub maximum_checked_reduced_residual: Option<f64>,
    pub minimum_checked_rank_agreement: Option<f64>,
    pub maximum_checked_gradient_relative_error: Option<f64>,
    pub maximum_checked_constraint_violation: Option<f64>,
    /// The report is an adaptive reduced preview; individual points retain
    /// their exact `RomFidelity`, including full-order promotions.
    pub provenance: ArtifactProvenance,
}

/// Augmented two-term static tail correction for the bound drive's pattern.
#[derive(Debug, Clone)]
struct AugmentedTail {
    z0: Vec<Complex64>,
    z1: Vec<Complex64>,
    sigma: f64,
}

/// Modal ROM of a [`FemBemCoupling`] model, bound to one drive.
///
/// Build with [`FemBemCoupling::build_rom`]; the ROM borrows the coupling
/// model (mesh, DtN builder) and owns the basis, the precomputed stiffness
/// block, and the projected aperture incidence.
pub struct FemBemCouplingRom<'a> {
    coupling: &'a FemBemCoupling,
    rom: AcousticRom,
    drive: CouplingDrive,
    /// Craig–Bampton attachment columns (n × a), real, M-orthonormal and
    /// M-orthogonal to the modal basis.
    attachment: DenseMatrix<f64>,
    /// Modal aperture incidence `C = Bᵀ Φ`, row-major (m_ap × m), real.
    aperture_basis: Vec<f64>,
    /// `F[t][j] = (1/3)·Σ_{i ∈ face_j} a_t(i)`, row-major (a × m_ap), real.
    attachment_face_avg: Vec<f64>,
    /// Augmented stiffness block `Φ̂ᵀ K Φ̂`, row-major ((m+a)²), real.
    stiffness_block: Vec<f64>,
    /// Static tail correction for the bound drive's RHS pattern.
    correction: Option<AugmentedTail>,
}

impl FemBemCoupling {
    /// Build a modal ROM of this coupled model for the given drive.
    ///
    /// One-time cost: the eigensolve plus one shifted-static solve per
    /// aperture face (attachment modes). `solver_config` drives the static
    /// solves. Reuse the ROM across the whole frequency sweep.
    pub fn build_rom(
        &self,
        config: &AcousticRomConfig,
        drive: CouplingDrive,
        solver_config: &LinearSolveConfig,
    ) -> Result<FemBemCouplingRom<'_>, RomCouplingError> {
        let rom = AcousticRom::build(&self.mesh, PolynomialDegree::P1, &[], config)?;
        let n = self.num_dofs();
        let m = rom.num_modes();
        let m_ap = self.aperture.faces.len();
        let basis = rom.basis();

        // Modal aperture incidence C[j][r] = (area_j/3)·Σ φ_r(face_j).
        let mut aperture_basis = vec![0.0; m_ap * m];
        for (j, (nodes, area)) in self.aperture.faces.iter().enumerate() {
            for r in 0..m {
                aperture_basis[j * m + r] = (area / 3.0)
                    * (basis[[nodes[0], r]] + basis[[nodes[1], r]] + basis[[nodes[2], r]]);
            }
        }

        // Attachment patterns: unit Neumann data b_j of each aperture face
        // (area_j/3 on its three vertices — exactly one column of B).
        let patterns: Vec<Vec<Complex64>> = self
            .aperture
            .faces
            .iter()
            .map(|(nodes, area)| {
                let mut b = vec![Complex64::new(0.0, 0.0); n];
                for &node in nodes {
                    b[node] = Complex64::new(area / 3.0, 0.0);
                }
                b
            })
            .collect();
        let attachment = rom.attachment_modes(&patterns, solver_config)?;
        let a = attachment.ncols();

        // Attachment face averages F[t][j] = (1/3)·Σ_{i ∈ face_j} a_t(i).
        let mut attachment_face_avg = vec![0.0; a * m_ap];
        for t in 0..a {
            for (j, (nodes, _)) in self.aperture.faces.iter().enumerate() {
                attachment_face_avg[t * m_ap + j] = (attachment[[nodes[0], t]]
                    + attachment[[nodes[1], t]]
                    + attachment[[nodes[2], t]])
                    / 3.0;
            }
        }

        // Augmented stiffness block Φ̂ᵀ K Φ̂ by matvecs:
        //   mode-mode:      diag(λ)
        //   mode-attach:    φ_rᵀ K a_t
        //   attach-attach:  a_sᵀ K a_t  (symmetric, one triangle computed)
        let m_tot = m + a;
        let mut stiffness_block = vec![0.0; m_tot * m_tot];
        for (r, &lambda) in rom.eigenvalues().iter().enumerate() {
            stiffness_block[r * m_tot + r] = lambda;
        }
        for t in 0..a {
            let col_t = attachment.column_to_vec(t);
            let ka_t = rom.stiffness_matvec(&col_t);
            for r in 0..m {
                let dot: f64 = (0..n).map(|i| basis[[i, r]] * ka_t[i]).sum();
                stiffness_block[r * m_tot + m + t] = dot;
                stiffness_block[(m + t) * m_tot + r] = dot;
            }
            for s in 0..=t {
                let col_s = attachment.column_to_vec(s);
                let dot: f64 = col_s.iter().zip(&ka_t).map(|(x, y)| x * y).sum();
                stiffness_block[(m + s) * m_tot + m + t] = dot;
                stiffness_block[(m + t) * m_tot + m + s] = dot;
            }
        }

        Ok(FemBemCouplingRom {
            coupling: self,
            rom,
            drive,
            attachment,
            aperture_basis,
            attachment_face_avg,
            stiffness_block,
            correction: None,
        })
    }
}

impl FemBemCouplingRom<'_> {
    /// The underlying interior modal ROM.
    pub fn rom(&self) -> &AcousticRom {
        &self.rom
    }

    /// The drive bound at construction.
    pub fn drive(&self) -> CouplingDrive {
        self.drive.clone()
    }

    /// `iωρ · v_drive` — the scale of the bound drive's RHS pattern.
    fn drive_scale(&self, omega: f64) -> Complex64 {
        let i_omega_rho = Complex64::new(0.0, omega * self.coupling.density);
        match &self.drive {
            CouplingDrive::None => Complex64::new(0.0, 0.0),
            CouplingDrive::DriverVelocity(v) => i_omega_rho * *v,
            CouplingDrive::DriverVelocityProfiles(profiles) => {
                i_omega_rho
                    * profiles
                        .iter()
                        .map(|profile| profile.value)
                        .sum::<Complex64>()
            }
            CouplingDrive::DriverPartialPiston { value, .. } => i_omega_rho * *value,
        }
    }

    fn num_modes_total(&self) -> usize {
        self.rom.num_modes() + self.attachment.ncols()
    }

    /// Project a full-space vector onto the augmented basis: `[Φᵀx; Aᵀx]`.
    fn project_augmented(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut out = self.rom.project_rhs(x);
        for t in 0..self.attachment.ncols() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (i, &x_i) in x.iter().enumerate() {
                let a = self.attachment[[i, t]];
                if a != 0.0 {
                    acc += a * x_i;
                }
            }
            out.push(acc);
        }
        out
    }

    /// Reconstruct `Φ q_m + A q_a` from augmented coefficients.
    fn reconstruct_augmented(&self, coefficients: &[Complex64]) -> Vec<Complex64> {
        let m = self.rom.num_modes();
        let mut out = self.rom.reconstruct(&coefficients[..m]);
        for t in 0..self.attachment.ncols() {
            let q_t = coefficients[m + t];
            if q_t.norm_sqr() > 0.0 {
                for (i, out_i) in out.iter_mut().enumerate() {
                    *out_i += self.attachment[[i, t]] * q_t;
                }
            }
        }
        out
    }

    /// Precompute the static tail correction for the bound drive (two
    /// shifted-static solves; see the module docs). No-op for
    /// [`CouplingDrive::None`].
    pub fn with_static_correction(
        mut self,
        solver_config: &LinearSolveConfig,
    ) -> Result<Self, RomCouplingError> {
        let scale0 = self.drive_scale(1.0);
        if scale0.norm_sqr() == 0.0 {
            return Ok(self);
        }
        // The Neumann RHS is scale(ω) × ω-independent pattern; recover the
        // pattern from one probe RHS.
        let probe = self.coupling.drive_rhs(1.0, self.drive.clone())?;
        let pattern: Vec<Complex64> = probe.iter().map(|&v| v / scale0).collect();

        let sigma = self.rom.correction_shift();
        let m_tot = self.num_modes_total();
        // Reduced shifted-static operator K̂K + σI (real → complex LU).
        let shifted = DenseMatrix::from_fn(m_tot, m_tot, |i, j| {
            Complex64::new(
                self.stiffness_block[i * m_tot + j] + if i == j { sigma } else { 0.0 },
                0.0,
            )
        });
        let lu = lu_factorize(&shifted).map_err(|_| RomError::SingularReducedOperator)?;
        let solve_reduced_static = |rhs: &[Complex64]| -> Result<Vec<Complex64>, RomCouplingError> {
            lu.solve(rhs)
                .map_err(|_| RomError::SingularReducedOperator.into())
        };

        // z0 = w0 − Φ̂(K̂K+σI)⁻¹Φ̂ᵀf, z1 = w1 − Φ̂(K̂K+σI)⁻¹Φ̂ᵀMw0.
        let w0 = self.rom.shifted_static_solve(&pattern, solver_config)?;
        let c0 = self.project_augmented(&pattern);
        let q0 = solve_reduced_static(&c0)?;
        let z0: Vec<Complex64> = w0
            .iter()
            .zip(self.reconstruct_augmented(&q0).iter())
            .map(|(w, p)| w - p)
            .collect();
        let m_w0 = self.rom.mass_matvec(&w0);
        let w1 = self.rom.shifted_static_solve(&m_w0, solver_config)?;
        let c1 = self.project_augmented(&m_w0);
        let q1 = solve_reduced_static(&c1)?;
        let z1: Vec<Complex64> = w1
            .iter()
            .zip(self.reconstruct_augmented(&q1).iter())
            .map(|(w, p)| w - p)
            .collect();

        self.correction = Some(AugmentedTail { z0, z1, sigma });
        Ok(self)
    }

    /// Reduced coupled operator at one frequency plus the DtN inverse
    /// `G = D⁻¹` when the model has an exterior.
    fn reduced_coupled_operator(
        &self,
        frequency_hz: f64,
    ) -> Result<(DenseMatrix<Complex64>, Option<Vec<Complex64>>), RomCouplingError> {
        let omega = std::f64::consts::TAU * frequency_hz;
        let k = Complex64::new(omega / self.coupling.speed_of_sound, 0.0);
        let k_sq = k * k;
        let m = self.rom.num_modes();
        let a = self.attachment.ncols();
        let m_tot = m + a;
        let m_ap = self.coupling.aperture.faces.len();

        // Φ̂ᵀ K Φ̂ − k²I.
        let mut op = DenseMatrix::from_fn(m_tot, m_tot, |i, j| {
            Complex64::new(
                self.stiffness_block[i * m_tot + j] - if i == j { k_sq.re } else { 0.0 },
                0.0,
            )
        });

        // Projected DtN aperture block −iωρ·Ĉᵀ·G·N·Ĉ with
        // Ĉ[j][r] = C[j][r] (modes), Ĉ[j][m+t] = area_j·F[t][j] (attach).
        let mut dtn_inverse = None;
        if let Some(exterior) = &self.coupling.exterior {
            let factors = exterior.factor(frequency_hz)?;
            let dtn = factors.build_dtn()?;
            let g = dtn.inverse()?;
            let c_hat = |j: usize, r: usize| -> f64 {
                if r < m {
                    self.aperture_basis[j * m + r]
                } else {
                    self.coupling.aperture.faces[j].1 * self.attachment_face_avg[(r - m) * m_ap + j]
                }
            };
            let minus_i_omega_rho = Complex64::new(0.0, -omega * self.coupling.density);
            for r in 0..m_tot {
                for s in 0..m_tot {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for j in 0..m_ap {
                        let c_jr = c_hat(j, r);
                        if c_jr == 0.0 {
                            continue;
                        }
                        for l in 0..m_ap {
                            let c_ls = c_hat(l, s);
                            if c_ls != 0.0 {
                                // N = diag(1/area_l) applied on the l side.
                                acc += c_jr * g[j * m_ap + l] * c_ls
                                    / self.coupling.aperture.faces[l].1;
                            }
                        }
                    }
                    op[[r, s]] += minus_i_omega_rho * acc;
                }
            }
            dtn_inverse = Some(g);
        }
        Ok((op, dtn_inverse))
    }

    /// Solve the coupled ROM at one frequency.
    pub fn solve_frequency(
        &self,
        frequency_hz: f64,
        check_residual: bool,
    ) -> Result<RomCoupledSolution, RomCouplingError> {
        let omega = std::f64::consts::TAU * frequency_hz;
        let k = Complex64::new(omega / self.coupling.speed_of_sound, 0.0);
        let rhs = self.coupling.drive_rhs(omega, self.drive.clone())?;

        let (op, dtn_inverse) = self.reduced_coupled_operator(frequency_hz)?;
        let rhs_reduced = self.project_augmented(&rhs);
        let coefficients = self.rom.solve_reduced(&op, &rhs_reduced)?;
        let mut pressure = self.reconstruct_augmented(&coefficients);
        if let Some(correction) = &self.correction {
            let scale = self.drive_scale(omega);
            let z1_coeff = scale * (correction.sigma + k * k);
            for (p, (z0, z1)) in pressure
                .iter_mut()
                .zip(correction.z0.iter().zip(correction.z1.iter()))
            {
                *p += scale * z0 + z1_coeff * z1;
            }
        }

        // Aperture element velocities, recovered exactly as in the full
        // solve: v = G · p̄ from the element averages of the pressure.
        let aperture_velocity = match &dtn_inverse {
            Some(g) => {
                let p_avg = self.coupling.aperture.element_average_pressures(&pressure);
                let m_ap = p_avg.len();
                let mut v = vec![Complex64::new(0.0, 0.0); m_ap];
                for (j, v_j) in v.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (l, &p_l) in p_avg.iter().enumerate() {
                        acc += g[j * m_ap + l] * p_l;
                    }
                    *v_j = acc;
                }
                v
            }
            None => Vec::new(),
        };

        let relative_residual = if check_residual {
            Some(self.coupled_relative_residual(k, &pressure, &aperture_velocity, &rhs))
        } else {
            None
        };

        Ok(RomCoupledSolution {
            solution: CoupledSolution {
                frequency_hz,
                pressure,
                aperture_velocity,
                solver_iterations: 0,
                relative_residual,
            },
            num_modes: self.rom.num_modes(),
            num_attachment_modes: self.attachment.ncols(),
            relative_residual,
        })
    }

    /// Solve the reduced Hermitian adjoint system `A(ω)ᴴ λ = g` for a
    /// full augmented-basis objective load `g`.
    ///
    /// The returned vector is in the same modal-plus-attachment coordinates
    /// as [`Self::solve_frequency`]. Static tail correction is a fixed output
    /// reconstruction and is intentionally not folded into this algebraic
    /// state solve; callers differentiating corrected outputs must include
    /// that linear reconstruction term separately.
    pub fn solve_adjoint_reduced(
        &self,
        frequency_hz: f64,
        objective_reduced: &[Complex64],
    ) -> Result<Vec<Complex64>, RomCouplingError> {
        let (operator, _) = self.reduced_coupled_operator(frequency_hz)?;
        self.rom
            .solve_reduced_adjoint(&operator, objective_reduced)
            .map_err(Into::into)
    }

    /// Solve a frequency sweep; `residual_check_every = n` computes the
    /// exact coupled residual on every n-th frequency (0 = never).
    pub fn solve_sweep(
        &self,
        frequencies: &[f64],
        residual_check_every: usize,
    ) -> Result<Vec<RomCoupledSolution>, RomCouplingError> {
        frequencies
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                self.solve_frequency(f, residual_check_every > 0 && i % residual_check_every == 0)
            })
            .collect()
    }

    /// Solve a frequency sweep with an exact residual gate and bounded
    /// multi-fidelity promotion. A reduced point whose full-space residual is
    /// above the policy limit is never silently accepted: it is replaced by
    /// the full FEM--BEM solve while promotion budget remains, otherwise the
    /// method returns an error.
    pub fn solve_sweep_with_promotion(
        &self,
        frequencies: &[f64],
        policy: RomPromotionPolicy,
        solver_config: &LinearSolveConfig,
    ) -> Result<RomPromotionReport, RomCouplingError> {
        let mut validator = NoopRomPromotionValidator;
        self.solve_sweep_with_promotion_and_validator(
            frequencies,
            policy,
            solver_config,
            &mut validator,
        )
    }

    /// Solve a frequency sweep with residual and product-specific agreement
    /// gates. The validator is called at the policy's agreement cadence and
    /// can supply rank, gradient, and constraint evidence from a reference or
    /// full-order evaluator. Any enabled gate with missing evidence promotes
    /// conservatively.
    pub fn solve_sweep_with_promotion_and_validator<V: RomPromotionValidator>(
        &self,
        frequencies: &[f64],
        policy: RomPromotionPolicy,
        solver_config: &LinearSolveConfig,
        validator: &mut V,
    ) -> Result<RomPromotionReport, RomCouplingError> {
        policy.validate()?;
        if frequencies
            .windows(2)
            .any(|window| !window[0].is_finite() || window[0] <= 0.0 || window[1] <= window[0])
            || frequencies
                .last()
                .is_some_and(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(RomCouplingError::InvalidPromotionFrequencyGrid);
        }

        let mut points = Vec::with_capacity(frequencies.len());
        let mut promoted_count = 0;
        let mut maximum_checked_reduced_residual = None;
        let mut minimum_checked_rank_agreement = None;
        let mut maximum_checked_gradient_relative_error = None;
        let mut maximum_checked_constraint_violation = None;
        for (index, &frequency_hz) in frequencies.iter().enumerate() {
            let check_residual = index % policy.residual_check_every == 0;
            let reduced = self.solve_frequency(frequency_hz, check_residual)?;
            let residual = reduced.relative_residual;
            if let Some(residual) = residual {
                maximum_checked_reduced_residual = Some(
                    maximum_checked_reduced_residual
                        .map_or(residual, |maximum: f64| maximum.max(residual)),
                );
            }
            let diagnostics = if index % policy.agreement_check_every == 0 {
                validator.validate(frequency_hz, &reduced)?
            } else {
                None
            };
            if let Some(diagnostics) = diagnostics {
                diagnostics.validate()?;
                minimum_checked_rank_agreement = Some(
                    minimum_checked_rank_agreement
                        .map_or(diagnostics.rank_agreement, |minimum: f64| {
                            minimum.min(diagnostics.rank_agreement)
                        }),
                );
                maximum_checked_gradient_relative_error = Some(
                    maximum_checked_gradient_relative_error
                        .map_or(diagnostics.gradient_relative_error, |maximum: f64| {
                            maximum.max(diagnostics.gradient_relative_error)
                        }),
                );
                maximum_checked_constraint_violation = Some(
                    maximum_checked_constraint_violation
                        .map_or(diagnostics.maximum_constraint_violation, |maximum: f64| {
                            maximum.max(diagnostics.maximum_constraint_violation)
                        }),
                );
            }
            let residual_reason = residual
                .is_some_and(|value| !value.is_finite() || value > policy.max_relative_residual);
            let agreement_reason = policy.agreement_reason(diagnostics)?;
            let promotion_reason = if residual_reason {
                Some(RomPromotionReason::ResidualExceeded)
            } else {
                agreement_reason
            };
            if let Some(promotion_reason) = promotion_reason {
                let residual_value = residual.unwrap_or(f64::NAN);
                if promoted_count >= policy.max_full_order_solves {
                    return Err(RomCouplingError::PromotionBudgetExceeded {
                        frequency_hz,
                        residual: residual_value,
                        limit: policy.max_relative_residual,
                        budget: policy.max_full_order_solves,
                        reason: promotion_reason,
                    });
                }
                let solution = self.coupling.solve_frequency(
                    frequency_hz,
                    self.drive.clone(),
                    solver_config,
                )?;
                promoted_count += 1;
                points.push(RomAdaptiveSolution {
                    solution,
                    fidelity: RomFidelity::FullOrderPromoted,
                    reduced_relative_residual: residual,
                    promotion_diagnostics: diagnostics,
                    promotion_reason: Some(promotion_reason),
                    num_modes: reduced.num_modes,
                    num_attachment_modes: reduced.num_attachment_modes,
                });
            } else {
                points.push(RomAdaptiveSolution {
                    solution: reduced.solution,
                    fidelity: RomFidelity::Reduced,
                    reduced_relative_residual: residual,
                    promotion_diagnostics: diagnostics,
                    promotion_reason: None,
                    num_modes: reduced.num_modes,
                    num_attachment_modes: reduced.num_attachment_modes,
                });
            }
        }
        Ok(RomPromotionReport {
            points,
            promoted_count,
            maximum_checked_reduced_residual,
            minimum_checked_rank_agreement,
            maximum_checked_gradient_relative_error,
            maximum_checked_constraint_violation,
            provenance: ArtifactProvenance::native_preview(),
        })
    }

    /// Full-space relative residual of the coupled system
    /// `r = A_fem p − iωρ·B·v − f` for a candidate `(p, v)`.
    fn coupled_relative_residual(
        &self,
        k: Complex64,
        pressure: &[Complex64],
        aperture_velocity: &[Complex64],
        rhs: &[Complex64],
    ) -> f64 {
        let a = self.rom.assemble_full_operator(k, &HashMap::new());
        let n = self.coupling.num_dofs();
        let mut residual = vec![Complex64::new(0.0, 0.0); n];
        for (i, r) in residual.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, v) in a.row_entries(i) {
                acc += v * pressure[j];
            }
            *r = acc - rhs[i];
        }
        // w = −iωρ · B · v: each aperture face contributes area_j/3 · v_j on
        // its three vertices.
        if !aperture_velocity.is_empty() {
            let minus_i_omega_rho = Complex64::new(
                0.0,
                -k.re * self.coupling.speed_of_sound * self.coupling.density,
            );
            for (j, &v_j) in aperture_velocity.iter().enumerate() {
                let (nodes, area) = self.coupling.aperture.faces[j];
                let w = minus_i_omega_rho * (area / 3.0) * v_j;
                for &node in &nodes {
                    residual[node] += w;
                }
            }
        }
        let r_norm = residual.iter().map(|r| r.norm_sqr()).sum::<f64>().sqrt();
        let f_norm = rhs.iter().map(|r| r.norm_sqr()).sum::<f64>().sqrt();
        if f_norm > 0.0 {
            r_norm / f_norm
        } else {
            r_norm
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem_bem_coupling::{
        CylinderFixtureDiscretization, cylindrical_box_with_external_port,
    };
    use sonium_fem::mesh::tetrahedralize_box;
    use sonium_solvers::{LinearSolveConfig, LinearSolverType};

    const C0: f64 = 343.0;
    const RHO: f64 = 1.204;

    fn solver_config() -> LinearSolveConfig {
        LinearSolveConfig {
            solver: LinearSolverType::GmresIlu,
            max_iterations: 2000,
            tolerance: 1e-9,
            restart: 100,
            exact_residual_check_interval: 0,
            verbosity: 0,
        }
    }

    fn relative_error(a: &[Complex64], b: &[Complex64]) -> f64 {
        let diff: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let denom: f64 = b.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        diff / denom
    }

    /// Sealed cavity through the coupled-ROM machinery (no exterior, no
    /// aperture): the augmented basis reduces to the modal one, and the
    /// reduced solve plus static correction matches the full coupled solve.
    #[test]
    fn sealed_coupled_rom_matches_full() {
        let mesh = tetrahedralize_box(0.0, 0.34, 0.0, 0.22, 0.0, 0.18, 8);
        let coupling = FemBemCoupling::sealed(mesh, -1, Some(6), C0, RHO);
        let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0));
        let rom = coupling
            .build_rom(
                &AcousticRomConfig::for_sweep(C0, 1100.0),
                drive.clone(),
                &solver_config(),
            )
            .unwrap()
            .with_static_correction(&solver_config())
            .unwrap();

        let config = solver_config();
        for &f in &[150.0, 620.0, 1050.0] {
            let rom_solution = rom.solve_frequency(f, true).unwrap();
            assert_eq!(rom_solution.num_attachment_modes, 0);
            let full = coupling.solve_frequency(f, drive.clone(), &config).unwrap();
            let err = relative_error(&rom_solution.solution.pressure, &full.pressure);
            let residual = rom_solution.relative_residual.unwrap();
            println!(
                "sealed coupled ROM at {f:.0} Hz: rel err {err:.3e}, coupled residual {residual:.3e}"
            );
            // Measured: 4.1e-5 … 8.0e-3 (uniform piston on the res-8 box).
            assert!(err < 2.0e-2, "sealed coupled ROM error {err:.3e} at {f} Hz");
            assert!(
                residual < 8.0e-2,
                "coupled residual {residual:.3e} at {f} Hz"
            );
        }
    }

    /// Coupled box+port at frequencies around the Helmholtz resonance: the
    /// attachment-augmented ROM reproduces the full coupled pressure field
    /// and aperture volume velocity in the quasi-static regime.
    #[test]
    fn port_coupled_rom_matches_full_near_resonance() {
        let fixture = cylindrical_box_with_external_port(
            0.15,
            0.25,
            0.03,
            0.10,
            CylinderFixtureDiscretization {
                n_circumference: 16,
                n_box_depth: 5,
                n_cap_radial: 3,
                n_aperture_radial: 2,
                n_port_depth: 4,
            },
        )
        .unwrap();
        let coupling = FemBemCoupling::from_fixture(&fixture, C0, RHO).unwrap();
        let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0));
        // The box's first interior mode is ~660 Hz; 250 Hz of band keeps
        // the zero mode plus headroom (quasi-static regime at 40–80 Hz).
        let rom = coupling
            .build_rom(
                &AcousticRomConfig::for_sweep(C0, 250.0),
                drive.clone(),
                &solver_config(),
            )
            .unwrap()
            .with_static_correction(&solver_config())
            .unwrap();

        let areas = coupling.aperture_element_areas();
        let config = solver_config();
        for &f in &[50.0, 70.0] {
            let rom_solution = rom.solve_frequency(f, true).unwrap();
            let full = coupling.solve_frequency(f, drive.clone(), &config).unwrap();
            let err = relative_error(&rom_solution.solution.pressure, &full.pressure);
            let u_rom = rom_solution
                .solution
                .aperture_volume_velocity(&areas)
                .norm();
            let u_full = full.aperture_volume_velocity(&areas).norm();
            let u_rel = (u_rom - u_full).abs() / u_full;
            let residual = rom_solution.relative_residual.unwrap();
            println!(
                "port coupled ROM at {f:.0} Hz ({} modes + {} attachment): field err {err:.3e}, |U| rel {u_rel:.3e}, coupled residual {residual:.3e}",
                rom_solution.num_modes, rom_solution.num_attachment_modes
            );
            // Quasi-static regime: attachment modes make the static limit
            // exact, the correction handles the driver tail.
            assert!(err < 2.0e-2, "port ROM field error {err:.3e} at {f} Hz");
            assert!(u_rel < 2.0e-2, "port ROM |U| off by {u_rel:.3e} at {f} Hz");
            assert!(
                residual < 5.0e-2,
                "coupled residual {residual:.3e} at {f} Hz"
            );

            let objective: Vec<Complex64> = (0..rom.num_modes_total())
                .map(|index| Complex64::new(0.25 + index as f64 * 0.01, -0.1))
                .collect();
            let adjoint = rom.solve_adjoint_reduced(f, &objective).unwrap();
            let (operator, _) = rom.reduced_coupled_operator(f).unwrap();
            let adjoint_residual = (0..operator.nrows())
                .map(|i| {
                    let lhs: Complex64 = (0..operator.ncols())
                        .map(|j| operator[[j, i]].conj() * adjoint[j])
                        .sum();
                    (lhs - objective[i]).norm()
                })
                .fold(0.0, f64::max);
            assert!(
                adjoint_residual < 1.0e-10,
                "coupled adjoint residual {adjoint_residual:.3e} at {f} Hz"
            );
            assert!(
                rom.solve_adjoint_reduced(f, &objective[..objective.len() - 1])
                    .is_err()
            );
        }
    }

    #[test]
    fn rom_promotion_replaces_a_residual_failure_with_full_order() {
        let mesh = tetrahedralize_box(0.0, 0.30, 0.0, 0.20, 0.0, 0.16, 6);
        let coupling = FemBemCoupling::sealed(mesh, -1, Some(4), C0, RHO);
        let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0));
        let config = solver_config();
        let rom = coupling
            .build_rom(&AcousticRomConfig::for_sweep(C0, 900.0), drive, &config)
            .unwrap();

        let report = rom
            .solve_sweep_with_promotion(
                &[180.0],
                RomPromotionPolicy {
                    residual_check_every: 1,
                    agreement_check_every: 1,
                    max_relative_residual: 1.0e-30,
                    minimum_rank_agreement: None,
                    max_gradient_relative_error: None,
                    max_constraint_violation: None,
                    max_full_order_solves: 1,
                },
                &config,
            )
            .unwrap();
        assert_eq!(report.promoted_count, 1);
        assert_eq!(report.provenance, ArtifactProvenance::native_preview());
        assert_eq!(report.points.len(), 1);
        assert_eq!(report.points[0].fidelity, RomFidelity::FullOrderPromoted);
        assert!(report.points[0].solution.solver_iterations > 0);
        assert!(
            report
                .maximum_checked_reduced_residual
                .is_some_and(|residual| residual > 1.0e-30)
        );

        let exhausted = rom.solve_sweep_with_promotion(
            &[180.0],
            RomPromotionPolicy {
                residual_check_every: 1,
                agreement_check_every: 1,
                max_relative_residual: 1.0e-30,
                minimum_rank_agreement: None,
                max_gradient_relative_error: None,
                max_constraint_violation: None,
                max_full_order_solves: 0,
            },
            &config,
        );
        assert!(matches!(
            exhausted,
            Err(RomCouplingError::PromotionBudgetExceeded { .. })
        ));
    }

    #[test]
    fn agreement_policy_explains_each_promotion_gate() {
        let policy = RomPromotionPolicy {
            residual_check_every: 1,
            agreement_check_every: 1,
            max_relative_residual: 1.0,
            minimum_rank_agreement: Some(0.9),
            max_gradient_relative_error: Some(0.1),
            max_constraint_violation: Some(0.1),
            max_full_order_solves: 1,
        };
        assert_eq!(
            policy
                .agreement_reason(Some(RomPromotionDiagnostics {
                    rank_agreement: 0.8,
                    gradient_relative_error: 0.01,
                    maximum_constraint_violation: 0.01,
                }))
                .unwrap(),
            Some(RomPromotionReason::RankAgreementBelowThreshold)
        );
        assert_eq!(
            policy
                .agreement_reason(Some(RomPromotionDiagnostics {
                    rank_agreement: 0.95,
                    gradient_relative_error: 0.2,
                    maximum_constraint_violation: 0.01,
                }))
                .unwrap(),
            Some(RomPromotionReason::GradientErrorExceeded)
        );
        assert_eq!(
            policy
                .agreement_reason(Some(RomPromotionDiagnostics {
                    rank_agreement: 0.95,
                    gradient_relative_error: 0.01,
                    maximum_constraint_violation: 0.2,
                }))
                .unwrap(),
            Some(RomPromotionReason::ConstraintViolationExceeded)
        );
        assert_eq!(
            policy.agreement_reason(None).unwrap(),
            Some(RomPromotionReason::DiagnosticsMissing)
        );
        assert!(matches!(
            policy.agreement_reason(Some(RomPromotionDiagnostics {
                rank_agreement: f64::NAN,
                gradient_relative_error: 0.0,
                maximum_constraint_violation: 0.0,
            })),
            Err(RomCouplingError::InvalidPromotionDiagnostics(_))
        ));
    }

    #[test]
    fn validator_diagnostics_promote_and_are_retained_in_the_report() {
        let mesh = tetrahedralize_box(0.0, 0.30, 0.0, 0.20, 0.0, 0.16, 6);
        let coupling = FemBemCoupling::sealed(mesh, -1, Some(4), C0, RHO);
        let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0));
        let config = solver_config();
        let rom = coupling
            .build_rom(&AcousticRomConfig::for_sweep(C0, 900.0), drive, &config)
            .unwrap();
        let diagnostics = RomPromotionDiagnostics {
            rank_agreement: 0.25,
            gradient_relative_error: 0.25,
            maximum_constraint_violation: 0.25,
        };
        let mut validator =
            |_frequency_hz: f64, _reduced: &RomCoupledSolution| Ok(Some(diagnostics));
        let report = rom
            .solve_sweep_with_promotion_and_validator(
                &[180.0],
                RomPromotionPolicy {
                    residual_check_every: 1,
                    agreement_check_every: 1,
                    // Leave residual headroom so agreement is the trigger.
                    max_relative_residual: 1.0e6,
                    minimum_rank_agreement: Some(0.9),
                    max_gradient_relative_error: Some(0.1),
                    max_constraint_violation: Some(0.1),
                    max_full_order_solves: 1,
                },
                &config,
                &mut validator,
            )
            .unwrap();
        assert_eq!(report.promoted_count, 1);
        assert_eq!(
            report.points[0].promotion_reason,
            Some(RomPromotionReason::RankAgreementBelowThreshold)
        );
        assert_eq!(report.points[0].promotion_diagnostics, Some(diagnostics));
        assert_eq!(report.minimum_checked_rank_agreement, Some(0.25));
        assert_eq!(report.maximum_checked_gradient_relative_error, Some(0.25));
        assert_eq!(report.maximum_checked_constraint_violation, Some(0.25));
    }

    #[test]
    fn enabled_agreement_gate_promotes_when_validator_has_no_evidence() {
        let mesh = tetrahedralize_box(0.0, 0.30, 0.0, 0.20, 0.0, 0.16, 6);
        let coupling = FemBemCoupling::sealed(mesh, -1, Some(4), C0, RHO);
        let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0));
        let config = solver_config();
        let rom = coupling
            .build_rom(&AcousticRomConfig::for_sweep(C0, 900.0), drive, &config)
            .unwrap();
        let mut validator = |_frequency_hz: f64, _reduced: &RomCoupledSolution| Ok(None);
        let report = rom
            .solve_sweep_with_promotion_and_validator(
                &[180.0],
                RomPromotionPolicy {
                    residual_check_every: 1,
                    agreement_check_every: 1,
                    max_relative_residual: 1.0e6,
                    minimum_rank_agreement: Some(0.9),
                    max_gradient_relative_error: None,
                    max_constraint_violation: None,
                    max_full_order_solves: 1,
                },
                &config,
                &mut validator,
            )
            .unwrap();
        assert_eq!(report.points[0].fidelity, RomFidelity::FullOrderPromoted);
        assert_eq!(
            report.points[0].promotion_reason,
            Some(RomPromotionReason::DiagnosticsMissing)
        );
        assert_eq!(report.points[0].promotion_diagnostics, None);
    }
}
