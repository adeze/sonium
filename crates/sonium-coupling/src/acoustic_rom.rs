//! Modal-basis reduced-order model (ROM) for fixed-topology interior acoustic
//! frequency sweeps (review item C5: `reviews/acoustics-20250730.md` Phase C
//! item 5, §9 P1 #5, §10 speaker modeling).
//!
//! Repeated Helmholtz solves on a fixed mesh share the same spatial operator;
//! only the coefficients change with frequency. A modal ROM exploits this:
//!
//! 1. **Basis construction (one-time).** The undamped rigid-wall problem
//!    `K φ = λ M φ` (pure Neumann, `λ = (ω/c)²`) is solved for the lowest `m`
//!    modes with the generalized LOBPCG eigensolver
//!    ([`lobpcg_generalized`]). The basis Φ is M-orthonormal
//!    (`Φᵀ M Φ = I`), and the λ = 0 rigid-body (compliance) mode of the pure
//!    Neumann problem is always included — it carries the quasi-static /
//!    Helmholtz-resonator compliance response exactly.
//! 2. **Reduced solve (per frequency).** With `A(ω) = K − k²M + Σ_j α_j(ω)
//!    M_Γj` the Galerkin projection `A_r = Φᵀ A Φ` is the small dense system
//!
//!    ```text
//!    A_r(ω) = diag(λ) − k²·I + Σ_j α_j(ω)·Φᵀ M_Γj Φ,   q = A_r⁻¹ Φᵀ f
//!    ```
//!
//!    The boundary projections `Φᵀ M_Γj Φ` are frequency-independent and
//!    computed once; only their coefficients `α_j(ω)` vary. The m×m complex
//!    system is solved with the dense LU of `sonium-solvers`, and the
//!    full-space pressure is reconstructed as `p = Φ q`. External dense
//!    blocks (e.g. the FEM-BEM DtN aperture block) project the same way and
//!    are added by the caller before factorization — see
//!    `FemBemCoupling::build_rom` in `sonium-sim-speaker-modeling`.
//!
//! # Mode count
//!
//! `m` is chosen from a Weyl asymptotic mode count at
//! `band_hz = oversampling × max_frequency_hz` (volume + surface terms, 1.5×
//! safety), clamped to `[min_modes, max_modes]`. After the eigensolve the top
//! retained mode is checked against `band_hz`; the request is doubled once
//! towards `max_modes` if the band is not covered, and
//! [`RomError::InsufficientBand`] is returned if even `max_modes` cannot
//! cover it.
//!
//! # Static tail correction
//!
//! Galerkin truncation drops the tail `Σ_{r>m} φ_r (φ_rᵀf)/(λ_r − k²)`. For
//! boundary-supported Neumann data (any piston drive) the tail decays slowly
//! — it is the static boundary-layer response at the driven face — and
//! dominates the ROM error away from resonances (measured: 5–12 % mid-band
//! without correction on the QA box). [`AcousticRom::static_correction`]
//! precomputes the first two terms of the tail's expansion in
//! `1/(λ − k²) = 1/(λ + σ_c) + (σ_c + k²)/(λ + σ_c)² + …` against the
//! shifted static operator `K + σ_c M` (`σ_c = 0.1·k_band²`): two real
//! solves per RHS *pattern*, reused for the whole sweep (the correction is
//! linear in the RHS). [`AcousticRom::solve_frequency_static`] then delivers
//! 5e-5 … 6e-3 error across the sealed-box band. For the coupled FEM–BEM
//! path the same correction is applied against the Craig–Bampton augmented
//! basis — see `fem_bem_coupling::rom` in `sonium-sim-speaker-modeling`.
//!
//! # Accuracy contract and limitations
//!
//! - Modes up to `oversampling × max_frequency_hz` are represented exactly
//!   (up to eigensolver tolerance), including the zero mode, so undamped and
//!   proportionally damped sweeps converge to the full FEM solution as
//!   `m` grows; ~1.4–1.5× band oversampling plus the static correction gives
//!   sub-percent accuracy (measured numbers below).
//! - The static correction assumes an ω-independent RHS *pattern* (true for
//!   velocity drives) and uses the *undamped* shifted-static operator: for
//!   damped walls the wall's own boundary-layer tail is only partly captured
//!   (measured: 1.6e-2 … 1.2e-1 for Z = ρc(4+2i), growing toward the band
//!   edge).
//! - Discontinuous drives (partial pistons with a hard edge) put most of
//!   the RHS energy in the far modal tail and converge 3–4× worse than the
//!   smooth piston used in the tests/QA — prefer smooth drive profiles, or
//!   widen the band.
//! - Frequency-dependent **boundary** terms (impedance walls, DtN blocks)
//!   are re-projected exactly each frequency, so moderate wall damping is
//!   handled well. Strongly non-proportional damping (e.g. a heavily damped
//!   wall whose modes differ qualitatively from the rigid-wall basis)
//!   degrades modal ROMs — the residual spot-check exists to detect this.
//! - **LOBPCG initial-block entropy (fixed in `sonium-solvers`).** The
//!   deterministic initial block in `sonium-solvers` `eigen/lobpcg.rs`
//!   previously had numerical rank ≈ 17 regardless of the requested block
//!   size (an XOR-Cayley structure with only 17 nonzero Walsh modes), so
//!   LOBPCG returned a contiguous initial segment of ≤ 17–20 eigenpairs with
//!   `converged = true` even when more were requested, capping the basis at
//!   ≈ 1.9 kHz on the QA boxes. The generator is now a seeded xorshift64*
//!   (full numerical rank; regression-tested up to block size 72), so the
//!   requested mode count is delivered and wider bands are supported. The
//!   retry/doubling loop in [`AcousticRom::build`] remains as a defensive
//!   fallback.
//! - The ROM covers the interior FEM only; in the coupled FEM-BEM path the
//!   exterior BEM/DtN stays full fidelity.
//! - Deliberately out of scope: rational Krylov / second-order Arnoldi
//!   bases (better for strongly damped or very wide bands), parametric
//!   (geometry-varying) ROMs.
//!
//! # Residual spot-check
//!
//! [`AcousticRom::solve_frequency`] and the coupled sweep accept an opt-in
//! residual check: the full operator is assembled and
//! `‖A(Φq) − f‖ / ‖f‖` is evaluated every k-th frequency. This follows the
//! exact/compressed sentinel philosophy of the B5 recycling work — the ROM
//! is a compressed surrogate, and the spot-check is the cheap exactness
//! tripwire.
//!
//! # Benchmarks
//!
//! Measured on the QA fixtures (Apple M4, `--release`; see
//! `qa-speaker-acoustic-rom`):
//!
//! | sweep | full FEM | ROM (build + per-freq) | speedup |
//! |-------|----------|------------------------|---------|
//! | sealed box 0.34×0.22×0.18 m (2 197 DOFs, 16 modes, 41 freqs) | 791 ms | 276 ms build + 14 ms sweep | 57× on the sweep, break-even ≈ 15 freqs |
//! | Helmholtz port fixture, coupled (618 DOFs, 48 aperture elements, 8 modes + 33 attachment, 21 freqs) | 5.1 s | 0.18 s build + 5.3 s sweep | none at this size — the BEM DtN condensation dominates both paths |
//!
//! Crossover: the one-time basis build (LOBPCG + two shifted-static solves)
//! pays for itself after ~15 frequencies on the sealed box; beyond that the
//! per-frequency cost is O(m³ + m·n) ≪ GMRES. In the coupled path the ROM
//! removes only the interior solve — the per-frequency BEM factorization and
//! DtN back-solves are unchanged, so the wall-clock win grows with the
//! interior mesh size, not with the aperture size.

use num_complex::Complex64;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::acoustics::pressure::{
    AcousticsAssembler, AcousticsAssemblerExt, build_acoustics_assembler_with_boundaries,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use sonium_solvers::{CsrMatrix, LobpcgConfig, lobpcg_generalized_preconditioned};
use std::collections::HashMap;

/// Configuration for [`AcousticRom::build`].
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticRomConfig {
    /// Speed of sound (m/s); converts eigenvalues to frequencies and sets
    /// `k = ω/c` in the reduced operator.
    pub speed_of_sound: f64,
    /// Highest frequency (Hz) of the sweeps this ROM will serve.
    pub max_frequency_hz: f64,
    /// Band oversampling: modes are retained up to
    /// `oversampling × max_frequency_hz` so the truncation tail does not
    /// pollute the band edge. Default 1.4.
    pub band_oversampling: f64,
    /// Minimum retained modes (floor on the Weyl estimate). Default 8.
    pub min_modes: usize,
    /// Hard cap on retained modes. Default 256.
    pub max_modes: usize,
    /// LOBPCG relative-residual tolerance for the eigenpairs. Default 1e-9.
    pub lobpcg_tolerance: f64,
    /// LOBPCG outer-iteration cap. Default 500.
    pub lobpcg_max_iterations: usize,
}

impl AcousticRomConfig {
    /// Config for sweeps up to `max_frequency_hz` with default tolerances.
    pub fn for_sweep(speed_of_sound: f64, max_frequency_hz: f64) -> Self {
        Self {
            speed_of_sound,
            max_frequency_hz,
            band_oversampling: 1.4,
            min_modes: 8,
            max_modes: 256,
            lobpcg_tolerance: 1e-9,
            lobpcg_max_iterations: 500,
        }
    }

    /// Highest mode frequency (Hz) the basis must cover.
    pub fn band_hz(&self) -> f64 {
        self.band_oversampling * self.max_frequency_hz
    }

    fn validate(&self) -> Result<(), RomError> {
        let finite_positive = [
            ("speed_of_sound", self.speed_of_sound),
            ("max_frequency_hz", self.max_frequency_hz),
            ("band_oversampling", self.band_oversampling),
            ("lobpcg_tolerance", self.lobpcg_tolerance),
        ];
        for (field, value) in finite_positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(RomError::InvalidConfig(format!(
                    "{field} must be finite and greater than zero (got {value})"
                )));
            }
        }
        if self.band_oversampling < 1.0 {
            return Err(RomError::InvalidConfig(
                "band_oversampling must be at least 1.0".to_string(),
            ));
        }
        if self.min_modes == 0 {
            return Err(RomError::InvalidConfig(
                "min_modes must be at least 1 (the zero mode is always needed)".to_string(),
            ));
        }
        if self.max_modes < self.min_modes {
            return Err(RomError::InvalidConfig(
                "max_modes must be no smaller than min_modes".to_string(),
            ));
        }
        Ok(())
    }
}

/// Errors raised by the acoustic ROM.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RomError {
    #[error("invalid ROM configuration: {0}")]
    InvalidConfig(String),
    #[error("LOBPCG eigensolve failed: {0}")]
    EigenSolve(String),
    #[error("LOBPCG did not converge after {iterations} iterations")]
    EigenNotConverged { iterations: usize },
    #[error(
        "basis covers {highest_mode_hz:.1} Hz but the requested band reaches {band_hz:.1} Hz; \
         raise max_modes"
    )]
    InsufficientBand { highest_mode_hz: f64, band_hz: f64 },
    #[error("reduced operator is singular at this frequency")]
    SingularReducedOperator,
    #[error("static tail correction solve failed: {0}")]
    StaticSolve(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Solution of one reduced solve.
#[derive(Debug, Clone)]
pub struct RomFrequencySolution {
    pub frequency_hz: f64,
    /// Modal coefficients `q` (length = number of modes).
    pub coefficients: Vec<Complex64>,
    /// Full-space nodal pressures `p = Φ q` (length = number of DOFs).
    pub pressure: Vec<Complex64>,
    /// Full-space relative residual `‖A(Φq) − f‖/‖f‖` when requested.
    ///
    /// Interpretation: for boundary-supported Neumann data this is dominated
    /// by the out-of-basis content of the RHS (the modal tail the ROM cannot
    /// represent), so it is a *quality metric*, not a pass/fail tripwire.
    pub relative_residual: Option<f64>,
    /// Projected cross-check residual `‖Φᵀ(A(Φq) − f)‖/‖Φᵀf‖` when
    /// requested. Uses the independently assembled full operator, so it
    /// detects mismatches between the reduced and full assembly paths
    /// (sign/coefficient errors in boundary terms); ~1e-10 in normal
    /// operation.
    pub projected_residual: Option<f64>,
}

/// Two-term static (mode-acceleration) tail correction for one fixed
/// right-hand-side *pattern*, precomputed by
/// [`AcousticRom::static_correction`].
///
/// Modal truncation drops the tail `Σ_{r>m} φ_r (φ_rᵀf)/(λ_r − k²)`. For
/// boundary-supported Neumann data (any piston drive) this tail decays
/// slowly — it is the static boundary-layer response at the driven face —
/// and dominates the ROM error away from resonances (measured: 5–12 %
/// mid-band without correction on the QA box). Expanding
/// `1/(λ − k²) = 1/(λ + σ_c) + (σ_c + k²)/(λ + σ_c)² + …` and evaluating the
/// series with the shifted static operator `K + σ_c M` (`σ_c = 0.1·k_band²`,
/// small enough that the first tail modes are captured accurately) gives
///
/// ```text
/// p ≈ Φ q + z0 + (σ_c + k²)·z1
/// ```
///
/// with `z0`, `z1` each requiring one real shifted-static solve per RHS
/// pattern (two solves total, then reused for the whole sweep — the
/// correction is linear in the RHS, so a sweep with `f(ω) = α(ω)·pattern`
/// scales it by `α(ω)`). Measured with correction: 5e-5 … 6e-3 error across
/// the sealed-box band. The correction uses the undamped basis; it remains
/// effective for moderate wall damping (validated in the test suite).
pub struct StaticCorrection {
    /// Unit RHS pattern the correction was built from.
    pattern: Vec<Complex64>,
    /// First expansion term `(K + σ_c M)⁻¹ f − Φ diag(1/(λ+σ_c)) Φᵀ f`.
    z0: Vec<Complex64>,
    /// Second expansion term (same, with one extra `1/(λ+σ_c)` factor).
    z1: Vec<Complex64>,
    /// Correction shift `σ_c`.
    sigma: f64,
}

impl StaticCorrection {
    /// The unit RHS pattern this correction was built from.
    pub fn pattern(&self) -> &[Complex64] {
        &self.pattern
    }

    /// The tail contribution `scale·(z0 + (σ_c + k²)·z1)` at wavenumber
    /// `k`, for `rhs = scale × pattern`. Used by callers that augment the
    /// reduced operator themselves (e.g. the coupled FEM–BEM ROM) and
    /// therefore cannot go through [`AcousticRom::solve_frequency_static`].
    pub fn tail_terms(&self, k: Complex64, scale: Complex64) -> Vec<Complex64> {
        let z1_coeff = scale * (self.sigma + k * k);
        (0..self.z0.len())
            .map(|i| scale * self.z0[i] + z1_coeff * self.z1[i])
            .collect()
    }
}

/// Modal reduced-order model of a fixed interior acoustic FEM topology.
pub struct AcousticRom {
    assembler: AcousticsAssembler,
    /// n × m real basis, M-orthonormal columns.
    basis: DenseMatrix<f64>,
    /// Eigenvalues `λ_j = (ω_j/c)²`, ascending, `λ_0 = 0` (rigid-wall mode).
    eigenvalues: Vec<f64>,
    /// `(tag, Φᵀ M_Γ Φ)` per boundary marker, frequency-independent.
    boundary_projections: Vec<(usize, DenseMatrix<f64>)>,
    speed_of_sound: f64,
    band_hz: f64,
    /// Shift `σ_c = 0.1·k_band²` used for static tail corrections.
    correction_shift: f64,
}

impl AcousticRom {
    /// Build the modal basis for a mesh (one-time cost per fixed topology).
    ///
    /// `boundary_tags` select which boundary markers get projected mass
    /// matrices `Φᵀ M_Γ Φ` for per-frequency impedance terms; pass an empty
    /// slice for purely rigid walls.
    pub fn build(
        mesh: &Mesh,
        degree: PolynomialDegree,
        boundary_tags: &[i32],
        config: &AcousticRomConfig,
    ) -> Result<Self, RomError> {
        config.validate()?;
        let assembler = build_acoustics_assembler_with_boundaries(mesh, degree, boundary_tags);
        let n = assembler.num_rows;
        let stiffness = assembler.assemble_real(&[1.0, 0.0]);
        let mass = assembler.assemble_real(&[0.0, 1.0]);

        // Weyl asymptotic mode count at the band edge: N(k) ≈ V k³/(6π²)
        // + S k²/(16π) (Neumann surface term), with a 1.5× safety factor.
        let volume = frobenius_sum(&mass);
        let surface_area = boundary_surface_area(mesh);
        let band_hz = config.band_hz();
        let estimate = weyl_mode_count(volume, surface_area, band_hz, config.speed_of_sound);

        let cap = config.max_modes.min(n).max(1);
        let mut m = estimate.max(config.min_modes).min(cap);
        // Spectral shift σ = k_band²: solve (K + σM)φ = λ̃ Mφ and recover
        // λ = λ̃ − σ. The eigenvectors are identical, all shifted eigenvalues
        // are positive, and — critically for the pure Neumann problem — the
        // zero (compliance) mode gets λ̃₀ = σ > 0, so LOBPCG's relative
        // residual criterion is well defined for it (unshifted it is 0/0 and
        // the pair never locks). The preconditioner is ILU(0) of the shifted
        // operator: unpreconditioned and Jacobi LOBPCG stagnate on the
        // clustered cavity spectrum, while ILU converged in every probe
        // (typically 40–100 iterations).
        //
        // Note on the eigensolver: LOBPCG's soft-locking can return a
        // *contiguous initial segment* shorter than the requested count on
        // these cavity pencils (historically ~17–20 pairs, caused by the old
        // rank-capped initial block in `eigen/lobpcg.rs`, now fixed; kept as
        // a defensive fallback). The loop below therefore treats the
        // returned count as the achieved basis size, stops when a retry
        // makes no progress, and every returned pair is residual-verified
        // before use.
        let k_band = std::f64::consts::TAU * band_hz / config.speed_of_sound;
        let sigma = k_band * k_band;
        let shifted = assembler.assemble_real(&[1.0, sigma]);
        let preconditioner = sonium_solvers::IluPreconditioner::from_csr(&shifted);
        let mut previous_highest = f64::NEG_INFINITY;
        let result = loop {
            let padding = (m / 4).clamp(2, 16).min(n.saturating_sub(m));
            let lobpcg_config = LobpcgConfig {
                num_eigenvalues: m,
                max_iterations: config.lobpcg_max_iterations,
                tolerance: config.lobpcg_tolerance,
                block_padding: padding,
            };
            let result =
                lobpcg_generalized_preconditioned(&shifted, &mass, &preconditioner, &lobpcg_config)
                    .map_err(|e| RomError::EigenSolve(e.to_string()))?;
            if !result.converged {
                return Err(RomError::EigenNotConverged {
                    iterations: result.iterations,
                });
            }
            let highest = mode_hz(
                result.eigenvalues.last().copied().unwrap_or(0.0) - sigma,
                config.speed_of_sound,
            );
            if highest >= band_hz {
                break result;
            }
            if m >= cap || highest <= previous_highest {
                // Either at the configured cap or the eigensolver returns the
                // same initial segment no matter what we ask for.
                return Err(RomError::InsufficientBand {
                    highest_mode_hz: highest,
                    band_hz,
                });
            }
            previous_highest = highest;
            m = (2 * m).min(cap);
        };

        let mut basis = result.eigenvectors;
        let mut eigenvalues: Vec<f64> = result
            .eigenvalues
            .iter()
            .map(|&l| (l - sigma).max(0.0))
            .collect();

        // Residual-verify every returned pair against the *unshifted* pencil
        // (defence in depth against the soft-locking quirks noted above).
        for (j, &lambda) in eigenvalues.iter().enumerate() {
            let col = basis.column_to_vec(j);
            let k_col = csr_matvec(&stiffness, &col);
            let m_col = csr_matvec(&mass, &col);
            let mut res_sq = 0.0;
            let mut scale_sq = 0.0;
            for i in 0..n {
                let ri = k_col[i] - lambda * m_col[i];
                res_sq += ri * ri;
                scale_sq += m_col[i] * m_col[i];
            }
            let spectral_scale = eigenvalues.last().copied().unwrap_or(0.0).max(1.0);
            let residual = res_sq.sqrt() / (spectral_scale * scale_sq.sqrt()).max(1e-300);
            if residual > 1e-6 {
                return Err(RomError::EigenSolve(format!(
                    "mode {j} failed residual verification: rel residual {residual:.3e}"
                )));
            }
        }

        // LOBPCG returns approximately M-orthonormal vectors (its own test
        // suite asserts 1e-10). Guard the ROM contract anyway: if the Gram
        // matrix drifts past 1e-8, re-orthonormalize in the M inner product
        // and refresh the Rayleigh quotients.
        if m_orthonormal_deviation(&mass, &basis) > 1e-8 {
            mgs_orthonormalize_m(&mass, &mut basis);
            for (j, lambda) in eigenvalues.iter_mut().enumerate() {
                let col = basis.column_to_vec(j);
                let k_col = csr_matvec(&stiffness, &col);
                let m_col = csr_matvec(&mass, &col);
                let num: f64 = col.iter().zip(&k_col).map(|(a, b)| a * b).sum();
                let den: f64 = col.iter().zip(&m_col).map(|(a, b)| a * b).sum();
                *lambda = if den > 0.0 { (num / den).max(0.0) } else { 0.0 };
            }
        }

        // Frequency-independent boundary projections Φᵀ M_Γ Φ.
        let mut boundary_projections = Vec::with_capacity(assembler.boundary_components.len());
        for (&tag, values) in &assembler.boundary_components {
            let csr = CsrMatrix::from_raw_parts(
                n,
                n,
                assembler.row_ptrs.clone(),
                assembler.col_indices.clone(),
                values.clone(),
            );
            let projected = project_sparse(&csr, &basis);
            boundary_projections.push((tag, projected));
        }

        Ok(Self {
            assembler,
            basis,
            eigenvalues,
            boundary_projections,
            speed_of_sound: config.speed_of_sound,
            band_hz,
            correction_shift: 0.1 * sigma,
        })
    }

    /// Number of retained modes `m`.
    pub fn num_modes(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Number of full-space DOFs `n`.
    pub fn num_dofs(&self) -> usize {
        self.basis.nrows()
    }

    /// Eigenvalues `λ_j = (ω_j/c)²`, ascending.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Modal frequencies (Hz) of the retained modes.
    pub fn mode_frequencies_hz(&self) -> Vec<f64> {
        self.eigenvalues
            .iter()
            .map(|&l| mode_hz(l, self.speed_of_sound))
            .collect()
    }

    /// The n × m M-orthonormal basis (columns are the modes).
    pub fn basis(&self) -> &DenseMatrix<f64> {
        &self.basis
    }

    /// Highest retained mode frequency (Hz); compare against
    /// [`AcousticRomConfig::band_hz`] for band coverage.
    pub fn highest_mode_hz(&self) -> f64 {
        mode_hz(
            self.eigenvalues.last().copied().unwrap_or(0.0),
            self.speed_of_sound,
        )
    }

    /// Band edge (Hz) the basis was required to cover at build time.
    pub fn band_hz(&self) -> f64 {
        self.band_hz
    }

    /// Reduced operator `A_r(k) = diag(λ) − k² I + Σ_j α_j Φᵀ M_Γj Φ`.
    ///
    /// `boundary_coeffs` maps boundary markers to their (possibly
    /// frequency-dependent) coefficients, exactly as in
    /// [`AcousticsAssemblerExt::assemble_helmholtz`]. The returned dense m×m
    /// matrix can be augmented by callers with other projected dense terms
    /// (e.g. the DtN aperture block) before calling
    /// [`AcousticRom::solve_reduced`].
    pub fn reduced_operator(
        &self,
        k: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> DenseMatrix<Complex64> {
        let m = self.num_modes();
        let k_sq = k * k;
        let mut op = DenseMatrix::from_fn(m, m, |i, j| {
            if i == j {
                Complex64::new(self.eigenvalues[i], 0.0) - k_sq
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        for (tag, projected) in &self.boundary_projections {
            let Some(&alpha) = boundary_coeffs.get(tag) else {
                continue;
            };
            if alpha.norm_sqr() == 0.0 {
                continue;
            }
            for i in 0..m {
                for j in 0..m {
                    op[[i, j]] += alpha * projected[[i, j]];
                }
            }
        }
        op
    }

    /// Project a full-space right-hand side: `f_r = Φᵀ f` (plain transpose —
    /// the Helmholtz system is complex-symmetric, not Hermitian).
    pub fn project_rhs(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            rhs.len(),
            self.num_dofs(),
            "rhs length must match the full-space dimension"
        );
        let m = self.num_modes();
        let mut out = vec![Complex64::new(0.0, 0.0); m];
        for (r, out_r) in out.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (i, &rhs_i) in rhs.iter().enumerate() {
                let phi = self.basis[[i, r]];
                if phi != 0.0 {
                    acc += phi * rhs_i;
                }
            }
            *out_r = acc;
        }
        out
    }

    /// Solve a reduced operator (as returned by
    /// [`AcousticRom::reduced_operator`], optionally augmented by the caller)
    /// against a reduced right-hand side.
    pub fn solve_reduced(
        &self,
        operator: &DenseMatrix<Complex64>,
        rhs_reduced: &[Complex64],
    ) -> Result<Vec<Complex64>, RomError> {
        validate_reduced_system(operator, rhs_reduced)?;
        let lu = lu_factorize(operator).map_err(|_| RomError::SingularReducedOperator)?;
        lu.solve(rhs_reduced)
            .map_err(|_| RomError::SingularReducedOperator)
    }

    /// Solve the Hermitian adjoint reduced system `Aᴴ λ = g`.
    ///
    /// Coupled objective derivatives use this solve after projecting the
    /// objective load into the same augmented modal basis as the forward
    /// state. Building the conjugate transpose here keeps the convention in
    /// one place and prevents callers from accidentally using `Aᵀ` for a
    /// complex, radiating operator.
    pub fn solve_reduced_adjoint(
        &self,
        operator: &DenseMatrix<Complex64>,
        objective_reduced: &[Complex64],
    ) -> Result<Vec<Complex64>, RomError> {
        validate_reduced_system(operator, objective_reduced)?;
        let adjoint = DenseMatrix::from_fn(operator.nrows(), operator.ncols(), |i, j| {
            operator[[j, i]].conj()
        });
        let lu = lu_factorize(&adjoint).map_err(|_| RomError::SingularReducedOperator)?;
        lu.solve(objective_reduced)
            .map_err(|_| RomError::SingularReducedOperator)
    }

    /// Reconstruct the full-space pressure `p = Φ q`.
    pub fn reconstruct(&self, coefficients: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            coefficients.len(),
            self.num_modes(),
            "coefficient vector length must match the mode count"
        );
        let n = self.num_dofs();
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (r, &q_r) in coefficients.iter().enumerate() {
                if q_r.norm_sqr() > 0.0 {
                    acc += self.basis[[i, r]] * q_r;
                }
            }
            *out_i = acc;
        }
        out
    }

    /// Shift `σ_c` used for static tail corrections and attachment modes.
    pub fn correction_shift(&self) -> f64 {
        self.correction_shift
    }

    /// Real shifted-static solve `(K + σ_c M)⁻¹ rhs` — the shared workhorse
    /// behind static tail corrections and attachment modes. `config` drives
    /// the solve; a CG/GMRES+ILU configuration with a tight tolerance
    /// (1e-10 or so) is recommended.
    pub fn shifted_static_solve(
        &self,
        rhs: &[Complex64],
        config: &sonium_solvers::LinearSolveConfig,
    ) -> Result<Vec<Complex64>, RomError> {
        if rhs.len() != self.num_dofs() {
            return Err(RomError::DimensionMismatch {
                expected: self.num_dofs(),
                got: rhs.len(),
            });
        }
        let shifted = complex_cast(&self.assembler.assemble_real(&[1.0, self.correction_shift]));
        let result = sonium_solvers::solve_complex(&shifted, rhs, config)
            .map_err(|e| RomError::StaticSolve(e.to_string()))?;
        if !result.converged {
            return Err(RomError::StaticSolve(format!(
                "static solve did not converge (residual {:.3e})",
                result.residual
            )));
        }
        Ok(result.x)
    }

    /// Mass-matrix matvec `M x`.
    pub fn mass_matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        csr_matvec_complex(&self.assembler.assemble_real(&[0.0, 1.0]), x)
    }

    /// Stiffness-matrix matvec `K x` (real) — used by callers that augment
    /// the basis and need the projected stiffness block `Φ̂ᵀ K Φ̂`.
    pub fn stiffness_matvec(&self, x: &[f64]) -> Vec<f64> {
        csr_matvec(&self.assembler.assemble_real(&[1.0, 0.0]), x)
    }

    /// Craig–Bampton attachment modes: for each unit RHS `pattern` (real in
    /// essence — e.g. the Neumann data of one aperture face), the static
    /// response `(K + σ_c M)⁻¹ pattern` minus its projection onto the modal
    /// basis, M-orthonormalized against the previously returned columns.
    /// Linearly dependent patterns (fully modal static response) are
    /// dropped. One shifted-static solve per pattern.
    ///
    /// Augmenting the basis with attachment modes puts the quasi-static
    /// boundary layer of each pattern's support *in* the trial space, which
    /// rigid-wall modes otherwise resolve at a slow 1/n rate — this is what
    /// makes the coupled FEM–BEM ROM accurate below the first cavity mode
    /// (see `fem_bem_coupling::rom` in `sonium-sim-speaker-modeling`).
    pub fn attachment_modes(
        &self,
        patterns: &[Vec<Complex64>],
        config: &sonium_solvers::LinearSolveConfig,
    ) -> Result<DenseMatrix<f64>, RomError> {
        let n = self.num_dofs();
        let mass = self.assembler.assemble_real(&[0.0, 1.0]);
        let mut kept: Vec<Vec<f64>> = Vec::new();
        for pattern in patterns {
            let w = self.shifted_static_solve(pattern, config)?;
            let coeff = self.project_rhs(&self.mass_matvec(&w));
            let modal = self.reconstruct(&coeff);
            let mut col: Vec<f64> = (0..n).map(|i| (w[i] - modal[i]).re).collect();
            // M-orthogonalize against previously kept attachment columns
            // (one MGS pass is enough — the caller's solve tolerance keeps
            // the columns nearly independent).
            for other in &kept {
                let m_col = csr_matvec(&mass, &col);
                let dot: f64 = other.iter().zip(&m_col).map(|(a, b)| a * b).sum();
                if dot != 0.0 {
                    for (c, &o) in col.iter_mut().zip(other) {
                        *c -= dot * o;
                    }
                }
            }
            let m_col = csr_matvec(&mass, &col);
            let m_norm: f64 = col
                .iter()
                .zip(&m_col)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .sqrt();
            // Drop (nearly) dependent patterns: their static response is
            // already spanned by the modal basis + kept columns.
            let w_norm = norm2(&w);
            if m_norm <= 1e-10 * w_norm.max(1e-300) {
                continue;
            }
            for c in &mut col {
                *c /= m_norm;
            }
            kept.push(col);
        }
        let mut basis = DenseMatrix::zeros(n, kept.len());
        for (j, col) in kept.iter().enumerate() {
            basis.set_column(j, col);
        }
        Ok(basis)
    }

    /// Build the two-term static tail correction for one fixed RHS pattern
    /// (one-time cost: two real shifted-static solves). See
    /// [`StaticCorrection`] for the formulation.
    ///
    /// `config` drives the static solves; a CG/GMRES+ILU configuration with
    /// a tight tolerance (1e-10 or so) is recommended.
    pub fn static_correction(
        &self,
        pattern: &[Complex64],
        config: &sonium_solvers::LinearSolveConfig,
    ) -> Result<StaticCorrection, RomError> {
        if pattern.len() != self.num_dofs() {
            return Err(RomError::DimensionMismatch {
                expected: self.num_dofs(),
                got: pattern.len(),
            });
        }
        let sigma = self.correction_shift;
        let n = self.num_dofs();

        // w0 = (K + σM)⁻¹ f, w1 = (K + σM)⁻¹ M w0.
        let w0 = self.shifted_static_solve(pattern, config)?;
        let m_w0 = self.mass_matvec(&w0);
        let w1 = self.shifted_static_solve(&m_w0, config)?;

        // Subtract the in-basis parts: u0_r = (Φᵀf)_r/(λ_r+σ),
        // u1_r = (Φᵀ M w0)_r/(λ_r+σ).
        let c = self.project_rhs(pattern);
        let u0: Vec<Complex64> = c
            .iter()
            .zip(self.eigenvalues.iter())
            .map(|(cr, &lambda)| *cr / (lambda + sigma))
            .collect();
        let phi_u0 = self.reconstruct(&u0);
        let m_w0_proj = self.project_rhs(&m_w0);
        let u1: Vec<Complex64> = m_w0_proj
            .iter()
            .zip(self.eigenvalues.iter())
            .map(|(cr, &lambda)| *cr / (lambda + sigma))
            .collect();
        let phi_u1 = self.reconstruct(&u1);

        let z0: Vec<Complex64> = (0..n).map(|i| w0[i] - phi_u0[i]).collect();
        let z1: Vec<Complex64> = (0..n).map(|i| w1[i] - phi_u1[i]).collect();
        Ok(StaticCorrection {
            pattern: pattern.to_vec(),
            z0,
            z1,
            sigma,
        })
    }

    /// Solve at one frequency including the static tail correction, with
    /// `rhs = rhs_scale × correction.pattern` (the correction scales
    /// linearly). The returned pressure is
    /// `Φ q + rhs_scale·(z0 + (σ_c + k²) z1)`.
    pub fn solve_frequency_static(
        &self,
        frequency_hz: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        correction: &StaticCorrection,
        rhs_scale: Complex64,
        check_residual: bool,
    ) -> Result<RomFrequencySolution, RomError> {
        let rhs: Vec<Complex64> = correction.pattern.iter().map(|p| rhs_scale * *p).collect();
        let mut solution =
            self.solve_frequency(frequency_hz, boundary_coeffs, &rhs, check_residual)?;
        let omega = std::f64::consts::TAU * frequency_hz;
        let k = Complex64::new(omega / self.speed_of_sound, 0.0);
        let tail_coeff = rhs_scale * (correction.sigma + k * k);
        for i in 0..solution.pressure.len() {
            solution.pressure[i] += rhs_scale * correction.z0[i] + tail_coeff * correction.z1[i];
        }
        // The residual diagnostics were computed on the uncorrected pressure;
        // refresh them when requested.
        if check_residual {
            let full = self.full_relative_residual(k, boundary_coeffs, &solution.pressure, &rhs);
            solution.relative_residual = Some(full);
        }
        Ok(solution)
    }

    /// Assemble the full-space operator `A = K − k²M + Σ α_j M_Γj` (used for
    /// residual spot-checks and for augmenting the reduced operator by
    /// callers that own extra dense blocks).
    pub fn assemble_full_operator(
        &self,
        k: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
    ) -> CsrMatrix<Complex64> {
        self.assembler.assemble_helmholtz(k, boundary_coeffs)
    }

    /// Full-space relative residual `‖A p − f‖ / ‖f‖` of a candidate
    /// solution (0 if both sides vanish).
    pub fn full_relative_residual(
        &self,
        k: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        pressure: &[Complex64],
        rhs: &[Complex64],
    ) -> f64 {
        let a = self.assemble_full_operator(k, boundary_coeffs);
        let n = self.num_dofs();
        let mut residual = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, v) in a.row_entries(i) {
                acc += v * pressure[j];
            }
            residual[i] = acc - rhs[i];
        }
        let r_norm = norm2(&residual);
        let f_norm = norm2(rhs);
        if f_norm > 0.0 {
            r_norm / f_norm
        } else if r_norm > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Solve at one frequency: project, dense-LU the m×m system, reconstruct.
    ///
    /// When `check_residual` is set, the full operator is assembled and the
    /// exact relative residual is returned in the solution (the opt-in
    /// spot-check described in the module docs).
    pub fn solve_frequency(
        &self,
        frequency_hz: f64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        rhs: &[Complex64],
        check_residual: bool,
    ) -> Result<RomFrequencySolution, RomError> {
        if rhs.len() != self.num_dofs() {
            return Err(RomError::DimensionMismatch {
                expected: self.num_dofs(),
                got: rhs.len(),
            });
        }
        let omega = std::f64::consts::TAU * frequency_hz;
        let k = Complex64::new(omega / self.speed_of_sound, 0.0);
        let operator = self.reduced_operator(k, boundary_coeffs);
        let rhs_reduced = self.project_rhs(rhs);
        let coefficients = self.solve_reduced(&operator, &rhs_reduced)?;
        let pressure = self.reconstruct(&coefficients);
        let (relative_residual, projected_residual) = if check_residual {
            let full = self.full_relative_residual(k, boundary_coeffs, &pressure, rhs);
            let a = self.assemble_full_operator(k, boundary_coeffs);
            let n = self.num_dofs();
            let mut residual = vec![Complex64::new(0.0, 0.0); n];
            for i in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for (j, v) in a.row_entries(i) {
                    acc += v * pressure[j];
                }
                residual[i] = acc - rhs[i];
            }
            let projected = self.project_rhs(&residual);
            let proj_norm = norm2(&projected);
            let rhs_proj_norm = norm2(&rhs_reduced);
            let projected_rel = if rhs_proj_norm > 0.0 {
                proj_norm / rhs_proj_norm
            } else if proj_norm > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            (Some(full), Some(projected_rel))
        } else {
            (None, None)
        };
        Ok(RomFrequencySolution {
            frequency_hz,
            coefficients,
            pressure,
            relative_residual,
            projected_residual,
        })
    }
}

fn mode_hz(lambda: f64, speed_of_sound: f64) -> f64 {
    speed_of_sound * lambda.max(0.0).sqrt() / std::f64::consts::TAU
}

fn norm2(x: &[Complex64]) -> f64 {
    x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
}

/// Sum of all matrix entries (`1ᵀ A 1`); for the P1 mass matrix this equals
/// the domain volume exactly (partition of unity).
fn frobenius_sum(matrix: &CsrMatrix<f64>) -> f64 {
    (0..matrix.num_rows)
        .map(|i| matrix.row_entries(i).map(|(_, v)| v).sum::<f64>())
        .sum()
}

/// Total boundary triangle area (for the Weyl surface term).
fn boundary_surface_area(mesh: &Mesh) -> f64 {
    mesh.boundaries
        .iter()
        .map(|face| {
            let pa = &mesh.nodes[face.nodes[0]];
            let pb = &mesh.nodes[face.nodes[1]];
            let pc = &mesh.nodes[face.nodes[2]];
            let v1 = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
            let v2 = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];
            0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
        })
        .sum()
}

/// Weyl asymptotic count of cavity modes below `band_hz`, with a 1.5×
/// safety factor.
fn weyl_mode_count(volume: f64, surface_area: f64, band_hz: f64, speed_of_sound: f64) -> usize {
    let k = std::f64::consts::TAU * band_hz / speed_of_sound;
    let volume_term = volume * k.powi(3) / (6.0 * std::f64::consts::PI.powi(2));
    let surface_term = surface_area * k.powi(2) / (16.0 * std::f64::consts::PI);
    (1.5 * (volume_term + surface_term) + 1.0).ceil() as usize
}

fn csr_matvec(matrix: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; matrix.num_rows];
    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, v) in matrix.row_entries(i) {
            acc += v * x[j];
        }
        *out_i = acc;
    }
    out
}

/// Real CSR × complex vector.
fn csr_matvec_complex(matrix: &CsrMatrix<f64>, x: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = Complex64::new(0.0, 0.0);
        for (j, v) in matrix.row_entries(i) {
            acc += v * x[j];
        }
        *out_i = acc;
    }
    out
}

/// Complex-cast a real CSR matrix (for static solves with complex RHS).
fn complex_cast(matrix: &CsrMatrix<f64>) -> CsrMatrix<Complex64> {
    let mut triplets = Vec::with_capacity(matrix.nnz());
    for i in 0..matrix.num_rows {
        for (j, v) in matrix.row_entries(i) {
            triplets.push((i, j, Complex64::new(v, 0.0)));
        }
    }
    CsrMatrix::from_triplets(matrix.num_rows, matrix.num_cols, triplets)
}

/// Sparse × dense product on the ROM's small matrices (one-time build cost).
fn validate_reduced_system(
    operator: &DenseMatrix<Complex64>,
    rhs: &[Complex64],
) -> Result<(), RomError> {
    if operator.nrows() != operator.ncols() {
        return Err(RomError::DimensionMismatch {
            expected: operator.nrows(),
            got: operator.ncols(),
        });
    }
    if rhs.len() != operator.nrows() {
        return Err(RomError::DimensionMismatch {
            expected: operator.nrows(),
            got: rhs.len(),
        });
    }
    Ok(())
}

fn csr_matmat(matrix: &CsrMatrix<f64>, dense: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let cols = dense.ncols();
    let mut out = DenseMatrix::zeros(matrix.num_rows, cols);
    for i in 0..matrix.num_rows {
        for (j, v) in matrix.row_entries(i) {
            for c in 0..cols {
                out[[i, c]] += v * dense[[j, c]];
            }
        }
    }
    out
}

/// Galerkin projection `Φᵀ B Φ` of a real sparse operator.
fn project_sparse(matrix: &CsrMatrix<f64>, basis: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let applied = csr_matmat(matrix, basis);
    basis.t().matmul(&applied)
}

/// Max absolute deviation of `Φᵀ M Φ` from the identity.
fn m_orthonormal_deviation(mass: &CsrMatrix<f64>, basis: &DenseMatrix<f64>) -> f64 {
    let gram = project_sparse(mass, basis);
    let m = basis.ncols();
    let mut deviation = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            let target = if i == j { 1.0 } else { 0.0 };
            deviation = deviation.max((gram[[i, j]] - target).abs());
        }
    }
    deviation
}

/// Modified Gram-Schmidt in the M inner product with one reorthogonalization
/// pass. Only used when the LOBPCG output drifts from M-orthonormality.
fn mgs_orthonormalize_m(mass: &CsrMatrix<f64>, basis: &mut DenseMatrix<f64>) {
    let m = basis.ncols();
    for _pass in 0..2 {
        for j in 0..m {
            let mut col = basis.column_to_vec(j);
            let m_col = csr_matvec(mass, &col);
            for i in 0..j {
                let other = basis.column_to_vec(i);
                let dot: f64 = other.iter().zip(&m_col).map(|(a, b)| a * b).sum();
                if dot != 0.0 {
                    for (c, &o) in col.iter_mut().zip(&other) {
                        *c -= dot * o;
                    }
                }
            }
            let m_col = csr_matvec(mass, &col);
            let norm: f64 = col
                .iter()
                .zip(&m_col)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for c in &mut col {
                    *c /= norm;
                }
            }
            basis.set_column(j, &col);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::boundary::generic::{FemSystem, apply_neumann_complex};
    use sonium_fem::mesh::tetrahedralize_box;
    use sonium_solvers::sparse::TripletMatrix;
    use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_complex};

    const C0: f64 = 343.0;
    const RHO: f64 = 1.204;
    const DRIVE_MARKER: i32 = 6;

    fn solver_config() -> LinearSolveConfig {
        LinearSolveConfig {
            solver: LinearSolverType::GmresIlu,
            max_iterations: 2000,
            // The full reference solve reaches a stable true-residual floor
            // just above 1e-9 on this mesh; keep the gate tight while leaving
            // room for that finite-precision floor.
            tolerance: 2e-9,
            restart: 100,
            exact_residual_check_interval: 0,
            verbosity: 0,
        }
    }

    fn box_mesh() -> Mesh {
        tetrahedralize_box(0.0, 0.34, 0.0, 0.22, 0.0, 0.18, 8)
    }

    /// Smooth tilted-cosine piston profile on the drive face: excites both
    /// even and odd modes but keeps the Neumann data C¹, which modal ROMs
    /// need — a discontinuous (partial-piston) drive puts most of the RHS
    /// energy in the far tail of the modal expansion and converges slowly
    /// (documented in the module limitations).
    /// ω-independent drive pattern (unit amplitude); the Neumann RHS is
    /// `iωρ·velocity × pattern`, which is exactly the scale-invariant form
    /// the static correction needs.
    fn drive_pattern(mesh: &Mesh, marker: i32) -> Vec<Complex64> {
        let n = mesh.num_nodes();
        let mut system = FemSystem::new(
            TripletMatrix::<Complex64>::new(n, n),
            vec![Complex64::new(0.0, 0.0); n],
        );
        apply_neumann_complex(&mut system, mesh, marker, &move |x, y, _| {
            let shape = (0.6 + 0.4 * (std::f64::consts::PI * x / 0.34).cos())
                * (0.6 + 0.4 * (std::f64::consts::PI * y / 0.22).cos());
            Complex64::new(shape, 0.0)
        });
        system.rhs
    }

    fn drive_rhs(mesh: &Mesh, marker: i32, omega: f64, velocity: Complex64) -> Vec<Complex64> {
        let h = Complex64::new(0.0, omega * RHO) * velocity;
        drive_pattern(mesh, marker).iter().map(|&p| h * p).collect()
    }

    fn full_solve(
        mesh: &Mesh,
        boundary_tags: &[i32],
        k: Complex64,
        boundary_coeffs: &HashMap<usize, Complex64>,
        rhs: &[Complex64],
    ) -> Vec<Complex64> {
        let assembler =
            build_acoustics_assembler_with_boundaries(mesh, PolynomialDegree::P1, boundary_tags);
        let matrix = assembler.assemble_helmholtz(k, boundary_coeffs);
        let result = solve_complex(&matrix, rhs, &solver_config()).expect("full solve runs");
        assert!(result.converged, "full solve did not converge");
        result.x
    }

    fn relative_error(a: &[Complex64], b: &[Complex64]) -> f64 {
        let diff: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).norm_sqr())
            .sum::<f64>()
            .sqrt();
        diff / norm2(b)
    }

    #[test]
    fn basis_is_m_orthonormal_and_covers_band() {
        let mesh = box_mesh();
        let config = AcousticRomConfig::for_sweep(C0, 1100.0);
        let rom = AcousticRom::build(&mesh, PolynomialDegree::P1, &[], &config).unwrap();

        assert!(rom.num_modes() >= config.min_modes);
        assert!(
            rom.highest_mode_hz() >= config.band_hz(),
            "highest mode {:.1} Hz below band {:.1} Hz",
            rom.highest_mode_hz(),
            config.band_hz()
        );
        // Zero mode (compliance) is the first retained mode.
        assert!(
            rom.eigenvalues()[0] < 1e-6,
            "first eigenvalue {} should be the rigid-wall zero mode",
            rom.eigenvalues()[0]
        );

        // Φᵀ M Φ ≈ I on the assembled mass matrix.
        let assembler = build_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &[]);
        let mass = assembler.assemble_real(&[0.0, 1.0]);
        let deviation = m_orthonormal_deviation(&mass, rom.basis());
        assert!(
            deviation < 1e-8,
            "basis not M-orthonormal: max deviation {deviation:.3e}"
        );
    }

    #[test]
    fn sealed_box_rom_matches_full_across_first_modes() {
        let mesh = box_mesh();
        let rom = AcousticRom::build(
            &mesh,
            PolynomialDegree::P1,
            &[],
            &AcousticRomConfig::for_sweep(C0, 1100.0),
        )
        .unwrap();

        // Static correction for the ω-independent drive pattern (two real
        // shifted-static solves, reused for the whole sweep).
        let pattern = drive_pattern(&mesh, DRIVE_MARKER);
        let correction = rom.static_correction(&pattern, &solver_config()).unwrap();

        // Off-mode frequencies (both models share the discrete poles, but
        // GMRES on the lossless full system is happier away from them).
        let frequencies = [150.0, 300.0, 470.0, 620.0, 760.0, 900.0, 1050.0];
        for &f in &frequencies {
            let omega = std::f64::consts::TAU * f;
            let rhs_scale = Complex64::new(0.0, omega * RHO * 1.0e-2);
            let rhs: Vec<Complex64> = pattern.iter().map(|&p| rhs_scale * p).collect();
            let k = Complex64::new(omega / C0, 0.0);
            let full = full_solve(&mesh, &[], k, &HashMap::new(), &rhs);
            let rom_solution = rom
                .solve_frequency_static(f, &HashMap::new(), &correction, rhs_scale, true)
                .unwrap();
            let err = relative_error(&rom_solution.pressure, &full);
            let residual = rom_solution.relative_residual.unwrap();
            let projected = rom_solution.projected_residual.unwrap();
            println!(
                "sealed ROM at {f:.0} Hz: rel err {err:.3e}, full residual {residual:.3e}, projected {projected:.3e}"
            );
            // Measured on the res-8 box: 4.6e-5 … 6.6e-3 (grows toward the
            // band edge); assert with ~2× headroom.
            assert!(err < 1.5e-2, "ROM error {err:.3e} at {f} Hz");
            assert!(residual < 8.0e-2, "full residual {residual:.3e} at {f} Hz");
            assert!(
                projected < 1.0e-8,
                "projected cross-check residual {projected:.3e} at {f} Hz"
            );
        }
    }

    #[test]
    fn error_decreases_with_mode_count() {
        let mesh = box_mesh();
        for f in [300.0_f64, 900.0] {
            error_decreases_at(&mesh, f);
        }
    }

    fn error_decreases_at(mesh: &Mesh, f: f64) {
        let omega = std::f64::consts::TAU * f;
        let rhs = drive_rhs(mesh, DRIVE_MARKER, omega, Complex64::new(1.0e-2, 0.0));
        let k = Complex64::new(omega / C0, 0.0);
        let full = full_solve(mesh, &[], k, &HashMap::new(), &rhs);

        // Truncated basis: band 140 Hz → only the zero mode is required, so
        // min_modes = 4 pins m to 4.
        let small = AcousticRom::build(
            mesh,
            PolynomialDegree::P1,
            &[],
            &AcousticRomConfig {
                min_modes: 4,
                max_modes: 8,
                ..AcousticRomConfig::for_sweep(C0, 100.0)
            },
        )
        .unwrap();
        assert!(small.num_modes() <= 8);
        let large = AcousticRom::build(
            mesh,
            PolynomialDegree::P1,
            &[],
            &AcousticRomConfig::for_sweep(C0, 1100.0),
        )
        .unwrap();

        let err_small = relative_error(
            &small
                .solve_frequency(f, &HashMap::new(), &rhs, false)
                .unwrap()
                .pressure,
            &full,
        );
        let err_large = relative_error(
            &large
                .solve_frequency(f, &HashMap::new(), &rhs, false)
                .unwrap()
                .pressure,
            &full,
        );
        println!(
            "convergence at {f} Hz: m={} → {err_small:.3e}, m={} → {err_large:.3e}",
            small.num_modes(),
            large.num_modes()
        );
        assert!(
            err_large < err_small,
            "error should decrease with modes: {err_large:.3e} !< {err_small:.3e}"
        );
        assert!(err_large < 9.0);
    }

    #[test]
    fn zero_mode_captures_quasi_static_compliance() {
        let mesh = box_mesh();
        let rom = AcousticRom::build(
            &mesh,
            PolynomialDegree::P1,
            &[],
            &AcousticRomConfig::for_sweep(C0, 200.0),
        )
        .unwrap();
        let f = 20.0;
        let omega = std::f64::consts::TAU * f;
        let velocity = Complex64::new(1.0e-2, 0.0);
        let rhs = drive_rhs(&mesh, DRIVE_MARKER, omega, velocity);
        let k = Complex64::new(omega / C0, 0.0);
        let full = full_solve(&mesh, &[], k, &HashMap::new(), &rhs);
        let rom_solution = rom
            .solve_frequency(f, &HashMap::new(), &rhs, false)
            .unwrap();
        let err = relative_error(&rom_solution.pressure, &full);
        assert!(
            err < 1.0e-3,
            "quasi-static ROM error {err:.3e} (zero mode must be exact)"
        );

        // Quasi-static magnitude: |p̄| = ρ c² |Q| / (ω V), Q = v · S_face ·
        // mean(shape); the tilted-cosine profile averages to 0.6·0.6 = 0.36.
        let driven_area = 0.34 * 0.22 * 0.36;
        let volume = 0.34 * 0.22 * 0.18;
        let analytic = RHO * C0 * C0 * velocity.norm() * driven_area / (omega * volume);
        let mean =
            rom_solution.pressure.iter().sum::<Complex64>() / rom_solution.pressure.len() as f64;
        let rel = (mean.norm() - analytic).abs() / analytic;
        println!(
            "quasi-static: |p̄| = {:.4} (ROM), {analytic:.4} (analytic), rel {rel:.3e}",
            mean.norm()
        );
        // Faceted drive area + P1 cavity: percent-level agreement.
        assert!(rel < 0.10, "quasi-static magnitude off by {rel:.3e}");
    }

    #[test]
    fn damped_wall_rom_matches_full() {
        let wall_marker: i32 = 5;
        let mesh = box_mesh();
        let boundary_tags = [wall_marker];
        let rom = AcousticRom::build(
            &mesh,
            PolynomialDegree::P1,
            &boundary_tags,
            &AcousticRomConfig::for_sweep(C0, 1100.0),
        )
        .unwrap();

        // Moderate wall impedance Z = ρc (4 + 2i): ∂p/∂n = ik(ρc/Z) p enters
        // the weak form as α = −i k ρc/Z on the wall boundary mass.
        let z_ratio = Complex64::new(4.0, 2.0);
        let pattern = drive_pattern(&mesh, DRIVE_MARKER);
        let correction = rom.static_correction(&pattern, &solver_config()).unwrap();
        let frequencies = [300.0, 504.4, 760.0, 1000.0];
        for &f in &frequencies {
            let omega = std::f64::consts::TAU * f;
            let k = omega / C0;
            let alpha = Complex64::new(0.0, -k) / z_ratio;
            let mut coeffs = HashMap::new();
            coeffs.insert(wall_marker as usize, alpha);
            let rhs_scale = Complex64::new(0.0, omega * RHO * 1.0e-2);
            let rhs: Vec<Complex64> = pattern.iter().map(|&p| rhs_scale * p).collect();
            let full = full_solve(&mesh, &boundary_tags, Complex64::new(k, 0.0), &coeffs, &rhs);
            let rom_solution = rom
                .solve_frequency_static(f, &coeffs, &correction, rhs_scale, false)
                .unwrap();
            let err = relative_error(&rom_solution.pressure, &full);
            println!("damped ROM at {f:.0} Hz: rel err {err:.3e}");
            // Measured: 1.6e-2 … 1.2e-1, growing with frequency — the static
            // correction uses the undamped static operator, so the damped
            // wall's boundary-layer tail is only partly captured (documented
            // limitation). Keep the QA gate at the calibrated 15 % ceiling.
            assert!(err < 1.5e-1, "damped ROM error {err:.3e} at {f} Hz");
        }
    }
}
