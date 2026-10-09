//! Product-independent monolithic interior/exterior acoustic coupling.
//!
//! This module owns the numeric core extracted from
//! `sonium-sim-speaker-modeling`'s `fem_bem_coupling::monolithic`: the
//! matrix-free `[p_interior, p_exterior, v_aperture]` operator with a
//! compressed exterior pressure block and compressed velocity RHS block, its
//! field-split preconditioner, and the dense-reference solution record.
//!
//! Product geometry, patch registries, driver networks, and fixture meshes
//! stay in the simulation crates. Those crates construct
//! [`CompressedMonolithicFemBemOperator`] through [`CompressedMonolithicFemBemOperator::new`]
//! (after validating their own convention and transfer contracts) and keep
//! their own product error enums wrapping [`MonolithicCouplingError`].
//!
//! The coupling coefficients use the shared `exp(+ikr)` phasor convention:
//! outward normal velocity `v` and pressure satisfy `dp/dn = iwr v_n`, so the
//! interior scatter carries `-i*omega*rho * area / 3` per P1 vertex.

use num_complex::Complex64;
use sonium_bem::core::assembly::hmatrix::{
    AcaVariant, HMatrix, LowRankPrecision, build_hmatrix_with_variant_precision_and_beta,
};
use sonium_bem::core::assembly::tbem::{HMatrixVelocityOperator, TbemVelocityOperatorError};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner};
use sonium_solvers::{FgmresConfig, IluPreconditioner, fgmres_preconditioned};

/// Errors raised by the product-independent monolithic acoustic operator.
///
/// Product crates keep their own error enums (fixture, mesh-bridge, design
/// failures) and convert these variants explicitly so no product failure mode
/// is silently re-pointed at a generic source.
#[derive(Debug, thiserror::Error)]
pub enum MonolithicCouplingError {
    #[error(transparent)]
    BemCoupling(#[from] sonium_bem::core::coupling::CouplingError),
    #[error(transparent)]
    BemOperator(#[from] TbemVelocityOperatorError),
    #[error("frequency must be finite and positive")]
    InvalidFrequency,
    #[error("dense monolithic system has {requested} DOFs, exceeding policy maximum {maximum}")]
    DenseSystemTooLarge { requested: usize, maximum: usize },
    #[error("monolithic LU factorization failed: {0}")]
    LuFactor(String),
    #[error("monolithic LU solve failed: {0}")]
    LuSolve(String),
    #[error("monolithic solve produced a non-finite state")]
    NonFiniteState,
    #[error("compressed monolithic RHS has {got} entries; expected {expected}")]
    InvalidCompressedRhs { expected: usize, got: usize },
    #[error("compressed monolithic field-split setup failed: {0}")]
    FieldSplitSetup(String),
    #[error(
        "compressed monolithic iterative solve did not converge after {iterations} iterations (residual {residual:e})"
    )]
    IterativeSolveNotConverged { iterations: usize, residual: f64 },
    #[error("aperture element {element} is not an active compressed BEM velocity DOF")]
    InvalidApertureElement { element: usize },
}
/// Build the H-matrix representation of the stabilized velocity-mode TBEM
/// block used by the dense monolithic reference.  The H-matrix velocity
/// adapter separately retains a pressure-mode block and converts it when it
/// builds velocity right-hand sides; the exterior pressure action itself must
/// use this velocity-mode assembly to preserve dense/compressed parity.
pub fn build_velocity_mode_hmatrix(
    elements: &[sonium_bem::core::types::Element],
    nodes: &[[f64; 3]],
    clusters: &[sonium_bem::core::types::Cluster],
    physics: &sonium_bem::core::types::PhysicsParams,
    aca_tolerance: f64,
    max_rank: usize,
    sign: f64,
    beta: Complex64,
) -> HMatrix {
    let mut normalized = elements.to_vec();
    for element in &mut normalized {
        if !element.property.is_evaluation() {
            element.boundary_condition =
                sonium_bem::core::types::BoundaryCondition::Velocity(vec![Complex64::default()]);
        }
    }
    build_hmatrix_with_variant_precision_and_beta(
        &normalized,
        nodes,
        clusters,
        physics,
        aca_tolerance,
        max_rank,
        AcaVariant::default(),
        LowRankPrecision::default(),
        sign,
        beta,
    )
}
/// The dense golden implementation deliberately has a bounded system size.
pub const DEFAULT_MONOLITHIC_MAX_DOFS: usize = 4_096;
/// One full state from the acoustic monolithic fixture solve.
#[derive(Debug, Clone)]
pub struct MonolithicCoupledSolution {
    pub frequency_hz: f64,
    pub interior_pressure: Vec<Complex64>,
    pub exterior_pressure: Vec<Complex64>,
    /// Outward aperture velocity, one entry per aperture triangle.
    pub aperture_velocity: Vec<Complex64>,
    /// Exact dense-system relative residual.
    pub relative_residual: f64,
    /// Interior Helmholtz/Neumann block residual, normalized by the full RHS.
    pub interior_relative_residual: f64,
    /// Exterior stabilized TBEM block residual, normalized by the full RHS.
    pub exterior_relative_residual: f64,
    /// Aperture pressure-continuity residual, normalized by the full RHS.
    pub interface_relative_residual: f64,
    /// Per-aperture pressure continuity residual `p_interior - p_exterior`.
    pub aperture_pressure_jump: Vec<Complex64>,
}

impl MonolithicCoupledSolution {
    /// Aperture volume velocity `Σ_j area_j v_j`.
    pub fn aperture_volume_velocity(&self, aperture_areas: &[f64]) -> Complex64 {
        self.aperture_velocity
            .iter()
            .zip(aperture_areas)
            .map(|(velocity, area)| *velocity * *area)
            .sum()
    }
}
/// Matrix-free monolithic acoustic fixture operator with a compressed
/// exterior pressure block and compressed velocity RHS block.
///
/// The unknown layout is `[p_interior, p_exterior, v_aperture]`.  This operator
/// is the bounded H-matrix seam for iterative callers: it retains the same
/// trace and `-iωρ` coupling coefficients without materializing the exterior
/// dense matrix.  Product electromechanical blocks layer on top of this
/// acoustic operator in the simulation crates.
pub struct CompressedMonolithicFemBemOperator {
    fem: CsrMatrix<Complex64>,
    exterior: HMatrix,
    velocity: HMatrixVelocityOperator,
    aperture_faces: Vec<([usize; 3], f64)>,
    aperture_bem_elements: Vec<usize>,
    aperture_velocity_slots: Vec<usize>,
    aperture_source_nodes: Vec<usize>,
    interface_weights: Vec<Vec<(usize, f64)>>,
    omega_density: Complex64,
}

impl CompressedMonolithicFemBemOperator {
    /// Assemble the operator from its validated blocks.
    ///
    /// The caller owns convention validation, transfer construction, and the
    /// aperture topology; see the product adapters in
    /// `sonium-sim-speaker-modeling` for the fixture-driven construction.
    pub fn new(
        fem: CsrMatrix<Complex64>,
        exterior: HMatrix,
        velocity: HMatrixVelocityOperator,
        aperture_faces: Vec<([usize; 3], f64)>,
        aperture_bem_elements: Vec<usize>,
        aperture_velocity_slots: Vec<usize>,
        aperture_source_nodes: Vec<usize>,
        interface_weights: Vec<Vec<(usize, f64)>>,
        omega_density: Complex64,
    ) -> Self {
        Self {
            fem,
            exterior,
            velocity,
            aperture_faces,
            aperture_bem_elements,
            aperture_velocity_slots,
            aperture_source_nodes,
            interface_weights,
            omega_density,
        }
    }

    /// Number of interior pressure unknowns.
    pub fn num_interior_dofs(&self) -> usize {
        self.fem.num_rows()
    }

    /// Number of exterior pressure unknowns.
    pub fn num_exterior_dofs(&self) -> usize {
        self.exterior.num_dofs
    }

    /// Number of explicit aperture velocity unknowns.
    pub fn num_aperture_dofs(&self) -> usize {
        self.aperture_faces.len()
    }

    /// Exterior velocity-block slot per aperture unknown, in aperture order.
    pub fn aperture_velocity_slots(&self) -> &[usize] {
        &self.aperture_velocity_slots
    }

    fn offsets(&self) -> (usize, usize, usize) {
        let interior = self.num_interior_dofs();
        let exterior = interior + self.num_exterior_dofs();
        let total = exterior + self.num_aperture_dofs();
        (interior, exterior, total)
    }

    fn pressure_from_aperture(&self, aperture_velocity: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            aperture_velocity.len(),
            self.num_aperture_dofs(),
            "compressed monolithic aperture input length"
        );
        let (interior_offset, exterior_offset, _) = self.offsets();
        let mut result = vec![Complex64::default(); exterior_offset];
        let mut full_velocity = vec![Complex64::default(); self.velocity.len()];
        for (aperture_index, &slot) in self.aperture_velocity_slots.iter().enumerate() {
            full_velocity[slot] = aperture_velocity[aperture_index];
        }
        let exterior_load = self
            .velocity
            .apply(&full_velocity)
            .expect("validated compressed velocity block input");
        for (row, value) in exterior_load.into_iter().enumerate() {
            result[interior_offset + row] -= value;
        }
        for (aperture_index, &(nodes, area)) in self.aperture_faces.iter().enumerate() {
            for node in nodes {
                result[node] -=
                    self.omega_density * (area / 3.0) * aperture_velocity[aperture_index];
            }
        }
        result
    }

    fn aperture_from_pressure(&self, pressure: &[Complex64]) -> Vec<Complex64> {
        let (interior_offset, exterior_offset, _) = self.offsets();
        assert_eq!(
            pressure.len(),
            exterior_offset,
            "compressed monolithic pressure input length"
        );
        self.aperture_bem_elements
            .iter()
            .enumerate()
            .map(|(aperture_index, &bem_element)| {
                let trace = self.interface_weights[aperture_index].iter().fold(
                    Complex64::default(),
                    |value, &(source, weight)| {
                        value
                            + Complex64::new(weight, 0.0)
                                * pressure[self.aperture_source_nodes[source]]
                    },
                );
                trace - pressure[interior_offset + bem_element]
            })
            .collect()
    }

    /// Build the pressure/velocity field-split preconditioner used by the
    /// compressed FGMRES product path.
    pub fn build_field_split_preconditioner(
        &self,
    ) -> Result<CompressedMonolithicFieldSplitPreconditioner, MonolithicCouplingError> {
        let interior = IluPreconditioner::try_from_csr(&self.fem)
            .map_err(|error| MonolithicCouplingError::FieldSplitSetup(error.to_string()))?;
        let interior_size = self.num_interior_dofs();
        let pressure_size = self.offsets().1;

        // Probe only the diagonal of the compressed exterior operator. This
        // keeps setup matrix-free while providing a bounded pressure-block
        // scaling for the H-matrix field.
        let mut exterior_inverse_diagonal = vec![Complex64::default(); self.num_exterior_dofs()];
        for (column, inverse) in exterior_inverse_diagonal.iter_mut().enumerate() {
            let mut basis = vec![Complex64::default(); self.num_exterior_dofs()];
            basis[column] = Complex64::new(1.0, 0.0);
            let response = LinearOperator::apply(&self.exterior, &basis);
            let diagonal = response[column];
            *inverse = if diagonal.norm() > 1.0e-14 && diagonal.norm().is_finite() {
                Complex64::new(1.0, 0.0) / diagonal
            } else {
                Complex64::new(1.0, 0.0)
            };
        }

        let aperture_size = self.num_aperture_dofs();
        let mut pressure_from_aperture =
            DenseMatrix::from_elem(pressure_size, aperture_size, Complex64::default());
        for aperture in 0..aperture_size {
            let mut unit = vec![Complex64::default(); aperture_size];
            unit[aperture] = Complex64::new(1.0, 0.0);
            pressure_from_aperture.set_column(aperture, &self.pressure_from_aperture(&unit));
        }

        let mut aperture_from_pressure =
            DenseMatrix::from_elem(aperture_size, pressure_size, Complex64::default());
        for pressure in 0..pressure_size {
            let mut unit = vec![Complex64::default(); pressure_size];
            unit[pressure] = Complex64::new(1.0, 0.0);
            aperture_from_pressure.set_column(pressure, &self.aperture_from_pressure(&unit));
        }

        let pressure_inverse = |rhs: &[Complex64]| {
            let mut result = vec![Complex64::default(); pressure_size];
            let interior_solution = interior.apply(&rhs[..interior_size]);
            result[..interior_size].copy_from_slice(&interior_solution);
            for (index, &inverse) in exterior_inverse_diagonal.iter().enumerate() {
                result[interior_size + index] = rhs[interior_size + index] * inverse;
            }
            result
        };

        // S = -C M_p^{-1} B is the small aperture Schur complement of the
        // pressure field split. Only this dense matrix is materialized.
        let mut schur = DenseMatrix::from_elem(aperture_size, aperture_size, Complex64::default());
        for aperture in 0..aperture_size {
            let mut b_column = vec![Complex64::default(); pressure_size];
            for row in 0..pressure_size {
                b_column[row] = pressure_from_aperture[[row, aperture]];
            }
            let pressure_response = pressure_inverse(&b_column);
            let aperture_response = aperture_from_pressure.matvec(&pressure_response);
            for row in 0..aperture_size {
                schur[[row, aperture]] = -aperture_response[row];
            }
        }
        let schur_factor = lu_factorize(&schur)
            .map_err(|error| MonolithicCouplingError::FieldSplitSetup(error.to_string()))?;
        let mut schur_inverse =
            DenseMatrix::from_elem(aperture_size, aperture_size, Complex64::default());
        for column in 0..aperture_size {
            let mut unit = vec![Complex64::default(); aperture_size];
            unit[column] = Complex64::new(1.0, 0.0);
            let solved = schur_factor
                .solve(&unit)
                .map_err(|error| MonolithicCouplingError::FieldSplitSetup(error.to_string()))?;
            schur_inverse.set_column(column, &solved);
        }

        Ok(CompressedMonolithicFieldSplitPreconditioner {
            interior,
            interior_size,
            pressure_size,
            exterior_inverse_diagonal,
            pressure_from_aperture,
            aperture_from_pressure,
            aperture_schur_inverse: schur_inverse,
        })
    }

    /// Solve one compressed monolithic acoustic system with field-split
    /// FGMRES and return the true global residual.
    pub fn solve_compressed(
        &self,
        rhs: &[Complex64],
        config: &FgmresConfig<f64>,
    ) -> Result<CompressedMonolithicSolution, MonolithicCouplingError> {
        if rhs.len() != self.num_rows() {
            return Err(MonolithicCouplingError::InvalidCompressedRhs {
                expected: self.num_rows(),
                got: rhs.len(),
            });
        }
        if rhs.iter().any(|value| !value.norm().is_finite()) {
            return Err(MonolithicCouplingError::NonFiniteState);
        }
        let preconditioner = self.build_field_split_preconditioner()?;
        let solved = fgmres_preconditioned(self, &preconditioner, rhs, None, config);
        let mut residual = self.apply(&solved.x);
        let rhs_norm = rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        for (value, expected) in residual.iter_mut().zip(rhs) {
            *value -= *expected;
        }
        let relative_residual = residual
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / rhs_norm;
        if !solved.converged
            || !relative_residual.is_finite()
            || relative_residual > config.tolerance
        {
            return Err(MonolithicCouplingError::IterativeSolveNotConverged {
                iterations: solved.iterations,
                residual: relative_residual,
            });
        }
        Ok(CompressedMonolithicSolution {
            state: solved.x,
            iterations: solved.iterations,
            relative_residual,
        })
    }
}

/// Field-split preconditioner for the compressed monolithic acoustic block.
///
/// The pressure field uses FEM ILU plus a matrix-free H-matrix diagonal, and
/// the explicit aperture velocity field uses a dense Schur complement. This
/// is a genuine single Krylov solve preconditioner; it does not partition the
/// accepted product state into independent solves.
pub struct CompressedMonolithicFieldSplitPreconditioner {
    interior: IluPreconditioner<Complex64>,
    interior_size: usize,
    pressure_size: usize,
    exterior_inverse_diagonal: Vec<Complex64>,
    pressure_from_aperture: DenseMatrix<Complex64>,
    aperture_from_pressure: DenseMatrix<Complex64>,
    aperture_schur_inverse: DenseMatrix<Complex64>,
}

impl CompressedMonolithicFieldSplitPreconditioner {
    fn pressure_inverse(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(rhs.len(), self.pressure_size);
        let mut result = vec![Complex64::default(); self.pressure_size];
        let interior_solution = self.interior.apply(&rhs[..self.interior_size]);
        result[..self.interior_size].copy_from_slice(&interior_solution);
        for (index, &inverse) in self.exterior_inverse_diagonal.iter().enumerate() {
            result[self.interior_size + index] = rhs[self.interior_size + index] * inverse;
        }
        result
    }
}

impl Preconditioner<Complex64> for CompressedMonolithicFieldSplitPreconditioner {
    fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            rhs.len(),
            self.pressure_size + self.aperture_from_pressure.nrows()
        );
        let (pressure_rhs, aperture_rhs) = rhs.split_at(self.pressure_size);
        let pressure_initial = self.pressure_inverse(pressure_rhs);
        let pressure_trace = self.aperture_from_pressure.matvec(&pressure_initial);
        let schur_rhs = aperture_rhs
            .iter()
            .zip(pressure_trace)
            .map(|(rhs, trace)| *rhs - trace)
            .collect::<Vec<_>>();
        let aperture = self.aperture_schur_inverse.matvec(&schur_rhs);
        let pressure_load = self.pressure_from_aperture.matvec(&aperture);
        let corrected_pressure = pressure_rhs
            .iter()
            .zip(pressure_load)
            .map(|(rhs, load)| *rhs - load)
            .collect::<Vec<_>>();
        let pressure = self.pressure_inverse(&corrected_pressure);
        let mut result = pressure;
        result.extend(aperture);
        result
    }
}

/// Result of one field-split compressed monolithic FGMRES solve.
#[derive(Debug, Clone)]
pub struct CompressedMonolithicSolution {
    /// Concatenated `[p_interior, p_exterior, v_aperture]` state.
    pub state: Vec<Complex64>,
    /// Number of Krylov iterations performed.
    pub iterations: usize,
    /// True global relative residual recomputed with the original operator.
    pub relative_residual: f64,
}

impl LinearOperator<Complex64> for CompressedMonolithicFemBemOperator {
    fn num_rows(&self) -> usize {
        self.offsets().2
    }

    fn num_cols(&self) -> usize {
        self.offsets().2
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let (interior_offset, aperture_offset, total) = self.offsets();
        assert_eq!(x.len(), total, "compressed monolithic input length");
        assert_eq!(y.len(), total, "compressed monolithic output length");
        y.fill(Complex64::default());

        self.fem
            .apply_into(&x[..interior_offset], &mut y[..interior_offset]);
        self.exterior.apply_into(
            &x[interior_offset..aperture_offset],
            &mut y[interior_offset..aperture_offset],
        );
        let mut full_velocity = vec![Complex64::default(); self.velocity.len()];
        for (aperture_index, &slot) in self.aperture_velocity_slots.iter().enumerate() {
            full_velocity[slot] = x[aperture_offset + aperture_index];
        }
        let velocity = self
            .velocity
            .apply(&full_velocity)
            .expect("validated compressed velocity block input");
        for (row, value) in velocity.into_iter().enumerate() {
            y[interior_offset + row] -= value;
        }

        for (aperture_index, (face, &bem_element)) in self
            .aperture_faces
            .iter()
            .zip(&self.aperture_bem_elements)
            .enumerate()
        {
            let (nodes, area) = *face;
            let value = x[aperture_offset + aperture_index];
            for node in nodes {
                y[node] -= self.omega_density * (area / 3.0) * value;
            }
            let interface_row = aperture_offset + aperture_index;
            for &(source, weight) in &self.interface_weights[aperture_index] {
                y[interface_row] +=
                    Complex64::new(weight, 0.0) * x[self.aperture_source_nodes[source]];
            }
            y[interface_row] -= x[interior_offset + bem_element];
        }
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let (interior_offset, aperture_offset, total) = self.offsets();
        assert_eq!(
            x.len(),
            total,
            "compressed monolithic transpose input length"
        );
        assert_eq!(
            y.len(),
            total,
            "compressed monolithic transpose output length"
        );
        y.fill(Complex64::default());

        self.fem
            .apply_transpose_into(&x[..interior_offset], &mut y[..interior_offset]);
        self.exterior.apply_transpose_into(
            &x[interior_offset..aperture_offset],
            &mut y[interior_offset..aperture_offset],
        );
        let velocity = self
            .velocity
            .apply_transpose(&x[interior_offset..aperture_offset])
            .expect("validated compressed velocity block input");
        for (aperture_index, &slot) in self.aperture_velocity_slots.iter().enumerate() {
            y[aperture_offset + aperture_index] -= velocity[slot];
        }

        for (aperture_index, (face, &bem_element)) in self
            .aperture_faces
            .iter()
            .zip(&self.aperture_bem_elements)
            .enumerate()
        {
            let (nodes, area) = *face;
            let interface_value = x[aperture_offset + aperture_index];
            for &(source, weight) in &self.interface_weights[aperture_index] {
                y[self.aperture_source_nodes[source]] +=
                    Complex64::new(weight, 0.0) * interface_value;
            }
            y[interior_offset + bem_element] -= interface_value;
            for node in nodes {
                y[aperture_offset + aperture_index] -= self.omega_density * (area / 3.0) * x[node];
            }
        }
    }

    fn apply_hermitian(&self, x: &[Complex64]) -> Vec<Complex64> {
        let (interior_offset, aperture_offset, total) = self.offsets();
        assert_eq!(
            x.len(),
            total,
            "compressed monolithic Hermitian input length"
        );
        let mut y = vec![Complex64::default(); total];
        let fem_hermitian = self.fem.apply_hermitian(&x[..interior_offset]);
        y[..interior_offset].copy_from_slice(&fem_hermitian);
        let exterior_hermitian = self
            .exterior
            .apply_hermitian(&x[interior_offset..aperture_offset]);
        y[interior_offset..aperture_offset].copy_from_slice(&exterior_hermitian);
        let velocity = self
            .velocity
            .apply_hermitian(&x[interior_offset..aperture_offset])
            .expect("validated compressed velocity block input");
        for (aperture_index, &slot) in self.aperture_velocity_slots.iter().enumerate() {
            y[aperture_offset + aperture_index] -= velocity[slot];
        }

        for (aperture_index, (face, &bem_element)) in self
            .aperture_faces
            .iter()
            .zip(&self.aperture_bem_elements)
            .enumerate()
        {
            let (nodes, area) = *face;
            let interface_value = x[aperture_offset + aperture_index];
            for &(source, weight) in &self.interface_weights[aperture_index] {
                y[self.aperture_source_nodes[source]] +=
                    Complex64::new(weight, 0.0) * interface_value;
            }
            y[interior_offset + bem_element] -= interface_value;
            for node in nodes {
                y[aperture_offset + aperture_index] -=
                    self.omega_density.conj() * (area / 3.0) * x[node];
            }
        }
        y
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// Dense reference operator for one fixture topology.
/// Matrix-free product FEM/BEM operator with explicit port and
/// electromechanical unknowns.
///
/// The unknown layout is `[p_interior, p_exterior, v_port, u_driver,
/// i_driver]`. Port and driver velocities are expanded onto their canonical
/// exterior BEM elements before the compressed velocity RHS action, while
/// the pressure, force, and electrical rows retain the dense reference's
/// work-conjugate coefficients.
/// Shared pressure/aperture field-split setup used by both the acoustic
/// operator's own [`CompressedMonolithicFemBemOperator::build_field_split_preconditioner`]
/// and the product operator's acoustic block. The caller supplies the
/// already-assembled blocks; only the small aperture Schur complement is
/// materialized.
pub fn build_pressure_port_field_split_preconditioner(
    fem: &CsrMatrix<Complex64>,
    exterior: &HMatrix,
    velocity: &HMatrixVelocityOperator,
    aperture_faces: &[([usize; 3], f64)],
    aperture_bem_elements: &[usize],
    aperture_velocity_slots: &[usize],
    aperture_source_nodes: &[usize],
    interface_weights: &[Vec<(usize, f64)>],
    omega_density: Complex64,
) -> Result<CompressedMonolithicFieldSplitPreconditioner, MonolithicCouplingError> {
    let interior = IluPreconditioner::try_from_csr(fem)
        .map_err(|error| MonolithicCouplingError::FieldSplitSetup(error.to_string()))?;
    let interior_size = fem.num_rows();
    let pressure_size = interior_size + exterior.num_dofs;

    let mut exterior_inverse_diagonal = vec![Complex64::default(); exterior.num_dofs];
    for (column, inverse) in exterior_inverse_diagonal.iter_mut().enumerate() {
        let mut basis = vec![Complex64::default(); exterior.num_dofs];
        basis[column] = Complex64::new(1.0, 0.0);
        let response = LinearOperator::apply(exterior, &basis);
        let diagonal = response[column];
        *inverse = if diagonal.norm() > 1.0e-14 && diagonal.norm().is_finite() {
            Complex64::new(1.0, 0.0) / diagonal
        } else {
            Complex64::new(1.0, 0.0)
        };
    }

    let pressure_from_aperture = |aperture_velocity: &[Complex64]| {
        let mut result = vec![Complex64::default(); pressure_size];
        let mut full_velocity = vec![Complex64::default(); velocity.len()];
        for (aperture_index, &slot) in aperture_velocity_slots.iter().enumerate() {
            full_velocity[slot] = aperture_velocity[aperture_index];
        }
        let exterior_load = velocity
            .apply(&full_velocity)
            .expect("validated compressed pressure/port velocity input");
        for (row, value) in exterior_load.into_iter().enumerate() {
            result[interior_size + row] -= value;
        }
        for (aperture_index, &(nodes, area)) in aperture_faces.iter().enumerate() {
            for node in nodes {
                result[node] -= omega_density * (area / 3.0) * aperture_velocity[aperture_index];
            }
        }
        result
    };
    let aperture_from_pressure = |pressure: &[Complex64]| {
        aperture_bem_elements
            .iter()
            .enumerate()
            .map(|(aperture_index, &bem_element)| {
                let trace = interface_weights[aperture_index].iter().fold(
                    Complex64::default(),
                    |value, &(source, weight)| {
                        value
                            + Complex64::new(weight, 0.0) * pressure[aperture_source_nodes[source]]
                    },
                );
                trace - pressure[interior_size + bem_element]
            })
            .collect::<Vec<_>>()
    };

    let mut pressure_from_aperture_matrix =
        DenseMatrix::from_elem(pressure_size, aperture_faces.len(), Complex64::default());
    for aperture in 0..aperture_faces.len() {
        let mut unit = vec![Complex64::default(); aperture_faces.len()];
        unit[aperture] = Complex64::new(1.0, 0.0);
        pressure_from_aperture_matrix.set_column(aperture, &pressure_from_aperture(&unit));
    }

    let mut aperture_from_pressure_matrix =
        DenseMatrix::from_elem(aperture_faces.len(), pressure_size, Complex64::default());
    for pressure in 0..pressure_size {
        let mut unit = vec![Complex64::default(); pressure_size];
        unit[pressure] = Complex64::new(1.0, 0.0);
        aperture_from_pressure_matrix.set_column(pressure, &aperture_from_pressure(&unit));
    }

    let pressure_inverse = |rhs: &[Complex64]| {
        let mut result = vec![Complex64::default(); pressure_size];
        let interior_solution = interior.apply(&rhs[..interior_size]);
        result[..interior_size].copy_from_slice(&interior_solution);
        for (index, &inverse) in exterior_inverse_diagonal.iter().enumerate() {
            result[interior_size + index] = rhs[interior_size + index] * inverse;
        }
        result
    };

    let mut aperture_schur_inverse = DenseMatrix::from_elem(
        aperture_faces.len(),
        aperture_faces.len(),
        Complex64::default(),
    );
    if !aperture_faces.is_empty() {
        let mut schur = DenseMatrix::from_elem(
            aperture_faces.len(),
            aperture_faces.len(),
            Complex64::default(),
        );
        for aperture in 0..aperture_faces.len() {
            let mut b_column = vec![Complex64::default(); pressure_size];
            for row in 0..pressure_size {
                b_column[row] = pressure_from_aperture_matrix[[row, aperture]];
            }
            let pressure_response = pressure_inverse(&b_column);
            let aperture_response = aperture_from_pressure_matrix.matvec(&pressure_response);
            for row in 0..aperture_faces.len() {
                schur[[row, aperture]] = -aperture_response[row];
            }
        }
        let schur_factor = lu_factorize(&schur)
            .map_err(|error| MonolithicCouplingError::FieldSplitSetup(error.to_string()))?;
        for column in 0..aperture_faces.len() {
            let mut unit = vec![Complex64::default(); aperture_faces.len()];
            unit[column] = Complex64::new(1.0, 0.0);
            let solved = schur_factor
                .solve(&unit)
                .map_err(|error| MonolithicCouplingError::FieldSplitSetup(error.to_string()))?;
            aperture_schur_inverse.set_column(column, &solved);
        }
    }

    Ok(CompressedMonolithicFieldSplitPreconditioner {
        interior,
        interior_size,
        pressure_size,
        exterior_inverse_diagonal,
        pressure_from_aperture: pressure_from_aperture_matrix,
        aperture_from_pressure: aperture_from_pressure_matrix,
        aperture_schur_inverse,
    })
}
