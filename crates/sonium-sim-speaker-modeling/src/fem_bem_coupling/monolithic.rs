//! Dense monolithic interior-FEM/exterior-BEM acoustic reference solve.
//!
//! Unlike [`super::coupled::FemBemCoupling`], this module retains exterior
//! pressure and aperture velocity as unknowns.  Its first fixture form solves
//!
//! ```text
//! x = [p_interior, p_exterior, v_aperture]
//! ```
//!
//! in one system.  A prescribed driver velocity remains a right-hand-side
//! load for this initial algebraic reference.  Driver mechanical/electrical
//! unknowns are added on top of this operator once the product patch registry
//! supplies distinct front/rear interfaces.
use num_complex::Complex64;
use sonium_bem::core::ConservativeTraceTransfer;
use sonium_bem::core::assembly::hmatrix::HMatrix;
use sonium_bem::core::assembly::tbem::{
    HMatrixVelocityOperator, StabilizedTbemVelocityOperator, TbemVelocityOperatorError,
};
use sonium_bem::core::solver::frequency_sweep::validate_frequency_grid;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::acoustics::pressure::{
    AcousticsAssemblerExt, build_acoustics_assembler_with_boundaries,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_factorize;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{AdjointCapability, LinearOperator, Preconditioner};
use sonium_solvers::{FgmresConfig, fgmres_preconditioned};
use std::collections::HashMap;

// Product-independent numeric core lives in the orchestration crate; this
// module keeps the product constructors, dense product references, and the
// product error superset. The re-exported names preserve the existing
// `fem_bem_coupling::monolithic` paths for downstream consumers.
use sonium_coupling::MonolithicCouplingError as GenericMonolithicError;
pub use sonium_coupling::{
    CompressedMonolithicFemBemOperator, CompressedMonolithicFieldSplitPreconditioner,
    CompressedMonolithicSolution, DEFAULT_MONOLITHIC_MAX_DOFS, MonolithicCoupledSolution,
    build_pressure_port_field_split_preconditioner, build_velocity_mode_hmatrix,
};

use super::coupled::{CouplingDrive, CouplingError, FemBemCoupling};
use super::fixtures::CouplingFixture;
use super::mesh_bridge::{
    MeshBridgeError, ProductCouplingMesh, ProductInterfaceTrace, build_exterior_bem_mesh,
};
use crate::product_acoustic_mesh::{ProductInteriorMesh, ProductInteriorPatchKind};
use crate::product_design::{ProductDesign, ProductDesignError};
use crate::structural_acoustic::StructuralAcousticModalBlock;
/// Errors raised by the dense monolithic acoustic reference.
#[derive(Debug, thiserror::Error)]
pub enum MonolithicCouplingError {
    #[error(transparent)]
    Coupling(#[from] CouplingError),
    #[error(transparent)]
    BemCoupling(#[from] sonium_bem::core::coupling::CouplingError),
    #[error(transparent)]
    MeshBridge(#[from] MeshBridgeError),
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
impl From<GenericMonolithicError> for MonolithicCouplingError {
    /// Map the orchestration-crate acoustic errors onto the identical product
    /// variants. Each arm is explicit so a new generic variant fails to
    /// compile here instead of silently landing on a wrong product error.
    fn from(error: GenericMonolithicError) -> Self {
        match error {
            GenericMonolithicError::BemCoupling(error) => Self::BemCoupling(error),
            GenericMonolithicError::BemOperator(error) => Self::BemOperator(error),
            GenericMonolithicError::InvalidFrequency => Self::InvalidFrequency,
            GenericMonolithicError::DenseSystemTooLarge { requested, maximum } => {
                Self::DenseSystemTooLarge { requested, maximum }
            }
            GenericMonolithicError::LuFactor(error) => Self::LuFactor(error),
            GenericMonolithicError::LuSolve(error) => Self::LuSolve(error),
            GenericMonolithicError::NonFiniteState => Self::NonFiniteState,
            GenericMonolithicError::InvalidCompressedRhs { expected, got } => {
                Self::InvalidCompressedRhs { expected, got }
            }
            GenericMonolithicError::FieldSplitSetup(error) => Self::FieldSplitSetup(error),
            GenericMonolithicError::IterativeSolveNotConverged {
                iterations,
                residual,
            } => Self::IterativeSolveNotConverged {
                iterations,
                residual,
            },
            GenericMonolithicError::InvalidApertureElement { element } => {
                Self::InvalidApertureElement { element }
            }
        }
    }
}
/// Block-diagonal field split for a compressed product solve. The acoustic
/// field split is reused for `[p_interior, p_exterior, v_port]`; each driver
/// uses the exact inverse of its coupled mechanical/electrical 2x2 block.
pub struct CompressedMonolithicProductFieldSplitPreconditioner {
    acoustic: CompressedMonolithicFieldSplitPreconditioner,
    acoustic_size: usize,
    driver_inverses: Vec<[[Complex64; 2]; 2]>,
    structural_inverses: Vec<Complex64>,
}

impl Preconditioner<Complex64> for CompressedMonolithicProductFieldSplitPreconditioner {
    fn apply(&self, rhs: &[Complex64]) -> Vec<Complex64> {
        let mut result = vec![Complex64::default(); rhs.len()];
        let acoustic = self.acoustic.apply(&rhs[..self.acoustic_size]);
        result[..self.acoustic_size].copy_from_slice(&acoustic);
        let n_drivers = self.driver_inverses.len();
        let current_offset = self.acoustic_size + n_drivers;
        let structural_offset = current_offset + n_drivers;
        for driver_index in 0..n_drivers {
            let velocity_rhs = rhs[self.acoustic_size + driver_index];
            let current_rhs = rhs[current_offset + driver_index];
            let inverse = self.driver_inverses[driver_index];
            result[self.acoustic_size + driver_index] =
                inverse[0][0] * velocity_rhs + inverse[0][1] * current_rhs;
            result[current_offset + driver_index] =
                inverse[1][0] * velocity_rhs + inverse[1][1] * current_rhs;
        }
        for (mode_index, &inverse) in self.structural_inverses.iter().enumerate() {
            result[structural_offset + mode_index] = inverse * rhs[structural_offset + mode_index];
        }
        result
    }
}

/// Result of a compressed product FGMRES solve. `state` uses the operator's
/// `[p_interior, p_exterior, v_port, u_driver, i_driver, q_structural]`
/// layout.
#[derive(Debug, Clone)]
pub struct CompressedMonolithicProductSolution {
    pub state: Vec<Complex64>,
    pub iterations: usize,
    pub relative_residual: f64,
    pub driver_mechanical_relative_residual: Vec<f64>,
    pub driver_electrical_relative_residual: Vec<f64>,
    /// Maximum relative residual across explicit structural modal rows.
    pub structural_relative_residual: f64,
}

/// One frequency-tagged point returned by a compressed product sweep.
#[derive(Debug, Clone)]
pub struct CompressedMonolithicProductSweepPoint {
    pub frequency_hz: f64,
    pub solution: CompressedMonolithicProductSolution,
}

pub struct CompressedMonolithicProductOperator {
    fem: CsrMatrix<Complex64>,
    exterior: HMatrix,
    velocity: HMatrixVelocityOperator,
    exterior_element_areas: Vec<f64>,
    port_faces: Vec<([usize; 3], f64)>,
    port_bem_elements: Vec<usize>,
    port_velocity_slots: Vec<usize>,
    driver_faces: Vec<Vec<([usize; 3], f64)>>,
    driver_bem_elements: Vec<Vec<usize>>,
    driver_velocity_slots: Vec<Vec<usize>>,
    driver_mechanical_impedances: Vec<Complex64>,
    driver_force_factors: Vec<Complex64>,
    driver_coil_impedances: Vec<Complex64>,
    structural_dynamic_stiffness: Vec<Complex64>,
    structural_pressure_weights: Vec<Vec<(usize, Complex64)>>,
    structural_velocity_weights: Vec<Vec<(usize, Complex64)>>,
    omega_density: Complex64,
}

impl CompressedMonolithicProductOperator {
    fn offsets(&self) -> (usize, usize, usize, usize, usize, usize) {
        let n_interior = self.fem.num_rows();
        let n_exterior = self.exterior.num_dofs;
        let n_port = self.port_faces.len();
        let n_driver = self.driver_faces.len();
        let port_offset = n_interior + n_exterior;
        let velocity_offset = port_offset + n_port;
        let current_offset = velocity_offset + n_driver;
        let structural_offset = current_offset + n_driver;
        (
            n_interior,
            n_exterior,
            port_offset,
            velocity_offset,
            current_offset,
            structural_offset + self.structural_dynamic_stiffness.len(),
        )
    }

    fn structural_offset(&self) -> usize {
        let (_, _, _, _, current_offset, _) = self.offsets();
        current_offset + self.driver_faces.len()
    }

    /// Number of explicit structural modal unknowns.
    pub fn num_structural_modes(&self) -> usize {
        self.structural_dynamic_stiffness.len()
    }

    /// Total number of product unknowns.
    pub fn num_rows(&self) -> usize {
        self.offsets().5
    }

    /// Total number of product unknowns.
    pub fn num_cols(&self) -> usize {
        self.offsets().5
    }

    /// Build the block-diagonal field-split preconditioner for the product
    /// operator. The pressure/port setup is the same field split used by the
    /// compressed acoustic operator; driver blocks remain exact 2x2 solves.
    pub fn build_field_split_preconditioner(
        &self,
    ) -> Result<CompressedMonolithicProductFieldSplitPreconditioner, MonolithicProductCouplingError>
    {
        let (n_interior, _n_exterior, _port_offset, velocity_offset, _current_offset, _total) =
            self.offsets();
        let aperture_source_nodes = (0..n_interior).collect::<Vec<_>>();
        let interface_weights = self
            .port_faces
            .iter()
            .map(|&(nodes, _)| {
                nodes
                    .into_iter()
                    .map(|node| (node, 1.0 / 3.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let acoustic = build_pressure_port_field_split_preconditioner(
            &self.fem,
            &self.exterior,
            &self.velocity,
            &self.port_faces,
            &self.port_bem_elements,
            &self.port_velocity_slots,
            &aperture_source_nodes,
            &interface_weights,
            self.omega_density,
        )
        .map_err(|error| {
            MonolithicProductCouplingError::CompressedFieldSplitSetup(error.to_string())
        })?;

        let mut driver_inverses = Vec::with_capacity(self.driver_faces.len());
        for driver_index in 0..self.driver_faces.len() {
            let mechanical = self.driver_mechanical_impedances[driver_index];
            let force = self.driver_force_factors[driver_index];
            let electrical = self.driver_coil_impedances[driver_index];
            let determinant = mechanical * electrical + force * force;
            if !determinant.is_finite() || determinant.norm() <= 1.0e-14 {
                return Err(MonolithicProductCouplingError::CompressedFieldSplitSetup(
                    format!("driver {driver_index} mechanical/electrical block is singular"),
                ));
            }
            driver_inverses.push([
                [electrical / determinant, force / determinant],
                [-force / determinant, mechanical / determinant],
            ]);
        }

        let structural_inverses = self
            .structural_dynamic_stiffness
            .iter()
            .map(|&dynamic| {
                if !dynamic.is_finite() || dynamic.norm_sqr() <= 1.0e-28 {
                    return Err(MonolithicProductCouplingError::CompressedFieldSplitSetup(
                        "structural modal block is singular".to_owned(),
                    ));
                }
                Ok(Complex64::new(1.0, 0.0) / dynamic)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(CompressedMonolithicProductFieldSplitPreconditioner {
            acoustic,
            acoustic_size: velocity_offset,
            driver_inverses,
            structural_inverses,
        })
    }

    /// Solve a compressed product system with right-preconditioned FGMRES.
    /// The caller supplies the complete RHS so this method is also useful for
    /// non-voltage product excitations; [`MonolithicSealedProductCoupling`]
    /// provides the crossover-voltage convenience wrapper.
    pub fn solve_compressed(
        &self,
        rhs: &[Complex64],
        config: &FgmresConfig<f64>,
    ) -> Result<CompressedMonolithicProductSolution, MonolithicProductCouplingError> {
        if rhs.len() != self.num_rows() {
            return Err(MonolithicProductCouplingError::InvalidCompressedRhs {
                expected: self.num_rows(),
                got: rhs.len(),
            });
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(MonolithicProductCouplingError::NonFiniteState);
        }
        let preconditioner = self.build_field_split_preconditioner()?;
        let solved = fgmres_preconditioned(self, &preconditioner, rhs, None, config);
        if solved.x.iter().any(|value| !value.is_finite()) {
            return Err(MonolithicProductCouplingError::NonFiniteState);
        }

        let (_, _, _, velocity_offset, current_offset, _) = self.offsets();
        let mut residual = self.apply(&solved.x);
        for (value, expected) in residual.iter_mut().zip(rhs) {
            *value -= *expected;
        }
        let rhs_norm = rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        let relative_residual = residual
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / rhs_norm;
        let relative_block_residual = |row: usize| residual[row].norm() / rhs[row].norm().max(1.0);
        let driver_mechanical_relative_residual = (0..self.driver_faces.len())
            .map(|driver_index| relative_block_residual(velocity_offset + driver_index))
            .collect::<Vec<_>>();
        let driver_electrical_relative_residual = (0..self.driver_faces.len())
            .map(|driver_index| relative_block_residual(current_offset + driver_index))
            .collect::<Vec<_>>();
        let structural_offset = self.structural_offset();
        let structural_relative_residual = (structural_offset..self.num_rows())
            .map(relative_block_residual)
            .fold(0.0_f64, f64::max);
        if !solved.converged
            || !relative_residual.is_finite()
            || relative_residual > config.tolerance
        {
            return Err(MonolithicProductCouplingError::IterativeSolveNotConverged {
                iterations: solved.iterations,
                residual: relative_residual,
            });
        }
        Ok(CompressedMonolithicProductSolution {
            state: solved.x,
            iterations: solved.iterations,
            relative_residual,
            driver_mechanical_relative_residual,
            driver_electrical_relative_residual,
            structural_relative_residual,
        })
    }

    fn solve_compressed_with_preconditioner(
        &self,
        rhs: &[Complex64],
        config: &FgmresConfig<f64>,
        preconditioner: &CompressedMonolithicProductFieldSplitPreconditioner,
    ) -> Result<CompressedMonolithicProductSolution, MonolithicProductCouplingError> {
        if rhs.len() != self.num_rows() {
            return Err(MonolithicProductCouplingError::InvalidCompressedRhs {
                expected: self.num_rows(),
                got: rhs.len(),
            });
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(MonolithicProductCouplingError::NonFiniteState);
        }

        let solved = fgmres_preconditioned(self, preconditioner, rhs, None, config);
        if solved.x.iter().any(|value| !value.is_finite()) {
            return Err(MonolithicProductCouplingError::NonFiniteState);
        }

        let (_, _, _, velocity_offset, current_offset, _) = self.offsets();
        let mut residual = self.apply(&solved.x);
        for (value, expected) in residual.iter_mut().zip(rhs) {
            *value -= *expected;
        }
        let rhs_norm = rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        let relative_residual = residual
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / rhs_norm;
        let relative_block_residual = |row: usize| residual[row].norm() / rhs[row].norm().max(1.0);
        let driver_mechanical_relative_residual = (0..self.driver_faces.len())
            .map(|driver_index| relative_block_residual(velocity_offset + driver_index))
            .collect::<Vec<_>>();
        let driver_electrical_relative_residual = (0..self.driver_faces.len())
            .map(|driver_index| relative_block_residual(current_offset + driver_index))
            .collect::<Vec<_>>();
        let structural_offset = self.structural_offset();
        let structural_relative_residual = (structural_offset..self.num_rows())
            .map(relative_block_residual)
            .fold(0.0_f64, f64::max);

        if !solved.converged
            || !relative_residual.is_finite()
            || relative_residual > config.tolerance
        {
            return Err(MonolithicProductCouplingError::IterativeSolveNotConverged {
                iterations: solved.iterations,
                residual: relative_residual,
            });
        }
        Ok(CompressedMonolithicProductSolution {
            state: solved.x,
            iterations: solved.iterations,
            relative_residual,
            driver_mechanical_relative_residual,
            driver_electrical_relative_residual,
            structural_relative_residual,
        })
    }

    /// Solve several complete RHS columns while reusing the same compressed
    /// H-matrix field-split preconditioner. Columns retain caller order.
    pub fn solve_compressed_many(
        &self,
        rhs_columns: &[Vec<Complex64>],
        config: &FgmresConfig<f64>,
    ) -> Result<Vec<CompressedMonolithicProductSolution>, MonolithicProductCouplingError> {
        for rhs in rhs_columns {
            if rhs.len() != self.num_rows() {
                return Err(MonolithicProductCouplingError::InvalidCompressedRhs {
                    expected: self.num_rows(),
                    got: rhs.len(),
                });
            }
            if rhs.iter().any(|value| !value.is_finite()) {
                return Err(MonolithicProductCouplingError::NonFiniteState);
            }
        }

        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }

        let preconditioner = self.build_field_split_preconditioner()?;
        rhs_columns
            .iter()
            .map(|rhs| self.solve_compressed_with_preconditioner(rhs, config, &preconditioner))
            .collect()
    }
}

impl LinearOperator<Complex64> for CompressedMonolithicProductOperator {
    fn num_rows(&self) -> usize {
        self.offsets().5
    }

    fn num_cols(&self) -> usize {
        self.offsets().5
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let (n_interior, _n_exterior, port_offset, velocity_offset, current_offset, total) =
            self.offsets();
        assert_eq!(x.len(), total, "compressed product input length");
        assert_eq!(y.len(), total, "compressed product output length");
        y.fill(Complex64::default());

        self.fem.apply_into(&x[..n_interior], &mut y[..n_interior]);
        self.exterior
            .apply_into(&x[n_interior..port_offset], &mut y[n_interior..port_offset]);

        let mut full_velocity = vec![Complex64::default(); self.velocity.len()];
        for (index, &slot) in self.port_velocity_slots.iter().enumerate() {
            full_velocity[slot] += x[port_offset + index];
        }
        for (driver_index, slots) in self.driver_velocity_slots.iter().enumerate() {
            for &slot in slots {
                full_velocity[slot] += x[velocity_offset + driver_index];
            }
        }
        let exterior_velocity = self
            .velocity
            .apply(&full_velocity)
            .expect("validated compressed product velocity input");
        for (row, value) in exterior_velocity.into_iter().enumerate() {
            y[n_interior + row] -= value;
        }

        for (index, &(nodes, area)) in self.port_faces.iter().enumerate() {
            let velocity = x[port_offset + index];
            for node in nodes {
                y[node] -= self.omega_density * (area / 3.0) * velocity;
            }
            let row = port_offset + index;
            y[row] += (x[nodes[0]] + x[nodes[1]] + x[nodes[2]]) / 3.0;
            y[row] -= x[n_interior + self.port_bem_elements[index]];
        }

        for (driver_index, faces) in self.driver_faces.iter().enumerate() {
            let velocity = x[velocity_offset + driver_index];
            let current = x[current_offset + driver_index];
            let row = velocity_offset + driver_index;
            for &(nodes, area) in faces {
                for node in nodes {
                    y[node] -= self.omega_density * (area / 3.0) * velocity;
                    y[row] -= Complex64::new(area / 3.0, 0.0) * x[node];
                }
            }
            for &element in &self.driver_bem_elements[driver_index] {
                y[row] += Complex64::new(self.exterior_element_areas[element], 0.0)
                    * x[n_interior + element];
            }
            y[row] += self.driver_mechanical_impedances[driver_index] * velocity
                - self.driver_force_factors[driver_index] * current;

            let electrical_row = current_offset + driver_index;
            y[electrical_row] += self.driver_force_factors[driver_index] * velocity
                + self.driver_coil_impedances[driver_index] * current;
        }

        let structural_offset = self.structural_offset();
        for (mode_index, &dynamic) in self.structural_dynamic_stiffness.iter().enumerate() {
            let structural_row = structural_offset + mode_index;
            y[structural_row] += dynamic * x[structural_row];
            for &(node, coefficient) in &self.structural_pressure_weights[mode_index] {
                y[structural_row] += coefficient * x[node];
            }
            for &(node, coefficient) in &self.structural_velocity_weights[mode_index] {
                y[node] += coefficient * x[structural_row];
            }
        }
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let (n_interior, _n_exterior, port_offset, velocity_offset, current_offset, total) =
            self.offsets();
        assert_eq!(x.len(), total, "compressed product transpose input length");
        assert_eq!(y.len(), total, "compressed product transpose output length");
        y.fill(Complex64::default());

        self.fem
            .apply_transpose_into(&x[..n_interior], &mut y[..n_interior]);
        self.exterior
            .apply_transpose_into(&x[n_interior..port_offset], &mut y[n_interior..port_offset]);
        let mut exterior_velocity = vec![Complex64::default(); self.velocity.len()];
        self.velocity
            .apply_transpose_into(&x[n_interior..port_offset], &mut exterior_velocity);
        for (index, &slot) in self.port_velocity_slots.iter().enumerate() {
            y[port_offset + index] -= exterior_velocity[slot];
        }
        for (driver_index, slots) in self.driver_velocity_slots.iter().enumerate() {
            for &slot in slots {
                y[velocity_offset + driver_index] -= exterior_velocity[slot];
            }
        }

        for (index, &(nodes, area)) in self.port_faces.iter().enumerate() {
            let row = port_offset + index;
            for node in nodes {
                y[port_offset + index] -= self.omega_density * (area / 3.0) * x[node];
                y[node] += Complex64::new(1.0 / 3.0, 0.0) * x[row];
            }
            y[n_interior + self.port_bem_elements[index]] -= x[row];
        }

        for (driver_index, faces) in self.driver_faces.iter().enumerate() {
            let velocity_row = velocity_offset + driver_index;
            let electrical_row = current_offset + driver_index;
            for &(nodes, area) in faces {
                for node in nodes {
                    y[velocity_offset + driver_index] -=
                        self.omega_density * (area / 3.0) * x[node];
                    y[node] -= Complex64::new(area / 3.0, 0.0) * x[velocity_row];
                }
            }
            for &element in &self.driver_bem_elements[driver_index] {
                y[n_interior + element] +=
                    Complex64::new(self.exterior_element_areas[element], 0.0) * x[velocity_row];
            }
            y[velocity_offset + driver_index] += self.driver_mechanical_impedances[driver_index]
                * x[velocity_row]
                + self.driver_force_factors[driver_index] * x[electrical_row];
            y[current_offset + driver_index] -=
                self.driver_force_factors[driver_index] * x[velocity_row];
            y[current_offset + driver_index] +=
                self.driver_coil_impedances[driver_index] * x[electrical_row];
        }
        let structural_offset = self.structural_offset();
        for (mode_index, &dynamic) in self.structural_dynamic_stiffness.iter().enumerate() {
            let structural_row = structural_offset + mode_index;
            y[structural_row] += dynamic * x[structural_row];
            for &(node, coefficient) in &self.structural_pressure_weights[mode_index] {
                y[node] += coefficient * x[structural_row];
            }
            for &(node, coefficient) in &self.structural_velocity_weights[mode_index] {
                y[structural_row] += coefficient * x[node];
            }
        }
    }

    fn apply_hermitian(&self, x: &[Complex64]) -> Vec<Complex64> {
        let (n_interior, _n_exterior, port_offset, velocity_offset, current_offset, total) =
            self.offsets();
        assert_eq!(x.len(), total, "compressed product Hermitian input length");
        let mut y = vec![Complex64::default(); total];

        let fem = self.fem.apply_hermitian(&x[..n_interior]);
        y[..n_interior].copy_from_slice(&fem);
        let exterior = self.exterior.apply_hermitian(&x[n_interior..port_offset]);
        y[n_interior..port_offset].copy_from_slice(&exterior);
        let exterior_velocity = self
            .velocity
            .apply_hermitian(&x[n_interior..port_offset])
            .expect("validated compressed product Hermitian input");
        for (index, &slot) in self.port_velocity_slots.iter().enumerate() {
            y[port_offset + index] -= exterior_velocity[slot];
        }
        for (driver_index, slots) in self.driver_velocity_slots.iter().enumerate() {
            for &slot in slots {
                y[velocity_offset + driver_index] -= exterior_velocity[slot];
            }
        }

        for (index, &(nodes, area)) in self.port_faces.iter().enumerate() {
            let row = port_offset + index;
            for node in nodes {
                y[port_offset + index] -= self.omega_density.conj() * (area / 3.0) * x[node];
                y[node] += Complex64::new(1.0 / 3.0, 0.0) * x[row];
            }
            y[n_interior + self.port_bem_elements[index]] -= x[row];
        }

        for (driver_index, faces) in self.driver_faces.iter().enumerate() {
            let velocity_row = velocity_offset + driver_index;
            let electrical_row = current_offset + driver_index;
            for &(nodes, area) in faces {
                for node in nodes {
                    y[velocity_offset + driver_index] -=
                        self.omega_density.conj() * (area / 3.0) * x[node];
                    y[node] -= Complex64::new(area / 3.0, 0.0) * x[velocity_row];
                }
            }
            for &element in &self.driver_bem_elements[driver_index] {
                y[n_interior + element] +=
                    Complex64::new(self.exterior_element_areas[element], 0.0) * x[velocity_row];
            }
            y[velocity_offset + driver_index] +=
                self.driver_mechanical_impedances[driver_index].conj() * x[velocity_row]
                    + self.driver_force_factors[driver_index].conj() * x[electrical_row];
            y[current_offset + driver_index] -=
                self.driver_force_factors[driver_index].conj() * x[velocity_row];
            y[current_offset + driver_index] +=
                self.driver_coil_impedances[driver_index].conj() * x[electrical_row];
        }
        let structural_offset = self.structural_offset();
        for (mode_index, &dynamic) in self.structural_dynamic_stiffness.iter().enumerate() {
            let structural_row = structural_offset + mode_index;
            y[structural_row] += dynamic.conj() * x[structural_row];
            for &(node, coefficient) in &self.structural_pressure_weights[mode_index] {
                y[node] += coefficient.conj() * x[structural_row];
            }
            for &(node, coefficient) in &self.structural_velocity_weights[mode_index] {
                y[structural_row] += coefficient.conj() * x[node];
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

pub struct MonolithicFemBemCoupling {
    interior: FemBemCoupling,
    exterior_mesh: sonium_bem::core::types::Mesh,
    aperture_bem_elements: Vec<usize>,
    aperture_source_nodes: Vec<usize>,
    aperture_transfer: ConservativeTraceTransfer,
    max_dofs: usize,
}

/// Dense sealed-product reference with acoustic, mechanical, and electrical
/// driver unknowns. It uses the canonical interior mesh's rear driver patches
/// as matching front BEM patches for a zero-thickness planar diaphragm.
/// Explicit port/interface unknowns are added by the resolved duct lowering;
/// this sealed path must not be used for vented products.
pub struct MonolithicSealedProductCoupling {
    interior_mesh: ProductInteriorMesh,
    exterior_mesh: sonium_bem::core::types::Mesh,
    driver_bem_elements: Vec<Vec<usize>>,
    port_bem_elements: Vec<usize>,
    port_fem_faces: Vec<usize>,
    assembler: sonium_fem::physics::acoustics::pressure::AcousticsAssembler,
    speed_of_sound: f64,
    density: f64,
    max_dofs: usize,
}

/// Errors raised by the sealed product monolithic reference.
#[derive(Debug, thiserror::Error)]
pub enum MonolithicProductCouplingError {
    #[error(
        "compressed product operator has no active velocity slot for exterior element {element}"
    )]
    InvalidCompressedProductElement { element: usize },
    #[error(transparent)]
    InvalidProduct(#[from] ProductDesignError),
    #[error(transparent)]
    MeshBridge(#[from] MeshBridgeError),
    #[error(transparent)]
    BemOperator(#[from] TbemVelocityOperatorError),
    #[error("the sealed product monolithic reference does not accept a port")]
    PortUnsupported,
    #[error("the sealed product monolithic reference does not accept braces")]
    BraceUnsupported,
    #[error("product mesh is missing the rear patch for driver {driver_index}")]
    MissingDriverPatch { driver_index: usize },
    #[error("product mesh driver patch {driver_index} has no canonical normal")]
    MissingDriverNormal { driver_index: usize },
    #[error("product mesh driver patch {driver_index} canonical normal mismatches the design")]
    DriverNormalMismatch { driver_index: usize },
    #[error("product mesh driver patch {driver_index} has no exterior BEM elements")]
    MissingDriverExterior { driver_index: usize },
    #[error("product mesh is missing the resolved port-mouth patch")]
    MissingPortPatch,
    #[error("product port-mouth patch has no matching exterior BEM elements")]
    MissingPortExterior,
    #[error("structural acoustic feedback failed: {0}")]
    StructuralFeedback(String),
    #[error(
        "product interface trace {patch:?} has {fem_faces} FEM faces but {bem_elements} BEM elements"
    )]
    InvalidTraceLength {
        patch: ProductInteriorPatchKind,
        fem_faces: usize,
        bem_elements: usize,
    },
    #[error("product interface trace {patch:?} references invalid exterior BEM element {element}")]
    InvalidTraceElement {
        patch: ProductInteriorPatchKind,
        element: usize,
    },
    #[error("frequency and input voltage must be finite, with frequency positive")]
    InvalidDrive,
    #[error("compressed product frequency grid must be finite, positive, and strictly increasing")]
    InvalidCompressedFrequencyGrid,
    #[error(
        "dense monolithic product system has {requested} DOFs, exceeding policy maximum {maximum}"
    )]
    DenseSystemTooLarge { requested: usize, maximum: usize },
    #[error("monolithic product LU factorization failed: {0}")]
    LuFactor(String),
    #[error("monolithic product LU solve failed: {0}")]
    LuSolve(String),
    #[error("monolithic product solve produced a non-finite state")]
    NonFiniteState,
    #[error("compressed product field-split setup failed: {0}")]
    CompressedFieldSplitSetup(String),
    #[error("compressed product RHS has {got} entries; expected {expected}")]
    InvalidCompressedRhs { expected: usize, got: usize },
    #[error(
        "compressed product iterative solve did not converge after {iterations} iterations (residual {residual:e})"
    )]
    IterativeSolveNotConverged { iterations: usize, residual: f64 },
}

/// Full state and residual evidence from the sealed product block solve.
#[derive(Debug, Clone)]
pub struct MonolithicSealedProductSolution {
    pub frequency_hz: f64,
    /// Explicit structural generalized displacement in BEM peak convention.
    pub structural_generalized_displacement_peak: Vec<Complex64>,
    /// Explicit structural surface-velocity contribution per mode in BEM peak convention.
    pub structural_velocity_peak: Vec<Complex64>,
    /// Relative residual of the explicit structural modal rows.
    pub structural_relative_residual: f64,
    pub interior_pressure: Vec<Complex64>,
    pub exterior_pressure: Vec<Complex64>,
    /// Outward velocity on every port-mouth interface triangle.
    pub port_velocity_m_per_s: Vec<Complex64>,
    /// Interior-minus-exterior pressure continuity residual per port face.
    pub port_pressure_jump_pa: Vec<Complex64>,
    /// Signed mouth volume velocity, positive from enclosure to exterior.
    pub port_volume_velocity_m3_per_s: Complex64,
    pub driver_terminal_voltage_v: Vec<Complex64>,
    pub driver_velocity_m_per_s: Vec<Complex64>,
    pub driver_current_a: Vec<Complex64>,
    /// Declared outward normal of each driver diaphragm interface.
    pub driver_normal_m: Vec<[f64; 3]>,
    pub driver_rear_pressure_force_n: Vec<Complex64>,
    pub driver_front_pressure_force_n: Vec<Complex64>,
    /// RMS electrical input power `Re{V I*}` summed over drivers.
    pub electrical_input_power_w: f64,
    /// Electromagnetic mechanical power `Re{BL I u*}` summed over drivers.
    pub motor_mechanical_power_w: f64,
    /// Mechanical-equation residual for each driver interface.
    pub driver_mechanical_relative_residual: Vec<f64>,
    /// Electrical-equation residual for each driver interface.
    pub driver_electrical_relative_residual: Vec<f64>,
    /// Driver diaphragm volume velocity for each driver interface.
    pub driver_volume_velocity_m3_per_s: Vec<Complex64>,
    /// Electrical work-balance residual for each driver.
    pub driver_electrical_power_residual_w: Vec<f64>,
    /// Mechanical/acoustic work-balance residual for each driver.
    pub driver_mechanical_power_residual_w: Vec<f64>,
    /// Real copper dissipation from the electrical block.
    pub coil_dissipation_w: f64,
    /// Real mechanical loss from the moving-system impedance block.
    pub mechanical_dissipation_w: f64,
    /// Work transferred from cavity/exterior pressure onto the drivers.
    pub acoustic_pressure_power_w: f64,
    /// `P_electrical - P_coil - P_motor`.
    pub electrical_power_residual_w: f64,
    /// `P_motor + P_acoustic - P_mechanical_loss`.
    pub mechanical_power_residual_w: f64,
    pub relative_residual: f64,
    pub interior_relative_residual: f64,
    pub exterior_relative_residual: f64,
    pub mechanical_relative_residual: f64,
    pub electrical_relative_residual: f64,
}

impl MonolithicFemBemCoupling {
    /// Build an uncondensed reference directly from a tagged FEM fixture.
    pub fn from_fixture(
        fixture: &CouplingFixture,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<Self, MonolithicCouplingError> {
        let mut wall_markers = vec![fixture.wall_marker];
        if let Some(driver_marker) = fixture.driver_marker {
            wall_markers.push(driver_marker);
        }
        let bridge =
            build_exterior_bem_mesh(&fixture.fem_mesh, fixture.aperture_marker, &wall_markers)?;
        let interior = FemBemCoupling::assemble_parts(
            fixture.fem_mesh.clone(),
            fixture.aperture_marker,
            fixture.driver_marker,
            None,
            speed_of_sound,
            density,
        );
        let aperture_faces = &interior.aperture.faces;
        let face_nodes = aperture_faces
            .iter()
            .map(|(nodes, _)| *nodes)
            .collect::<Vec<_>>();
        let face_areas = aperture_faces
            .iter()
            .map(|(_, area)| *area)
            .collect::<Vec<_>>();
        let (aperture_transfer, aperture_source_nodes) =
            ConservativeTraceTransfer::try_from_tri3_vertex_faces(&face_nodes, &face_areas)?;
        Ok(Self {
            interior,
            exterior_mesh: bridge.mesh,
            aperture_bem_elements: bridge.aperture_elements,
            aperture_source_nodes,
            aperture_transfer,
            max_dofs: DEFAULT_MONOLITHIC_MAX_DOFS,
        })
    }

    /// Override the dense golden-system size policy.
    pub fn with_max_dofs(mut self, max_dofs: usize) -> Self {
        self.max_dofs = max_dofs;
        self
    }

    /// Build the bounded matrix-free H-matrix monolithic operator at one
    /// frequency. The supplied clusters must cover the exterior constant
    /// panels in the same DOF layout as the BEM mesh.
    pub fn build_compressed_operator(
        &self,
        frequency_hz: f64,
        clusters: &[sonium_bem::core::types::Cluster],
        aca_tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedMonolithicFemBemOperator, MonolithicCouplingError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MonolithicCouplingError::InvalidFrequency);
        }
        let omega = std::f64::consts::TAU * frequency_hz;
        let wavenumber = omega / self.interior.speed_of_sound;
        let fem = self
            .interior
            .assembler
            .assemble_helmholtz(Complex64::new(wavenumber, 0.0), &Default::default());
        let physics = sonium_bem::core::types::PhysicsParams::new(
            frequency_hz,
            self.interior.speed_of_sound,
            self.interior.density,
            false,
        );
        let beta = physics.burton_miller_beta_scaled(0.2);
        let velocity = HMatrixVelocityOperator::build(
            &self.exterior_mesh.elements,
            &self.exterior_mesh.nodes,
            clusters,
            &physics,
            aca_tolerance,
            max_rank,
            1.0,
            beta,
        )?;
        let aperture_velocity_slots = self
            .aperture_bem_elements
            .iter()
            .map(|&element| {
                velocity
                    .velocity_element_indices()
                    .iter()
                    .position(|&active_element| active_element == element)
                    .ok_or(MonolithicCouplingError::InvalidApertureElement { element })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let exterior = build_velocity_mode_hmatrix(
            &self.exterior_mesh.elements,
            &self.exterior_mesh.nodes,
            clusters,
            &physics,
            aca_tolerance,
            max_rank,
            1.0,
            beta,
        );
        Ok(CompressedMonolithicFemBemOperator::new(
            fem,
            exterior,
            velocity,
            self.interior.aperture.faces.clone(),
            self.aperture_bem_elements.clone(),
            aperture_velocity_slots,
            self.aperture_source_nodes.clone(),
            self.aperture_transfer.weights.clone(),
            Complex64::new(0.0, omega * self.interior.density),
        ))
    }

    pub fn num_interior_dofs(&self) -> usize {
        self.interior.num_dofs()
    }

    pub fn num_exterior_dofs(&self) -> usize {
        self.exterior_mesh.elements.len()
    }

    pub fn num_aperture_dofs(&self) -> usize {
        self.aperture_bem_elements.len()
    }

    pub fn aperture_element_areas(&self) -> Vec<f64> {
        self.interior.aperture_element_areas()
    }

    /// Solve the full FEM/BEM/interface block system at one frequency.
    pub fn solve_frequency(
        &self,
        frequency_hz: f64,
        drive: CouplingDrive,
    ) -> Result<MonolithicCoupledSolution, MonolithicCouplingError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MonolithicCouplingError::InvalidFrequency);
        }
        let n_interior = self.num_interior_dofs();
        let n_exterior = self.num_exterior_dofs();
        let n_aperture = self.num_aperture_dofs();
        let total = n_interior + n_exterior + n_aperture;
        if total > self.max_dofs {
            return Err(MonolithicCouplingError::DenseSystemTooLarge {
                requested: total,
                maximum: self.max_dofs,
            });
        }

        let omega = std::f64::consts::TAU * frequency_hz;
        let wavenumber = omega / self.interior.speed_of_sound;
        let fem = self
            .interior
            .assembler
            .assemble_helmholtz(Complex64::new(wavenumber, 0.0), &Default::default());
        let physics = sonium_bem::core::types::PhysicsParams::new(
            frequency_hz,
            self.interior.speed_of_sound,
            self.interior.density,
            false,
        );
        let beta = physics.burton_miller_beta_scaled(0.2);
        let exterior = StabilizedTbemVelocityOperator::new(
            &self.exterior_mesh.elements,
            &self.exterior_mesh.nodes,
            &physics,
            1.0,
            beta,
        )?;
        debug_assert_eq!(exterior.len(), n_exterior);

        let aperture_faces = &self.interior.aperture.faces;
        debug_assert_eq!(aperture_faces.len(), n_aperture);
        let mut matrix = DenseMatrix::zeros(total, total);
        let mut rhs = vec![Complex64::default(); total];

        // A_i p_i - iωρ B_p v_p = f_driver.
        for row in 0..n_interior {
            for (column, value) in fem.row_entries(row) {
                matrix[[row, column]] += value;
            }
        }
        let i_omega_rho = Complex64::new(0.0, omega * self.interior.density);
        for (aperture_index, (nodes, area)) in aperture_faces.iter().enumerate() {
            let column = n_interior + n_exterior + aperture_index;
            for &node in nodes {
                matrix[[node, column]] -= i_omega_rho * (*area / 3.0);
            }
        }
        let drive_rhs = self.interior.drive_rhs(omega, drive)?;
        rhs[..n_interior].copy_from_slice(&drive_rhs);

        // A_e p_e - Q_p v_p = 0.  The raw BEM API accepts one velocity
        // entry per exterior constant panel; only aperture columns are active.
        let exterior_offset = n_interior;
        for row in 0..n_exterior {
            for column in 0..n_exterior {
                matrix[[exterior_offset + row, exterior_offset + column]] +=
                    exterior.pressure_matrix()[[row, column]];
            }
        }
        for (aperture_index, &bem_element) in self.aperture_bem_elements.iter().enumerate() {
            let mut unit_velocity = vec![Complex64::default(); n_exterior];
            unit_velocity[bem_element] = Complex64::new(1.0, 0.0);
            let column_rhs = exterior.velocity_rhs(&unit_velocity)?;
            let column = n_interior + n_exterior + aperture_index;
            for (row, value) in column_rhs.iter().enumerate() {
                matrix[[exterior_offset + row, column]] -= *value;
            }
        }

        // P_i p_i - p_e = 0 on every aperture element.  The bridge's
        // constant-panel BEM ordering is aligned with aperture_faces.
        let interface_offset = n_interior + n_exterior;
        for (aperture_index, _) in aperture_faces.iter().enumerate() {
            let row = interface_offset + aperture_index;
            for &(source, weight) in &self.aperture_transfer.weights[aperture_index] {
                let node = self.aperture_source_nodes[source];
                matrix[[row, node]] += Complex64::new(weight, 0.0);
            }
            matrix[[
                row,
                exterior_offset + self.aperture_bem_elements[aperture_index],
            ]] -= Complex64::new(1.0, 0.0);
        }

        let factor = lu_factorize(&matrix)
            .map_err(|error| MonolithicCouplingError::LuFactor(error.to_string()))?;
        let state = factor
            .solve(&rhs)
            .map_err(|error| MonolithicCouplingError::LuSolve(error.to_string()))?;
        if state.iter().any(|value| !value.is_finite()) {
            return Err(MonolithicCouplingError::NonFiniteState);
        }
        let residual = relative_residual(&matrix, &state, &rhs);
        let interior_relative_residual =
            block_relative_residual(&matrix, &state, &rhs, 0, n_interior);
        let exterior_relative_residual =
            block_relative_residual(&matrix, &state, &rhs, exterior_offset, interface_offset);
        let interface_relative_residual =
            block_relative_residual(&matrix, &state, &rhs, interface_offset, total);
        let interior_pressure = state[..n_interior].to_vec();
        let exterior_pressure = state[exterior_offset..interface_offset].to_vec();
        let aperture_velocity = state[interface_offset..].to_vec();
        let interior_trace = self
            .aperture_source_nodes
            .iter()
            .map(|&node| interior_pressure[node])
            .collect::<Vec<_>>();
        let transferred_interior_trace = self.aperture_transfer.apply(&interior_trace)?;
        let aperture_pressure_jump = transferred_interior_trace
            .iter()
            .zip(&self.aperture_bem_elements)
            .map(|(&interior_pressure, &bem_element)| {
                interior_pressure - exterior_pressure[bem_element]
            })
            .collect();
        Ok(MonolithicCoupledSolution {
            frequency_hz,
            interior_pressure,
            exterior_pressure,
            aperture_velocity,
            relative_residual: residual,
            interior_relative_residual,
            exterior_relative_residual,
            interface_relative_residual,
            aperture_pressure_jump,
        })
    }
}

impl MonolithicSealedProductCoupling {
    /// Build a compressed product operator retaining the dense reference's
    /// explicit port, driver-velocity, and driver-current unknowns.
    pub fn build_compressed_product_operator(
        &self,
        design: &ProductDesign,
        frequency_hz: f64,
        clusters: &[sonium_bem::core::types::Cluster],
        aca_tolerance: f64,
        max_rank: usize,
    ) -> Result<CompressedMonolithicProductOperator, MonolithicProductCouplingError> {
        design.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        if self.driver_bem_elements.len() != design.drivers.len() {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }

        let omega = std::f64::consts::TAU * frequency_hz;
        let wavenumber = omega / self.speed_of_sound;
        let mut boundary_coeffs = structural_boundary_coefficients(
            design,
            &self.interior_mesh,
            frequency_hz,
            self.density,
        )?;
        let structural_block = design
            .structural_acoustic
            .as_ref()
            .map(|feedback| StructuralAcousticModalBlock::from_feedback(feedback, frequency_hz))
            .transpose()
            .map_err(|error| {
                MonolithicProductCouplingError::StructuralFeedback(error.to_string())
            })?;
        if structural_block.is_some() {
            for patch in &self.interior_mesh.patches {
                if matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)) {
                    boundary_coeffs.remove(&(patch.marker as usize));
                }
            }
        }
        let fem = self
            .assembler
            .assemble_helmholtz(Complex64::new(wavenumber, 0.0), &boundary_coeffs);
        let physics = sonium_bem::core::types::PhysicsParams::new(
            frequency_hz,
            self.speed_of_sound,
            self.density,
            false,
        );
        let beta = physics.burton_miller_beta_scaled(0.2);
        let velocity = HMatrixVelocityOperator::build(
            &self.exterior_mesh.elements,
            &self.exterior_mesh.nodes,
            clusters,
            &physics,
            aca_tolerance,
            max_rank,
            1.0,
            beta,
        )?;
        let active_elements = velocity.velocity_element_indices().to_vec();
        let slot_for = |element: usize| {
            active_elements
                .iter()
                .position(|&active| active == element)
                .ok_or(MonolithicProductCouplingError::InvalidCompressedProductElement { element })
        };

        let port_faces = self
            .port_fem_faces
            .iter()
            .map(|&face| {
                triangular_face(&self.interior_mesh.mesh, face)
                    .ok_or(MonolithicProductCouplingError::MissingPortPatch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let port_bem_elements = self.port_bem_elements.clone();
        let port_velocity_slots = port_bem_elements
            .iter()
            .copied()
            .map(&slot_for)
            .collect::<Result<Vec<_>, _>>()?;

        let mut driver_faces = Vec::with_capacity(design.drivers.len());
        for driver_index in 0..design.drivers.len() {
            let patch = self
                .interior_mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
            let faces = patch
                .boundary_faces
                .iter()
                .map(|&face| {
                    triangular_face(&self.interior_mesh.mesh, face)
                        .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })
                })
                .collect::<Result<Vec<_>, _>>()?;
            driver_faces.push(faces);
        }
        let driver_bem_elements = self.driver_bem_elements.clone();
        let driver_velocity_slots = driver_bem_elements
            .iter()
            .map(|elements| {
                elements
                    .iter()
                    .copied()
                    .map(&slot_for)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let omega_density = Complex64::new(0.0, omega * self.density);
        let mut structural_dynamic_stiffness = Vec::new();
        let mut structural_pressure_weights = Vec::new();
        let mut structural_velocity_weights = Vec::new();
        if let Some(block) = structural_block.as_ref() {
            let structural_area: f64 = self
                .interior_mesh
                .patches
                .iter()
                .filter(|patch| matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)))
                .map(|patch| patch.area_m2)
                .sum();
            if !structural_area.is_finite() || structural_area <= 0.0 {
                return Err(MonolithicProductCouplingError::StructuralFeedback(
                    "compressed structural block has no rigid-wall area".to_owned(),
                ));
            }
            for mode_index in 0..block.mode_count() {
                let velocity_per_displacement =
                    block.generalized_to_surface_velocity()[mode_index].conj();
                let pressure_force =
                    Complex64::new(block.pressure_to_generalized_force()[mode_index], 0.0);
                let mut pressure_weights = Vec::new();
                let mut velocity_weights = Vec::new();
                for patch in
                    self.interior_mesh.patches.iter().filter(|patch| {
                        matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_))
                    })
                {
                    for &face_index in &patch.boundary_faces {
                        let (nodes, area) = triangular_face(&self.interior_mesh.mesh, face_index)
                            .ok_or_else(|| {
                            MonolithicProductCouplingError::StructuralFeedback(
                                "compressed structural block requires triangular rigid-wall faces"
                                    .to_owned(),
                            )
                        })?;
                        let pressure_weight = -pressure_force * (area / (3.0 * structural_area));
                        let velocity_weight =
                            -omega_density * velocity_per_displacement * (area / 3.0);
                        for node in nodes {
                            pressure_weights.push((node, pressure_weight));
                            velocity_weights.push((node, velocity_weight));
                        }
                    }
                }
                structural_dynamic_stiffness.push(block.dynamic_stiffness()[mode_index].conj());
                structural_pressure_weights.push(pressure_weights);
                structural_velocity_weights.push(velocity_weights);
            }
        }

        let exterior = build_velocity_mode_hmatrix(
            &self.exterior_mesh.elements,
            &self.exterior_mesh.nodes,
            clusters,
            &physics,
            aca_tolerance,
            max_rank,
            1.0,
            beta,
        );

        Ok(CompressedMonolithicProductOperator {
            fem,
            exterior,
            velocity,
            exterior_element_areas: self
                .exterior_mesh
                .elements
                .iter()
                .map(|element| element.area)
                .collect(),
            port_faces,
            port_bem_elements,
            port_velocity_slots,
            driver_faces,
            driver_bem_elements,
            driver_velocity_slots,
            driver_mechanical_impedances: design
                .drivers
                .iter()
                .map(|driver| driver.mechanical_impedance_n_s_per_m())
                .collect(),
            driver_force_factors: design
                .drivers
                .iter()
                .map(|driver| Complex64::new(driver.force_factor_tm, 0.0))
                .collect(),
            driver_coil_impedances: design
                .drivers
                .iter()
                .map(|driver| driver.coil_impedance_ohm())
                .collect(),
            structural_dynamic_stiffness,
            structural_pressure_weights,
            structural_velocity_weights,
            omega_density,
        })
    }

    /// Borrow the canonical body-fitted product mesh retained by this
    /// coupling. Product-level reference adapters use the same mesh for
    /// geometry identity and promotion evidence as the assembled solver.
    /// Solve one product frequency with the compressed operator and a
    /// crossover-derived terminal-voltage RHS.
    pub fn solve_compressed(
        &self,
        design: &ProductDesign,
        frequency_hz: f64,
        input_voltage_rms: Complex64,
        clusters: &[sonium_bem::core::types::Cluster],
        aca_tolerance: f64,
        max_rank: usize,
        config: &FgmresConfig<f64>,
    ) -> Result<CompressedMonolithicProductSolution, MonolithicProductCouplingError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || !input_voltage_rms.is_finite() {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        design.validate()?;
        let operator = self.build_compressed_product_operator(
            design,
            frequency_hz,
            clusters,
            aca_tolerance,
            max_rank,
        )?;
        let (_, _, _, _, current_offset, total) = operator.offsets();
        let structural_offset = operator.structural_offset();
        let transfers = design
            .crossover
            .transfers_at(design.drivers.len(), frequency_hz)?;
        let terminal_voltage = transfers
            .into_iter()
            .map(|transfer| transfer * input_voltage_rms)
            .collect::<Vec<_>>();
        let mut rhs = vec![Complex64::default(); total];
        rhs[current_offset..structural_offset].copy_from_slice(&terminal_voltage);
        operator.solve_compressed(&rhs, config)
    }

    /// Solve several crossover-voltage excitations at one frequency while
    /// reusing the assembled compressed operator and field-split setup.
    /// Returned solutions retain the exact order of `input_voltages_rms`.
    pub fn solve_compressed_many(
        &self,
        design: &ProductDesign,
        frequency_hz: f64,
        input_voltages_rms: &[Complex64],
        clusters: &[sonium_bem::core::types::Cluster],
        aca_tolerance: f64,
        max_rank: usize,
        config: &FgmresConfig<f64>,
    ) -> Result<Vec<CompressedMonolithicProductSolution>, MonolithicProductCouplingError> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        if input_voltages_rms
            .iter()
            .any(|input_voltage| !input_voltage.is_finite())
        {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        design.validate()?;
        if input_voltages_rms.is_empty() {
            return Ok(Vec::new());
        }

        let operator = self.build_compressed_product_operator(
            design,
            frequency_hz,
            clusters,
            aca_tolerance,
            max_rank,
        )?;
        let (_, _, _, _, current_offset, total) = operator.offsets();
        let structural_offset = operator.structural_offset();
        let transfers = design
            .crossover
            .transfers_at(design.drivers.len(), frequency_hz)?;
        let rhs_columns = input_voltages_rms
            .iter()
            .map(|&input_voltage_rms| {
                let terminal_voltage = transfers
                    .iter()
                    .map(|&transfer| transfer * input_voltage_rms)
                    .collect::<Vec<_>>();
                let mut rhs = vec![Complex64::default(); total];
                rhs[current_offset..structural_offset].copy_from_slice(&terminal_voltage);
                rhs
            })
            .collect::<Vec<_>>();
        operator.solve_compressed_many(&rhs_columns, config)
    }

    /// Dispatch a frequency grid and reuse one compressed field-split setup
    /// for every RHS column at each frequency. Outer and inner result order
    /// match `frequencies` and `input_voltages_rms`, respectively.
    pub fn solve_compressed_sweep_many(
        &self,
        design: &ProductDesign,
        frequencies: &[f64],
        input_voltages_rms: &[Complex64],
        clusters: &[sonium_bem::core::types::Cluster],
        aca_tolerance: f64,
        max_rank: usize,
        config: &FgmresConfig<f64>,
    ) -> Result<Vec<Vec<CompressedMonolithicProductSolution>>, MonolithicProductCouplingError> {
        if input_voltages_rms
            .iter()
            .any(|input_voltage| !input_voltage.is_finite())
        {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        design.validate()?;
        if frequencies.is_empty() {
            return Ok(Vec::new());
        }
        validate_frequency_grid(frequencies)
            .map_err(|_| MonolithicProductCouplingError::InvalidCompressedFrequencyGrid)?;
        if input_voltages_rms.is_empty() {
            return Ok(frequencies.iter().map(|_| Vec::new()).collect());
        }

        frequencies
            .iter()
            .copied()
            .map(|frequency_hz| {
                self.solve_compressed_many(
                    design,
                    frequency_hz,
                    input_voltages_rms,
                    clusters,
                    aca_tolerance,
                    max_rank,
                    config,
                )
            })
            .collect()
    }

    /// Sequentially dispatch a compressed product solve over a frequency grid.
    ///
    /// Each frequency assembles its own frequency-dependent H-matrix operator;
    /// the caller-owned cluster topology, ACA controls, and Krylov policy are
    /// applied consistently to every point. The tagged result keeps the
    /// frequency associated with the state for promotion or downstream
    /// comparison code.
    pub fn solve_compressed_sweep(
        &self,
        design: &ProductDesign,
        frequencies: &[f64],
        input_voltage_rms: Complex64,
        clusters: &[sonium_bem::core::types::Cluster],
        aca_tolerance: f64,
        max_rank: usize,
        config: &FgmresConfig<f64>,
    ) -> Result<Vec<CompressedMonolithicProductSweepPoint>, MonolithicProductCouplingError> {
        if !input_voltage_rms.is_finite() {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        design.validate()?;
        if !frequencies.is_empty() {
            validate_frequency_grid(frequencies)
                .map_err(|_| MonolithicProductCouplingError::InvalidCompressedFrequencyGrid)?;
        }

        frequencies
            .iter()
            .copied()
            .map(|frequency_hz| {
                self.solve_compressed(
                    design,
                    frequency_hz,
                    input_voltage_rms,
                    clusters,
                    aca_tolerance,
                    max_rank,
                    config,
                )
                .map(|solution| CompressedMonolithicProductSweepPoint {
                    frequency_hz,
                    solution,
                })
            })
            .collect()
    }

    pub fn interior_mesh(&self) -> &ProductInteriorMesh {
        &self.interior_mesh
    }

    /// Build the monolithic reference from the canonical product coupling
    /// artifact, retaining its validated FEM/BEM trace maps.
    pub fn from_coupling_mesh(
        design: &ProductDesign,
        coupling_mesh: ProductCouplingMesh,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<Self, MonolithicProductCouplingError> {
        design.validate()?;

        let ProductCouplingMesh {
            interior,
            exterior,
            traces,
            ..
        } = coupling_mesh;

        let port_patch = if design.cabinet.port.is_some() {
            Some(
                interior
                    .patch(ProductInteriorPatchKind::PortMouth)
                    .ok_or(MonolithicProductCouplingError::MissingPortPatch)?,
            )
        } else {
            None
        };

        for driver_index in 0..design.drivers.len() {
            let patch = interior
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
            let canonical_normal = patch
                .canonical_normal_m
                .ok_or(MonolithicProductCouplingError::MissingDriverNormal { driver_index })?;
            let expected_normal = design.drivers[driver_index].normal;
            let normal_error = canonical_normal
                .into_iter()
                .zip(expected_normal)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f64>()
                .sqrt();
            if normal_error > 1.0e-9 {
                return Err(MonolithicProductCouplingError::DriverNormalMismatch { driver_index });
            }
        }

        let validate_trace = |trace: &ProductInterfaceTrace| {
            if trace.fem_boundary_faces.len() != trace.bem_elements.len() {
                return Err(MonolithicProductCouplingError::InvalidTraceLength {
                    patch: trace.patch,
                    fem_faces: trace.fem_boundary_faces.len(),
                    bem_elements: trace.bem_elements.len(),
                });
            }
            for &element in &trace.bem_elements {
                if element >= exterior.mesh.elements.len() {
                    return Err(MonolithicProductCouplingError::InvalidTraceElement {
                        patch: trace.patch,
                        element,
                    });
                }
            }
            Ok(())
        };

        let mut driver_bem_elements = Vec::with_capacity(design.drivers.len());
        for driver_index in 0..design.drivers.len() {
            let trace = traces
                .get(&ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverExterior { driver_index })?;
            validate_trace(trace)?;
            if trace.bem_elements.is_empty() {
                return Err(MonolithicProductCouplingError::MissingDriverExterior { driver_index });
            }
            driver_bem_elements.push(trace.bem_elements.clone());
        }

        let (port_fem_faces, port_bem_elements) = if port_patch.is_some() {
            let trace = traces
                .get(&ProductInteriorPatchKind::PortMouth)
                .ok_or(MonolithicProductCouplingError::MissingPortExterior)?;
            validate_trace(trace)?;
            if trace.bem_elements.is_empty() {
                return Err(MonolithicProductCouplingError::MissingPortExterior);
            }
            (trace.fem_boundary_faces.clone(), trace.bem_elements.clone())
        } else {
            (Vec::new(), Vec::new())
        };

        let boundary_tags = structural_boundary_tags(&interior);
        let assembler = build_acoustics_assembler_with_boundaries(
            &interior.mesh,
            PolynomialDegree::P1,
            &boundary_tags,
        );

        Ok(Self {
            interior_mesh: interior,
            exterior_mesh: exterior.mesh,
            driver_bem_elements,
            port_bem_elements,
            port_fem_faces,
            assembler,
            speed_of_sound,
            density,
            max_dofs: DEFAULT_MONOLITHIC_MAX_DOFS,
        })
    }

    /// Build the sealed product reference from the canonical interior mesh.
    pub fn new(
        design: &ProductDesign,
        interior_mesh: ProductInteriorMesh,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<Self, MonolithicProductCouplingError> {
        design.validate()?;
        let port_patch = if design.cabinet.port.is_some() {
            Some(
                interior_mesh
                    .patch(ProductInteriorPatchKind::PortMouth)
                    .ok_or(MonolithicProductCouplingError::MissingPortPatch)?,
            )
        } else {
            None
        };
        // Validate orientation metadata before constructing the exterior
        // bridge so stale meshes fail closed without entering the BEM path.
        for driver_index in 0..design.drivers.len() {
            let patch = interior_mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
            let canonical_normal = patch
                .canonical_normal_m
                .ok_or(MonolithicProductCouplingError::MissingDriverNormal { driver_index })?;
            let expected_normal = design.drivers[driver_index].normal;
            let normal_error = canonical_normal
                .into_iter()
                .zip(expected_normal)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f64>()
                .sqrt();
            if normal_error > 1.0e-9 {
                return Err(MonolithicProductCouplingError::DriverNormalMismatch { driver_index });
            }
        }
        let wall_markers: Vec<i32> = interior_mesh
            .patches
            .iter()
            .filter(|patch| {
                !matches!(
                    patch.kind,
                    ProductInteriorPatchKind::PortMouth
                        | ProductInteriorPatchKind::PortWall
                        | ProductInteriorPatchKind::WaveguideWall { .. }
                        | ProductInteriorPatchKind::BraceWall { .. }
                )
            })
            .map(|patch| patch.marker)
            .collect();
        let aperture_marker = port_patch.map_or(i32::MIN, |patch| patch.marker);
        let bridge = build_exterior_bem_mesh(&interior_mesh.mesh, aperture_marker, &wall_markers)?;
        let fem_to_bem: HashMap<usize, usize> = bridge
            .bem_to_fem_boundary
            .iter()
            .enumerate()
            .map(|(bem, &fem)| (fem, bem))
            .collect();
        let mut driver_bem_elements = Vec::with_capacity(design.drivers.len());
        for driver_index in 0..design.drivers.len() {
            let patch = interior_mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
            let elements: Vec<usize> = patch
                .boundary_faces
                .iter()
                .map(|face| fem_to_bem.get(face).copied())
                .collect::<Option<_>>()
                .ok_or(MonolithicProductCouplingError::MissingDriverExterior { driver_index })?;
            if elements.is_empty() {
                return Err(MonolithicProductCouplingError::MissingDriverExterior { driver_index });
            }
            driver_bem_elements.push(elements);
        }
        let port_fem_faces = port_patch
            .map(|patch| patch.boundary_faces.clone())
            .unwrap_or_default();
        let port_bem_elements: Vec<usize> = port_fem_faces
            .iter()
            .map(|face| fem_to_bem.get(face).copied())
            .collect::<Option<_>>()
            .ok_or(MonolithicProductCouplingError::MissingPortExterior)?;
        if port_patch.is_some() && port_bem_elements.is_empty() {
            return Err(MonolithicProductCouplingError::MissingPortExterior);
        }
        let boundary_tags = structural_boundary_tags(&interior_mesh);
        let assembler = build_acoustics_assembler_with_boundaries(
            &interior_mesh.mesh,
            PolynomialDegree::P1,
            &boundary_tags,
        );
        Ok(Self {
            interior_mesh,
            exterior_mesh: bridge.mesh,
            driver_bem_elements,
            port_bem_elements,
            port_fem_faces,
            assembler,
            speed_of_sound,
            density,
            max_dofs: DEFAULT_MONOLITHIC_MAX_DOFS,
        })
    }

    /// Override the dense golden-system size policy.
    pub fn with_max_dofs(mut self, max_dofs: usize) -> Self {
        self.max_dofs = max_dofs;
        self
    }

    /// Solve `[p_i, p_e, v_port, u_d, i_d]` for one product frequency.
    pub fn solve_frequency(
        &self,
        design: &ProductDesign,
        frequency_hz: f64,
        input_voltage_rms: Complex64,
    ) -> Result<MonolithicSealedProductSolution, MonolithicProductCouplingError> {
        self.solve_frequency_with_coil_resistances(design, frequency_hz, input_voltage_rms, None)
    }

    /// Solve one product frequency with optional per-driver hot copper
    /// resistances. The canonical block is unchanged; only the real copper
    /// term and its work ledger use the supplied values.
    pub fn solve_frequency_with_coil_resistances(
        &self,
        design: &ProductDesign,
        frequency_hz: f64,
        input_voltage_rms: Complex64,
        coil_resistances_ohm: Option<&[f64]>,
    ) -> Result<MonolithicSealedProductSolution, MonolithicProductCouplingError> {
        design.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || !input_voltage_rms.is_finite() {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        if let Some(resistances) = coil_resistances_ohm
            && (resistances.len() != design.drivers.len()
                || resistances
                    .iter()
                    .any(|resistance| !resistance.is_finite() || *resistance <= 0.0))
        {
            return Err(MonolithicProductCouplingError::InvalidDrive);
        }
        let coil_resistance = |driver_index: usize| {
            coil_resistances_ohm
                .map(|resistances| resistances[driver_index])
                .unwrap_or(design.drivers[driver_index].coil_resistance_ohm)
        };
        let n_interior = self.interior_mesh.mesh.num_nodes();
        let n_exterior = self.exterior_mesh.elements.len();
        let n_port = self.port_fem_faces.len();
        let n_drivers = design.drivers.len();
        let structural_mode_count = design
            .structural_acoustic
            .as_ref()
            .map_or(0, |feedback| feedback.modes.len());
        let port_velocity_offset = n_interior + n_exterior;
        let velocity_offset = port_velocity_offset + n_port;
        let current_offset = velocity_offset + n_drivers;
        let structural_offset = current_offset + n_drivers;
        let total = structural_offset + structural_mode_count;
        if total > self.max_dofs {
            return Err(MonolithicProductCouplingError::DenseSystemTooLarge {
                requested: total,
                maximum: self.max_dofs,
            });
        }

        let omega = std::f64::consts::TAU * frequency_hz;
        let wavenumber = omega / self.speed_of_sound;
        let mut boundary_coeffs = structural_boundary_coefficients(
            design,
            &self.interior_mesh,
            frequency_hz,
            self.density,
        )?;
        let structural_block = design
            .structural_acoustic
            .as_ref()
            .map(|feedback| StructuralAcousticModalBlock::from_feedback(feedback, frequency_hz))
            .transpose()
            .map_err(|error| {
                MonolithicProductCouplingError::StructuralFeedback(error.to_string())
            })?;
        if structural_block.is_some() {
            for patch in &self.interior_mesh.patches {
                if matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)) {
                    boundary_coeffs.remove(&(patch.marker as usize));
                }
            }
        }
        let fem = self
            .assembler
            .assemble_helmholtz(Complex64::new(wavenumber, 0.0), &boundary_coeffs);
        let physics = sonium_bem::core::types::PhysicsParams::new(
            frequency_hz,
            self.speed_of_sound,
            self.density,
            false,
        );
        let exterior = StabilizedTbemVelocityOperator::new(
            &self.exterior_mesh.elements,
            &self.exterior_mesh.nodes,
            &physics,
            1.0,
            physics.burton_miller_beta_scaled(0.2),
        )?;
        let mut matrix = DenseMatrix::zeros(total, total);
        let mut rhs = vec![Complex64::default(); total];

        for row in 0..n_interior {
            for (column, value) in fem.row_entries(row) {
                matrix[[row, column]] += value;
            }
        }
        let exterior_offset = n_interior;
        for row in 0..n_exterior {
            for column in 0..n_exterior {
                matrix[[exterior_offset + row, exterior_offset + column]] +=
                    exterior.pressure_matrix()[[row, column]];
            }
        }

        let i_omega_rho = Complex64::new(0.0, omega * self.density);
        // Assemble every port/driver velocity column as one validated block.
        // The dense reference keeps the same TBEM coefficients, while the
        // block shape is also the seam used by compressed H/FMM backends.
        let velocity_rhs_count = n_port + n_drivers;
        let mut velocity_rhs_inputs = DenseMatrix::zeros(n_exterior, velocity_rhs_count);
        for (port_index, &bem_element) in self.port_bem_elements.iter().enumerate() {
            velocity_rhs_inputs[[bem_element, port_index]] = Complex64::new(1.0, 0.0);
        }
        for (driver_index, driver_elements) in self.driver_bem_elements.iter().enumerate() {
            let column = n_port + driver_index;
            for &element in driver_elements {
                velocity_rhs_inputs[[element, column]] = Complex64::new(1.0, 0.0);
            }
        }
        let exterior_velocity_rhs = exterior.velocity_rhs_block(&velocity_rhs_inputs)?;
        for (port_index, (&face_index, &bem_element)) in self
            .port_fem_faces
            .iter()
            .zip(&self.port_bem_elements)
            .enumerate()
        {
            let (nodes, area) = triangular_face(&self.interior_mesh.mesh, face_index)
                .ok_or(MonolithicProductCouplingError::MissingPortPatch)?;
            let velocity_column = port_velocity_offset + port_index;
            for node in nodes {
                matrix[[node, velocity_column]] -= i_omega_rho * (area / 3.0);
            }
            for row in 0..n_exterior {
                matrix[[exterior_offset + row, velocity_column]] -=
                    exterior_velocity_rhs[[row, port_index]];
            }
            // The port-velocity unknown's row is its pressure-continuity
            // constraint.  Do not add another interface-row range: that
            // would introduce unconstrained dummy columns.
            let interface_row = port_velocity_offset + port_index;
            for node in nodes {
                matrix[[interface_row, node]] += Complex64::new(1.0 / 3.0, 0.0);
            }
            matrix[[interface_row, exterior_offset + bem_element]] -= Complex64::new(1.0, 0.0);
        }
        for (driver_index, driver) in design.drivers.iter().enumerate() {
            let patch = self
                .interior_mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
            let velocity_column = velocity_offset + driver_index;
            let mechanical_row = velocity_column;
            let current_column = current_offset + driver_index;
            let electrical_row = current_column;
            let coil_resistance_ohm = coil_resistance(driver_index);

            // Interior kinematic lifting and its work-conjugate rear pressure
            // force.  The same P1 face quadrature supplies both operators.
            for &face_index in &patch.boundary_faces {
                let (nodes, area) = triangular_face(&self.interior_mesh.mesh, face_index)
                    .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
                for node in nodes {
                    matrix[[node, velocity_column]] -= i_omega_rho * (area / 3.0);
                    matrix[[mechanical_row, node]] -= Complex64::new(area / 3.0, 0.0);
                }
            }

            // Exterior kinematic lifting plus the work-conjugate front
            // pressure force.  Every driver patch element has one constant
            // BEM velocity/pressure DOF.
            for &element in &self.driver_bem_elements[driver_index] {
                matrix[[mechanical_row, exterior_offset + element]] +=
                    Complex64::new(self.exterior_mesh.elements[element].area, 0.0);
            }
            for row in 0..n_exterior {
                matrix[[exterior_offset + row, velocity_column]] -=
                    exterior_velocity_rhs[[row, n_port + driver_index]];
            }

            matrix[[mechanical_row, velocity_column]] += driver.mechanical_impedance_n_s_per_m();
            matrix[[mechanical_row, current_column]] -= Complex64::new(driver.force_factor_tm, 0.0);
            matrix[[electrical_row, velocity_column]] +=
                Complex64::new(driver.force_factor_tm, 0.0);
            matrix[[electrical_row, current_column]] +=
                Complex64::new(coil_resistance_ohm, driver.coil_reactance_ohm);
        }

        // Add the explicit structural modal block in the BEM peak convention.
        // The acoustic pressure is projected onto all rigid-wall faces, while
        // each modal generalized displacement contributes a uniform modal
        // normal velocity on those same faces. This is the non-condensed
        // counterpart of the legacy structural Robin coefficient.
        if let Some(block) = structural_block.as_ref() {
            let structural_area: f64 = self
                .interior_mesh
                .patches
                .iter()
                .filter(|patch| matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)))
                .map(|patch| patch.area_m2)
                .sum();
            if !structural_area.is_finite() || structural_area <= 0.0 {
                return Err(MonolithicProductCouplingError::StructuralFeedback(
                    "explicit structural block has no rigid-wall area".to_owned(),
                ));
            }
            for mode_index in 0..block.mode_count() {
                let structural_column = structural_offset + mode_index;
                matrix[[structural_column, structural_column]] +=
                    block.dynamic_stiffness()[mode_index].conj();
                let velocity_per_displacement =
                    block.generalized_to_surface_velocity()[mode_index].conj();
                let pressure_force =
                    Complex64::new(block.pressure_to_generalized_force()[mode_index], 0.0);
                for patch in
                    self.interior_mesh.patches.iter().filter(|patch| {
                        matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_))
                    })
                {
                    for &face_index in &patch.boundary_faces {
                        let (nodes, area) = triangular_face(&self.interior_mesh.mesh, face_index)
                            .ok_or_else(|| {
                            MonolithicProductCouplingError::StructuralFeedback(
                                "explicit structural block requires triangular rigid-wall faces"
                                    .to_owned(),
                            )
                        })?;
                        let face_weight = area / (3.0 * structural_area);
                        for node in nodes {
                            matrix[[node, structural_column]] -=
                                i_omega_rho * velocity_per_displacement * (area / 3.0);
                            matrix[[structural_column, node]] -= pressure_force * face_weight;
                        }
                    }
                }
            }
        }
        let transfers = design.crossover.transfers_at(n_drivers, frequency_hz)?;
        let terminal_voltage: Vec<Complex64> = transfers
            .into_iter()
            .map(|transfer| transfer * input_voltage_rms)
            .collect();
        rhs[current_offset..structural_offset].copy_from_slice(&terminal_voltage);

        let zero_rows: Vec<usize> = (0..total)
            .filter(|&row| {
                matrix
                    .row(row)
                    .iter()
                    .all(|value: &Complex64| value.norm_sqr() <= 1.0e-60)
            })
            .collect();
        let zero_columns: Vec<usize> = (0..total)
            .filter(|&column| (0..total).all(|row| matrix[[row, column]].norm_sqr() <= 1.0e-60))
            .collect();
        let factor = lu_factorize(&matrix).map_err(|error| {
            MonolithicProductCouplingError::LuFactor(format!(
                "{error}; n_interior={n_interior}; n_exterior={n_exterior}; n_port={n_port}; \
                 port_velocity_offset={port_velocity_offset}; velocity_offset={velocity_offset}; \
                 current_offset={current_offset}; \
                 zero_rows={zero_rows:?}; zero_columns={zero_columns:?}"
            ))
        })?;
        let state = factor
            .solve(&rhs)
            .map_err(|error| MonolithicProductCouplingError::LuSolve(error.to_string()))?;
        if state.iter().any(|value| !value.is_finite()) {
            return Err(MonolithicProductCouplingError::NonFiniteState);
        }
        let port_velocity_m_per_s = state[port_velocity_offset..velocity_offset].to_vec();
        let driver_velocity_m_per_s = state[velocity_offset..current_offset].to_vec();
        let driver_current_a = state[current_offset..total].to_vec();
        let interior_pressure = state[..n_interior].to_vec();
        let exterior_pressure = state[exterior_offset..port_velocity_offset].to_vec();
        let port_pressure_jump_pa = self
            .port_fem_faces
            .iter()
            .zip(&self.port_bem_elements)
            .map(|(&face_index, &bem_element)| {
                let (nodes, _) = triangular_face(&self.interior_mesh.mesh, face_index)
                    .expect("validated port patch contains triangular faces");
                (interior_pressure[nodes[0]]
                    + interior_pressure[nodes[1]]
                    + interior_pressure[nodes[2]])
                    / 3.0
                    - exterior_pressure[bem_element]
            })
            .collect();
        let port_volume_velocity_m3_per_s = self
            .port_fem_faces
            .iter()
            .zip(&port_velocity_m_per_s)
            .map(|(&face_index, velocity)| {
                let (_, area) = triangular_face(&self.interior_mesh.mesh, face_index)
                    .expect("validated port patch contains triangular faces");
                *velocity * area
            })
            .sum();
        let mut driver_rear_pressure_force_n: Vec<Complex64> = Vec::with_capacity(n_drivers);
        let mut driver_front_pressure_force_n: Vec<Complex64> = Vec::with_capacity(n_drivers);
        for driver_index in 0..n_drivers {
            let patch = self
                .interior_mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                .ok_or(MonolithicProductCouplingError::MissingDriverPatch { driver_index })?;
            let rear: Complex64 = patch
                .boundary_faces
                .iter()
                .map(|&face_index| {
                    let (nodes, area) = triangular_face(&self.interior_mesh.mesh, face_index)
                        .expect("validated product patch contains triangular faces");
                    (interior_pressure[nodes[0]]
                        + interior_pressure[nodes[1]]
                        + interior_pressure[nodes[2]])
                        * (area / 3.0)
                })
                .sum();
            let front: Complex64 = self.driver_bem_elements[driver_index]
                .iter()
                .map(|&element| {
                    exterior_pressure[element] * self.exterior_mesh.elements[element].area
                })
                .sum();
            driver_rear_pressure_force_n.push(rear);
            driver_front_pressure_force_n.push(front);
        }
        let electrical_input_power_w = terminal_voltage
            .iter()
            .zip(&driver_current_a)
            .map(|(voltage, current)| (*voltage * current.conj()).re)
            .sum();
        let motor_mechanical_power_w = design
            .drivers
            .iter()
            .zip(&driver_current_a)
            .zip(&driver_velocity_m_per_s)
            .map(|((driver, current), velocity)| {
                (Complex64::new(driver.force_factor_tm, 0.0) * *current * velocity.conj()).re
            })
            .sum();
        let coil_dissipation_w = design
            .drivers
            .iter()
            .zip(&driver_current_a)
            .enumerate()
            .map(|(driver_index, (_driver, current))| {
                coil_resistance(driver_index) * current.norm_sqr()
            })
            .sum();
        let mechanical_dissipation_w = design
            .drivers
            .iter()
            .zip(&driver_velocity_m_per_s)
            .map(|(driver, velocity)| driver.mechanical_resistance_n_s_per_m * velocity.norm_sqr())
            .sum();
        let acoustic_pressure_power_w: f64 = driver_rear_pressure_force_n
            .iter()
            .zip(&driver_front_pressure_force_n)
            .zip(&driver_velocity_m_per_s)
            .map(|((rear, front), velocity)| ((*rear - *front) * velocity.conj()).re)
            .sum();
        let driver_mechanical_relative_residual = (0..n_drivers)
            .map(|driver_index| {
                block_relative_residual(
                    &matrix,
                    &state,
                    &rhs,
                    velocity_offset + driver_index,
                    velocity_offset + driver_index + 1,
                )
            })
            .collect();
        let driver_electrical_relative_residual = (0..n_drivers)
            .map(|driver_index| {
                block_relative_residual(
                    &matrix,
                    &state,
                    &rhs,
                    current_offset + driver_index,
                    current_offset + driver_index + 1,
                )
            })
            .collect();
        let driver_volume_velocity_m3_per_s = (0..n_drivers)
            .map(|driver_index| {
                let area = self
                    .interior_mesh
                    .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                    .expect("validated driver patch exists")
                    .area_m2;
                driver_velocity_m_per_s[driver_index] * area
            })
            .collect();
        let driver_electrical_power_residual_w = design
            .drivers
            .iter()
            .enumerate()
            .map(|(driver_index, driver)| {
                let input_power =
                    (terminal_voltage[driver_index] * driver_current_a[driver_index].conj()).re;
                let motor_power = (Complex64::new(driver.force_factor_tm, 0.0)
                    * driver_current_a[driver_index]
                    * driver_velocity_m_per_s[driver_index].conj())
                .re;
                input_power
                    - coil_resistance(driver_index) * driver_current_a[driver_index].norm_sqr()
                    - motor_power
            })
            .collect();
        let driver_mechanical_power_residual_w = design
            .drivers
            .iter()
            .enumerate()
            .map(|(driver_index, driver)| {
                let motor_power = (Complex64::new(driver.force_factor_tm, 0.0)
                    * driver_current_a[driver_index]
                    * driver_velocity_m_per_s[driver_index].conj())
                .re;
                let acoustic_power = ((driver_rear_pressure_force_n[driver_index]
                    - driver_front_pressure_force_n[driver_index])
                    * driver_velocity_m_per_s[driver_index].conj())
                .re;
                motor_power + acoustic_power
                    - driver.mechanical_resistance_n_s_per_m
                        * driver_velocity_m_per_s[driver_index].norm_sqr()
            })
            .collect();
        let structural_generalized_displacement_peak = if structural_block.is_some() {
            state[structural_offset..total].to_vec()
        } else {
            Vec::new()
        };
        let structural_velocity_peak = structural_block
            .as_ref()
            .map(|block| {
                block
                    .generalized_to_surface_velocity()
                    .iter()
                    .zip(&structural_generalized_displacement_peak)
                    .map(|(coefficient, displacement)| coefficient.conj() * *displacement)
                    .collect()
            })
            .unwrap_or_default();
        let structural_relative_residual = structural_block
            .as_ref()
            .map(|_| block_relative_residual(&matrix, &state, &rhs, structural_offset, total))
            .unwrap_or(0.0);

        Ok(MonolithicSealedProductSolution {
            frequency_hz,
            structural_generalized_displacement_peak,
            structural_velocity_peak,
            structural_relative_residual,
            interior_pressure,
            exterior_pressure,
            port_velocity_m_per_s,
            port_pressure_jump_pa,
            port_volume_velocity_m3_per_s,
            driver_terminal_voltage_v: terminal_voltage,
            driver_velocity_m_per_s,
            driver_current_a,
            driver_normal_m: design.drivers.iter().map(|driver| driver.normal).collect(),
            driver_rear_pressure_force_n,
            driver_front_pressure_force_n,
            electrical_input_power_w,
            motor_mechanical_power_w,
            driver_mechanical_relative_residual,
            driver_electrical_relative_residual,
            driver_volume_velocity_m3_per_s,
            driver_electrical_power_residual_w,
            driver_mechanical_power_residual_w,
            coil_dissipation_w,
            mechanical_dissipation_w,
            acoustic_pressure_power_w,
            electrical_power_residual_w: electrical_input_power_w
                - coil_dissipation_w
                - motor_mechanical_power_w,
            mechanical_power_residual_w: motor_mechanical_power_w + acoustic_pressure_power_w
                - mechanical_dissipation_w,
            relative_residual: relative_residual(&matrix, &state, &rhs),
            interior_relative_residual: block_relative_residual(
                &matrix, &state, &rhs, 0, n_interior,
            ),
            exterior_relative_residual: block_relative_residual(
                &matrix,
                &state,
                &rhs,
                exterior_offset,
                port_velocity_offset,
            ),
            mechanical_relative_residual: block_relative_residual(
                &matrix,
                &state,
                &rhs,
                velocity_offset,
                current_offset,
            ),
            electrical_relative_residual: block_relative_residual(
                &matrix,
                &state,
                &rhs,
                current_offset,
                total,
            ),
        })
    }
}

fn structural_boundary_tags(mesh: &ProductInteriorMesh) -> Vec<i32> {
    mesh.patches
        .iter()
        .filter_map(|patch| {
            matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)).then_some(patch.marker)
        })
        .collect()
}

/// Lower the product modal wall response into the FEM weak Robin coefficient.
///
/// The monolithic acoustic pressure uses the BEM `e^{-iωt}` peak convention.
/// `StructuralAcousticFeedback` consumes modeling `e^{+iωt}` RMS pressure, so
/// a unit BEM pressure corresponds to `1 / sqrt(2)` modeling RMS pressure.
/// For `-Δp-k²p=0`, the weak boundary term is `-∂n p`; momentum gives
/// `∂n p = iωρv`, hence the assembled coefficient is `-iωρ(v/p)`.
fn structural_boundary_coefficients(
    design: &ProductDesign,
    mesh: &ProductInteriorMesh,
    frequency_hz: f64,
    density: f64,
) -> Result<HashMap<usize, Complex64>, MonolithicProductCouplingError> {
    let omega = std::f64::consts::TAU * frequency_hz;
    let structural_coefficient = if let Some(feedback) = &design.structural_acoustic {
        let modeling_unit_pressure = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        let response = feedback
            .response_at(frequency_hz, modeling_unit_pressure)
            .map_err(|error| {
                MonolithicProductCouplingError::StructuralFeedback(error.to_string())
            })?;
        Some(Complex64::new(0.0, -omega * density) * response.velocity_bem_peak())
    } else {
        None
    };
    let waveguide_coefficient = design.waveguide.as_ref().map(|waveguide| {
        Complex64::new(
            0.0,
            -omega * density / waveguide.surface_specific_impedance_pa_s_per_m,
        )
    });
    let mut coefficients = HashMap::new();
    for patch in &mesh.patches {
        let coefficient = match patch.kind {
            ProductInteriorPatchKind::RigidWall(_) => structural_coefficient,
            ProductInteriorPatchKind::WaveguideWall { .. } => waveguide_coefficient,
            _ => None,
        };
        if let Some(coefficient) = coefficient {
            if !coefficient.is_finite() {
                return Err(MonolithicProductCouplingError::StructuralFeedback(
                    "non-finite FEM boundary coefficient".to_owned(),
                ));
            }
            coefficients.insert(patch.marker as usize, coefficient);
        }
    }
    Ok(coefficients)
}

fn triangular_face(mesh: &sonium_fem::mesh::Mesh, face_index: usize) -> Option<([usize; 3], f64)> {
    let face = mesh.boundaries.get(face_index)?;
    let [a, b, c] = [
        *face.nodes.first()?,
        *face.nodes.get(1)?,
        *face.nodes.get(2)?,
    ];
    if face.nodes.len() != 3 {
        return None;
    }
    let pa = mesh.nodes.get(a)?;
    let pb = mesh.nodes.get(b)?;
    let pc = mesh.nodes.get(c)?;
    let ux = pb.x - pa.x;
    let uy = pb.y - pa.y;
    let uz = pb.z - pa.z;
    let vx = pc.x - pa.x;
    let vy = pc.y - pa.y;
    let vz = pc.z - pa.z;
    let area = 0.5
        * ((uy * vz - uz * vy).powi(2) + (uz * vx - ux * vz).powi(2) + (ux * vy - uy * vx).powi(2))
            .sqrt();
    (area > 0.0).then_some(([a, b, c], area))
}

fn relative_residual(
    matrix: &DenseMatrix<Complex64>,
    state: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let mut residual_norm_sq = 0.0;
    let mut rhs_norm_sq = 0.0;
    for (row, rhs_value) in rhs.iter().enumerate() {
        let mut value = -*rhs_value;
        for (column, coefficient) in matrix.row(row).iter().enumerate() {
            value += *coefficient * state[column];
        }
        residual_norm_sq += value.norm_sqr();
        rhs_norm_sq += rhs_value.norm_sqr();
    }
    residual_norm_sq.sqrt() / rhs_norm_sq.sqrt().max(1.0)
}

fn block_relative_residual(
    matrix: &DenseMatrix<Complex64>,
    state: &[Complex64],
    rhs: &[Complex64],
    row_start: usize,
    row_end: usize,
) -> f64 {
    let rhs_norm = rhs
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    let mut residual_norm_sq = 0.0;
    for (row, rhs_value) in rhs
        .iter()
        .enumerate()
        .skip(row_start)
        .take(row_end - row_start)
    {
        let mut value = -*rhs_value;
        for (column, coefficient) in matrix.row(row).iter().enumerate() {
            value += *coefficient * state[column];
        }
        residual_norm_sq += value.norm_sqr();
    }
    residual_norm_sq.sqrt() / rhs_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabinet_design::{CabinetDesign, PortDesign, PortPlacement};
    use crate::conventions::ArtifactProvenance;
    use crate::fem_bem_coupling::fixtures::CylinderFixtureDiscretization;
    use crate::product_acoustic_mesh::{
        ProductInteriorMeshConfig, ProductInteriorPatchKind,
        build_body_fitted_sealed_product_interior_mesh, build_ported_product_interior_mesh,
        build_sealed_product_interior_mesh,
    };
    use crate::product_design::{
        CrossoverBranch, CrossoverKind, CrossoverNetwork, DriverMountingFace,
        MonolithicEvidenceGate, ProductDriver, ProductReferencePoint, ProductReferenceSweep,
        adjacent_mesh_convergence_error,
    };
    use crate::structural_acoustic::{StructuralAcousticFeedback, StructuralAcousticMode};
    use sonium_solvers::{FgmresConfig, LinearSolveConfig};

    #[test]
    fn compressed_product_with_structural_modes_closes_modal_rows() {
        let baseline_design = sealed_product();
        let baseline_mesh = build_sealed_product_interior_mesh(
            &baseline_design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.08,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("baseline product mesh");
        let mut feedback_design = sealed_product();
        feedback_design.structural_acoustic = Some(StructuralAcousticFeedback {
            area_m2: 0.02,
            modes: vec![StructuralAcousticMode {
                natural_frequency_hz: 250.0,
                effective_mass_kg: 0.2,
                damping_ratio: 0.08,
                pressure_participation: 1.0,
                stress_per_displacement_pa_per_m: 1_000.0,
            }],
            max_iterations: 12,
            velocity_tolerance: 1.0e-6,
            under_relaxation: 0.8,
        });
        let coupling =
            MonolithicSealedProductCoupling::new(&baseline_design, baseline_mesh, 343.0, 1.204)
                .expect("monolithic setup");
        let n_exterior = coupling.exterior_mesh.elements.len();
        let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
        cluster.element_indices = (0..n_exterior).collect();
        cluster.num_elements = n_exterior;
        cluster.radius = 1.0;
        cluster.near_clusters = vec![0];
        let clusters = vec![cluster];
        let operator = coupling
            .build_compressed_product_operator(&feedback_design, 120.0, &clusters, 1.0e-10, 16)
            .expect("compressed structural operator");

        let (_, _, _, _, current_offset, total) = operator.offsets();
        let structural_offset = operator.structural_offset();
        assert_eq!(operator.num_structural_modes(), 1);
        assert_eq!(
            structural_offset,
            current_offset + feedback_design.drivers.len()
        );
        assert_eq!(total, structural_offset + operator.num_structural_modes());

        let state = (0..operator.num_rows())
            .map(|index| Complex64::new(0.03 + index as f64 * 1.0e-4, -0.02))
            .collect::<Vec<_>>();
        let left = (0..operator.num_rows())
            .map(|index| Complex64::new(0.07, 0.01 + index as f64 * 2.0e-4))
            .collect::<Vec<_>>();
        let applied = operator.apply(&state);
        let transpose = operator.apply_transpose(&left);
        let transpose_lhs = left
            .iter()
            .zip(&applied)
            .map(|(a, b)| *a * *b)
            .sum::<Complex64>();
        let transpose_rhs = transpose
            .iter()
            .zip(&state)
            .map(|(a, b)| *a * *b)
            .sum::<Complex64>();
        assert!((transpose_lhs - transpose_rhs).norm() < 1.0e-10);
        let hermitian = operator.apply_hermitian(&left);
        let hermitian_lhs = left
            .iter()
            .zip(&applied)
            .map(|(a, b)| a.conj() * *b)
            .sum::<Complex64>();
        let hermitian_rhs = hermitian
            .iter()
            .zip(&state)
            .map(|(a, b)| a.conj() * *b)
            .sum::<Complex64>();
        assert!((hermitian_lhs - hermitian_rhs).norm() < 1.0e-10);

        let config = FgmresConfig {
            max_iterations: 60,
            restart: 80,
            tolerance: 1.0e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };
        let compressed = coupling
            .solve_compressed(
                &feedback_design,
                120.0,
                Complex64::new(2.0, 0.0),
                &clusters,
                1.0e-10,
                16,
                &config,
            )
            .expect("compressed structural solve");
        assert!(compressed.relative_residual < 1.0e-8);
        assert!(compressed.structural_relative_residual < 1.0e-8);
        assert!(compressed.state.iter().all(|value| value.is_finite()));
        assert!(compressed.state[structural_offset].norm() > 0.0);

        let dense = coupling
            .solve_frequency(&feedback_design, 120.0, Complex64::new(2.0, 0.0))
            .expect("dense structural solve");
        let mut dense_state = dense.interior_pressure.clone();
        dense_state.extend_from_slice(&dense.exterior_pressure);
        dense_state.extend_from_slice(&dense.port_velocity_m_per_s);
        dense_state.extend_from_slice(&dense.driver_velocity_m_per_s);
        dense_state.extend_from_slice(&dense.driver_current_a);
        dense_state.extend_from_slice(&dense.structural_generalized_displacement_peak);
        let max_state_difference = compressed
            .state
            .iter()
            .zip(&dense_state)
            .map(|(compressed, dense)| (*compressed - *dense).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_state_difference < 1.0e-6);
    }

    #[test]
    fn compressed_monolithic_operator_matches_dense_velocity_block() {
        let fixture = super::super::fixtures::cylindrical_box_with_circular_aperture(
            0.11,
            0.16,
            0.04,
            CylinderFixtureDiscretization {
                n_circumference: 8,
                n_box_depth: 1,
                n_cap_radial: 1,
                n_aperture_radial: 1,
                n_port_depth: 1,
            },
        )
        .expect("fixture");
        let monolithic = MonolithicFemBemCoupling::from_fixture(&fixture, 343.0, 1.204)
            .expect("monolithic setup");
        let n_exterior = monolithic.num_exterior_dofs();
        let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
        cluster.element_indices = (0..n_exterior).collect();
        cluster.num_elements = n_exterior;
        cluster.radius = 1.0;
        cluster.near_clusters = vec![0];

        let compressed = monolithic
            .build_compressed_operator(120.0, &[cluster], 1.0e-10, 16)
            .expect("compressed monolithic operator");
        assert_eq!(
            compressed.num_rows(),
            monolithic.num_interior_dofs() + n_exterior + monolithic.num_aperture_dofs()
        );

        let aperture_offset = monolithic.num_interior_dofs() + n_exterior;
        let mut state = vec![Complex64::default(); compressed.num_rows()];
        for aperture_index in 0..monolithic.num_aperture_dofs() {
            state[aperture_offset + aperture_index] =
                Complex64::new(0.2 + 0.03 * aperture_index as f64, -0.1);
        }
        let applied = compressed.apply(&state);
        let dense_velocity = StabilizedTbemVelocityOperator::new(
            &monolithic.exterior_mesh.elements,
            &monolithic.exterior_mesh.nodes,
            &sonium_bem::core::types::PhysicsParams::new(120.0, 343.0, 1.204, false),
            1.0,
            sonium_bem::core::types::PhysicsParams::new(120.0, 343.0, 1.204, false)
                .burton_miller_beta_scaled(0.2),
        )
        .expect("dense velocity operator");
        let mut velocities = vec![Complex64::default(); dense_velocity.len()];
        for (aperture_index, &slot) in compressed.aperture_velocity_slots().iter().enumerate() {
            velocities[slot] = state[aperture_offset + aperture_index];
        }
        let expected = dense_velocity
            .velocity_rhs(&velocities)
            .expect("dense velocity RHS");
        for (row, value) in expected.iter().enumerate() {
            assert!((applied[monolithic.num_interior_dofs() + row] + *value).norm() < 1.0e-8);
        }
        assert!(
            applied[aperture_offset..]
                .iter()
                .all(|value| value.norm() < 1.0e-10)
        );

        let left = (0..compressed.num_rows())
            .map(|index| Complex64::new(0.07 * index as f64, -0.03 * index as f64))
            .collect::<Vec<_>>();
        let transpose = compressed.apply_transpose(&left);
        let transpose_lhs = left
            .iter()
            .zip(&applied)
            .map(|(a, b)| *a * *b)
            .sum::<Complex64>();
        let transpose_rhs = transpose
            .iter()
            .zip(&state)
            .map(|(a, b)| *a * *b)
            .sum::<Complex64>();
        assert!(
            (transpose_lhs - transpose_rhs).norm() < 1.0e-8,
            "transpose error: lhs={transpose_lhs:?} rhs={transpose_rhs:?}"
        );

        let hermitian = compressed.apply_hermitian(&left);
        let hermitian_lhs = left
            .iter()
            .zip(&applied)
            .map(|(a, b)| a.conj() * *b)
            .sum::<Complex64>();
        let hermitian_rhs = hermitian
            .iter()
            .zip(&state)
            .map(|(a, b)| a.conj() * *b)
            .sum::<Complex64>();
        assert!((hermitian_lhs - hermitian_rhs).norm() < 1.0e-8);
    }

    #[test]
    fn compressed_monolithic_field_split_fgmres_solves_fixture() {
        let fixture = super::super::fixtures::cylindrical_box_with_circular_aperture(
            0.11,
            0.16,
            0.04,
            CylinderFixtureDiscretization {
                n_circumference: 8,
                n_box_depth: 1,
                n_cap_radial: 1,
                n_aperture_radial: 1,
                n_port_depth: 1,
            },
        )
        .expect("fixture");
        let monolithic = MonolithicFemBemCoupling::from_fixture(&fixture, 343.0, 1.204)
            .expect("monolithic setup");
        let n_exterior = monolithic.num_exterior_dofs();
        let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
        cluster.element_indices = (0..n_exterior).collect();
        cluster.num_elements = n_exterior;
        cluster.radius = 1.0;
        cluster.near_clusters = vec![0];
        let compressed = monolithic
            .build_compressed_operator(120.0, &[cluster], 1.0e-10, 16)
            .expect("compressed monolithic operator");
        let truth = (0..compressed.num_rows())
            .map(|index| Complex64::new(0.05 + index as f64 * 1.0e-4, -0.02))
            .collect::<Vec<_>>();
        let rhs = compressed.apply(&truth);
        let solved = compressed
            .solve_compressed(
                &rhs,
                &FgmresConfig {
                    max_iterations: 250,
                    restart: 40,
                    tolerance: 1.0e-8,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    exact_residual_check_interval: 1,
                },
            )
            .expect("field-split compressed solve");
        assert!(solved.iterations > 0);
        assert!(solved.relative_residual < 1.0e-8);
    }

    #[test]
    fn monolithic_fixture_matches_dtn_condensed_solution() {
        let fixture = super::super::fixtures::cylindrical_box_with_circular_aperture(
            0.11,
            0.16,
            0.04,
            CylinderFixtureDiscretization {
                n_circumference: 12,
                n_box_depth: 2,
                n_cap_radial: 2,
                n_aperture_radial: 2,
                n_port_depth: 2,
            },
        )
        .expect("fixture");
        let monolithic = MonolithicFemBemCoupling::from_fixture(&fixture, 343.0, 1.204)
            .expect("monolithic setup");
        let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-3, 0.0));
        let solution = monolithic
            .solve_frequency(120.0, drive.clone())
            .expect("monolithic solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.interface_relative_residual < 1.0e-10);
        assert!(
            solution
                .aperture_pressure_jump
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );

        let condensed = FemBemCoupling::from_fixture(&fixture, 343.0, 1.204)
            .expect("condensed setup")
            .solve_frequency(
                120.0,
                drive,
                &LinearSolveConfig {
                    tolerance: 1.0e-10,
                    max_iterations: 1_000,
                    ..LinearSolveConfig::default()
                },
            )
            .expect("condensed solve");
        assert_eq!(solution.interior_pressure.len(), condensed.pressure.len());
        for (actual, expected) in solution.interior_pressure.iter().zip(&condensed.pressure) {
            assert!(
                (*actual - *expected).norm() < 1.0e-6,
                "monolithic={actual:?}, condensed={expected:?}"
            );
        }
        for (actual, expected) in solution
            .aperture_velocity
            .iter()
            .zip(&condensed.aperture_velocity)
        {
            assert!((*actual - *expected).norm() < 1.0e-6);
        }

        let direct_condensed = FemBemCoupling::from_fixture(&fixture, 343.0, 1.204)
            .expect("direct condensed setup")
            .solve_frequency_direct_dense(
                120.0,
                CouplingDrive::DriverVelocity(Complex64::new(1.0e-3, 0.0)),
            )
            .expect("direct condensed solve");
        assert!(direct_condensed.relative_residual.unwrap() < 1.0e-10);
        for (actual, expected) in solution
            .interior_pressure
            .iter()
            .zip(&direct_condensed.pressure)
        {
            assert!(
                (*actual - *expected).norm() < 1.0e-10,
                "monolithic={actual:?}, direct-condensed={expected:?}"
            );
        }
        for (actual, expected) in solution
            .aperture_velocity
            .iter()
            .zip(&direct_condensed.aperture_velocity)
        {
            assert!((*actual - *expected).norm() < 1.0e-10);
        }
    }

    #[test]
    fn monolithic_dense_policy_rejects_oversized_fixture() {
        let fixture = super::super::fixtures::cylindrical_box_with_circular_aperture(
            0.11,
            0.16,
            0.04,
            CylinderFixtureDiscretization {
                n_circumference: 12,
                n_box_depth: 2,
                n_cap_radial: 2,
                n_aperture_radial: 2,
                n_port_depth: 2,
            },
        )
        .expect("fixture");
        let monolithic = MonolithicFemBemCoupling::from_fixture(&fixture, 343.0, 1.204)
            .expect("monolithic setup")
            .with_max_dofs(1);
        assert!(matches!(
            monolithic.solve_frequency(120.0, CouplingDrive::None),
            Err(MonolithicCouplingError::DenseSystemTooLarge { .. })
        ));
    }

    fn sealed_product() -> ProductDesign {
        ProductDesign {
            cabinet: CabinetDesign {
                port: None,
                ..CabinetDesign::default()
            },
            drivers: vec![ProductDriver {
                id: "woofer".into(),
                radius_m: 0.045,
                position_m: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 6.0,
                coil_resistance_ohm: 6.0,
                coil_reactance_ohm: 0.2,
                mechanical_resistance_n_s_per_m: 1.0,
                mechanical_reactance_n_s_per_m: 0.1,
                crossover_branch: 0,
            }],
            crossover: CrossoverNetwork {
                branches: vec![CrossoverBranch::default()],
            },
            waveguide: None,
            structural_acoustic: None,
        }
    }

    #[test]
    fn compressed_monolithic_product_operator_closes_electromechanical_blocks() {
        let mut design = sealed_product();
        design.drivers[0].position_m[0] = -0.06;
        design.cabinet.port = Some(PortDesign {
            diameter_m: 0.05,
            length_m: 0.15,
            placement: PortPlacement::FrontPanel,
            position_m: [0.06, 0.0],
            wall_thickness_m: Some(0.018),
        });
        let coupling_mesh =
            ProductCouplingMesh::build(&design, ProductInteriorMeshConfig::default())
                .expect("ported product coupling mesh");
        let mesh_exterior_dofs = coupling_mesh.exterior.mesh.elements.len();
        let coupling = MonolithicSealedProductCoupling::from_coupling_mesh(
            &design,
            coupling_mesh,
            343.0,
            1.204,
        )
        .expect("ported product monolithic coupling");
        let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
        cluster.element_indices = (0..mesh_exterior_dofs).collect();
        cluster.num_elements = mesh_exterior_dofs;
        cluster.radius = 1.0;
        cluster.near_clusters = vec![0];
        let operator = coupling
            .build_compressed_product_operator(&design, 120.0, &[cluster], 1.0e-10, 16)
            .expect("compressed product operator");
        let (
            n_interior,
            operator_exterior_dofs,
            port_offset,
            velocity_offset,
            current_offset,
            total,
        ) = operator.offsets();
        assert_eq!(operator_exterior_dofs, mesh_exterior_dofs);
        assert_eq!(port_offset, n_interior + operator_exterior_dofs);
        assert!(
            velocity_offset > port_offset,
            "ported product has no port DOF"
        );
        assert_eq!(current_offset, velocity_offset + 1);
        assert_eq!(operator.num_rows(), total);
        assert_eq!(operator.num_cols(), total);

        let state = (0..total)
            .map(|index| Complex64::new(0.03 * (index + 1) as f64, -0.02 * index as f64))
            .collect::<Vec<_>>();
        let left = (0..total)
            .map(|index| Complex64::new(0.07 * index as f64, 0.01 * (index + 2) as f64))
            .collect::<Vec<_>>();
        let applied = operator.apply(&state);
        assert!(applied.iter().all(|value| value.is_finite()));

        let transpose = operator.apply_transpose(&left);
        let transpose_lhs = left
            .iter()
            .zip(&applied)
            .map(|(a, b)| *a * *b)
            .sum::<Complex64>();
        let transpose_rhs = transpose
            .iter()
            .zip(&state)
            .map(|(a, b)| *a * *b)
            .sum::<Complex64>();
        let transpose_error = (transpose_lhs - transpose_rhs).norm();
        let transpose_scale = transpose_lhs.norm().max(transpose_rhs.norm()).max(1.0);
        assert!(
            transpose_error / transpose_scale < 1.0e-10,
            "transpose error: lhs={transpose_lhs:?} rhs={transpose_rhs:?}"
        );

        let hermitian = operator.apply_hermitian(&left);
        let hermitian_lhs = left
            .iter()
            .zip(&applied)
            .map(|(a, b)| a.conj() * *b)
            .sum::<Complex64>();
        let hermitian_rhs = hermitian
            .iter()
            .zip(&state)
            .map(|(a, b)| a.conj() * *b)
            .sum::<Complex64>();
        let hermitian_error = (hermitian_lhs - hermitian_rhs).norm();
        let hermitian_scale = hermitian_lhs.norm().max(hermitian_rhs.norm()).max(1.0);
        assert!(
            hermitian_error / hermitian_scale < 1.0e-10,
            "Hermitian error: lhs={hermitian_lhs:?} rhs={hermitian_rhs:?}"
        );

        let mut current_probe = vec![Complex64::default(); total];
        current_probe[current_offset] = Complex64::new(1.0, 0.25);
        let current_action = operator.apply(&current_probe);
        assert!(current_action[velocity_offset].norm() > 1.0e-10);
        assert!(current_action[current_offset].norm() > 1.0e-10);

        let mut velocity_probe = vec![Complex64::default(); total];
        velocity_probe[velocity_offset] = Complex64::new(1.0, -0.25);
        let velocity_action = operator.apply(&velocity_probe);
        assert!(velocity_action[velocity_offset].norm() > 1.0e-10);
        assert!(velocity_action[current_offset].norm() > 1.0e-10);

        let invalid_many = operator.solve_compressed_many(
            &[vec![Complex64::default(); total - 1]],
            &FgmresConfig {
                max_iterations: 1,
                restart: 1,
                tolerance: 1.0e-8,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 1,
            },
        );
        assert!(matches!(
            invalid_many,
            Err(MonolithicProductCouplingError::InvalidCompressedRhs { .. })
        ));
        let empty_many = operator
            .solve_compressed_many(
                &[],
                &FgmresConfig {
                    max_iterations: 1,
                    restart: 1,
                    tolerance: 1.0e-8,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    exact_residual_check_interval: 1,
                },
            )
            .expect("empty compressed operator batch");
        assert!(empty_many.is_empty());
    }

    #[test]
    fn compressed_product_fgmres_matches_dense_voltage_solve() {
        let design = sealed_product();
        let coupling_mesh =
            ProductCouplingMesh::build(&design, ProductInteriorMeshConfig::default())
                .expect("sealed product coupling mesh");
        let exterior_dofs = coupling_mesh.exterior.mesh.elements.len();
        let coupling = MonolithicSealedProductCoupling::from_coupling_mesh(
            &design,
            coupling_mesh,
            343.0,
            1.204,
        )
        .expect("sealed product monolithic setup");
        let dense = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("dense product solve");

        let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
        cluster.element_indices = (0..exterior_dofs).collect();
        cluster.num_elements = exterior_dofs;
        cluster.radius = 1.0;
        cluster.near_clusters = vec![0];
        let clusters = vec![cluster];
        let config = FgmresConfig {
            max_iterations: 60,
            restart: 80,
            tolerance: 1.0e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };
        let compressed_batch = coupling
            .solve_compressed_many(
                &design,
                120.0,
                &[Complex64::new(2.0, 0.0), Complex64::new(-0.75, 0.25)],
                &clusters,
                1.0e-10,
                exterior_dofs,
                &config,
            )
            .expect("compressed product multi-RHS FGMRES solve");
        assert_eq!(compressed_batch.len(), 2);
        assert!(
            compressed_batch
                .iter()
                .all(|solution| solution.relative_residual < 1.0e-8)
        );
        assert!(
            compressed_batch
                .iter()
                .all(|solution| solution.state.iter().all(|value| value.is_finite()))
        );
        assert!(
            compressed_batch[0]
                .state
                .iter()
                .zip(&compressed_batch[1].state)
                .any(|(first, second)| (*first - *second).norm() > 1.0e-10)
        );
        let compressed = &compressed_batch[0];
        let operator = coupling
            .build_compressed_product_operator(&design, 120.0, &clusters, 1.0e-10, exterior_dofs)
            .expect("compressed product operator");
        let physics = sonium_bem::core::types::PhysicsParams::new(
            120.0,
            coupling.speed_of_sound,
            coupling.density,
            false,
        );
        let dense_velocity = StabilizedTbemVelocityOperator::new(
            &coupling.exterior_mesh.elements,
            &coupling.exterior_mesh.nodes,
            &physics,
            1.0,
            physics.burton_miller_beta_scaled(0.2),
        )
        .expect("dense velocity operator");
        for (driver_index, slots) in operator.driver_velocity_slots.iter().enumerate() {
            let mut velocity = vec![Complex64::default(); operator.velocity.len()];
            for &slot in slots {
                velocity[slot] = Complex64::new(1.0, 0.0);
            }
            let compressed_rhs = operator
                .velocity
                .apply(&velocity)
                .expect("compressed velocity RHS");
            let dense_rhs = dense_velocity
                .velocity_rhs(&velocity)
                .expect("dense velocity RHS");
            let max_difference = compressed_rhs
                .iter()
                .zip(&dense_rhs)
                .map(|(compressed, dense)| (*compressed - *dense).norm())
                .fold(0.0_f64, f64::max);
            assert!(
                max_difference < 1.0e-8,
                "driver {driver_index} velocity RHS mismatch: {max_difference:.6e}"
            );
        }

        let n_interior = dense.interior_pressure.len();
        let n_exterior = dense.exterior_pressure.len();
        let n_port = dense.port_velocity_m_per_s.len();
        let n_drivers = dense.driver_velocity_m_per_s.len();
        let velocity_offset = n_interior + n_exterior + n_port;
        let current_offset = velocity_offset + n_drivers;
        let mut dense_state = dense.interior_pressure.clone();
        dense_state.extend_from_slice(&dense.exterior_pressure);
        dense_state.extend_from_slice(&dense.port_velocity_m_per_s);
        dense_state.extend_from_slice(&dense.driver_velocity_m_per_s);
        dense_state.extend_from_slice(&dense.driver_current_a);
        let terminal_voltage = design
            .crossover
            .transfers_at(n_drivers, 120.0)
            .expect("crossover transfers")
            .into_iter()
            .map(|transfer| transfer * Complex64::new(2.0, 0.0))
            .collect::<Vec<_>>();
        let mut voltage_rhs = vec![Complex64::default(); dense_state.len()];
        voltage_rhs[current_offset..].copy_from_slice(&terminal_voltage);
        let dense_state_residual = operator
            .apply(&dense_state)
            .iter()
            .zip(&voltage_rhs)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            dense_state_residual < 1.0e-8,
            "dense state residual under compressed operator: {dense_state_residual:.6e}"
        );
        let max_state_difference = compressed
            .state
            .iter()
            .zip(&dense_state)
            .map(|(compressed, dense)| (*compressed - *dense).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_state_difference < 1.0e-6,
            "compressed/dense state mismatch: {max_state_difference:.6e}"
        );
        assert!(compressed.relative_residual < 1.0e-8);
        assert!(compressed.iterations > 0);
        assert!(
            compressed
                .driver_mechanical_relative_residual
                .iter()
                .all(|residual| *residual < 1.0e-8)
        );
        assert!(
            compressed
                .driver_electrical_relative_residual
                .iter()
                .all(|residual| *residual < 1.0e-8)
        );
        assert_eq!(compressed.state.len(), current_offset + n_drivers);
        assert_eq!(
            compressed.driver_mechanical_relative_residual.len(),
            dense.driver_mechanical_relative_residual.len()
        );
        assert_eq!(
            compressed.driver_electrical_relative_residual.len(),
            dense.driver_electrical_relative_residual.len()
        );
        assert!(dense.relative_residual < 1.0e-10);
        assert!(
            dense
                .driver_mechanical_relative_residual
                .iter()
                .chain(&dense.driver_electrical_relative_residual)
                .all(|residual| *residual < 1.0e-10)
        );
        assert!(compressed.state.iter().all(|value| value.is_finite()));
        assert!(
            compressed.state[current_offset..]
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        assert!(
            dense
                .driver_current_a
                .iter()
                .any(|value| value.norm() > 0.0)
        );
    }

    #[test]
    fn compressed_product_sweep_dispatches_frequency_tagged_points() {
        let design = sealed_product();
        let coupling_mesh =
            ProductCouplingMesh::build(&design, ProductInteriorMeshConfig::default())
                .expect("sealed product coupling mesh");
        let exterior_dofs = coupling_mesh.exterior.mesh.elements.len();
        let coupling = MonolithicSealedProductCoupling::from_coupling_mesh(
            &design,
            coupling_mesh,
            343.0,
            1.204,
        )
        .expect("sealed product monolithic setup");

        let mut cluster = sonium_bem::core::types::Cluster::new([0.0; 3]);
        cluster.element_indices = (0..exterior_dofs).collect();
        cluster.num_elements = exterior_dofs;
        cluster.radius = 1.0;
        cluster.near_clusters = vec![0];
        let config = FgmresConfig {
            max_iterations: 60,
            restart: 80,
            tolerance: 1.0e-8,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
        };

        let sweep = coupling
            .solve_compressed_sweep(
                &design,
                &[100.0, 120.0],
                Complex64::new(2.0, 0.0),
                &[cluster.clone()],
                1.0e-10,
                16,
                &config,
            )
            .expect("compressed product frequency sweep");
        assert_eq!(sweep.len(), 2);
        assert_eq!(sweep[0].frequency_hz, 100.0);
        assert_eq!(sweep[1].frequency_hz, 120.0);
        assert!(
            sweep
                .iter()
                .all(|point| point.solution.relative_residual < 1.0e-8)
        );
        assert!(
            sweep
                .iter()
                .all(|point| point.solution.state.iter().all(|value| value.is_finite()))
        );
        assert!(
            sweep[0]
                .solution
                .state
                .iter()
                .zip(&sweep[1].solution.state)
                .any(|(first, second)| (*first - *second).norm() > 1.0e-10)
        );

        let invalid_voltage = coupling.solve_compressed_many(
            &design,
            120.0,
            &[Complex64::new(f64::NAN, 0.0)],
            &[cluster.clone()],
            1.0e-10,
            16,
            &config,
        );
        assert!(matches!(
            invalid_voltage,
            Err(MonolithicProductCouplingError::InvalidDrive)
        ));
        let empty_batch = coupling
            .solve_compressed_many(
                &design,
                120.0,
                &[],
                &[cluster.clone()],
                1.0e-10,
                16,
                &config,
            )
            .expect("empty compressed product batch");
        assert!(empty_batch.is_empty());
        let empty_rhs_sweep = coupling
            .solve_compressed_sweep_many(
                &design,
                &[100.0, 120.0],
                &[],
                &[cluster.clone()],
                1.0e-10,
                16,
                &config,
            )
            .expect("empty compressed product RHS sweep");
        assert_eq!(empty_rhs_sweep.len(), 2);
        assert!(empty_rhs_sweep.iter().all(|batch| batch.is_empty()));
        let empty_frequency_sweep = coupling
            .solve_compressed_sweep_many(
                &design,
                &[],
                &[Complex64::new(2.0, 0.0)],
                &[cluster.clone()],
                1.0e-10,
                16,
                &config,
            )
            .expect("empty compressed product frequency sweep");
        assert!(empty_frequency_sweep.is_empty());

        let invalid_grid = coupling.solve_compressed_sweep(
            &design,
            &[120.0, 120.0],
            Complex64::new(2.0, 0.0),
            &[cluster],
            1.0e-10,
            16,
            &config,
        );
        assert!(matches!(
            invalid_grid,
            Err(MonolithicProductCouplingError::InvalidCompressedFrequencyGrid)
        ));
    }

    #[test]
    fn sealed_product_monolithic_applies_structural_wall_feedback() {
        let baseline_design = sealed_product();
        let baseline_mesh = build_sealed_product_interior_mesh(
            &baseline_design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.08,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("baseline product mesh");
        let mut feedback_design = sealed_product();
        feedback_design.structural_acoustic = Some(StructuralAcousticFeedback {
            area_m2: 0.02,
            modes: vec![StructuralAcousticMode {
                natural_frequency_hz: 250.0,
                effective_mass_kg: 0.2,
                damping_ratio: 0.08,
                pressure_participation: 1.0,
                stress_per_displacement_pa_per_m: 1_000.0,
            }],
            max_iterations: 12,
            velocity_tolerance: 1.0e-6,
            under_relaxation: 0.8,
        });
        let coupling =
            MonolithicSealedProductCoupling::new(&baseline_design, baseline_mesh, 343.0, 1.204)
                .expect("baseline monolithic setup");
        let baseline = coupling
            .solve_frequency(&baseline_design, 120.0, Complex64::new(2.0, 0.0))
            .expect("baseline monolithic solve");
        let feedback = coupling
            .solve_frequency(&feedback_design, 120.0, Complex64::new(2.0, 0.0))
            .expect("feedback monolithic solve");

        assert!(baseline.relative_residual < 1.0e-10);
        assert!(feedback.relative_residual < 1.0e-10);
        assert!(feedback.interior_relative_residual < 1.0e-10);
        assert_eq!(feedback.structural_generalized_displacement_peak.len(), 1);
        assert_eq!(feedback.structural_velocity_peak.len(), 1);
        assert!(feedback.structural_velocity_peak[0].norm() > 0.0);
        let expected_structural_velocity = Complex64::new(
            0.0,
            -std::f64::consts::TAU
                * 120.0
                * feedback_design.structural_acoustic.as_ref().unwrap().modes[0]
                    .pressure_participation,
        ) * feedback.structural_generalized_displacement_peak[0];
        assert!(
            (feedback.structural_velocity_peak[0] - expected_structural_velocity).norm() < 1.0e-12
        );
        assert!(feedback.structural_relative_residual < 1.0e-10);
        let point = ProductReferencePoint::from_monolithic_solution_with_structural_feedback(
            &feedback,
            coupling.interior_mesh(),
            0.0,
            feedback_design.structural_acoustic.as_ref(),
        )
        .expect("structural monolithic evidence should be retained");
        let modal_residual = point
            .structural_evidence
            .as_ref()
            .expect("structural evidence")
            .explicit_modal_relative_residual
            .expect("explicit modal residual");
        assert!(modal_residual.is_finite());
        assert!(modal_residual < 1.0e-10);
        let pressure_change = baseline
            .interior_pressure
            .iter()
            .zip(&feedback.interior_pressure)
            .map(|(before, after)| (*before - *after).norm())
            .fold(0.0, f64::max);
        assert!(
            pressure_change > 1.0e-8,
            "structural feedback was not applied"
        );
    }

    #[test]
    fn sealed_product_monolithic_closes_all_acoustic_and_driver_blocks() {
        let design = sealed_product();
        let mesh = build_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.08,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("product coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("product solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 1);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 1);
        assert!(solution.driver_current_a[0].norm() > 0.0);
        assert!(solution.driver_velocity_m_per_s[0].norm() > 0.0);
        assert!(solution.electrical_input_power_w.is_finite());
        assert!(solution.motor_mechanical_power_w.is_finite());
        assert!(solution.electrical_power_residual_w.abs() < 1.0e-10);
        assert!(solution.mechanical_power_residual_w.abs() < 1.0e-10);
        assert_eq!(solution.driver_normal_m, vec![design.drivers[0].normal]);
    }

    #[test]
    fn body_fitted_waveguide_monolithic_applies_impedance_boundary() {
        let baseline_design = sealed_product();
        let baseline_mesh = build_body_fitted_sealed_product_interior_mesh(
            &baseline_design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("baseline body-fitted product mesh");

        let mut waveguide_design = sealed_product();
        waveguide_design.waveguide = Some(crate::ProductWaveguideDesign {
            driver_id: "woofer".into(),
            horizontal_spline: vec![[0.0, 0.06], [0.5, 0.052], [1.0, 0.045]],
            vertical_spline: vec![[0.0, 0.055], [0.5, 0.05], [1.0, 0.045]],
            depth_m: 0.02,
            surface_specific_impedance_pa_s_per_m: 2.5e6,
        });
        let waveguide_mesh = build_body_fitted_sealed_product_interior_mesh(
            &waveguide_design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("waveguide body-fitted product mesh");
        assert!(
            waveguide_mesh
                .patch(ProductInteriorPatchKind::WaveguideWall { driver_index: 0 })
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );

        let baseline =
            MonolithicSealedProductCoupling::new(&baseline_design, baseline_mesh, 343.0, 1.204)
                .expect("baseline body-fitted coupling")
                .solve_frequency(&baseline_design, 120.0, Complex64::new(2.0, 0.0))
                .expect("baseline body-fitted solve");
        let waveguide =
            MonolithicSealedProductCoupling::new(&waveguide_design, waveguide_mesh, 343.0, 1.204)
                .expect("waveguide body-fitted coupling")
                .solve_frequency(&waveguide_design, 120.0, Complex64::new(2.0, 0.0))
                .expect("waveguide body-fitted solve");

        assert!(waveguide.relative_residual < 1.0e-8);
        assert!(waveguide.interior_relative_residual < 1.0e-8);
        assert!(waveguide.exterior_relative_residual < 1.0e-8);
        assert!(waveguide.driver_velocity_m_per_s[0].norm() > 0.0);
        let max_pressure_delta = baseline
            .interior_pressure
            .iter()
            .zip(&waveguide.interior_pressure)
            .map(|(before, after)| (*before - *after).norm())
            .fold(0.0, f64::max);
        assert!(
            max_pressure_delta > 1.0e-8,
            "waveguide impedance must perturb the interior solution"
        );
    }

    #[test]
    fn monolithic_rejects_stale_driver_orientation_metadata() {
        let mut design = sealed_product();
        let tilt = 0.3_f64;
        design.drivers[0].normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];
        let mut mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("tilted body-fitted product mesh");
        let patch = mesh
            .patches
            .iter_mut()
            .find(|patch| patch.kind == ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("tilted driver patch");
        patch.canonical_normal_m = Some([0.0, 0.0, 1.0]);
        assert!(matches!(
            MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204),
            Err(MonolithicProductCouplingError::DriverNormalMismatch { driver_index: 0 })
        ));

        let mut mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("tilted body-fitted product mesh");
        mesh.patches
            .iter_mut()
            .find(|patch| patch.kind == ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("tilted driver patch")
            .canonical_normal_m = None;
        assert!(matches!(
            MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204),
            Err(MonolithicProductCouplingError::MissingDriverNormal { driver_index: 0 })
        ));
    }

    #[test]
    fn body_fitted_sealed_product_monolithic_closes_driver_blocks() {
        let design = sealed_product();
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("body-fitted product coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("body-fitted product solve");

        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 1);
        assert!(solution.driver_velocity_m_per_s[0].norm() > 0.0);
    }

    #[test]
    fn body_fitted_tilted_driver_monolithic_closes_all_blocks() {
        let mut design = sealed_product();
        let tilt = 0.3_f64;
        design.drivers[0].normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];
        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("tilted body-fitted product mesh");
        let driver_patch = mesh
            .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
            .expect("tilted driver patch");
        assert_eq!(
            driver_patch.canonical_normal_m,
            Some(design.drivers[0].normal)
        );
        assert!(driver_patch.area_m2 > 0.0);
        assert!(
            driver_patch
                .boundary_faces
                .iter()
                .flat_map(|&face_index| mesh.mesh.boundaries[face_index].nodes.iter())
                .map(|&node| mesh.mesh.nodes[node].z)
                .any(|z| z.abs() > 1.0e-4),
            "tilted driver must not be coplanar with the front wall"
        );

        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("tilted body-fitted product coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("tilted body-fitted product solve");
        // The tilted body-fitted FEM block is mildly conditioned; dense LU
        // remains within 1e-9 while the other coupled blocks stay at 1e-10.
        assert!(solution.relative_residual < 1.0e-9);
        assert!(solution.interior_relative_residual < 1.0e-9);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert!(solution.driver_current_a[0].norm() > 0.0);
        assert!(solution.driver_velocity_m_per_s[0].norm() > 0.0);
        assert_eq!(solution.driver_normal_m, vec![design.drivers[0].normal]);
        assert!(solution.electrical_power_residual_w.abs() < 1.0e-10);
        assert!(solution.mechanical_power_residual_w.abs() < 1.0e-10);
    }

    #[test]
    fn body_fitted_tilted_driver_monolithic_closes_every_face() {
        let base = sealed_product();
        let w = base.cabinet.box_width_m;
        let h = base.cabinet.box_height_m;
        let d = base.cabinet.box_depth_m;
        let tilt = 0.3_f64;
        let normal_axis = (1.0 - tilt * tilt).sqrt();
        let cases = [
            (
                DriverMountingFace::Front,
                [tilt, 0.0, normal_axis],
                [0.0, 0.0, 0.0],
            ),
            (
                DriverMountingFace::Back,
                [tilt, 0.0, -normal_axis],
                [0.0, 0.0, -d],
            ),
            (
                DriverMountingFace::Right,
                [normal_axis, 0.0, tilt],
                [w * 0.5, 0.0, -d * 0.5],
            ),
            (
                DriverMountingFace::Left,
                [-normal_axis, 0.0, tilt],
                [-w * 0.5, 0.0, -d * 0.5],
            ),
            (
                DriverMountingFace::Top,
                [0.0, normal_axis, tilt],
                [0.0, h * 0.5, -d * 0.5],
            ),
            (
                DriverMountingFace::Bottom,
                [0.0, -normal_axis, tilt],
                [0.0, -h * 0.5, -d * 0.5],
            ),
        ];

        for (face, normal, position_m) in cases {
            let mut design = base.clone();
            design.drivers[0].normal = normal;
            design.drivers[0].position_m = position_m;
            let mesh = build_body_fitted_sealed_product_interior_mesh(
                &design,
                ProductInteriorMeshConfig::default(),
            )
            .unwrap_or_else(|error| panic!("{face:?} tilted mesh: {error}"));
            let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
                .unwrap_or_else(|error| panic!("{face:?} tilted coupling: {error}"));
            let solution = coupling
                .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
                .unwrap_or_else(|error| panic!("{face:?} tilted solve: {error}"));

            // The tilted wall fans are intentionally more oblique than the
            // axis-aligned baseline; retain a tight but conditioning-aware
            // algebraic gate for this geometry-breadth regression.
            assert!(
                solution.relative_residual < 1.0e-8,
                "{face:?} relative residual {}",
                solution.relative_residual
            );
            assert!(
                solution.interior_relative_residual < 1.0e-8,
                "{face:?} interior residual {}",
                solution.interior_relative_residual
            );
            assert!(
                solution.exterior_relative_residual < 1.0e-8,
                "{face:?} exterior residual {}",
                solution.exterior_relative_residual
            );
            assert!(
                solution.mechanical_relative_residual < 1.0e-8,
                "{face:?} mechanical residual {}",
                solution.mechanical_relative_residual
            );
            assert!(
                solution.electrical_relative_residual < 1.0e-8,
                "{face:?} electrical residual {}",
                solution.electrical_relative_residual
            );
            assert_eq!(solution.driver_normal_m, vec![normal], "{face:?}");
            assert!(solution.driver_current_a[0].norm() > 0.0, "{face:?}");
            assert!(solution.driver_velocity_m_per_s[0].norm() > 0.0, "{face:?}");
        }
    }

    #[test]
    fn body_fitted_sealed_product_monolithic_closes_every_axis_aligned_driver_blocks() {
        let base = sealed_product();
        let w = base.cabinet.box_width_m;
        let h = base.cabinet.box_height_m;
        let d = base.cabinet.box_depth_m;
        let cases = [
            ("front", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ("back", [0.0, 0.0, -1.0], [0.0, 0.0, -d]),
            ("right", [1.0, 0.0, 0.0], [w * 0.5, 0.0, -d * 0.5]),
            ("left", [-1.0, 0.0, 0.0], [-w * 0.5, 0.0, -d * 0.5]),
            ("top", [0.0, 1.0, 0.0], [0.0, h * 0.5, -d * 0.5]),
            ("bottom", [0.0, -1.0, 0.0], [0.0, -h * 0.5, -d * 0.5]),
        ];

        for (name, normal, position_m) in cases {
            let mut design = base.clone();
            design.drivers[0].normal = normal;
            design.drivers[0].position_m = position_m;
            let mesh = build_body_fitted_sealed_product_interior_mesh(
                &design,
                ProductInteriorMeshConfig::default(),
            )
            .unwrap_or_else(|error| panic!("{name} body-fitted product mesh: {error}"));
            let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
                .unwrap_or_else(|error| panic!("{name} body-fitted product coupling: {error}"));
            let solution = coupling
                .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
                .unwrap_or_else(|error| panic!("{name} body-fitted product solve: {error}"));

            assert!(solution.relative_residual < 1.0e-10, "{name}");
            assert!(solution.interior_relative_residual < 1.0e-10, "{name}");
            assert!(solution.exterior_relative_residual < 1.0e-10, "{name}");
            assert!(solution.mechanical_relative_residual < 1.0e-10, "{name}");
            assert!(solution.electrical_relative_residual < 1.0e-10, "{name}");
            assert_eq!(solution.driver_current_a.len(), 1, "{name}");
            assert_eq!(solution.driver_velocity_m_per_s.len(), 1, "{name}");
            assert!(solution.driver_current_a[0].norm() > 0.0, "{name}");
            assert!(solution.driver_velocity_m_per_s[0].norm() > 0.0, "{name}");
        }
    }

    #[test]
    fn body_fitted_sealed_product_monolithic_closes_distinct_wall_driver_blocks() {
        let mut design = sealed_product();
        let mut second = design.drivers[0].clone();
        second.id = "side-tweeter".into();
        second.radius_m = 0.03;
        second.crossover_branch = 1;
        second.normal = [1.0, 0.0, 0.0];
        second.position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("two-wall body-fitted product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("two-wall body-fitted product coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("two-wall body-fitted product solve");

        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 2);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 2);
        assert!(
            solution
                .driver_current_a
                .iter()
                .all(|current| current.norm() > 0.0)
        );
        assert!(
            solution
                .driver_velocity_m_per_s
                .iter()
                .all(|velocity| velocity.norm() > 0.0)
        );
    }

    #[test]
    fn body_fitted_sealed_product_monolithic_closes_tilted_distinct_wall_driver_blocks() {
        let mut design = sealed_product();
        let tilt: f64 = 0.2;
        let normal_axis = (1.0 - tilt * tilt).sqrt();
        design.drivers[0].normal = [tilt, 0.0, normal_axis];

        let mut second = design.drivers[0].clone();
        second.id = "tilted-side-tweeter".into();
        second.radius_m = 0.03;
        second.crossover_branch = 1;
        second.normal = [normal_axis, 0.0, tilt];
        second.position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("two-wall tilted body-fitted product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("two-wall tilted body-fitted coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("two-wall tilted body-fitted solve");

        // Oblique two-wall throat facets mildly condition the assembled
        // interior block; the geometry-breadth gate remains two orders tighter
        // than the product promotion budget while the other blocks retain the
        // strict residual checks below.
        assert!(solution.relative_residual < 1.0e-8);
        assert!(solution.interior_relative_residual < 1.0e-8);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_normal_m.len(), 2);
        assert_eq!(solution.driver_normal_m[0], design.drivers[0].normal);
        assert_eq!(solution.driver_normal_m[1], design.drivers[1].normal);
        assert!(
            solution
                .driver_current_a
                .iter()
                .all(|current| current.norm() > 0.0)
        );
        assert!(
            solution
                .driver_velocity_m_per_s
                .iter()
                .all(|velocity| velocity.norm() > 0.0)
        );
    }

    #[test]
    fn body_fitted_sealed_product_monolithic_closes_same_wall_driver_blocks() {
        let mut design = sealed_product();
        design.cabinet.box_width_m = 0.4;
        design.cabinet.baffle_width_m = 0.44;
        design.drivers[0].position_m[0] = -0.1;
        let mut second = design.drivers[0].clone();
        second.id = "second-front-driver".into();
        second.crossover_branch = 1;
        second.position_m = [0.1, 0.0, 0.0];
        design.drivers.push(second);
        design.crossover.branches.push(CrossoverBranch::default());

        let mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("same-wall body-fitted product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("same-wall body-fitted product coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("same-wall body-fitted product solve");

        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 2);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 2);
        assert!(
            solution
                .driver_current_a
                .iter()
                .all(|current| current.norm() > 0.0)
        );
        assert!(
            solution
                .driver_velocity_m_per_s
                .iter()
                .all(|velocity| velocity.norm() > 0.0)
        );
    }

    #[test]
    fn sealed_product_monolithic_closes_side_wall_driver_blocks() {
        let mut design = sealed_product();
        design.drivers[0].normal = [1.0, 0.0, 0.0];
        design.drivers[0].position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        let mesh = build_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.08,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("side-wall product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("side-wall product coupling");
        let solution = coupling
            .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
            .expect("side-wall product solve");

        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 1);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 1);
    }

    #[test]
    fn ported_product_monolithic_closes_port_pressure_continuity() {
        let mut design = sealed_product();
        design.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        let interior =
            build_ported_product_interior_mesh(&design, ProductInteriorMeshConfig::default())
                .expect("ported product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, interior, 343.0, 1.204)
            .expect("ported monolithic setup");
        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("ported monolithic solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(!solution.port_velocity_m_per_s.is_empty());
        assert!(
            solution
                .port_velocity_m_per_s
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );
        assert!(solution.port_volume_velocity_m3_per_s.is_finite());
        let point = ProductReferencePoint::from_monolithic_solution(
            &solution,
            &coupling.interior_mesh,
            0.0,
        )
        .expect("mesh-convergence evidence should be valid");
        assert!(
            ProductReferencePoint::from_monolithic_solution(
                &solution,
                &coupling.interior_mesh,
                f64::NAN,
            )
            .is_err()
        );
        assert_eq!(point.driver_velocities_rms_m_per_s.len(), 1);
        assert_eq!(
            point
                .monolithic_evidence
                .as_ref()
                .expect("evidence")
                .mesh_geometry_fingerprint,
            coupling.interior_mesh.geometry_fingerprint
        );
    }

    #[test]
    #[ignore = "body-fitted product coupling validation; run via `just qa-nightly-regressions`"]
    fn body_fitted_ported_product_monolithic_closes_port_pressure_continuity() {
        let mut design = sealed_product();
        design.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        let interior = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted ported product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, interior, 343.0, 1.204)
            .expect("body-fitted ported monolithic setup");
        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("body-fitted ported monolithic solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert!(!solution.port_velocity_m_per_s.is_empty());
        assert!(
            solution
                .port_velocity_m_per_s
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );
        assert!(solution.port_volume_velocity_m3_per_s.is_finite());
    }

    #[test]
    fn body_fitted_port_and_brace_monolithic_closes_interfaces() {
        let mut design = sealed_product();
        design.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        design
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        let interior_mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("body-fitted port and brace mesh");
        assert!(
            interior_mesh
                .patch(crate::product_acoustic_mesh::ProductInteriorPatchKind::PortMouth)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(
            interior_mesh
                .patch(
                    crate::product_acoustic_mesh::ProductInteriorPatchKind::BraceWall {
                        brace_index: 0
                    }
                )
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );

        let coupling = MonolithicSealedProductCoupling::new(&design, interior_mesh, 343.0, 1.204)
            .expect("body-fitted port and brace monolithic interface construction");
        assert!(!coupling.port_bem_elements.is_empty());
        assert_eq!(coupling.driver_bem_elements.len(), 1);

        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("body-fitted port and brace monolithic solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert!(!solution.port_velocity_m_per_s.is_empty());
        assert!(solution.port_volume_velocity_m3_per_s.is_finite());
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );
    }

    #[test]
    fn body_fitted_tilted_port_and_brace_constructs_monolithic_interfaces() {
        let mut design = sealed_product();
        let tilt: f64 = 0.2;
        design.drivers[0].position_m[0] = -0.06;
        design.drivers[0].normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        design
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        let interior_mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("tilted body-fitted port and brace mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, interior_mesh, 343.0, 1.204)
            .expect("tilted body-fitted port and brace coupling");

        assert_eq!(coupling.driver_bem_elements.len(), 1);
        assert!(!coupling.driver_bem_elements[0].is_empty());
        assert!(!coupling.port_bem_elements.is_empty());
        assert_eq!(
            coupling.port_bem_elements.len(),
            coupling.port_fem_faces.len()
        );
    }

    #[test]
    fn body_fitted_tilted_port_and_brace_monolithic_closes_interfaces() {
        let mut design = sealed_product();
        let tilt: f64 = 0.2;
        design.drivers[0].position_m[0] = -0.06;
        design.drivers[0].normal = [tilt, 0.0, (1.0 - tilt * tilt).sqrt()];
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        design
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        let interior_mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.025,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("tilted body-fitted port and brace mesh");
        assert_eq!(
            interior_mesh
                .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
                .expect("tilted driver patch")
                .canonical_normal_m,
            Some(design.drivers[0].normal)
        );
        assert!(
            !interior_mesh
                .patch(ProductInteriorPatchKind::PortMouth)
                .expect("port mouth patch")
                .boundary_faces
                .is_empty()
        );
        assert!(
            !interior_mesh
                .patch(ProductInteriorPatchKind::BraceWall { brace_index: 0 })
                .expect("brace wall patch")
                .boundary_faces
                .is_empty()
        );

        let coupling = MonolithicSealedProductCoupling::new(&design, interior_mesh, 343.0, 1.204)
            .expect("tilted body-fitted port and brace coupling");
        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("tilted body-fitted port and brace solve");
        assert!(solution.relative_residual < 1.0e-8);
        assert!(solution.interior_relative_residual < 1.0e-8);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert!(!solution.port_velocity_m_per_s.is_empty());
        assert!(solution.port_volume_velocity_m3_per_s.is_finite());
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );
    }

    #[test]
    fn body_fitted_ported_product_monolithic_closes_two_driver_interfaces() {
        let mut design = sealed_product();
        design.cabinet.box_width_m = 0.4;
        design.cabinet.baffle_width_m = 0.44;
        design.drivers[0].position_m[0] = -0.10;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.0, 0.0];
        design.cabinet.port = Some(port);

        let mut side_driver = design.drivers[0].clone();
        side_driver.id = "front-tweeter".into();
        side_driver.radius_m = 0.03;
        side_driver.position_m[0] = 0.10;
        side_driver.crossover_branch = 1;
        design.drivers.push(side_driver);
        design.crossover.branches.push(CrossoverBranch::default());

        let interior_mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted two-driver ported mesh");
        assert!(
            interior_mesh
                .patch(ProductInteriorPatchKind::PortMouth)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        assert!(
            interior_mesh
                .patch(ProductInteriorPatchKind::PortWall)
                .is_some_and(|patch| patch.area_m2 > 0.0)
        );
        for driver_index in 0..2 {
            assert!(
                interior_mesh
                    .patch(ProductInteriorPatchKind::DriverRear { driver_index })
                    .is_some_and(|patch| patch.area_m2 > 0.0),
                "missing driver patch {driver_index}"
            );
        }

        let coupling = MonolithicSealedProductCoupling::new(&design, interior_mesh, 343.0, 1.204)
            .expect("body-fitted two-driver ported coupling");
        assert_eq!(coupling.driver_bem_elements.len(), 2);
        assert!(!coupling.port_bem_elements.is_empty());

        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("body-fitted two-driver ported solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 2);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 2);
        assert!(
            solution
                .driver_current_a
                .iter()
                .all(|current| current.norm() > 0.0)
        );
        assert!(
            solution
                .driver_velocity_m_per_s
                .iter()
                .all(|velocity| velocity.norm() > 0.0)
        );
        assert!(!solution.port_velocity_m_per_s.is_empty());
        assert!(solution.port_volume_velocity_m3_per_s.is_finite());
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );
        assert!(solution.electrical_input_power_w.is_finite());
        assert!(solution.motor_mechanical_power_w.is_finite());
        assert!(solution.electrical_power_residual_w.abs() < 1.0e-10);
        assert!(solution.mechanical_power_residual_w.abs() < 1.0e-10);
        assert_eq!(solution.driver_mechanical_relative_residual.len(), 2);
        assert_eq!(solution.driver_electrical_relative_residual.len(), 2);
        assert_eq!(solution.driver_volume_velocity_m3_per_s.len(), 2);
        assert_eq!(solution.driver_electrical_power_residual_w.len(), 2);
        assert_eq!(solution.driver_mechanical_power_residual_w.len(), 2);
        assert!(
            solution
                .driver_mechanical_relative_residual
                .iter()
                .chain(&solution.driver_electrical_relative_residual)
                .all(|residual| *residual < 1.0e-10)
        );

        let point = ProductReferencePoint::from_monolithic_solution(
            &solution,
            &coupling.interior_mesh,
            0.0,
        )
        .expect("two-driver interface evidence");
        let interfaces = point.interface_evidence.as_ref().expect("interfaces");
        assert_eq!(interfaces.len(), 3);
        assert_eq!(interfaces[0].interface_id, "port-mouth");
        assert_eq!(interfaces[1].interface_id, "driver-aperture-0");
        assert_eq!(interfaces[2].interface_id, "driver-aperture-1");
        let mut frequency_solutions = vec![solution];
        for frequency_hz in [60.0, 120.0] {
            frequency_solutions.push(
                coupling
                    .solve_frequency(&design, frequency_hz, Complex64::new(2.0, 0.0))
                    .expect("two-driver ported frequency solve"),
            );
        }
        frequency_solutions.sort_by(|left, right| left.frequency_hz.total_cmp(&right.frequency_hz));
        for solution in &frequency_solutions {
            assert!(solution.relative_residual < 1.0e-10);
            assert!(solution.interior_relative_residual < 1.0e-10);
            assert!(solution.exterior_relative_residual < 1.0e-10);
            assert!(solution.mechanical_relative_residual < 1.0e-10);
            assert!(solution.electrical_relative_residual < 1.0e-10);
            assert!(solution.port_volume_velocity_m3_per_s.is_finite());
            assert!(
                solution
                    .port_pressure_jump_pa
                    .iter()
                    .all(|jump| jump.norm() < 1.0e-10)
            );
        }
        let points = frequency_solutions
            .iter()
            .map(|solution| {
                ProductReferencePoint::from_monolithic_solution(
                    solution,
                    &coupling.interior_mesh,
                    0.0,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("three-frequency two-driver evidence");
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: frequency_solutions[1].interior_pressure.len(),
            aperture_elements: coupling.port_bem_elements.len(),
            points,
            driver_count: 2,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        sweep
            .validate_monolithic_product_evidence(MonolithicEvidenceGate::default())
            .expect("two-driver ported evidence should validate");
    }

    #[test]
    fn body_fitted_ported_product_monolithic_closes_three_driver_interfaces() {
        let mut design = sealed_product();
        design.cabinet.box_width_m = 0.4;
        design.cabinet.baffle_width_m = 0.44;
        design.cabinet.baffle_height_m = 0.44;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.0, 0.14];
        design.cabinet.port = Some(port);
        design.drivers[0].id = "left-woofer".into();
        design.drivers[0].radius_m = 0.03;
        design.drivers[0].position_m = [-0.12, -0.08, 0.0];

        let mut centre_driver = design.drivers[0].clone();
        centre_driver.id = "centre-tweeter".into();
        centre_driver.radius_m = 0.025;
        centre_driver.position_m = [0.0, 0.0, 0.0];
        centre_driver.crossover_branch = 1;
        design.drivers.push(centre_driver);

        let mut right_driver = design.drivers[0].clone();
        right_driver.id = "right-woofer".into();
        right_driver.position_m = [0.12, -0.08, 0.0];
        right_driver.crossover_branch = 2;
        design.drivers.push(right_driver);
        design.crossover.branches = vec![CrossoverBranch::default(); 3];

        let interior_mesh = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted three-driver ported mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, interior_mesh, 343.0, 1.204)
            .expect("body-fitted three-driver ported coupling");
        assert_eq!(coupling.driver_bem_elements.len(), 3);
        assert!(!coupling.port_bem_elements.is_empty());

        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("body-fitted three-driver ported solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 3);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 3);
        assert!(
            solution
                .driver_current_a
                .iter()
                .all(|current| current.norm() > 0.0)
        );
        assert!(
            solution
                .driver_velocity_m_per_s
                .iter()
                .all(|velocity| velocity.norm() > 0.0)
        );
        assert!(
            solution
                .driver_mechanical_relative_residual
                .iter()
                .chain(&solution.driver_electrical_relative_residual)
                .all(|residual| *residual < 1.0e-10)
        );
        assert!(solution.port_volume_velocity_m3_per_s.is_finite());
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );

        let point = ProductReferencePoint::from_monolithic_solution(
            &solution,
            &coupling.interior_mesh,
            0.0,
        )
        .expect("three-driver interface evidence");
        let interfaces = point.interface_evidence.as_ref().expect("interfaces");
        assert_eq!(interfaces.len(), 4);
        assert!(
            interfaces
                .iter()
                .map(|interface| interface.interface_id.as_str())
                .any(|id| id == "driver-aperture-2")
        );

        let thermal = design
            .thermal_reference_evidence(
                &solution,
                crate::DriverThermalNetworkConfig::default(),
                &[0.0, 0.0, 0.0],
            )
            .expect("three-driver thermal evidence");
        assert_eq!(
            thermal.driver_ids,
            vec!["left-woofer", "centre-tweeter", "right-woofer",]
        );
        assert_eq!(thermal.states.len(), 3);
        assert!(
            thermal
                .states
                .iter()
                .all(|state| state.maximum_balance_residual_w < 1.0e-8)
        );
        let thermal_point = design
            .reference_point_from_monolithic_solution_with_thermal(
                &solution,
                &coupling.interior_mesh,
                0.0,
                crate::DriverThermalNetworkConfig::default(),
                &[0.0, 0.0, 0.0],
            )
            .expect("three-driver thermal reference point");
        assert!(thermal_point.thermal_evidence.is_some());
    }

    #[test]
    fn body_fitted_ported_three_driver_frequency_grid_closes_crossover_and_power() {
        let mut design = sealed_product();
        design.cabinet.box_width_m = 0.4;
        design.cabinet.baffle_width_m = 0.44;
        design.cabinet.baffle_height_m = 0.44;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.0, 0.14];
        design.cabinet.port = Some(port);

        design.drivers[0].id = "left-woofer".into();
        design.drivers[0].radius_m = 0.03;
        design.drivers[0].position_m = [-0.12, -0.08, 0.0];
        let mut centre_driver = design.drivers[0].clone();
        centre_driver.id = "centre-tweeter".into();
        centre_driver.radius_m = 0.025;
        centre_driver.position_m = [0.0, 0.0, 0.0];
        centre_driver.crossover_branch = 1;
        let mut right_driver = design.drivers[0].clone();
        right_driver.id = "right-woofer".into();
        right_driver.position_m = [0.12, -0.08, 0.0];
        right_driver.crossover_branch = 2;
        design.drivers.extend([centre_driver, right_driver]);
        design.crossover = CrossoverNetwork {
            branches: vec![
                CrossoverBranch {
                    kind: CrossoverKind::LowPass,
                    corner_frequency_hz: 900.0,
                    gain: 1.0,
                    delay_s: 0.0,
                },
                CrossoverBranch {
                    kind: CrossoverKind::HighPass,
                    corner_frequency_hz: 900.0,
                    gain: 0.8,
                    delay_s: 0.0001,
                },
                CrossoverBranch {
                    kind: CrossoverKind::FullRange,
                    corner_frequency_hz: 900.0,
                    gain: 0.5,
                    delay_s: 0.0002,
                },
            ],
        };

        let interior = build_body_fitted_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig::default(),
        )
        .expect("body-fitted three-driver frequency-grid mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, interior, 343.0, 1.204)
            .expect("body-fitted three-driver frequency-grid coupling");

        let mut solutions = Vec::new();
        for frequency_hz in [60.0, 900.0, 4_000.0] {
            let solution = coupling
                .solve_frequency(&design, frequency_hz, Complex64::new(2.0, 0.0))
                .expect("body-fitted three-driver frequency-grid solve");
            let expected_terminal = design
                .drive_voltages_at(frequency_hz, Complex64::new(2.0, 0.0))
                .expect("crossover terminal drives");

            assert_eq!(solution.driver_terminal_voltage_v, expected_terminal);
            assert_eq!(solution.driver_terminal_voltage_v.len(), 3);
            assert_eq!(solution.driver_current_a.len(), 3);
            assert_eq!(solution.driver_velocity_m_per_s.len(), 3);
            assert!(
                solution
                    .driver_terminal_voltage_v
                    .windows(2)
                    .any(|pair| (pair[0] - pair[1]).norm() > 1.0e-6),
                "crossover branches collapsed at {frequency_hz} Hz"
            );
            assert!(solution.relative_residual < 1.0e-10);
            assert!(solution.interior_relative_residual < 1.0e-10);
            assert!(solution.exterior_relative_residual < 1.0e-10);
            assert!(solution.mechanical_relative_residual < 1.0e-10);
            assert!(solution.electrical_relative_residual < 1.0e-10);
            assert!(solution.electrical_input_power_w.is_finite());
            assert!(solution.electrical_input_power_w > 0.0);
            assert!(solution.coil_dissipation_w.is_finite());
            assert!(solution.mechanical_dissipation_w.is_finite());
            assert!(solution.electrical_power_residual_w.abs() < 1.0e-8);
            assert!(solution.mechanical_power_residual_w.abs() < 1.0e-8);
            assert!(
                solution
                    .driver_electrical_power_residual_w
                    .iter()
                    .all(|residual| residual.abs() < 1.0e-8)
            );
            assert!(
                solution
                    .driver_mechanical_power_residual_w
                    .iter()
                    .all(|residual| residual.abs() < 1.0e-8)
            );
            solutions.push(solution);
        }

        let points = solutions
            .iter()
            .map(|solution| {
                ProductReferencePoint::from_monolithic_solution(
                    solution,
                    &coupling.interior_mesh,
                    0.0,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("three-driver crossover product evidence");
        let sweep = ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: coupling.interior_mesh.mesh.num_nodes(),
            aperture_elements: coupling.port_bem_elements.len(),
            points,
            driver_count: 3,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        };
        sweep
            .validate_monolithic_product_evidence(MonolithicEvidenceGate::default())
            .expect("three-driver crossover product evidence should validate");
    }

    #[test]
    fn body_fitted_port_and_brace_monolithic_refinement_has_bounded_change() {
        let mut design = sealed_product();
        design.drivers[0].position_m[0] = -0.06;
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        design
            .cabinet
            .braces
            .push(crate::cabinet_design::BraceDesign {
                panel: crate::cabinet_design::PanelId::Top,
                split_along: crate::cabinet_design::PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(crate::cabinet_design::BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            });

        let solve = |max_element_size_m| {
            let interior_mesh = build_body_fitted_sealed_product_interior_mesh(
                &design,
                ProductInteriorMeshConfig {
                    max_element_size_m,
                    ..ProductInteriorMeshConfig::default()
                },
            )
            .expect("body-fitted refinement mesh");
            let coupling =
                MonolithicSealedProductCoupling::new(&design, interior_mesh, 343.0, 1.204)
                    .expect("body-fitted refinement coupling");
            let solution = coupling
                .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
                .expect("body-fitted refinement solve");
            (coupling.interior_mesh, solution)
        };

        let (coarse_mesh, coarse) = solve(0.004);
        let (medium_mesh, medium) = solve(0.003);
        let (fine_mesh, fine) = solve(0.002);
        assert!(medium.interior_pressure.len() > coarse.interior_pressure.len());
        assert!(fine.interior_pressure.len() > medium.interior_pressure.len());
        for (name, solution) in [("coarse", &coarse), ("medium", &medium), ("fine", &fine)] {
            assert!(
                solution.relative_residual < 1.0e-10,
                "{name} overall residual"
            );
            assert!(
                solution.interior_relative_residual < 1.0e-10,
                "{name} interior residual"
            );
            assert!(
                solution.exterior_relative_residual < 1.0e-10,
                "{name} exterior residual"
            );
            assert!(
                solution.mechanical_relative_residual < 1.0e-10,
                "{name} mechanical residual"
            );
            assert!(
                solution.electrical_relative_residual < 1.0e-10,
                "{name} electrical residual"
            );
            assert!(solution.port_volume_velocity_m3_per_s.is_finite());
            assert!(
                solution
                    .port_pressure_jump_pa
                    .iter()
                    .all(|jump| jump.norm() < 1.0e-10),
                "{name} port pressure continuity"
            );
        }

        let relative_flow_change = |reference: &Complex64, candidate: &Complex64| {
            (*candidate - *reference).norm() / candidate.norm().max(1.0e-30)
        };
        let coarse_to_medium = relative_flow_change(
            &coarse.port_volume_velocity_m3_per_s,
            &medium.port_volume_velocity_m3_per_s,
        );
        let medium_to_fine = relative_flow_change(
            &medium.port_volume_velocity_m3_per_s,
            &fine.port_volume_velocity_m3_per_s,
        );
        let coarse_to_fine = relative_flow_change(
            &coarse.port_volume_velocity_m3_per_s,
            &fine.port_volume_velocity_m3_per_s,
        );
        // The adjacent medium-to-fine observable is the product promotion
        // convergence gate; the coarse-to-fine value remains diagnostic
        // evidence for detecting a non-convergent refinement ladder.
        assert!(
            medium_to_fine < 1.0e-2,
            "braced port flow did not meet the adjacent sub-percent mesh gate: {medium_to_fine:e}"
        );
        assert!(
            coarse_to_fine.is_finite()
                && coarse_to_medium.is_finite()
                && medium_to_fine.is_finite(),
            "body-fitted refinement evidence must be finite"
        );

        let medium_point =
            ProductReferencePoint::from_monolithic_solution(&medium, &medium_mesh, 0.0)
                .expect("medium body-fitted reference evidence should be valid");
        let fine_point = ProductReferencePoint::from_monolithic_solution(&fine, &fine_mesh, 0.0)
            .expect("fine body-fitted reference evidence should be valid");
        let computed_mesh_error = adjacent_mesh_convergence_error(&medium_point, &fine_point)
            .expect("adjacent body-fitted mesh evidence should compare");
        assert!(
            computed_mesh_error < 1.0e-2,
            "all retained body-fitted observables must meet the mesh gate: {computed_mesh_error:e}"
        );
        let fine_point =
            ProductReferencePoint::from_monolithic_solution(&fine, &fine_mesh, computed_mesh_error)
                .expect("promoted fine body-fitted reference evidence should be valid");
        assert_eq!(
            fine_point
                .interface_evidence
                .as_ref()
                .expect("port interface evidence")
                .first()
                .and_then(|evidence| evidence.mesh_convergence_error),
            Some(computed_mesh_error)
        );
        ProductReferenceSweep {
            speed_of_sound_m_per_s: 343.0,
            density_kg_per_m3: 1.204,
            interior_dofs: fine_mesh.mesh.num_nodes(),
            aperture_elements: fine_mesh
                .patch(ProductInteriorPatchKind::PortMouth)
                .map_or(1, |patch| patch.boundary_faces.len()),
            points: vec![fine_point],
            driver_count: 1,
            provenance: ArtifactProvenance::native_golden(),
            rom_promotion: None,
            mesh_convergence: Vec::new(),
        }
        .validate_monolithic_product_evidence(MonolithicEvidenceGate::default())
        .expect("sub-percent body-fitted reference should promote");
        assert!(fine_mesh.diagnostics.node_count > medium_mesh.diagnostics.node_count);
        assert!(medium_mesh.diagnostics.node_count > coarse_mesh.diagnostics.node_count);
    }

    #[test]
    fn ported_product_monolithic_closes_side_wall_driver_and_port_blocks() {
        let mut design = sealed_product();
        design.drivers[0].normal = [1.0, 0.0, 0.0];
        design.drivers[0].position_m = [
            design.cabinet.box_width_m * 0.5,
            0.0,
            -design.cabinet.box_depth_m * 0.5,
        ];
        let mut port = CabinetDesign::default().port.expect("default port");
        port.position_m = [0.06, 0.0];
        design.cabinet.port = Some(port);
        let interior =
            build_ported_product_interior_mesh(&design, ProductInteriorMeshConfig::default())
                .expect("ported side-wall product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, interior, 343.0, 1.204)
            .expect("ported side-wall monolithic setup");
        let solution = coupling
            .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
            .expect("ported side-wall monolithic solve");

        assert!(solution.relative_residual < 1.0e-10);
        assert!(solution.interior_relative_residual < 1.0e-10);
        assert!(solution.exterior_relative_residual < 1.0e-10);
        assert!(solution.mechanical_relative_residual < 1.0e-10);
        assert!(solution.electrical_relative_residual < 1.0e-10);
        assert!(!solution.port_velocity_m_per_s.is_empty());
        assert!(
            solution
                .port_pressure_jump_pa
                .iter()
                .all(|jump| jump.norm() < 1.0e-10)
        );
    }

    #[test]
    fn sealed_three_driver_product_retains_independent_electromechanical_unknowns() {
        let mut design = sealed_product();
        design.cabinet.box_height_m = 0.18;
        design.cabinet.box_depth_m = 0.12;
        design.drivers[0].id = "left".into();
        design.drivers[0].radius_m = 0.025;
        design.drivers[0].position_m = [-0.07, 0.0, 0.0];
        design.drivers[0].crossover_branch = 0;
        for (index, x) in [0.0, 0.07].into_iter().enumerate() {
            let mut driver = design.drivers[0].clone();
            driver.id = match index {
                0 => "centre".into(),
                _ => "right".into(),
            };
            driver.position_m[0] = x;
            driver.crossover_branch = index + 1;
            design.drivers.push(driver);
        }
        design.crossover.branches = vec![
            CrossoverBranch::default(),
            CrossoverBranch::default(),
            CrossoverBranch::default(),
        ];
        let mesh = build_sealed_product_interior_mesh(
            &design,
            ProductInteriorMeshConfig {
                max_element_size_m: 0.04,
                ..ProductInteriorMeshConfig::default()
            },
        )
        .expect("three-driver product mesh");
        let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
            .expect("three-driver product coupling");
        let solution = coupling
            .solve_frequency(&design, 150.0, Complex64::new(2.0, 0.0))
            .expect("three-driver product solve");
        assert!(solution.relative_residual < 1.0e-10);
        assert_eq!(solution.driver_current_a.len(), 3);
        assert_eq!(solution.driver_velocity_m_per_s.len(), 3);
        assert!(
            solution
                .driver_current_a
                .iter()
                .all(|value| value.norm() > 0.0)
        );
        assert!(
            solution
                .driver_velocity_m_per_s
                .iter()
                .all(|value| value.norm() > 0.0)
        );
        assert!(solution.electrical_input_power_w > 0.0);
        assert!(solution.electrical_power_residual_w.abs() < 1.0e-10);
        assert!(solution.mechanical_power_residual_w.abs() < 1.0e-10);
    }
}
