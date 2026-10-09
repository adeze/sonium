//! Named generalized-eigen wrappers for common FEM physics workflows.
//!
//! These helpers expose standard domain problems as `A x = lambda B x` systems
//! backed by the existing assembly and LOBPCG infrastructure.

use std::collections::{HashMap, HashSet};

use num_complex::Complex64;
use sonium_solvers::{
    ComplexArnoldiConfig, ComplexTargetedEigenResult, CsrMatrix, LobpcgConfig, LobpcgResult,
    ShiftInvertConfig, SolverError, TargetedEigenResult, TripletMatrix,
    complex_shift_invert_arnoldi_generalized, dense::DenseMatrix, eigenpairs_in_interval,
    lobpcg_generalized, shift_invert_lanczos_generalized,
};

use crate::assembly::{assemble_bilinear, assemble_boundary_mass_generic};
use crate::basis::PolynomialDegree;
use crate::boundary::periodic::{MpcError, MpcSystem, PeriodicDofConstraint};
use crate::forms::{
    MassForm, VariableLaplacianForm, VariableMassForm, VectorMassForm, WeightedCurlCurlForm,
    WeightedEdgeMassForm, WeightedLaplacianForm, WeightedMassForm,
};
use crate::mesh::{Mesh, interpolate_scalar_at_point};
use crate::physics::electromagnetics::maxwell::EmMaterial;
use crate::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use crate::physics::thermal::heat_conduction::{ConductionForm, HeatCapacityForm, ThermalMaterial};
use crate::quadrature;
use crate::spaces::{FunctionSpace, H1Space, HcurlSpace, VectorH1Space};

/// Physical interpretation of a generalized eigenvalue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenPhysicsKind {
    /// General scalar elliptic/Sturm--Liouville eigenproblem.
    ScalarGeneralized,
    StructuralModal,
    LinearBuckling,
    ThermalStability,
    AcousticEigenmode,
    ElectromagneticEigenmode,
    /// Reduced radial Coulomb Hamiltonian for a fixed angular momentum.
    RadialSchrodinger,
}

/// One homogeneous generalized Robin contribution on a marked boundary.
///
/// A term on the stiffness side contributes coefficient times the boundary
/// mass matrix to A; a term on the mass side contributes it to B.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarRobinTerm {
    /// Boundary marker to integrate.
    pub marker: i32,
    /// Constant multiplying the boundary mass matrix.
    pub coefficient: f64,
}

/// One strong scalar Dirichlet condition for an eigenproblem.
///
/// Eigenproblems are homogeneous by definition: a nonzero prescribed value
/// makes the operator affine rather than linear and is rejected by the
/// checked builders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarDirichletTerm {
    pub marker: i32,
    pub value: f64,
}

/// Checked scalar eigenproblem construction failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EigenproblemBuildError {
    #[error("eigenproblem Dirichlet data must be homogeneous; marker {marker} has value {value}")]
    InhomogeneousDirichlet { marker: i32, value: f64 },
    #[error(transparent)]
    Periodic(#[from] MpcError),
}

/// Assembly options for a scalar generalized eigenproblem.
#[derive(Debug, Clone)]
pub struct ScalarEigenproblemConfig {
    /// Polynomial degree of the scalar H1 space.
    pub degree: PolynomialDegree,
    /// Strong homogeneous Dirichlet boundary markers.
    pub dirichlet_boundary_markers: Vec<i32>,
    /// Checked Dirichlet terms. Zero-valued terms are equivalent to entries
    /// in `dirichlet_boundary_markers`; nonzero or non-finite values are
    /// rejected.
    pub dirichlet_boundary_values: Vec<ScalarDirichletTerm>,
    /// Robin terms added to the stiffness-like operator.
    pub robin_stiffness_terms: Vec<ScalarRobinTerm>,
    /// Eigenvalue-dependent Robin terms added to the mass-like operator.
    pub robin_mass_terms: Vec<ScalarRobinTerm>,
    /// Real periodic or anti-periodic scalar DOF constraints.
    ///
    /// Complex Bloch phases are rejected by the checked builder because the
    /// current generalized LOBPCG path is real-valued.
    pub periodic_constraints: Vec<PeriodicDofConstraint>,
    /// Multiplier from sqrt(lambda) to angular frequency.
    pub angular_frequency_scale: f64,
}

impl Default for ScalarEigenproblemConfig {
    fn default() -> Self {
        Self {
            degree: PolynomialDegree::P1,
            dirichlet_boundary_markers: Vec::new(),
            dirichlet_boundary_values: Vec::new(),
            robin_stiffness_terms: Vec::new(),
            robin_mass_terms: Vec::new(),
            periodic_constraints: Vec::new(),
            angular_frequency_scale: 1.0,
        }
    }
}

/// Build [-div(diffusion grad u) + potential u] = lambda * weight * u.
///
/// Coefficients are evaluated in physical coordinates at quadrature points.
/// Homogeneous Dirichlet and generalized Robin boundary terms are configured
/// through ScalarEigenproblemConfig. Robin assembly currently supports P1
/// boundary traces, matching assemble_boundary_mass_generic.
pub fn build_scalar_eigenproblem<D, V, W>(
    mesh: &Mesh,
    diffusion: D,
    potential: V,
    weight: W,
    config: &ScalarEigenproblemConfig,
) -> EigenPhysicsProblem
where
    D: Fn(f64, f64, f64) -> f64 + Send + Sync,
    V: Fn(f64, f64, f64) -> f64 + Send + Sync,
    W: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    try_build_scalar_eigenproblem(mesh, diffusion, potential, weight, config)
        .expect("invalid scalar generalized eigenproblem constraints")
}

/// Checked variant of build_scalar_eigenproblem.
pub fn try_build_scalar_eigenproblem<D, V, W>(
    mesh: &Mesh,
    diffusion: D,
    potential: V,
    weight: W,
    config: &ScalarEigenproblemConfig,
) -> Result<EigenPhysicsProblem, EigenproblemBuildError>
where
    D: Fn(f64, f64, f64) -> f64 + Send + Sync,
    V: Fn(f64, f64, f64) -> f64 + Send + Sync,
    W: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let mut dirichlet_markers = config.dirichlet_boundary_markers.clone();
    for term in &config.dirichlet_boundary_values {
        if !term.value.is_finite() || term.value != 0.0 {
            return Err(EigenproblemBuildError::InhomogeneousDirichlet {
                marker: term.marker,
                value: term.value,
            });
        }
        dirichlet_markers.push(term.marker);
    }
    assert!(
        (config.robin_stiffness_terms.is_empty() && config.robin_mass_terms.is_empty())
            || config.degree == PolynomialDegree::P1,
        "scalar Robin eigenproblem assembly currently supports P1 traces"
    );
    let space = H1Space::new(config.degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(config.degree.degree()) + 1;
    let quad_mass = quadrature::required_order_for_mass(config.degree.degree()) + 1;
    let mut stiffness = assemble_bilinear(
        &VariableLaplacianForm::new(diffusion),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let potential = assemble_bilinear(
        &VariableMassForm::new(potential),
        &space,
        &space,
        mesh,
        quad_mass,
    );
    append_triplets(&mut stiffness, &potential);
    let mut mass = assemble_bilinear(
        &VariableMassForm::new(weight),
        &space,
        &space,
        mesh,
        quad_mass,
    );
    for term in &config.robin_stiffness_terms {
        let boundary = assemble_boundary_mass_generic(mesh, term.marker);
        append_scaled_triplets(&mut stiffness, &boundary, term.coefficient);
    }
    for term in &config.robin_mass_terms {
        let boundary = assemble_boundary_mass_generic(mesh, term.marker);
        append_scaled_triplets(&mut mass, &boundary, term.coefficient);
    }
    let constrained = boundary_dofs(mesh, &space, &dirichlet_markers);
    let stiffness = stiffness.to_csr();
    let mass = mass.to_csr();
    let (stiffness_like, mass_like) = if config.periodic_constraints.is_empty() {
        (
            eliminate_csr_dofs(&stiffness, &constrained),
            eliminate_csr_dofs(&mass, &constrained),
        )
    } else {
        let mpc = MpcSystem::from_periodic(stiffness.num_rows, &config.periodic_constraints)?;
        let stiffness = mpc.condense_real_matrix(&stiffness)?;
        let mass = mpc.condense_real_matrix(&mass)?;
        let reduced_constrained = map_constrained_dofs(&mpc, &constrained);
        (
            eliminate_csr_dofs(&stiffness, &reduced_constrained),
            eliminate_csr_dofs(&mass, &reduced_constrained),
        )
    };
    Ok(EigenPhysicsProblem {
        kind: EigenPhysicsKind::ScalarGeneralized,
        stiffness_like,
        mass_like,
        angular_frequency_scale: config.angular_frequency_scale,
    })
}

/// Configuration for a one-dimensional P1 interval Laplacian eigenproblem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalLaplacianConfig {
    /// Left interval coordinate.
    pub x_min: f64,
    /// Right interval coordinate.
    pub x_max: f64,
    /// Number of equal P1 elements.
    pub elements: usize,
    /// Eliminate the left endpoint value.
    pub dirichlet_left: bool,
    /// Eliminate the right endpoint value.
    pub dirichlet_right: bool,
}

impl Default for IntervalLaplacianConfig {
    fn default() -> Self {
        Self {
            x_min: 0.0,
            x_max: 1.0,
            elements: 80,
            dirichlet_left: true,
            dirichlet_right: true,
        }
    }
}

/// Build `-u'' = lambda u` on an interval with consistent P1 mass.
///
/// This specialized builder gives one-dimensional Sturm--Liouville examples a
/// first-class path without pretending the 2D/3D unstructured mesh topology
/// supports zero-dimensional boundary faces.
pub fn build_interval_laplacian_problem(config: &IntervalLaplacianConfig) -> EigenPhysicsProblem {
    assert!(
        config.x_min < config.x_max,
        "interval bounds must be ordered"
    );
    assert!(
        config.elements > 1,
        "interval eigenproblem needs at least two elements"
    );
    let nodes = config.elements + 1;
    let h = (config.x_max - config.x_min) / config.elements as f64;
    let mut stiffness = TripletMatrix::new(nodes, nodes);
    let mut mass = TripletMatrix::new(nodes, nodes);
    for element in 0..config.elements {
        let dofs = [element, element + 1];
        for local_row in 0..2 {
            for local_column in 0..2 {
                let stiffness_value = if local_row == local_column {
                    1.0 / h
                } else {
                    -1.0 / h
                };
                let mass_value = if local_row == local_column {
                    h / 3.0
                } else {
                    h / 6.0
                };
                stiffness.push(dofs[local_row], dofs[local_column], stiffness_value);
                mass.push(dofs[local_row], dofs[local_column], mass_value);
            }
        }
    }
    let mut constrained = Vec::with_capacity(2);
    if config.dirichlet_left {
        constrained.push(0);
    }
    if config.dirichlet_right {
        constrained.push(nodes - 1);
    }
    EigenPhysicsProblem {
        kind: EigenPhysicsKind::ScalarGeneralized,
        stiffness_like: eliminate_dofs(&stiffness, &constrained).to_csr(),
        mass_like: eliminate_dofs(&mass, &constrained).to_csr(),
        angular_frequency_scale: 1.0,
    }
}

/// A named generalized eigenproblem `stiffness_like x = lambda mass_like x`.
#[derive(Debug, Clone)]
pub struct EigenPhysicsProblem {
    pub kind: EigenPhysicsKind,
    pub stiffness_like: CsrMatrix<f64>,
    pub mass_like: CsrMatrix<f64>,
    /// Multiplier from `sqrt(lambda)` to angular frequency. Use 0 for
    /// non-oscillatory eigenvalues such as buckling loads and thermal decay.
    pub angular_frequency_scale: f64,
}

impl EigenPhysicsProblem {
    pub fn solve(&self, config: &LobpcgConfig) -> Result<LobpcgResult, SolverError> {
        lobpcg_generalized(&self.stiffness_like, &self.mass_like, config)
    }

    pub fn eigenvalues_to_angular_frequencies(&self, eigenvalues: &[f64]) -> Vec<f64> {
        eigenvalues
            .iter()
            .map(|&lambda| self.angular_frequency_scale * lambda.max(0.0).sqrt())
            .collect()
    }

    pub fn eigenvalues_to_frequencies_hz(&self, eigenvalues: &[f64]) -> Vec<f64> {
        self.eigenvalues_to_angular_frequencies(eigenvalues)
            .into_iter()
            .map(|omega| omega / (2.0 * std::f64::consts::PI))
            .collect()
    }

    /// Solve generalized eigenpairs nearest the configured shift.
    pub fn solve_near(
        &self,
        config: &ShiftInvertConfig,
    ) -> Result<TargetedEigenResult, SolverError> {
        shift_invert_lanczos_generalized(&self.stiffness_like, &self.mass_like, config)
    }

    /// Solve generalized eigenpairs inside an inclusive interval.
    pub fn solve_interval(
        &self,
        lower: f64,
        upper: f64,
        config: &ShiftInvertConfig,
    ) -> Result<TargetedEigenResult, SolverError> {
        eigenpairs_in_interval(&self.stiffness_like, &self.mass_like, lower, upper, config)
    }
}

/// Rayleigh/proportional damping coefficients for a quadratic wave pencil.
///
/// For `M u_ddot + C u_dot + K u = 0`, this represents
/// `C = mass_damping M + stiffness_damping K`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProportionalDamping {
    /// Coefficient multiplying the mass matrix.
    pub mass_damping: f64,
    /// Coefficient multiplying the stiffness matrix.
    pub stiffness_damping: f64,
}

impl Default for ProportionalDamping {
    fn default() -> Self {
        Self {
            mass_damping: 0.0,
            stiffness_damping: 0.0,
        }
    }
}

/// Complex temporal exponents of a proportionally damped wave problem.
#[derive(Debug, Clone)]
pub struct DampedEigenResult {
    /// Undamped spatial eigenvalues `K phi = lambda M phi`.
    pub spatial_eigenvalues: Vec<f64>,
    /// Two temporal roots per spatial mode, ordered as conjugate pairs.
    pub temporal_eigenvalues: Vec<Complex64>,
    /// Spatial mass-normalized eigenvectors.
    pub eigenvectors: DenseMatrix<f64>,
    /// Whether the underlying spatial solve converged.
    pub converged: bool,
}

/// General complex first-order pencil, including linearized quadratic waves.
#[derive(Debug, Clone)]
pub struct NonsymmetricComplexEigenProblem {
    /// Operator in `A x = lambda B x`.
    pub operator: CsrMatrix<Complex64>,
    /// Generalized mass matrix in `A x = lambda B x`.
    pub mass_like: CsrMatrix<Complex64>,
}

impl NonsymmetricComplexEigenProblem {
    /// Solve eigenpairs nearest a complex target with shift-invert Arnoldi.
    pub fn solve_near(
        &self,
        config: &ComplexArnoldiConfig,
    ) -> Result<ComplexTargetedEigenResult, SolverError> {
        complex_shift_invert_arnoldi_generalized(&self.operator, &self.mass_like, config)
    }
}

/// Linearize `M s^2 + C s + K` as a first-order generalized pencil.
///
/// The state is `[u, s u]` and the companion pencil is
/// `[[0,I],[-K,-C]] z = s [[I,0],[0,M]] z`. Unlike the proportional shortcut,
/// this accepts arbitrary real sparse damping matrices.
pub fn linearize_quadratic_eigenproblem(
    stiffness: &CsrMatrix<f64>,
    damping: &CsrMatrix<f64>,
    mass: &CsrMatrix<f64>,
) -> Result<NonsymmetricComplexEigenProblem, SolverError> {
    let n = stiffness.num_rows;
    if n == 0
        || stiffness.num_cols != n
        || damping.num_rows != n
        || damping.num_cols != n
        || mass.num_rows != n
        || mass.num_cols != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: damping
                .num_rows
                .max(damping.num_cols)
                .max(mass.num_rows)
                .max(mass.num_cols),
        });
    }
    let mut operator = Vec::with_capacity(n + stiffness.nnz() + damping.nnz());
    let mut generalized_mass = Vec::with_capacity(n + mass.nnz());
    for index in 0..n {
        operator.push((index, n + index, Complex64::new(1.0, 0.0)));
        generalized_mass.push((index, index, Complex64::new(1.0, 0.0)));
    }
    for row in 0..n {
        operator.extend(
            stiffness
                .row_entries(row)
                .map(|(column, value)| (n + row, column, Complex64::new(-value, 0.0))),
        );
        operator.extend(
            damping
                .row_entries(row)
                .map(|(column, value)| (n + row, n + column, Complex64::new(-value, 0.0))),
        );
        generalized_mass.extend(
            mass.row_entries(row)
                .map(|(column, value)| (n + row, n + column, Complex64::new(value, 0.0))),
        );
    }
    Ok(NonsymmetricComplexEigenProblem {
        operator: CsrMatrix::from_triplets(2 * n, 2 * n, operator),
        mass_like: CsrMatrix::from_triplets(2 * n, 2 * n, generalized_mass),
    })
}

/// Build a two-field nonsymmetric reaction-diffusion eigenproblem.
///
/// Given a scalar spatial pencil `K phi = mu M phi`, this assembles
/// `[[K, a M], [b M, d K]] z = lambda diag(M, M) z`. Opposite-sign
/// couplings can produce genuinely complex conjugate spectra.
pub fn build_coupled_reaction_diffusion_eigenproblem(
    spatial: &EigenPhysicsProblem,
    second_diffusion: f64,
    coupling_12: f64,
    coupling_21: f64,
) -> Result<NonsymmetricComplexEigenProblem, SolverError> {
    let n = spatial.stiffness_like.num_rows;
    if n == 0
        || spatial.stiffness_like.num_cols != n
        || spatial.mass_like.num_rows != n
        || spatial.mass_like.num_cols != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: spatial.mass_like.num_rows.max(spatial.mass_like.num_cols),
        });
    }
    if !second_diffusion.is_finite()
        || !coupling_12.is_finite()
        || !coupling_21.is_finite()
        || second_diffusion <= 0.0
    {
        return Err(SolverError::OperationFailed {
            operation: "coupled reaction-diffusion eigenproblem assembly",
            reason: "diffusion must be positive and all coefficients finite".into(),
        });
    }
    let mut operator =
        Vec::with_capacity(2 * spatial.stiffness_like.nnz() + 2 * spatial.mass_like.nnz());
    let mut mass = Vec::with_capacity(2 * spatial.mass_like.nnz());
    for row in 0..n {
        for (column, value) in spatial.stiffness_like.row_entries(row) {
            operator.push((row, column, Complex64::new(value, 0.0)));
            operator.push((
                n + row,
                n + column,
                Complex64::new(second_diffusion * value, 0.0),
            ));
        }
        for (column, value) in spatial.mass_like.row_entries(row) {
            operator.push((row, n + column, Complex64::new(coupling_12 * value, 0.0)));
            operator.push((n + row, column, Complex64::new(coupling_21 * value, 0.0)));
            mass.push((row, column, Complex64::new(value, 0.0)));
            mass.push((n + row, n + column, Complex64::new(value, 0.0)));
        }
    }
    Ok(NonsymmetricComplexEigenProblem {
        operator: CsrMatrix::from_triplets(2 * n, 2 * n, operator),
        mass_like: CsrMatrix::from_triplets(2 * n, 2 * n, mass),
    })
}

impl EigenPhysicsProblem {
    /// Solve a quadratic wave eigenproblem with proportional damping.
    ///
    /// Each spatial mode diagonalizes `M`, `C`, and `K`, reducing the
    /// quadratic pencil to `s^2 + (alpha + beta lambda) s + lambda = 0`.
    pub fn solve_proportionally_damped(
        &self,
        damping: ProportionalDamping,
        config: &LobpcgConfig,
    ) -> Result<DampedEigenResult, SolverError> {
        if !damping.mass_damping.is_finite()
            || !damping.stiffness_damping.is_finite()
            || damping.mass_damping < 0.0
            || damping.stiffness_damping < 0.0
        {
            return Err(SolverError::OperationFailed {
                operation: "proportional damped eigensolve",
                reason: "damping coefficients must be finite and non-negative".into(),
            });
        }
        let spatial = self.solve(config)?;
        let mut temporal_eigenvalues = Vec::with_capacity(2 * spatial.eigenvalues.len());
        for &lambda in &spatial.eigenvalues {
            let damping_rate = damping.mass_damping + damping.stiffness_damping * lambda;
            let discriminant =
                Complex64::new(damping_rate * damping_rate - 4.0 * lambda, 0.0).sqrt();
            temporal_eigenvalues.push((-damping_rate + discriminant) / 2.0);
            temporal_eigenvalues.push((-damping_rate - discriminant) / 2.0);
        }
        Ok(DampedEigenResult {
            spatial_eigenvalues: spatial.eigenvalues,
            temporal_eigenvalues,
            eigenvectors: spatial.eigenvectors,
            converged: spatial.converged,
        })
    }
}

/// Hermitian complex generalized eigenproblem.
///
/// The current solve path realifies the Hermitian pencil and calls the
/// production real LOBPCG implementation. Each complex eigenpair appears
/// twice in the real representation; solve_eigenvalues folds those duplicate
/// copies while preserving physical multiplicity.
#[derive(Debug, Clone)]
pub struct ComplexEigenPhysicsProblem {
    /// Physical interpretation of the eigenvalues.
    pub kind: EigenPhysicsKind,
    /// Hermitian stiffness-like operator.
    pub stiffness_like: CsrMatrix<Complex64>,
    /// Hermitian positive-definite mass-like operator.
    pub mass_like: CsrMatrix<Complex64>,
    /// Multiplier from sqrt(lambda) to angular frequency.
    pub angular_frequency_scale: f64,
}

/// Eigenpair result for a complex Hermitian pencil.
#[derive(Debug, Clone)]
pub struct ComplexEigenResult {
    /// Smallest requested eigenvalues, with realification duplicates folded.
    pub eigenvalues: Vec<f64>,
    /// Maximum residual of each folded realification pair.
    pub residual_norms: Vec<f64>,
    /// Complex eigenvectors as columns, mass-normalized by LOBPCG.
    pub eigenvectors: DenseMatrix<Complex64>,
    /// Number of real LOBPCG iterations.
    pub iterations: usize,
    /// Whether every realified pair converged.
    pub converged: bool,
}

/// Global mode correspondence between independently meshed geometries.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossMeshModeAssignment {
    /// For each previous mode, the matching column in the current modes.
    pub current_indices: Vec<usize>,
    /// Sign that aligns the selected current mode with the previous mode.
    pub phase_signs: Vec<f64>,
    /// Absolute normalized overlap for each selected match.
    pub overlaps: Vec<f64>,
    /// Number of common-domain sample points used.
    pub common_samples: usize,
}

/// Project a real field onto mass-normalized generalized eigenmodes.
///
/// For mode matrix `Phi` and mass matrix `M`, this returns `Phi^T M field`.
pub fn project_onto_real_modes(
    mass: &CsrMatrix<f64>,
    modes: &DenseMatrix<f64>,
    field: &[f64],
) -> Result<Vec<f64>, SolverError> {
    let rows = modes.nrows();
    if mass.num_rows != rows || mass.num_cols != rows || field.len() != rows {
        return Err(SolverError::DimensionMismatch {
            expected: rows,
            got: mass.num_rows.max(mass.num_cols).max(field.len()),
        });
    }
    let mut mass_field = vec![0.0; rows];
    for (row, value) in mass_field.iter_mut().enumerate() {
        *value = mass
            .row_entries(row)
            .map(|(column, entry)| entry * field[column])
            .sum();
    }
    Ok((0..modes.ncols())
        .map(|mode| {
            (0..rows)
                .map(|row| modes[[row, mode]] * mass_field[row])
                .sum()
        })
        .collect())
}

/// Reconstruct a real field from modal columns and coefficients.
pub fn reconstruct_from_real_modes(
    modes: &DenseMatrix<f64>,
    coefficients: &[f64],
) -> Result<Vec<f64>, SolverError> {
    if coefficients.len() > modes.ncols() {
        return Err(SolverError::DimensionMismatch {
            expected: modes.ncols(),
            got: coefficients.len(),
        });
    }
    Ok((0..modes.nrows())
        .map(|row| {
            coefficients
                .iter()
                .enumerate()
                .map(|(mode, coefficient)| coefficient * modes[[row, mode]])
                .sum()
        })
        .collect())
}

/// Advance modal coefficients for `u_tt + A u = 0`.
///
/// Initial velocity is expressed in the same mass-normalized modal basis. A
/// zero eigenvalue uses the exact rigid/constant-mode limit `a + t b`.
pub fn evolve_wave_mode_coefficients(
    eigenvalues: &[f64],
    initial_displacement: &[f64],
    initial_velocity: &[f64],
    time: f64,
) -> Result<Vec<f64>, SolverError> {
    if eigenvalues.len() != initial_displacement.len()
        || eigenvalues.len() != initial_velocity.len()
        || !time.is_finite()
        || eigenvalues
            .iter()
            .any(|value| !value.is_finite() || *value < -1.0e-12)
    {
        return Err(SolverError::OperationFailed {
            operation: "wave modal evolution",
            reason: "modal arrays, time, or nonnegative eigenvalues are invalid".into(),
        });
    }
    Ok(eigenvalues
        .iter()
        .zip(initial_displacement)
        .zip(initial_velocity)
        .map(|((&eigenvalue, &displacement), &velocity)| {
            if eigenvalue <= 1.0e-12 {
                displacement + time * velocity
            } else {
                let omega = eigenvalue.sqrt();
                displacement * (omega * time).cos() + velocity * (omega * time).sin() / omega
            }
        })
        .collect())
}

/// Advance modal coefficients for `u_t + diffusivity A u = 0`.
pub fn evolve_heat_mode_coefficients(
    eigenvalues: &[f64],
    initial: &[f64],
    diffusivity: f64,
    time: f64,
) -> Result<Vec<f64>, SolverError> {
    if eigenvalues.len() != initial.len()
        || !diffusivity.is_finite()
        || diffusivity < 0.0
        || !time.is_finite()
        || time < 0.0
        || eigenvalues
            .iter()
            .any(|value| !value.is_finite() || *value < -1.0e-12)
    {
        return Err(SolverError::OperationFailed {
            operation: "heat modal evolution",
            reason: "modal arrays, diffusivity, time, or nonnegative eigenvalues are invalid"
                .into(),
        });
    }
    Ok(eigenvalues
        .iter()
        .zip(initial)
        .map(|(&eigenvalue, &coefficient)| {
            coefficient * (-diffusivity * eigenvalue.max(0.0) * time).exp()
        })
        .collect())
}

/// Return unconstrained P1 nodal DOFs in scalar-builder reduction order.
pub fn homogeneous_dirichlet_free_dofs(mesh: &Mesh, boundary_markers: &[i32]) -> Vec<usize> {
    let constrained: HashSet<_> =
        boundary_dofs(mesh, &H1Space::new(PolynomialDegree::P1), boundary_markers)
            .into_iter()
            .collect();
    (0..mesh.num_nodes())
        .filter(|dof| !constrained.contains(dof))
        .collect()
}

/// Expand reduced P1 modes after homogeneous Dirichlet elimination.
///
/// Boundary values are restored as zero and unconstrained rows retain the
/// ordering used by [`build_scalar_eigenproblem`]. The resulting full nodal
/// fields can be interpolated or tracked across unrelated meshes.
pub fn expand_homogeneous_dirichlet_modes(
    mesh: &Mesh,
    boundary_markers: &[i32],
    reduced_modes: &DenseMatrix<f64>,
) -> Result<DenseMatrix<f64>, SolverError> {
    let free_dofs = homogeneous_dirichlet_free_dofs(mesh, boundary_markers);
    if reduced_modes.nrows() != free_dofs.len() {
        return Err(SolverError::DimensionMismatch {
            expected: free_dofs.len(),
            got: reduced_modes.nrows(),
        });
    }
    let mut expanded = DenseMatrix::zeros(mesh.num_nodes(), reduced_modes.ncols());
    for (reduced_row, &full_row) in free_dofs.iter().enumerate() {
        for column in 0..reduced_modes.ncols() {
            expanded[[full_row, column]] = reduced_modes[[reduced_row, column]];
        }
    }
    Ok(expanded)
}

/// Match real P1 nodal modes across unrelated meshes over their common domain.
///
/// Both mode matrices store nodal fields as columns. A Cartesian integration
/// grid covers the intersection of the mesh bounding boxes; samples outside
/// either actual mesh are discarded. Normalized cross-overlaps are assigned
/// globally with the Hungarian algorithm, preventing duplicate matches near
/// crossings. The returned signs remove the arbitrary eigenvector sign.
pub fn track_real_modes_across_meshes(
    previous_mesh: &Mesh,
    previous_modes: &DenseMatrix<f64>,
    current_mesh: &Mesh,
    current_modes: &DenseMatrix<f64>,
    samples_per_axis: usize,
) -> Result<CrossMeshModeAssignment, SolverError> {
    let modes = previous_modes.ncols();
    if modes == 0
        || current_modes.ncols() != modes
        || previous_modes.nrows() != previous_mesh.num_nodes()
        || current_modes.nrows() != current_mesh.num_nodes()
        || samples_per_axis < 2
    {
        return Err(SolverError::OperationFailed {
            operation: "cross-mesh eigenmode tracking",
            reason: "mode counts, nodal layouts, or sampling resolution are invalid".into(),
        });
    }
    let bounds = |mesh: &Mesh| {
        mesh.nodes.iter().fold(
            ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
            |(mut minimum, mut maximum), point| {
                minimum[0] = minimum[0].min(point.x);
                minimum[1] = minimum[1].min(point.y);
                maximum[0] = maximum[0].max(point.x);
                maximum[1] = maximum[1].max(point.y);
                (minimum, maximum)
            },
        )
    };
    let (previous_minimum, previous_maximum) = bounds(previous_mesh);
    let (current_minimum, current_maximum) = bounds(current_mesh);
    let minimum = [
        previous_minimum[0].max(current_minimum[0]),
        previous_minimum[1].max(current_minimum[1]),
    ];
    let maximum = [
        previous_maximum[0].min(current_maximum[0]),
        previous_maximum[1].min(current_maximum[1]),
    ];
    if minimum[0] >= maximum[0] || minimum[1] >= maximum[1] {
        return Err(SolverError::OperationFailed {
            operation: "cross-mesh eigenmode tracking",
            reason: "meshes have no overlapping bounding box".into(),
        });
    }
    let previous_columns: Vec<_> = (0..modes)
        .map(|mode| previous_modes.column_to_vec(mode))
        .collect();
    let current_columns: Vec<_> = (0..modes)
        .map(|mode| current_modes.column_to_vec(mode))
        .collect();
    let mut cross = vec![vec![0.0; modes]; modes];
    let mut previous_norms = vec![0.0; modes];
    let mut current_norms = vec![0.0; modes];
    let mut common_samples = 0;
    for y_index in 0..samples_per_axis {
        for x_index in 0..samples_per_axis {
            let point = [
                minimum[0]
                    + (x_index as f64 + 0.5) / samples_per_axis as f64 * (maximum[0] - minimum[0]),
                minimum[1]
                    + (y_index as f64 + 0.5) / samples_per_axis as f64 * (maximum[1] - minimum[1]),
                0.0,
            ];
            let previous_values: Option<Vec<_>> = previous_columns
                .iter()
                .map(|values| interpolate_scalar_at_point(previous_mesh, values, point))
                .collect();
            let current_values: Option<Vec<_>> = current_columns
                .iter()
                .map(|values| interpolate_scalar_at_point(current_mesh, values, point))
                .collect();
            let (Some(previous_values), Some(current_values)) = (previous_values, current_values)
            else {
                continue;
            };
            common_samples += 1;
            for previous in 0..modes {
                previous_norms[previous] += previous_values[previous].powi(2);
                for current in 0..modes {
                    cross[previous][current] += previous_values[previous] * current_values[current];
                }
            }
            for current in 0..modes {
                current_norms[current] += current_values[current].powi(2);
            }
        }
    }
    if common_samples == 0 {
        return Err(SolverError::OperationFailed {
            operation: "cross-mesh eigenmode tracking",
            reason: "meshes have no sampled common domain".into(),
        });
    }
    let normalized: Vec<Vec<f64>> = (0..modes)
        .map(|previous| {
            (0..modes)
                .map(|current| {
                    let denominator = (previous_norms[previous] * current_norms[current]).sqrt();
                    if denominator > f64::MIN_POSITIVE {
                        cross[previous][current] / denominator
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let weights: Vec<Vec<f64>> = normalized
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).collect())
        .collect();
    let current_indices = maximum_weight_assignment(&weights);
    let phase_signs = current_indices
        .iter()
        .enumerate()
        .map(|(previous, &current)| normalized[previous][current].signum())
        .collect();
    let overlaps = current_indices
        .iter()
        .enumerate()
        .map(|(previous, &current)| normalized[previous][current].abs())
        .collect();
    Ok(CrossMeshModeAssignment {
        current_indices,
        phase_signs,
        overlaps,
        common_samples,
    })
}

/// Uniform-grid P1 discretization of a reduced radial Coulomb Hamiltonian.
///
/// The solved field is `u(r) = r R(r)`, which removes the first derivative
/// from the three-dimensional radial Schrodinger equation:
///
/// `[-1/2 u'' + l(l+1)/(2 r^2) u - Z/r u] = E u`.
///
/// Homogeneous conditions at the origin and at `outer_radius` are built in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialSchrodingerConfig {
    /// Nuclear charge `Z` in atomic units.
    pub nuclear_charge: f64,
    /// Orbital angular momentum quantum number `l`.
    pub angular_momentum: usize,
    /// Truncation radius in Bohr radii.
    pub outer_radius: f64,
    /// Number of uniform P1 elements.
    pub elements: usize,
}

impl Default for RadialSchrodingerConfig {
    fn default() -> Self {
        Self {
            nuclear_charge: 1.0,
            angular_momentum: 0,
            outer_radius: 60.0,
            elements: 600,
        }
    }
}

/// Build the reduced radial hydrogen-like Schrodinger eigenproblem.
///
/// Atomic units are used, so the exact infinite-domain energies are
/// `-Z^2 / (2 n^2)` for `n > l`. Gauss points never sample `r = 0`; combined
/// with the strongly imposed `u(0) = 0`, the Coulomb and centrifugal weak
/// terms are integrable on the first element.
pub fn build_radial_schrodinger_problem(config: &RadialSchrodingerConfig) -> EigenPhysicsProblem {
    assert!(config.nuclear_charge.is_finite() && config.nuclear_charge > 0.0);
    assert!(config.outer_radius.is_finite() && config.outer_radius > 0.0);
    assert!(config.elements >= 2);

    let ndof = config.elements - 1;
    let h = config.outer_radius / config.elements as f64;
    let mut stiffness = TripletMatrix::new(ndof, ndof);
    let mut mass = TripletMatrix::new(ndof, ndof);
    let gauss_points = [
        (-0.861_136_311_594_052_6, 0.347_854_845_137_453_8),
        (-0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
        (0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
        (0.861_136_311_594_052_6, 0.347_854_845_137_453_8),
    ];
    let centrifugal = (config.angular_momentum * (config.angular_momentum + 1)) as f64 / 2.0;

    for element in 0..config.elements {
        let r0 = element as f64 * h;
        let global = [element, element + 1];
        let derivatives = [-1.0 / h, 1.0 / h];
        for local_row in 0..2 {
            let row_node = global[local_row];
            if row_node == 0 || row_node == config.elements {
                continue;
            }
            for local_col in 0..2 {
                let col_node = global[local_col];
                if col_node == 0 || col_node == config.elements {
                    continue;
                }
                let mut local_stiffness = 0.5 * derivatives[local_row] * derivatives[local_col] * h;
                let mut local_mass = 0.0;
                for &(xi, weight) in &gauss_points {
                    let r = r0 + 0.5 * h * (1.0 + xi);
                    let phi = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
                    let jacobian_weight = 0.5 * h * weight;
                    let potential = centrifugal / (r * r) - config.nuclear_charge / r;
                    let product = phi[local_row] * phi[local_col] * jacobian_weight;
                    local_stiffness += potential * product;
                    local_mass += product;
                }
                stiffness.push(row_node - 1, col_node - 1, local_stiffness);
                mass.push(row_node - 1, col_node - 1, local_mass);
            }
        }
    }

    EigenPhysicsProblem {
        kind: EigenPhysicsKind::RadialSchrodinger,
        stiffness_like: stiffness.to_csr(),
        mass_like: mass.to_csr(),
        angular_frequency_scale: 0.0,
    }
}

impl ComplexEigenResult {
    /// Reorder and phase-align modes to maximize mass overlap with a previous
    /// result.
    ///
    /// The returned vector maps each previous mode index to the original
    /// current mode index. Assignment is global (Hungarian algorithm), so two
    /// previous modes cannot select the same current mode at a crossing.
    pub fn track_from(
        &mut self,
        previous: &ComplexEigenResult,
        mass: &CsrMatrix<Complex64>,
    ) -> Vec<usize> {
        let modes = previous.eigenvalues.len();
        assert_eq!(self.eigenvalues.len(), modes, "mode count mismatch");
        assert_eq!(previous.eigenvectors.nrows(), self.eigenvectors.nrows());
        assert_eq!(mass.num_rows, self.eigenvectors.nrows());
        assert_eq!(mass.num_cols, self.eigenvectors.nrows());

        let mut overlaps = vec![vec![Complex64::new(0.0, 0.0); modes]; modes];
        for current in 0..modes {
            let vector = self.eigenvectors.column_to_vec(current);
            let mut mass_vector = vec![Complex64::new(0.0, 0.0); vector.len()];
            for (row, destination) in mass_vector.iter_mut().enumerate() {
                *destination = mass
                    .row_entries(row)
                    .map(|(column, value)| value * vector[column])
                    .sum();
            }
            for (prior, row) in overlaps.iter_mut().enumerate() {
                row[current] = previous
                    .eigenvectors
                    .column_to_vec(prior)
                    .iter()
                    .zip(&mass_vector)
                    .map(|(&left, &right)| left.conj() * right)
                    .sum();
            }
        }
        let weights: Vec<Vec<f64>> = overlaps
            .iter()
            .map(|row| row.iter().map(|value| value.norm()).collect())
            .collect();
        let assignment = maximum_weight_assignment(&weights);
        let old_values = self.eigenvalues.clone();
        let old_residuals = self.residual_norms.clone();
        let old_vectors = self.eigenvectors.clone();
        self.eigenvalues = assignment.iter().map(|&mode| old_values[mode]).collect();
        self.residual_norms = assignment.iter().map(|&mode| old_residuals[mode]).collect();
        self.eigenvectors =
            DenseMatrix::from_fn(old_vectors.nrows(), modes, |row, tracked_mode| {
                let original_mode = assignment[tracked_mode];
                let overlap = overlaps[tracked_mode][original_mode];
                let phase = if overlap.norm() > 0.0 {
                    overlap.conj() / overlap.norm()
                } else {
                    Complex64::new(1.0, 0.0)
                };
                old_vectors[[row, original_mode]] * phase
            });
        assignment
    }
}

impl ComplexEigenPhysicsProblem {
    /// Solve the smallest complex-Hermitian generalized eigenvalues.
    pub fn solve_eigenvalues(
        &self,
        config: &LobpcgConfig,
    ) -> Result<ComplexEigenResult, SolverError> {
        let stiffness = hermitian_to_real_block(&self.stiffness_like);
        let mass = hermitian_to_real_block(&self.mass_like);
        let real_config = LobpcgConfig {
            num_eigenvalues: 2 * config.num_eigenvalues,
            block_padding: 2 * config.block_padding,
            ..config.clone()
        };
        let result = lobpcg_generalized(&stiffness, &mass, &real_config)?;
        let n = self.stiffness_like.num_rows;
        let eigenvalues = result
            .eigenvalues
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        let residual_norms = result
            .residual_norms
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| pair[0].max(pair[1]))
            .collect();
        let eigenvectors = DenseMatrix::from_fn(n, config.num_eigenvalues, |row, column| {
            let real_column = 2 * column;
            Complex64::new(
                result.eigenvectors[[row, real_column]],
                result.eigenvectors[[n + row, real_column]],
            )
        });
        Ok(ComplexEigenResult {
            eigenvalues,
            residual_norms,
            eigenvectors,
            iterations: result.iterations,
            converged: result.converged,
        })
    }
}

/// Build a complex Hermitian scalar eigenproblem with arbitrary Bloch phases.
///
/// Volume coefficients and Robin terms are real. Complex values enter through
/// periodic multi-point constraints, producing the condensed Hermitian pencil
/// P^H A P.
pub fn try_build_complex_scalar_eigenproblem<D, V, W>(
    mesh: &Mesh,
    diffusion: D,
    potential: V,
    weight: W,
    config: &ScalarEigenproblemConfig,
) -> Result<ComplexEigenPhysicsProblem, EigenproblemBuildError>
where
    D: Fn(f64, f64, f64) -> f64 + Send + Sync,
    V: Fn(f64, f64, f64) -> f64 + Send + Sync,
    W: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let mut dirichlet_markers = config.dirichlet_boundary_markers.clone();
    for term in &config.dirichlet_boundary_values {
        if !term.value.is_finite() || term.value != 0.0 {
            return Err(EigenproblemBuildError::InhomogeneousDirichlet {
                marker: term.marker,
                value: term.value,
            });
        }
        dirichlet_markers.push(term.marker);
    }
    assert!(
        (config.robin_stiffness_terms.is_empty() && config.robin_mass_terms.is_empty())
            || config.degree == PolynomialDegree::P1,
        "scalar Robin eigenproblem assembly currently supports P1 traces"
    );
    let space = H1Space::new(config.degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(config.degree.degree()) + 1;
    let quad_mass = quadrature::required_order_for_mass(config.degree.degree()) + 1;
    let mut stiffness = assemble_bilinear(
        &VariableLaplacianForm::new(diffusion),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let potential = assemble_bilinear(
        &VariableMassForm::new(potential),
        &space,
        &space,
        mesh,
        quad_mass,
    );
    append_triplets(&mut stiffness, &potential);
    let mut mass = assemble_bilinear(
        &VariableMassForm::new(weight),
        &space,
        &space,
        mesh,
        quad_mass,
    );
    for term in &config.robin_stiffness_terms {
        append_scaled_triplets(
            &mut stiffness,
            &assemble_boundary_mass_generic(mesh, term.marker),
            term.coefficient,
        );
    }
    for term in &config.robin_mass_terms {
        append_scaled_triplets(
            &mut mass,
            &assemble_boundary_mass_generic(mesh, term.marker),
            term.coefficient,
        );
    }
    let constrained = boundary_dofs(mesh, &space, &dirichlet_markers);
    let stiffness = real_to_complex(&stiffness.to_csr());
    let mass = real_to_complex(&mass.to_csr());
    let (stiffness_like, mass_like) = if config.periodic_constraints.is_empty() {
        (
            eliminate_complex_csr_dofs(&stiffness, &constrained),
            eliminate_complex_csr_dofs(&mass, &constrained),
        )
    } else {
        let mpc = MpcSystem::from_periodic(stiffness.num_rows, &config.periodic_constraints)?;
        let stiffness = mpc.condense_matrix(&stiffness)?;
        let mass = mpc.condense_matrix(&mass)?;
        let reduced_constrained = map_constrained_dofs(&mpc, &constrained);
        (
            eliminate_complex_csr_dofs(&stiffness, &reduced_constrained),
            eliminate_complex_csr_dofs(&mass, &reduced_constrained),
        )
    };
    Ok(ComplexEigenPhysicsProblem {
        kind: EigenPhysicsKind::ScalarGeneralized,
        stiffness_like,
        mass_like,
        angular_frequency_scale: config.angular_frequency_scale,
    })
}

#[derive(Debug, Clone)]
pub struct StructuralModalConfig {
    pub degree: PolynomialDegree,
    pub components: usize,
    pub density: f64,
    pub plane: Option<PlaneAssumption>,
    pub fixed_boundary_markers: Vec<i32>,
}

impl StructuralModalConfig {
    pub fn plane_stress(density: f64, fixed_boundary_markers: Vec<i32>) -> Self {
        Self {
            degree: PolynomialDegree::P1,
            components: 2,
            density,
            plane: Some(PlaneAssumption::PlaneStress),
            fixed_boundary_markers,
        }
    }
}

pub fn build_structural_modal_problem(
    mesh: &Mesh,
    material: IsotropicMaterial,
    config: &StructuralModalConfig,
) -> EigenPhysicsProblem {
    let space = VectorH1Space::new(config.degree, config.components);
    let stiffness_form = match config.plane {
        Some(plane) => ElasticityForm::new_2d(material, plane),
        None => ElasticityForm::new_3d(material),
    };
    let quad_stiffness = quadrature::required_order_for_stiffness(config.degree.degree());
    let quad_mass = quadrature::required_order_for_mass(config.degree.degree());
    let k = assemble_bilinear(&stiffness_form, &space, &space, mesh, quad_stiffness);
    let m = assemble_bilinear(
        &VectorMassForm {
            coefficient: config.density,
        },
        &space,
        &space,
        mesh,
        quad_mass,
    );
    let constrained = boundary_dofs(mesh, &space, &config.fixed_boundary_markers);
    EigenPhysicsProblem {
        kind: EigenPhysicsKind::StructuralModal,
        stiffness_like: eliminate_dofs(&k, &constrained).to_csr(),
        mass_like: eliminate_dofs(&m, &constrained).to_csr(),
        angular_frequency_scale: 1.0,
    }
}

#[derive(Debug, Clone)]
pub struct AcousticEigenmodeConfig {
    pub degree: PolynomialDegree,
    pub sound_speed: f64,
    pub pressure_release_boundary_markers: Vec<i32>,
}

impl AcousticEigenmodeConfig {
    pub fn new(sound_speed: f64) -> Self {
        Self {
            degree: PolynomialDegree::P1,
            sound_speed,
            pressure_release_boundary_markers: Vec::new(),
        }
    }
}

pub fn build_acoustic_eigenmode_problem(
    mesh: &Mesh,
    config: &AcousticEigenmodeConfig,
) -> EigenPhysicsProblem {
    let space = H1Space::new(config.degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(config.degree.degree());
    let quad_mass = quadrature::required_order_for_mass(config.degree.degree());
    let k = assemble_bilinear(
        &WeightedLaplacianForm { coefficient: 1.0 },
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let m = assemble_bilinear(&MassForm, &space, &space, mesh, quad_mass);
    let constrained = boundary_dofs(mesh, &space, &config.pressure_release_boundary_markers);
    EigenPhysicsProblem {
        kind: EigenPhysicsKind::AcousticEigenmode,
        stiffness_like: eliminate_dofs(&k, &constrained).to_csr(),
        mass_like: eliminate_dofs(&m, &constrained).to_csr(),
        angular_frequency_scale: config.sound_speed,
    }
}

#[derive(Debug, Clone)]
pub struct ThermalStabilityConfig {
    pub degree: PolynomialDegree,
    /// Linearized reaction/stability coefficient added to conduction.
    pub reaction_coefficient: f64,
    pub fixed_temperature_boundary_markers: Vec<i32>,
}

impl Default for ThermalStabilityConfig {
    fn default() -> Self {
        Self {
            degree: PolynomialDegree::P1,
            reaction_coefficient: 0.0,
            fixed_temperature_boundary_markers: Vec::new(),
        }
    }
}

pub fn build_thermal_stability_problem(
    mesh: &Mesh,
    material: ThermalMaterial,
    config: &ThermalStabilityConfig,
) -> EigenPhysicsProblem {
    let space = H1Space::new(config.degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(config.degree.degree());
    let quad_mass = quadrature::required_order_for_mass(config.degree.degree());
    let mut k = assemble_bilinear(
        &ConductionForm::new(&material),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    if config.reaction_coefficient != 0.0 {
        let reaction = assemble_bilinear(
            &WeightedMassForm {
                coefficient: config.reaction_coefficient,
            },
            &space,
            &space,
            mesh,
            quad_mass,
        );
        append_triplets(&mut k, &reaction);
    }
    let m = assemble_bilinear(
        &HeatCapacityForm::new(&material),
        &space,
        &space,
        mesh,
        quad_mass,
    );
    let constrained = boundary_dofs(mesh, &space, &config.fixed_temperature_boundary_markers);
    EigenPhysicsProblem {
        kind: EigenPhysicsKind::ThermalStability,
        stiffness_like: eliminate_dofs(&k, &constrained).to_csr(),
        mass_like: eliminate_dofs(&m, &constrained).to_csr(),
        angular_frequency_scale: 0.0,
    }
}

#[derive(Debug, Clone)]
pub struct ElectromagneticEigenmodeConfig {
    pub order: usize,
    pub pec_boundary_markers: Vec<i32>,
}

impl Default for ElectromagneticEigenmodeConfig {
    fn default() -> Self {
        Self {
            order: 1,
            pec_boundary_markers: Vec::new(),
        }
    }
}

pub fn build_electromagnetic_eigenmode_problem(
    mesh: &Mesh,
    material: EmMaterial,
    config: &ElectromagneticEigenmodeConfig,
) -> EigenPhysicsProblem {
    let space = if config.order == 2 {
        HcurlSpace::order2()
    } else {
        HcurlSpace::new()
    };
    let degree = if config.order == 2 {
        PolynomialDegree::P2
    } else {
        PolynomialDegree::P1
    };
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    let k = assemble_bilinear(
        &WeightedCurlCurlForm {
            coefficient: 1.0 / material.permeability(),
        },
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let m = assemble_bilinear(
        &WeightedEdgeMassForm {
            coefficient: material.complex_permittivity(1.0).re,
        },
        &space,
        &space,
        mesh,
        quad_mass,
    );
    let constrained = boundary_dofs(mesh, &space, &config.pec_boundary_markers);
    EigenPhysicsProblem {
        kind: EigenPhysicsKind::ElectromagneticEigenmode,
        stiffness_like: eliminate_dofs(&k, &constrained).to_csr(),
        mass_like: eliminate_dofs(&m, &constrained).to_csr(),
        angular_frequency_scale: 1.0,
    }
}

pub fn build_linear_buckling_problem(
    elastic_stiffness: &TripletMatrix<f64>,
    geometric_stiffness: &TripletMatrix<f64>,
    constrained_dofs: &[usize],
) -> EigenPhysicsProblem {
    EigenPhysicsProblem {
        kind: EigenPhysicsKind::LinearBuckling,
        stiffness_like: eliminate_dofs(elastic_stiffness, constrained_dofs).to_csr(),
        mass_like: eliminate_dofs(geometric_stiffness, constrained_dofs).to_csr(),
        angular_frequency_scale: 0.0,
    }
}

pub fn euler_column_buckling_load(
    young_modulus: f64,
    second_moment_area: f64,
    length: f64,
    effective_length_factor: f64,
) -> f64 {
    std::f64::consts::PI.powi(2) * young_modulus * second_moment_area
        / (effective_length_factor * length).powi(2)
}

pub fn critical_loads(eigenvalues: &[f64]) -> Vec<f64> {
    eigenvalues.to_vec()
}

pub fn thermal_decay_rates(eigenvalues: &[f64]) -> Vec<f64> {
    eigenvalues.to_vec()
}

fn boundary_dofs<S: FunctionSpace>(mesh: &Mesh, space: &S, markers: &[i32]) -> Vec<usize> {
    let mut set = HashSet::new();
    for &marker in markers {
        set.extend(space.boundary_dofs(mesh, marker));
    }
    let mut dofs: Vec<_> = set.into_iter().collect();
    dofs.sort_unstable();
    dofs
}

fn append_triplets(dst: &mut TripletMatrix<f64>, src: &TripletMatrix<f64>) {
    for i in 0..src.nnz() {
        dst.push(src.rows[i], src.cols[i], src.values[i]);
    }
}

fn append_scaled_triplets(dst: &mut TripletMatrix<f64>, src: &TripletMatrix<f64>, scale: f64) {
    for i in 0..src.nnz() {
        dst.push(src.rows[i], src.cols[i], scale * src.values[i]);
    }
}

fn eliminate_dofs(matrix: &TripletMatrix<f64>, constrained_dofs: &[usize]) -> TripletMatrix<f64> {
    if constrained_dofs.is_empty() {
        return matrix.clone();
    }
    let constrained: HashSet<_> = constrained_dofs.iter().copied().collect();
    let mut map = HashMap::with_capacity(matrix.num_rows);
    let mut next = 0;
    for old in 0..matrix.num_rows {
        if !constrained.contains(&old) {
            map.insert(old, next);
            next += 1;
        }
    }
    let mut reduced = TripletMatrix::with_capacity(next, next, matrix.nnz());
    for i in 0..matrix.nnz() {
        if let (Some(&row), Some(&col)) = (map.get(&matrix.rows[i]), map.get(&matrix.cols[i])) {
            reduced.push(row, col, matrix.values[i]);
        }
    }
    reduced
}

fn eliminate_csr_dofs(matrix: &CsrMatrix<f64>, constrained_dofs: &[usize]) -> CsrMatrix<f64> {
    if constrained_dofs.is_empty() {
        return matrix.clone();
    }
    let constrained: HashSet<_> = constrained_dofs.iter().copied().collect();
    let mut map = HashMap::with_capacity(matrix.num_rows);
    let mut next = 0;
    for old in 0..matrix.num_rows {
        if !constrained.contains(&old) {
            map.insert(old, next);
            next += 1;
        }
    }
    let mut triplets = Vec::with_capacity(matrix.nnz());
    for old_row in 0..matrix.num_rows {
        let Some(&row) = map.get(&old_row) else {
            continue;
        };
        for (old_column, value) in matrix.row_entries(old_row) {
            if let Some(&column) = map.get(&old_column) {
                triplets.push((row, column, value));
            }
        }
    }
    CsrMatrix::from_triplets(next, next, triplets)
}

fn map_constrained_dofs(mpc: &MpcSystem, constrained: &[usize]) -> Vec<usize> {
    let mut reduced: Vec<_> = constrained
        .iter()
        .flat_map(|&dof| mpc.prolongation_row(dof).iter().map(|&(index, _)| index))
        .collect();
    reduced.sort_unstable();
    reduced.dedup();
    reduced
}

fn real_to_complex(matrix: &CsrMatrix<f64>) -> CsrMatrix<Complex64> {
    let mut triplets = Vec::with_capacity(matrix.nnz());
    for row in 0..matrix.num_rows {
        for (column, value) in matrix.row_entries(row) {
            triplets.push((row, column, Complex64::new(value, 0.0)));
        }
    }
    CsrMatrix::from_triplets(matrix.num_rows, matrix.num_cols, triplets)
}

fn eliminate_complex_csr_dofs(
    matrix: &CsrMatrix<Complex64>,
    constrained_dofs: &[usize],
) -> CsrMatrix<Complex64> {
    if constrained_dofs.is_empty() {
        return matrix.clone();
    }
    let constrained: HashSet<_> = constrained_dofs.iter().copied().collect();
    let mut map = HashMap::with_capacity(matrix.num_rows);
    let mut next = 0;
    for old in 0..matrix.num_rows {
        if !constrained.contains(&old) {
            map.insert(old, next);
            next += 1;
        }
    }
    let mut triplets = Vec::with_capacity(matrix.nnz());
    for old_row in 0..matrix.num_rows {
        let Some(&row) = map.get(&old_row) else {
            continue;
        };
        for (old_column, value) in matrix.row_entries(old_row) {
            if let Some(&column) = map.get(&old_column) {
                triplets.push((row, column, value));
            }
        }
    }
    CsrMatrix::from_triplets(next, next, triplets)
}

fn hermitian_to_real_block(matrix: &CsrMatrix<Complex64>) -> CsrMatrix<f64> {
    let n = matrix.num_rows;
    let mut triplets = Vec::with_capacity(4 * matrix.nnz());
    for row in 0..n {
        for (column, value) in matrix.row_entries(row) {
            triplets.push((row, column, value.re));
            triplets.push((row, n + column, -value.im));
            triplets.push((n + row, column, value.im));
            triplets.push((n + row, n + column, value.re));
        }
    }
    CsrMatrix::from_triplets(2 * n, 2 * n, triplets)
}

fn maximum_weight_assignment(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    assert!(weights.iter().all(|row| row.len() == n));
    if n == 0 {
        return Vec::new();
    }
    // Hungarian minimization with cost = -overlap.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let current = -weights[i0 - 1][j - 1] - u[i0] - v[j];
                if current < minv[j] {
                    minv[j] = current;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut assignment = vec![0usize; n];
    for column in 1..=n {
        assignment[p[column] - 1] = column - 1;
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::periodic::{
        PeriodicTransform, expand_node_periodic_constraints, pair_marked_boundary_nodes,
    };
    use crate::mesh::generators::rectangular_mesh_triangles;
    use num_complex::Complex64;

    #[test]
    fn acoustic_eigen_problem_builds_reduced_dirichlet_system() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 4, 1);
        let problem = build_acoustic_eigenmode_problem(
            &mesh,
            &AcousticEigenmodeConfig {
                degree: PolynomialDegree::P1,
                sound_speed: 343.0,
                pressure_release_boundary_markers: vec![1, 2],
            },
        );
        assert_eq!(problem.kind, EigenPhysicsKind::AcousticEigenmode);
        assert!(problem.stiffness_like.num_rows < mesh.num_nodes());
        assert_eq!(problem.stiffness_like.num_rows, problem.mass_like.num_rows);
    }

    #[test]
    fn euler_column_load_matches_textbook_formula() {
        let load = euler_column_buckling_load(200.0e9, 1.0e-8, 2.0, 1.0);
        assert!((load - std::f64::consts::PI.powi(2) * 500.0).abs() < 1.0e-9);
    }

    #[test]
    fn scalar_potential_shifts_dirichlet_spectrum() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 12, 12);
        let config = ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1, 2, 3, 4],
            ..Default::default()
        };
        let solve_config = LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 200,
            tolerance: 1.0e-8,
            block_padding: 2,
        };
        let base =
            build_scalar_eigenproblem(&mesh, |_, _, _| 1.0, |_, _, _| 0.0, |_, _, _| 1.0, &config)
                .solve(&solve_config)
                .unwrap();
        let shifted =
            build_scalar_eigenproblem(&mesh, |_, _, _| 1.0, |_, _, _| 5.0, |_, _, _| 1.0, &config)
                .solve(&solve_config)
                .unwrap();
        assert!((shifted.eigenvalues[0] - base.eigenvalues[0] - 5.0).abs() < 1.0e-8);
    }

    #[test]
    fn scalar_interval_solve_finds_only_interior_modes() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 28, 22);
        let problem = build_scalar_eigenproblem(
            &mesh,
            |_, _, _| 1.0,
            |_, _, _| 5.0,
            |_, _, _| 1.0,
            &ScalarEigenproblemConfig {
                dirichlet_boundary_markers: vec![1, 2, 3, 4],
                ..Default::default()
            },
        );
        let result = problem
            .solve_interval(
                70.0,
                112.0,
                &ShiftInvertConfig {
                    target: 91.0,
                    num_eigenvalues: 8,
                    krylov_dimension: 36,
                    tolerance: 1.0e-8,
                    breakdown_tolerance: 1.0e-14,
                },
            )
            .unwrap();
        let exact = [
            (1.0 + 4.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
            (4.0 + 4.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
            (9.0 + 1.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
        ];
        let mut actual = result.eigenvalues;
        actual.sort_by(f64::total_cmp);
        assert_eq!(actual.len(), exact.len());
        for (&actual, &expected) in actual.iter().zip(&exact) {
            assert!(
                (actual - expected).abs() / expected < 0.02,
                "interval mode: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn radial_hydrogen_bound_states_match_balmer_energies() {
        let problem = build_radial_schrodinger_problem(&RadialSchrodingerConfig::default());
        let result = problem
            .solve_interval(
                -0.55,
                -0.02,
                &ShiftInvertConfig {
                    target: -0.2,
                    num_eigenvalues: 10,
                    krylov_dimension: 48,
                    tolerance: 1.0e-8,
                    breakdown_tolerance: 1.0e-14,
                },
            )
            .unwrap();
        let mut bound_states: Vec<_> = result
            .eigenvalues
            .into_iter()
            .filter(|energy| *energy < -0.02)
            .collect();
        bound_states.sort_by(f64::total_cmp);
        let exact = [-0.5, -0.125, -1.0 / 18.0, -0.03125];
        assert_eq!(bound_states.len(), exact.len());
        for (&actual, &expected) in bound_states.iter().zip(&exact) {
            assert!(
                (actual - expected).abs() / expected.abs() < 0.01,
                "radial energy: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn proportional_damping_produces_quadratic_conjugate_roots() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 20, 16);
        let problem = build_scalar_eigenproblem(
            &mesh,
            |_, _, _| 1.0,
            |_, _, _| 0.0,
            |_, _, _| 1.0,
            &ScalarEigenproblemConfig {
                dirichlet_boundary_markers: vec![1, 2, 3, 4],
                ..Default::default()
            },
        );
        let result = problem
            .solve_proportionally_damped(
                ProportionalDamping {
                    mass_damping: 0.2,
                    stiffness_damping: 0.0,
                },
                &LobpcgConfig {
                    num_eigenvalues: 2,
                    max_iterations: 300,
                    tolerance: 1.0e-8,
                    block_padding: 2,
                },
            )
            .unwrap();
        assert!(result.converged);
        for (mode, pair) in result.temporal_eigenvalues.chunks_exact(2).enumerate() {
            assert!((pair[0].re + 0.1).abs() < 1.0e-10);
            assert!((pair[1] - pair[0].conj()).norm() < 1.0e-10);
            let lambda = result.spatial_eigenvalues[mode];
            for &root in pair {
                assert!((root * root + 0.2 * root + lambda).norm() < 1.0e-8);
            }
        }
    }

    #[test]
    fn general_quadratic_linearization_recovers_damped_roots() {
        let stiffness = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 4.0)]);
        let damping = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 0.4)]);
        let mass = CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0)]);
        let problem = linearize_quadratic_eigenproblem(&stiffness, &damping, &mass).unwrap();
        let result = problem
            .solve_near(&ComplexArnoldiConfig {
                target: Complex64::new(0.0, 0.8),
                num_eigenvalues: 2,
                krylov_dimension: 2,
                tolerance: 1.0e-8,
                max_qr_iterations: 2_000,
                breakdown_tolerance: 1.0e-14,
            })
            .unwrap();
        assert!(result.converged, "{result:?}");
        let imaginary = (4.0_f64 - 0.2_f64.powi(2)).sqrt();
        let exact = [
            Complex64::new(-0.2, imaginary),
            Complex64::new(-0.2, -imaginary),
        ];
        for (&actual, &expected) in result.eigenvalues.iter().zip(&exact) {
            assert!((actual - expected).norm() < 1.0e-8, "{result:?}");
            assert!((actual * actual + 0.4 * actual + 4.0).norm() < 1.0e-8);
        }
    }

    #[test]
    fn coupled_reaction_diffusion_has_complex_nonsymmetric_spectrum() {
        let spatial = EigenPhysicsProblem {
            kind: EigenPhysicsKind::ScalarGeneralized,
            stiffness_like: CsrMatrix::from_triplets(1, 1, vec![(0, 0, 30.0)]),
            mass_like: CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0)]),
            angular_frequency_scale: 0.0,
        };
        let problem =
            build_coupled_reaction_diffusion_eigenproblem(&spatial, 2.0, 20.0, -20.0).unwrap();
        let result = problem
            .solve_near(&ComplexArnoldiConfig {
                target: Complex64::new(45.0, 13.0),
                num_eigenvalues: 2,
                krylov_dimension: 2,
                tolerance: 1.0e-9,
                max_qr_iterations: 2_000,
                breakdown_tolerance: 1.0e-14,
            })
            .unwrap();
        assert!(result.converged, "{result:?}");
        let imaginary = (1600.0_f64 - 30.0_f64.powi(2)).sqrt() / 2.0;
        let exact = [
            Complex64::new(45.0, imaginary),
            Complex64::new(45.0, -imaginary),
        ];
        for (&actual, &expected) in result.eigenvalues.iter().zip(&exact) {
            assert!((actual - expected).norm() < 1.0e-8, "{result:?}");
        }
    }

    #[test]
    fn scalar_periodic_constraints_compose_with_dirichlet_elimination() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 16, 12);
        let pairs = pair_marked_boundary_nodes(
            &mesh,
            1,
            2,
            PeriodicTransform {
                translation: [1.0, 0.0, 0.0],
                ..PeriodicTransform::IDENTITY
            },
            1.0e-12,
        )
        .unwrap();
        let config = ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![3, 4],
            periodic_constraints: expand_node_periodic_constraints(
                &pairs,
                1,
                Complex64::new(1.0, 0.0),
            ),
            ..Default::default()
        };
        let problem = try_build_scalar_eigenproblem(
            &mesh,
            |_, _, _| 1.0,
            |_, _, _| 0.0,
            |_, _, _| 1.0,
            &config,
        )
        .unwrap();
        let result = problem
            .solve(&LobpcgConfig {
                num_eigenvalues: 1,
                max_iterations: 300,
                tolerance: 1.0e-8,
                block_padding: 2,
            })
            .unwrap();
        let exact = (std::f64::consts::PI / 0.8).powi(2);
        assert!((result.eigenvalues[0] - exact).abs() / exact < 0.015);
    }

    #[test]
    fn complex_bloch_phase_matches_rectangular_spectrum() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 20, 16);
        let pairs = pair_marked_boundary_nodes(
            &mesh,
            1,
            2,
            PeriodicTransform {
                translation: [1.0, 0.0, 0.0],
                ..PeriodicTransform::IDENTITY
            },
            1.0e-12,
        )
        .unwrap();
        let theta = std::f64::consts::PI / 3.0;
        let config = ScalarEigenproblemConfig {
            periodic_constraints: expand_node_periodic_constraints(
                &pairs,
                1,
                Complex64::from_polar(1.0, theta),
            ),
            ..Default::default()
        };
        let problem = try_build_complex_scalar_eigenproblem(
            &mesh,
            |_, _, _| 1.0,
            |_, _, _| 5.0,
            |_, _, _| 1.0,
            &config,
        )
        .unwrap();
        let result = problem
            .solve_eigenvalues(&LobpcgConfig {
                num_eigenvalues: 3,
                max_iterations: 500,
                tolerance: 1.0e-8,
                block_padding: 3,
            })
            .unwrap();
        let exact = [
            5.0 + theta.powi(2),
            5.0 + theta.powi(2) + (std::f64::consts::PI / 0.8).powi(2),
            5.0 + (theta - 2.0 * std::f64::consts::PI).powi(2),
        ];
        for (&actual, &expected) in result.eigenvalues.iter().zip(&exact) {
            assert!((actual - expected).abs() / expected < 0.02);
        }
        for mode in 0..result.eigenvalues.len() {
            let vector = result.eigenvectors.column_to_vec(mode);
            let mut residual_norm_sq = 0.0;
            let mut mass_norm = Complex64::new(0.0, 0.0);
            for row in 0..problem.stiffness_like.num_rows {
                let ax: Complex64 = problem
                    .stiffness_like
                    .row_entries(row)
                    .map(|(column, value)| value * vector[column])
                    .sum();
                let bx: Complex64 = problem
                    .mass_like
                    .row_entries(row)
                    .map(|(column, value)| value * vector[column])
                    .sum();
                residual_norm_sq += (ax - result.eigenvalues[mode] * bx).norm_sqr();
                mass_norm += vector[row].conj() * bx;
            }
            assert!(residual_norm_sq.sqrt() < 1.0e-5);
            assert!((mass_norm.re - 1.0).abs() < 1.0e-8);
            assert!(mass_norm.im.abs() < 1.0e-10);
        }
    }

    #[test]
    fn complex_mode_tracking_globally_assigns_and_phase_aligns() {
        let previous = ComplexEigenResult {
            eigenvalues: vec![1.0, 2.0],
            residual_norms: vec![1.0e-9, 2.0e-9],
            eigenvectors: DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            ),
            iterations: 1,
            converged: true,
        };
        let mut current = ComplexEigenResult {
            eigenvalues: vec![2.0, 1.0],
            residual_norms: vec![2.0e-9, 1.0e-9],
            eigenvectors: DenseMatrix::from_row_major_vec(
                2,
                2,
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            ),
            iterations: 1,
            converged: true,
        };
        let mass = CsrMatrix::from_triplets(
            2,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 0.0)),
                (1, 1, Complex64::new(1.0, 0.0)),
            ],
        );
        assert_eq!(current.track_from(&previous, &mass), vec![1, 0]);
        assert_eq!(current.eigenvalues, vec![1.0, 2.0]);
        for mode in 0..2 {
            for row in 0..2 {
                assert!(
                    (current.eigenvectors[[row, mode]] - previous.eigenvectors[[row, mode]]).norm()
                        < 1.0e-14
                );
            }
        }
    }

    #[test]
    fn real_mode_tracking_matches_fields_on_independent_meshes() {
        let previous_mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 12, 10);
        let current_mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 17, 13);
        let previous_modes = DenseMatrix::from_fn(previous_mesh.num_nodes(), 2, |node, mode| {
            let point = &previous_mesh.nodes[node];
            if mode == 0 { point.x } else { point.y }
        });
        let current_modes = DenseMatrix::from_fn(current_mesh.num_nodes(), 2, |node, mode| {
            let point = &current_mesh.nodes[node];
            if mode == 0 { -point.y } else { point.x }
        });

        let assignment = track_real_modes_across_meshes(
            &previous_mesh,
            &previous_modes,
            &current_mesh,
            &current_modes,
            31,
        )
        .unwrap();

        assert_eq!(assignment.current_indices, vec![1, 0]);
        assert_eq!(assignment.phase_signs, vec![1.0, -1.0]);
        assert!(
            assignment
                .overlaps
                .iter()
                .all(|overlap| *overlap > 1.0 - 1.0e-12)
        );
        assert!(assignment.common_samples > 800);
    }

    #[test]
    fn modal_projection_reconstruction_and_time_evolution_are_consistent() {
        let mass = CsrMatrix::from_triplets(3, 3, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 4.0)]);
        let modes = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 1.0 / 2.0_f64.sqrt(), 0.0, 0.0, 0.0, 0.5],
        );
        let field = vec![2.0, 3.0, 4.0];
        let coefficients = project_onto_real_modes(&mass, &modes, &field).unwrap();
        let reconstructed = reconstruct_from_real_modes(&modes, &coefficients).unwrap();
        for (&actual, &expected) in reconstructed.iter().zip(&field) {
            assert!((actual - expected).abs() < 1.0e-14);
        }

        let wave =
            evolve_wave_mode_coefficients(&[0.0, 4.0], &[1.0, 2.0], &[0.5, 3.0], 0.25).unwrap();
        assert!((wave[0] - 1.125).abs() < 1.0e-14);
        assert!((wave[1] - (2.0 * 0.5_f64.cos() + 1.5 * 0.5_f64.sin())).abs() < 1.0e-14);

        let heat = evolve_heat_mode_coefficients(&[0.0, 4.0], &[1.0, 2.0], 0.25, 0.5).unwrap();
        assert!((heat[0] - 1.0).abs() < 1.0e-14);
        assert!((heat[1] - 2.0 * (-0.5_f64).exp()).abs() < 1.0e-14);
    }

    #[test]
    fn interval_laplacian_matches_dirichlet_spectrum() {
        let problem = build_interval_laplacian_problem(&IntervalLaplacianConfig::default());
        let result = problem
            .solve(&LobpcgConfig {
                num_eigenvalues: 4,
                max_iterations: 400,
                tolerance: 1.0e-10,
                block_padding: 4,
            })
            .unwrap();
        for (index, &actual) in result.eigenvalues.iter().enumerate() {
            let mode = (index + 1) as f64;
            let exact = (mode * std::f64::consts::PI).powi(2);
            assert!(
                (actual - exact).abs() / exact < 0.003,
                "interval mode {}: actual={actual}, exact={exact}",
                index + 1
            );
        }
    }
}
