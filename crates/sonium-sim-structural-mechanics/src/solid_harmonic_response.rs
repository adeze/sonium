//! Direct assembled 3-D solid harmonic-response evidence.
//!
//! This module assembles the production P1 solid stiffness and consistent
//! vector mass matrices, applies the existing fixed-face/loaded-face policy,
//! and solves the complex dynamic system independently at each requested
//! frequency. It is a bounded CPU/f64 frequency-response reference, not a
//! nonlinear, transient, distributed, or accelerator harmonic solver.

use crate::{
    Solid3DStudy, StructuralExecutionProvenance, StructuralMechanicsError,
    StructuralModelAssumption, StructuralPolynomialOrder, families::apply_solid3d_traction,
    reduce_triplets,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_bilinear_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::standard::VectorMassForm;
use sonium_fem::physics::solid::elasticity::ElasticityForm;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{
    CsrMatrix, MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, factorize_sparse,
};

/// Schema version for the direct 3-D harmonic-response result.
pub const SOLID3D_HARMONIC_RESPONSE_SCHEMA_VERSION: u32 = 1;

/// Direct assembled 3-D harmonic-response controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DHarmonicResponseStudy {
    pub name: String,
    pub base: Solid3DStudy,
    pub frequencies_hz: Vec<f64>,
    /// Real/imaginary total force phasor for each Cartesian direction.
    pub force_phasors_n: [[f64; 2]; 3],
    /// Mass-proportional damping uses `C = 2 ζ ω_ref M`.
    pub damping_ratio: f64,
    pub damping_reference_frequency_hz: f64,
    pub observation_node: usize,
    pub observation_component: usize,
    pub residual_tolerance: f64,
    pub allowable_displacement_m: f64,
}

impl Default for Solid3DHarmonicResponseStudy {
    fn default() -> Self {
        Self::introductory()
    }
}

impl Solid3DHarmonicResponseStudy {
    /// Small bounded direct-frequency-response fixture.
    pub fn introductory() -> Self {
        Self {
            name: "solid-3d-harmonic-response-reference".to_string(),
            base: Solid3DStudy::introductory(),
            frequencies_hz: vec![1.0, 10.0, 100.0],
            force_phasors_n: [[100.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            damping_ratio: 0.02,
            damping_reference_frequency_hz: 10.0,
            observation_node: 5,
            observation_component: 0,
            residual_tolerance: 1.0e-9,
            allowable_displacement_m: 1.0,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        if self.name.trim().is_empty() {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_harmonic_response.name",
                value: "empty".to_string(),
            });
        }
        if self.frequencies_hz.is_empty()
            || self.frequencies_hz.len() > 1_000_001
            || self
                .frequencies_hz
                .windows(2)
                .any(|window| !window[0].is_finite() || window[1] <= window[0])
            || self
                .frequencies_hz
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_harmonic_response.frequency_grid",
                value: "finite, positive, strictly increasing bounded grid required".to_string(),
            });
        }
        if self
            .force_phasors_n
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
            || self.force_phasors_n.iter().all(|component| {
                component[0].abs() <= f64::EPSILON && component[1].abs() <= f64::EPSILON
            })
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_harmonic_response.force_phasors_n",
                value: "finite non-zero force phasor required".to_string(),
            });
        }
        for (field, value) in [
            (
                "solid3d_harmonic_response.damping_ratio",
                self.damping_ratio,
            ),
            (
                "solid3d_harmonic_response.damping_reference_frequency_hz",
                self.damping_reference_frequency_hz,
            ),
            (
                "solid3d_harmonic_response.residual_tolerance",
                self.residual_tolerance,
            ),
            (
                "solid3d_harmonic_response.allowable_displacement_m",
                self.allowable_displacement_m,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.observation_component >= 3 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_harmonic_response.observation_component",
                value: self.observation_component.to_string(),
            });
        }
        Ok(())
    }
}

/// One direct complex 3-D frequency-response sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Solid3DHarmonicResponsePoint {
    pub frequency_hz: f64,
    pub displacement_phasor_m: [f64; 2],
    pub displacement_abs_m: f64,
    /// Principal-value lag for the `exp(iωt)` convention.
    pub phase_lag_rad: f64,
    pub relative_solver_residual: f64,
}

/// Versioned direct assembled 3-D harmonic-response result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DHarmonicResponseResult {
    pub schema_version: u32,
    pub study: Solid3DHarmonicResponseStudy,
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub points: Vec<Solid3DHarmonicResponsePoint>,
    /// Full nodal complex displacement fields, indexed by frequency then node
    /// then Cartesian component. This is numerical solver output, not measured
    /// provenance.
    pub displacement_field_phasors_m: Vec<Vec<[[f64; 2]; 3]>>,
    pub peak_displacement_abs_m: f64,
    pub peak_frequency_hz: f64,
    pub max_factorization_memory_bytes: usize,
    pub factorizations: usize,
    pub converged: bool,
    pub feasible: bool,
    pub model_assumption: StructuralModelAssumption,
    pub polynomial_order: StructuralPolynomialOrder,
    pub execution: StructuralExecutionProvenance,
}

/// Run the direct assembled P1 3-D solid harmonic response.
pub fn run_solid3d_harmonic_response(
    study: &Solid3DHarmonicResponseStudy,
) -> Result<Solid3DHarmonicResponseResult, StructuralMechanicsError> {
    study.validate()?;
    let mesh = study.base.mesh()?;
    let polynomial_order = StructuralPolynomialOrder::Linear;
    let degree = PolynomialDegree::P1;
    let space = VectorH1Space::new(degree, 3);
    let dofs = space.num_global_dofs(&mesh);
    if study.observation_node >= mesh.num_nodes() {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d_harmonic_response.observation_node",
            value: study.observation_node.to_string(),
        });
    }

    let stiffness_full = assemble_bilinear_checked(
        &ElasticityForm::new_3d(study.base.material.as_fem()),
        &space,
        &space,
        &mesh,
        degree.degree() + 1,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d harmonic stiffness assembly",
        message: error.to_string(),
    })?;
    let mass_full = assemble_bilinear_checked(
        &VectorMassForm {
            coefficient: study.base.material.density_kg_m3,
        },
        &space,
        &space,
        &mesh,
        degree.degree() + 1,
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "solid-3d harmonic mass assembly",
        message: error.to_string(),
    })?;
    let fixed_dofs = space.boundary_dofs(&mesh, study.base.loads.fixed_boundary_marker);
    let (stiffness, free_to_full) = reduce_triplets(&stiffness_full, &fixed_dofs);
    let (mass, mass_free_to_full) = reduce_triplets(&mass_full, &fixed_dofs);
    if free_to_full != mass_free_to_full {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "solid-3d harmonic reduction",
            message: "stiffness and mass free-DOF maps disagree".to_string(),
        });
    }

    let mut real_force = vec![0.0; dofs];
    let mut imaginary_force = vec![0.0; dofs];
    apply_solid3d_traction(
        &mut real_force,
        &mesh,
        study.base.loads.loaded_boundary_marker,
        std::array::from_fn(|component| study.force_phasors_n[component][0]),
        study.base.geometry.loaded_face_area_m2(),
        polynomial_order,
    )?;
    apply_solid3d_traction(
        &mut imaginary_force,
        &mesh,
        study.base.loads.loaded_boundary_marker,
        std::array::from_fn(|component| study.force_phasors_n[component][1]),
        study.base.geometry.loaded_face_area_m2(),
        polynomial_order,
    )?;
    let rhs = free_to_full
        .iter()
        .map(|&full_dof| Complex64::new(real_force[full_dof], imaginary_force[full_dof]))
        .collect::<Vec<_>>();
    let rhs_norm = complex_norm(&rhs).max(1.0);
    let damping_coefficient =
        2.0 * study.damping_ratio * std::f64::consts::TAU * study.damping_reference_frequency_hz;
    let solver_config = SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SupernodalLu,
        ordering: Ordering::Auto,
        ..SparseDirectConfig::default()
    };
    let mut points = Vec::with_capacity(study.frequencies_hz.len());
    let mut displacement_field_phasors_m = Vec::with_capacity(study.frequencies_hz.len());
    let mut peak_displacement_abs_m = 0.0;
    let mut peak_frequency_hz = study.frequencies_hz[0];
    let mut max_factorization_memory_bytes = 0;

    for &frequency_hz in &study.frequencies_hz {
        let omega = std::f64::consts::TAU * frequency_hz;
        let mut triplets = Vec::with_capacity(stiffness.nnz() + mass.nnz());
        append_scaled_real_matrix(&mut triplets, &stiffness, Complex64::new(1.0, 0.0));
        append_scaled_real_matrix(
            &mut triplets,
            &mass,
            Complex64::new(-omega * omega, omega * damping_coefficient),
        );
        let operator = CsrMatrix::from_triplets(stiffness.num_rows, stiffness.num_cols, triplets);
        let factor = factorize_sparse(&operator, &solver_config, None).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "solid-3d harmonic factorization",
                message: format!("{frequency_hz} Hz: {error}"),
            }
        })?;
        max_factorization_memory_bytes =
            max_factorization_memory_bytes.max(factor.factor_memory_bytes());
        let solution =
            factor
                .solve(&rhs)
                .map_err(|error| StructuralMechanicsError::SolverFailure {
                    stage: "solid-3d harmonic solve",
                    message: format!("{frequency_hz} Hz: {error}"),
                })?;
        let residual = complex_residual_norm(&operator, &solution, &rhs) / rhs_norm;
        if !residual.is_finite() || residual > study.residual_tolerance {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "solid3d_harmonic.relative_solver_residual",
                value: residual,
            });
        }

        let mut full_solution = vec![Complex64::new(0.0, 0.0); dofs];
        for (reduced, &full_dof) in free_to_full.iter().enumerate() {
            full_solution[full_dof] = solution[reduced];
        }
        let field = full_solution
            .as_chunks::<3>()
            .0
            .iter()
            .map(|node| {
                [
                    [node[0].re, node[0].im],
                    [node[1].re, node[1].im],
                    [node[2].re, node[2].im],
                ]
            })
            .collect::<Vec<_>>();
        if field
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "solid-3d harmonic field reconstruction",
                message: format!("non-finite nodal displacement at {frequency_hz} Hz"),
            });
        }
        displacement_field_phasors_m.push(field);
        let observed = full_solution[study.observation_node * 3 + study.observation_component];
        let displacement_phasor_m = [observed.re, observed.im];
        let displacement_abs_m = observed.norm();
        if !displacement_abs_m.is_finite() {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "solid-3d harmonic response reconstruction",
                message: format!("non-finite displacement at {frequency_hz} Hz"),
            });
        }
        if displacement_abs_m > peak_displacement_abs_m {
            peak_displacement_abs_m = displacement_abs_m;
            peak_frequency_hz = frequency_hz;
        }
        points.push(Solid3DHarmonicResponsePoint {
            frequency_hz,
            displacement_phasor_m,
            displacement_abs_m,
            phase_lag_rad: (-observed.im).atan2(observed.re),
            relative_solver_residual: residual,
        });
    }

    let mut execution = StructuralExecutionProvenance::cpu_reference();
    execution.capability_id =
        "sonium-structural.cpu-reference.f64.solid3d-harmonic-response.v1".to_string();
    execution.limitations.extend([
        "The direct response uses assembled P1 stiffness and consistent vector mass matrices on the CPU/f64 reference path.".to_string(),
        "Damping is caller-configured mass proportional damping; geometric nonlinearity, contact, transient loading, and modal coupling are not represented.".to_string(),
        "This is bounded 3-D harmonic FEM evidence, not product-scale convergence, measured correlation, external solver parity, or accelerator evidence.".to_string(),
    ]);

    Ok(Solid3DHarmonicResponseResult {
        schema_version: SOLID3D_HARMONIC_RESPONSE_SCHEMA_VERSION,
        study: study.clone(),
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        points,
        displacement_field_phasors_m,
        peak_displacement_abs_m,
        peak_frequency_hz,
        max_factorization_memory_bytes,
        factorizations: study.frequencies_hz.len(),
        converged: true,
        feasible: peak_displacement_abs_m <= study.allowable_displacement_m,
        model_assumption: StructuralModelAssumption::Solid3DHarmonicResponse,
        polynomial_order,
        execution,
    })
}

fn append_scaled_real_matrix(
    output: &mut Vec<(usize, usize, Complex64)>,
    matrix: &CsrMatrix<f64>,
    scale: Complex64,
) {
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            output.push((
                row,
                matrix.col_indices[index],
                Complex64::new(matrix.values[index], 0.0) * scale,
            ));
        }
    }
}

fn complex_norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn complex_residual_norm(
    matrix: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    let residual = (0..matrix.num_rows)
        .map(|row| {
            let value = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * solution[matrix.col_indices[index]])
                .sum::<Complex64>();
            value - rhs[row]
        })
        .collect::<Vec<_>>();
    complex_norm(&residual)
}
