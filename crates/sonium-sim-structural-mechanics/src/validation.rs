//! Structural validation catalogue and analytical thick-cylinder fixture.
//!
//! The NAFEMS-labelled cases in this catalogue are verification axes aligned
//! with the review's static, modal, thermal-stress, and nonlinear requirements.
//! They deliberately identify their native references as analytical surrogates:
//! passing this suite is not evidence of official NAFEMS parity until versioned
//! NAFEMS decks and reference values are added through the external adapter
//! layer.

use crate::{
    Beam2DStudy, Hyperelastic3DStudy, StructuralExecutionProvenance, StructuralMechanicsError,
    ThermoelasticStudy, run_beam2d_modal, run_beam2d_static, run_hyperelastic_3d,
    run_thermoelastic_static,
};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use sonium_fem::assembly::assemble_bilinear;
#[cfg(test)]
use sonium_fem::basis::PolynomialDegree;
#[cfg(test)]
use sonium_fem::boundary::generic::FemSystem;
#[cfg(test)]
use sonium_fem::mesh::rectangular_mesh_triangles;
#[cfg(test)]
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
#[cfg(test)]
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
#[cfg(test)]
use sonium_solvers::{
    MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, factorize_sparse,
};

/// Schema version for the native structural validation catalogue.
pub const STRUCTURAL_VALIDATION_SCHEMA_VERSION: u32 = 1;

/// Validation axis represented by a catalogue case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NafemsBenchmarkKind {
    Static,
    Modal,
    ThermalStress,
    Nonlinear,
}

/// Provenance of the reference used by a native validation case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationReferenceKind {
    /// Closed-form or manufactured reference evaluated in this crate.
    AnalyticalSurrogate,
}

/// One machine-readable NAFEMS-axis validation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NafemsBenchmarkCase {
    pub id: String,
    pub kind: NafemsBenchmarkKind,
    pub reference_kind: ValidationReferenceKind,
    pub reference_source: String,
    pub computed_value: f64,
    pub reference_value: f64,
    pub relative_error: f64,
    pub tolerance: f64,
    pub solver_residual: Option<f64>,
    pub energy_error: Option<f64>,
    pub passed: bool,
    pub applicability: String,
}

/// Complete native validation catalogue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NafemsBenchmarkSuite {
    pub schema_version: u32,
    pub execution: StructuralExecutionProvenance,
    pub cases: Vec<NafemsBenchmarkCase>,
    pub all_passed: bool,
    pub official_reference_data_loaded: bool,
    pub applicability_limits: Vec<String>,
}

impl NafemsBenchmarkSuite {
    /// Return true only when native checks and official reference data are
    /// both present. Analytical surrogates therefore cannot pass this gate.
    pub fn official_parity_passed(&self) -> bool {
        self.official_reference_data_loaded && self.all_passed
    }
}

/// Run static, modal, thermal-stress, and nonlinear NAFEMS-axis checks.
pub fn run_nafems_benchmark_suite() -> Result<NafemsBenchmarkSuite, StructuralMechanicsError> {
    let mut cases = Vec::with_capacity(4);

    let mut static_study = Beam2DStudy::introductory();
    static_study.geometry.elements = 4;
    let static_result = run_beam2d_static(&static_study)?;
    cases.push(case_report(
        "nafems-static-cantilever-axis",
        NafemsBenchmarkKind::Static,
        "NAFEMS static verification axis; Euler-Bernoulli cantilever surrogate",
        static_result.tip_deflection_m.abs(),
        static_result.analytical_tip_deflection_m.abs(),
        static_result.analytical_relative_error,
        1.0e-8,
        Some(static_result.solver_residual),
        Some(static_result.energy_balance_relative_error),
        "Assembled beam static equilibrium and energy check; not an official NAFEMS LE deck.",
    ));

    let mut modal_study = Beam2DStudy::introductory();
    modal_study.geometry.elements = 4;
    modal_study.modal_modes = 2;
    let modal_result = run_beam2d_modal(&modal_study)?;
    let beta = 1.875_104_068_711_961_f64;
    let modal_reference = beta.powi(2)
        * (modal_study.material.youngs_modulus_pa * modal_study.geometry.second_moment_m4()
            / (modal_study.material.density_kg_m3
                * modal_study.geometry.area_m2()
                * modal_study.geometry.length_m.powi(4)))
        .sqrt()
        / (2.0 * std::f64::consts::PI);
    let modal_computed = modal_result.frequencies_hz[0];
    let modal_error = relative_error(modal_computed, modal_reference);
    cases.push(case_report(
        "nafems-modal-cantilever-axis",
        NafemsBenchmarkKind::Modal,
        "NAFEMS modal verification axis; Euler-Bernoulli cantilever first-frequency surrogate",
        modal_computed,
        modal_reference,
        modal_error,
        1.0e-2,
        Some(modal_result.residual_norms.iter().copied().fold(0.0, f64::max)),
        Some(
            modal_result
                .mass_normalization_error
                .max(modal_result.orthogonality_error),
        ),
        "Assembled generalized eigenproblem with mass normalization; not an official NAFEMS modal deck.",
    ));

    let thermal_study = ThermoelasticStudy::introductory();
    let thermal_result = run_thermoelastic_static(&thermal_study)?;
    let thermal_reference = thermal_result.expected_corner_displacement_m[0]
        .hypot(thermal_result.expected_corner_displacement_m[1]);
    let thermal_computed = thermal_result.actual_corner_displacement_m[0]
        .hypot(thermal_result.actual_corner_displacement_m[1]);
    cases.push(case_report(
        "nafems-thermal-stress-free-expansion-axis",
        NafemsBenchmarkKind::ThermalStress,
        "NAFEMS thermal-stress verification axis; uniform eigenstrain surrogate",
        thermal_computed,
        thermal_reference,
        relative_error(thermal_computed, thermal_reference),
        1.0e-8,
        Some(thermal_result.solver_residual),
        Some(
            thermal_result
                .energy_balance_relative_error
                .max(thermal_result.max_affine_displacement_error_m),
        ),
        "Plane-stress free expansion with a pin; constrained thermal-stress cases require an independent deck.",
    ));

    let nonlinear_study = Hyperelastic3DStudy::introductory();
    let nonlinear_result = run_hyperelastic_3d(&nonlinear_study)?;
    cases.push(case_report(
        "nafems-nonlinear-extension-axis",
        NafemsBenchmarkKind::Nonlinear,
        "NAFEMS nonlinear verification axis; homogeneous Neo-Hookean extension surrogate",
        nonlinear_result.max_displacement_m,
        nonlinear_study.prescribed_extension_m,
        relative_error(
            nonlinear_result.max_displacement_m,
            nonlinear_study.prescribed_extension_m,
        ),
        5.0e-2,
        Some(nonlinear_result.final_residual),
        Some(
            nonlinear_result
                .relative_energy_error
                .max((nonlinear_result.taylor_remainder_ratio - 0.25).abs()),
        ),
        "Displacement-controlled total-Lagrangian Newton/Taylor check; not official post-buckling or NAFEMS NL parity.",
    ));

    Ok(NafemsBenchmarkSuite {
        schema_version: STRUCTURAL_VALIDATION_SCHEMA_VERSION,
        execution: StructuralExecutionProvenance::cpu_reference(),
        all_passed: cases.iter().all(|case| case.passed),
        cases,
        official_reference_data_loaded: false,
        applicability_limits: vec![
            "Native cases are analytical surrogates aligned to NAFEMS verification axes, not official NAFEMS result files.".to_string(),
            "Official static/modal/thermal/nonlinear deck parity requires versioned reference data and an explicitly enabled external run.".to_string(),
            "The thick-cylinder fixture below is analytical and does not replace an axisymmetric FEM convergence study.".to_string(),
        ],
    })
}

/// Thick-walled cylinder inputs for the Lamé analytical fixture.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThickWalledCylinderStudy {
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
    pub internal_pressure_pa: f64,
    pub external_pressure_pa: f64,
}

impl ThickWalledCylinderStudy {
    pub fn introductory() -> Self {
        Self {
            inner_radius_m: 0.05,
            outer_radius_m: 0.10,
            internal_pressure_pa: 2.0e6,
            external_pressure_pa: 0.25e6,
        }
    }

    fn validate(self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("cylinder.inner_radius_m", self.inner_radius_m),
            ("cylinder.outer_radius_m", self.outer_radius_m),
            ("cylinder.internal_pressure_pa", self.internal_pressure_pa),
            ("cylinder.external_pressure_pa", self.external_pressure_pa),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.inner_radius_m <= 0.0 || self.outer_radius_m <= self.inner_radius_m {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "cylinder.radii",
                value: format!("{} >= {}", self.inner_radius_m, self.outer_radius_m),
            });
        }
        Ok(())
    }
}

/// Lamé stresses at the inner and outer walls of a thick cylinder.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThickWalledCylinderResult {
    pub inner_radial_stress_pa: f64,
    pub outer_radial_stress_pa: f64,
    pub inner_hoop_stress_pa: f64,
    pub outer_hoop_stress_pa: f64,
    pub radial_boundary_error_pa: f64,
    pub hoop_stress_difference_pa: f64,
    pub converged: bool,
}

/// Evaluate the exact Lamé solution used by cylinder verification fixtures.
pub fn run_thick_walled_cylinder_analytical(
    study: &ThickWalledCylinderStudy,
) -> Result<ThickWalledCylinderResult, StructuralMechanicsError> {
    study.validate()?;
    let ri2 = study.inner_radius_m.powi(2);
    let ro2 = study.outer_radius_m.powi(2);
    let denominator = ro2 - ri2;
    let a = (study.internal_pressure_pa * ri2 - study.external_pressure_pa * ro2) / denominator;
    let b = ri2 * ro2 * (study.internal_pressure_pa - study.external_pressure_pa) / denominator;
    let inner_radial = a - b / ri2;
    let outer_radial = a - b / ro2;
    let inner_hoop = a + b / ri2;
    let outer_hoop = a + b / ro2;
    let radial_boundary_error = (inner_radial + study.internal_pressure_pa)
        .abs()
        .max((outer_radial + study.external_pressure_pa).abs());
    Ok(ThickWalledCylinderResult {
        inner_radial_stress_pa: inner_radial,
        outer_radial_stress_pa: outer_radial,
        inner_hoop_stress_pa: inner_hoop,
        outer_hoop_stress_pa: outer_hoop,
        radial_boundary_error_pa: radial_boundary_error,
        hoop_stress_difference_pa: inner_hoop - outer_hoop,
        converged: radial_boundary_error
            < 1.0e-8
                * study
                    .internal_pressure_pa
                    .max(study.external_pressure_pa)
                    .max(1.0),
    })
}

/// Assemble and solve a full-field affine plane-stress patch.
///
/// A P1 displacement field reproduces an affine field exactly on every
/// triangle. The test uses this property to catch element-local indexing,
/// global scatter, and Dirichlet-value elimination errors independently of a
/// beam or plate benchmark.
#[cfg(test)]
#[test]
fn affine_plane_stress_patch_test() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let material = IsotropicMaterial::new(2.0e6, 0.25);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let dofs = space.num_global_dofs(&mesh);
    let prescribed: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(node, point)| {
            let ux = 1.0e-3 * point.x + 2.0e-4 * point.y;
            let uy = -4.0e-4 * point.x + 3.0e-4 * point.y;
            [(2 * node, ux), (2 * node + 1, uy)]
        })
        .collect();
    let mut system = FemSystem::new(stiffness, vec![0.0; dofs]);
    apply_affine_dirichlet_values(&mut system, &prescribed);
    let factorization = factorize_sparse(
        &system.matrix.to_csr(),
        &SparseDirectConfig {
            matrix_class: MatrixClass::SymmetricPositiveDefinite,
            method: SparseDirectMethod::SimplicialCholesky,
            ordering: Ordering::Auto,
            ..SparseDirectConfig::default()
        },
        None,
    )
    .expect("affine patch matrix must factorize");
    let solved = factorization
        .solve(&system.rhs)
        .expect("affine patch system must solve");
    for (dof, expected) in prescribed {
        assert!((solved[dof] - expected).abs() < 1.0e-12);
    }
}

#[cfg(test)]
fn apply_affine_dirichlet_values(system: &mut FemSystem<f64>, prescribed: &[(usize, f64)]) {
    for index in 0..system.matrix.values.len() {
        let row = system.matrix.rows[index];
        let column = system.matrix.cols[index];
        if !prescribed.iter().any(|(dof, _)| *dof == row)
            && let Some((_, value)) = prescribed.iter().find(|(dof, _)| *dof == column)
        {
            system.rhs[row] -= system.matrix.values[index] * value;
        }
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for index in 0..system.matrix.values.len() {
        let row = system.matrix.rows[index];
        let column = system.matrix.cols[index];
        if !prescribed.iter().any(|(dof, _)| *dof == row)
            && !prescribed.iter().any(|(dof, _)| *dof == column)
        {
            rows.push(row);
            cols.push(column);
            values.push(system.matrix.values[index]);
        }
    }
    for &(dof, value) in prescribed {
        rows.push(dof);
        cols.push(dof);
        values.push(1.0);
        system.rhs[dof] = value;
        system.dirichlet_dofs.insert(dof);
    }
    system.matrix.rows = rows;
    system.matrix.cols = cols;
    system.matrix.values = values;
}

#[allow(clippy::too_many_arguments)]
fn case_report(
    id: &str,
    kind: NafemsBenchmarkKind,
    reference_source: &str,
    computed_value: f64,
    reference_value: f64,
    relative_error: f64,
    tolerance: f64,
    solver_residual: Option<f64>,
    energy_error: Option<f64>,
    applicability: &str,
) -> NafemsBenchmarkCase {
    let passed = computed_value.is_finite()
        && reference_value.is_finite()
        && relative_error.is_finite()
        && relative_error <= tolerance
        && solver_residual.is_none_or(|value| value.is_finite())
        && energy_error.is_none_or(|value| value.is_finite());
    NafemsBenchmarkCase {
        id: id.to_string(),
        kind,
        reference_kind: ValidationReferenceKind::AnalyticalSurrogate,
        reference_source: reference_source.to_string(),
        computed_value,
        reference_value,
        relative_error,
        tolerance,
        solver_residual,
        energy_error,
        passed,
        applicability: applicability.to_string(),
    }
}

fn relative_error(computed: f64, reference: f64) -> f64 {
    (computed - reference).abs() / reference.abs().max(1.0e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nafems_axis_catalogue_is_explicit_about_reference_provenance() {
        let suite = run_nafems_benchmark_suite().unwrap();
        assert!(suite.all_passed, "{suite:#?}");
        assert_eq!(suite.schema_version, STRUCTURAL_VALIDATION_SCHEMA_VERSION);
        assert_eq!(
            suite.execution,
            StructuralExecutionProvenance::cpu_reference()
        );
        assert_eq!(suite.cases.len(), 4);
        assert!(!suite.official_reference_data_loaded);
        assert!(!suite.official_parity_passed());
        assert!(suite.cases.iter().all(|case| {
            case.reference_kind == ValidationReferenceKind::AnalyticalSurrogate
                && case.passed
                && case.solver_residual.is_some()
        }));
    }

    #[test]
    fn thick_cylinder_lame_solution_satisfies_pressure_boundary_conditions() {
        let result =
            run_thick_walled_cylinder_analytical(&ThickWalledCylinderStudy::introductory())
                .unwrap();
        assert!(result.converged);
        assert!(result.radial_boundary_error_pa < 1.0e-8);
        assert!(result.inner_hoop_stress_pa > result.outer_hoop_stress_pa);
    }

    #[test]
    fn thick_cylinder_rejects_reversed_radii() {
        let mut study = ThickWalledCylinderStudy::introductory();
        study.outer_radius_m = study.inner_radius_m;
        assert!(matches!(
            run_thick_walled_cylinder_analytical(&study),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "cylinder.radii",
                ..
            })
        ));
    }
}
