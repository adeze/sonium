//! Focused CPU evidence for the production constant-panel far-field path.

use num_complex::Complex64;
use sonium_bem::core::postprocess::compute_constant_panel_far_field_checked;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolution, BemSolver, IncidentField, SolverMethod,
};

const RADIUS_M: f64 = 0.2;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.5;
const MAX_RECIPROCITY_RELATIVE_ERROR: f64 = 2.0e-2;

fn frequency_hz() -> f64 {
    KA * SPEED_OF_SOUND_M_PER_S / (2.0 * std::f64::consts::PI * RADIUS_M)
}

fn solve(direction: [f64; 3]) -> Result<BemSolution, String> {
    let problem = BemProblem::try_rigid_sphere_scattering(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
    )
    .map_err(|error| format!("sphere admission failed: {error:?}"))?
    .with_incident_field(IncidentField::plane_wave(direction, 1.0));

    BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::BiCgStab)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve(&problem)
        .map_err(|error| format!("sphere solve failed: {error:?}"))
}

fn relative_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

#[test]
fn rigid_sphere_far_field_obeys_reciprocity() {
    let forward = solve([0.0, 0.0, 1.0]).expect("forward production solve");
    let reverse = solve([0.0, 0.0, -1.0]).expect("reverse production solve");

    let forward_amplitude = compute_constant_panel_far_field_checked(
        &[[0.0, 0.0, -1.0]],
        &forward.elements,
        &forward.nodes,
        &forward.surface_pressure,
        None,
        &forward.physics,
    )
    .expect("forward far-field evaluation")[0];
    let reverse_amplitude = compute_constant_panel_far_field_checked(
        &[[0.0, 0.0, 1.0]],
        &reverse.elements,
        &reverse.nodes,
        &reverse.surface_pressure,
        None,
        &reverse.physics,
    )
    .expect("reverse far-field evaluation")[0];

    let error = relative_error(forward_amplitude, reverse_amplitude);
    assert!(
        error.is_finite() && error <= MAX_RECIPROCITY_RELATIVE_ERROR,
        "far-field reciprocity error {error:.3e} exceeds {MAX_RECIPROCITY_RELATIVE_ERROR:.3e}"
    );
    assert!(forward_amplitude.norm() > f64::MIN_POSITIVE);
}

#[test]
fn far_field_checked_path_rejects_malformed_inputs() {
    let problem = BemProblem::try_rigid_sphere_scattering(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
    )
    .expect("sphere admission");

    let invalid_direction = compute_constant_panel_far_field_checked(
        &[[0.0, 0.0, 0.0]],
        &problem.mesh.elements,
        &problem.mesh.nodes,
        &[],
        None,
        &problem.physics,
    )
    .expect_err("zero direction must be rejected");
    assert!(matches!(
        invalid_direction,
        sonium_bem::core::postprocess::FarFieldError::InvalidDirection { index: 0 }
    ));

    let pressure_mismatch = compute_constant_panel_far_field_checked(
        &[[0.0, 0.0, 1.0]],
        &problem.mesh.elements,
        &problem.mesh.nodes,
        &[],
        None,
        &problem.physics,
    )
    .expect_err("pressure count must be checked");
    assert!(matches!(
        pressure_mismatch,
        sonium_bem::core::postprocess::FarFieldError::SurfacePressureCount { .. }
    ));
}
