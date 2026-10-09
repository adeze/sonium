//! Independent regular-versus-Burton--Miller formulation reference.
//!
//! Both formulations solve the same rigid-sphere boundary problem.  The
//! production fields are compared against the independent Mie-series total
//! pressure at several exterior angles, and against each other.  This is a
//! bounded CPU equivalence gate; it does not claim equivalence for every
//! geometry, backend, or discretisation order.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolution, BemSolver, IncidentField, SolverMethod,
};
use sonium_wave::analytical::sphere_scattering_3d;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.2;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.5;
const MAX_REFERENCE_RELATIVE_ERROR: f64 = 1.5e-1;
const MAX_FORMULATION_RELATIVE_ERROR: f64 = 8.0e-2;

fn frequency_hz() -> f64 {
    KA * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M)
}

fn solve(use_burton_miller: bool) -> BemSolution {
    let problem = BemProblem::try_rigid_sphere_scattering(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
    )
    .expect("rigid-sphere problem admission")
    .with_incident_field(IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0))
    .with_burton_miller(use_burton_miller);

    BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve(&problem)
        .expect("production sphere solve")
}

fn relative_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / reference.norm().max(1.0e-12)
}

pub fn run_regular_and_burton_miller_reference() {
    let regular = solve(false);
    let burton_miller = solve(true);
    let eval_radius = (12.0 * RADIUS_M).max(RADIUS_M + 1.0);
    let angles = [0.3_f64, 1.0, 2.0, 2.8];
    let mie = sphere_scattering_3d(
        regular.physics.wave_number,
        RADIUS_M,
        40,
        vec![eval_radius],
        angles.to_vec(),
    );

    let mut max_reference_error = 0.0_f64;
    let mut max_formulation_error = 0.0_f64;
    for (index, &theta) in angles.iter().enumerate() {
        let point = [eval_radius * theta.sin(), 0.0, eval_radius * theta.cos()];
        let regular_pressure = regular.evaluate_pressure(&point);
        let burton_pressure = burton_miller.evaluate_pressure(&point);
        let reference = mie.pressure[index];
        assert!(regular_pressure.re.is_finite() && regular_pressure.im.is_finite());
        assert!(burton_pressure.re.is_finite() && burton_pressure.im.is_finite());
        assert!(reference.re.is_finite() && reference.im.is_finite());
        max_reference_error = max_reference_error.max(relative_error(regular_pressure, reference));
        max_reference_error = max_reference_error.max(relative_error(burton_pressure, reference));
        max_formulation_error =
            max_formulation_error.max(relative_error(regular_pressure, burton_pressure));
    }

    assert!(
        max_reference_error <= MAX_REFERENCE_RELATIVE_ERROR,
        "regular/Burton-Miller Mie error {max_reference_error:.3e} exceeds {MAX_REFERENCE_RELATIVE_ERROR:.3e}"
    );
    assert!(
        max_formulation_error <= MAX_FORMULATION_RELATIVE_ERROR,
        "regular/Burton-Miller formulation error {max_formulation_error:.3e} exceeds {MAX_FORMULATION_RELATIVE_ERROR:.3e}"
    );
}

#[cfg_attr(test, test)]
pub fn regular_and_burton_miller_match_independent_mie_reference() {
    run_regular_and_burton_miller_reference();
}
