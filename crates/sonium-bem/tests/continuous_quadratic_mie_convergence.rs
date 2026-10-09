//! Physical convergence gate for the globally continuous quadratic CPU path.
//!
//! Unlike the topology-only continuous-Q2 fixtures, this test solves two
//! closed rigid-sphere meshes through production TBEM, evaluates the exterior
//! field, and compares it with an independent spherical-Mie reference.

use num_complex::Complex64;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolution, BemSolver, IncidentField, SolverMethod,
};
use sonium_wave::analytical::sphere_scattering_3d;
use std::collections::BTreeSet;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.1;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.35;
const COARSE_N_THETA: usize = 3;
const COARSE_N_PHI: usize = 6;
const FINE_N_THETA: usize = 4;
const FINE_N_PHI: usize = 8;
const EVALUATION_RADIUS_M: f64 = 1.2;
const MIE_TERMS: usize = 32;

// This is a deliberately small, low-ka CPU release slice. The limits are not
// intended as universal high-frequency BEM acceptance tolerances.
const MAX_FINE_L2_ERROR: f64 = 0.01;
const MAX_FINE_POINTWISE_ERROR: f64 = 0.02;
const MAX_AXISYMMETRY_ERROR: f64 = 0.01;
const MAX_REFINEMENT_ERROR_RATIO: f64 = 0.85;

#[derive(Debug, Clone, Copy)]
pub struct ContinuousQuadraticMieMetrics {
    pub coarse_l2_error: f64,
    pub coarse_pointwise_error: f64,
    pub fine_l2_error: f64,
    pub fine_pointwise_error: f64,
    pub axisymmetry_error: f64,
    pub coarse_dofs: usize,
    pub fine_dofs: usize,
}

fn frequency_hz() -> f64 {
    KA * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M)
}

fn solve(n_theta: usize, n_phi: usize) -> BemSolution {
    let problem = BemProblem::try_rigid_sphere_scattering_custom(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
        n_theta,
        n_phi,
    )
    .expect("checked sphere problem admission")
    .with_incident_field(IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0));

    BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .with_max_iterations(500)
        .with_tolerance(1.0e-9)
        .solve(&problem)
        .expect("continuous-Q2 production TBEM solve")
}

fn polar_angles() -> Vec<f64> {
    vec![0.25, 0.7, 1.15, 1.6, 2.05, 2.5, 2.9]
}

fn evaluation_points() -> Vec<[f64; 3]> {
    polar_angles()
        .into_iter()
        .map(|theta| {
            [
                EVALUATION_RADIUS_M * theta.sin(),
                0.0,
                EVALUATION_RADIUS_M * theta.cos(),
            ]
        })
        .collect()
}

fn mie_reference() -> Vec<Complex64> {
    sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        MIE_TERMS,
        vec![EVALUATION_RADIUS_M],
        polar_angles(),
    )
    .pressure
}

fn relative_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

fn field_errors(solution: &BemSolution, reference: &[Complex64]) -> (f64, f64) {
    let actual: Vec<_> = solution
        .evaluate_pressure_field(&evaluation_points())
        .into_iter()
        .map(|field| field.p_total)
        .collect();
    let reference_norm = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let l2_error = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| (*actual - *reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm.max(f64::MIN_POSITIVE);
    let pointwise_error = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| relative_error(actual, reference))
        .fold(0.0, f64::max);
    (l2_error, pointwise_error)
}

fn unique_solution_dofs(solution: &BemSolution) -> usize {
    solution
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .flat_map(|element| element.dof_addresses.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
}

pub fn run_continuous_quadratic_mie_convergence() -> ContinuousQuadraticMieMetrics {
    let reference = mie_reference();
    let coarse = solve(COARSE_N_THETA, COARSE_N_PHI);
    let fine = solve(FINE_N_THETA, FINE_N_PHI);

    let coarse_dofs = unique_solution_dofs(&coarse);
    let fine_dofs = unique_solution_dofs(&fine);
    assert_eq!(coarse_dofs, coarse.num_dofs());
    assert_eq!(fine_dofs, fine.num_dofs());
    assert!(
        fine_dofs > coarse_dofs,
        "refinement must add global Q2 DOFs"
    );
    assert!(
        fine.surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "continuous-Q2 solve produced non-finite Cauchy data"
    );

    let (coarse_l2_error, coarse_pointwise_error) = field_errors(&coarse, &reference);
    let (fine_l2_error, fine_pointwise_error) = field_errors(&fine, &reference);

    let theta = 0.82_f64;
    let z = EVALUATION_RADIUS_M * theta.cos();
    let radius_xy = EVALUATION_RADIUS_M * theta.sin();
    let symmetry_points = [
        [radius_xy, 0.0, z],
        [0.0, radius_xy, z],
        [-radius_xy, 0.0, z],
        [0.0, -radius_xy, z],
    ];
    let symmetry_fields = fine.evaluate_pressure_field(&symmetry_points);
    let axisymmetry_error = symmetry_fields
        .iter()
        .skip(1)
        .map(|field| relative_error(field.p_total, symmetry_fields[0].p_total))
        .fold(0.0, f64::max);

    assert!(
        fine_l2_error / coarse_l2_error <= MAX_REFINEMENT_ERROR_RATIO,
        "continuous-Q2 L2 field error did not contract enough: coarse={coarse_l2_error:.3e}, fine={fine_l2_error:.3e}"
    );
    assert!(
        fine_pointwise_error / coarse_pointwise_error <= MAX_REFINEMENT_ERROR_RATIO,
        "continuous-Q2 maximum field error did not contract enough: coarse={coarse_pointwise_error:.3e}, fine={fine_pointwise_error:.3e}"
    );
    assert!(
        fine_l2_error <= MAX_FINE_L2_ERROR,
        "continuous-Q2 fine L2 error {fine_l2_error:.3e} exceeds {MAX_FINE_L2_ERROR:.3e}"
    );
    assert!(
        fine_pointwise_error <= MAX_FINE_POINTWISE_ERROR,
        "continuous-Q2 fine pointwise error {fine_pointwise_error:.3e} exceeds {MAX_FINE_POINTWISE_ERROR:.3e}"
    );
    assert!(
        axisymmetry_error <= MAX_AXISYMMETRY_ERROR,
        "continuous-Q2 rotational-invariance error {axisymmetry_error:.3e} exceeds {MAX_AXISYMMETRY_ERROR:.3e}"
    );

    ContinuousQuadraticMieMetrics {
        coarse_l2_error,
        coarse_pointwise_error,
        fine_l2_error,
        fine_pointwise_error,
        axisymmetry_error,
        coarse_dofs,
        fine_dofs,
    }
}

#[test]
fn continuous_quadratic_sphere_converges_to_independent_mie_field() {
    let metrics = run_continuous_quadratic_mie_convergence();
    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.continuous-quadratic-mie-convergence.v1\",\"status\":\"passed\",\"coarse_l2_error\":{:.6e},\"coarse_pointwise_error\":{:.6e},\"fine_l2_error\":{:.6e},\"fine_pointwise_error\":{:.6e},\"axisymmetry_error\":{:.6e},\"coarse_dofs\":{},\"fine_dofs\":{}}}",
        metrics.coarse_l2_error,
        metrics.coarse_pointwise_error,
        metrics.fine_l2_error,
        metrics.fine_pointwise_error,
        metrics.axisymmetry_error,
        metrics.coarse_dofs,
        metrics.fine_dofs,
    );
}
