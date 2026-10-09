//! Physical Mie convergence for globally continuous Q2 Quad4 panels.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolution, BemSolver, IncidentField, PhysicsParams, SolverMethod,
};
use sonium_wave::analytical::sphere_scattering_3d;
use std::collections::BTreeSet;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.1;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.35;
const EVALUATION_RADIUS_M: f64 = 1.2;
const MIE_TERMS: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct ContinuousQuadraticQuad4MieMetrics {
    pub coarse_l2_error: f64,
    pub coarse_pointwise_error: f64,
    pub fine_l2_error: f64,
    pub fine_pointwise_error: f64,
    pub axisymmetry_error: f64,
    pub coarse_dofs: usize,
    pub fine_dofs: usize,
    pub coarse_panels: usize,
    pub fine_panels: usize,
}

fn frequency_hz() -> f64 {
    KA * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M)
}

fn solve(face_resolution: usize) -> BemSolution {
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(RADIUS_M, face_resolution),
        physics: PhysicsParams::try_new(
            frequency_hz(),
            SPEED_OF_SOUND_M_PER_S,
            DENSITY_KG_PER_M3,
            false,
        )
        .expect("checked sphere physics"),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    problem.validate().expect("checked cubed-sphere problem");
    BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .with_tolerance(1.0e-9)
        .solve(&problem)
        .expect("continuous-Q2 Quad4 production solve")
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
    let l2 = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| (actual - reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm.max(f64::MIN_POSITIVE);
    let pointwise = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| relative_error(actual, reference))
        .fold(0.0, f64::max);
    (l2, pointwise)
}

fn global_dofs(solution: &BemSolution) -> usize {
    solution
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .flat_map(|element| element.dof_addresses.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
}

pub fn run_continuous_quadratic_quad4_mie_convergence() -> ContinuousQuadraticQuad4MieMetrics {
    let reference = sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        MIE_TERMS,
        vec![EVALUATION_RADIUS_M],
        polar_angles(),
    )
    .pressure;
    let coarse = solve(2);
    let fine = solve(3);
    let (coarse_l2_error, coarse_pointwise_error) = field_errors(&coarse, &reference);
    let (fine_l2_error, fine_pointwise_error) = field_errors(&fine, &reference);
    let coarse_dofs = global_dofs(&coarse);
    let fine_dofs = global_dofs(&fine);

    assert_eq!(coarse_dofs, coarse.num_dofs());
    assert_eq!(fine_dofs, fine.num_dofs());
    assert_eq!(coarse_dofs, 98);
    assert_eq!(fine_dofs, 218);
    assert!(fine_l2_error / coarse_l2_error <= 0.85);
    assert!(fine_pointwise_error / coarse_pointwise_error <= 0.85);

    let theta = 0.82_f64;
    let z = EVALUATION_RADIUS_M * theta.cos();
    let radial = EVALUATION_RADIUS_M * theta.sin();
    let symmetry_points = [
        [radial, 0.0, z],
        [0.0, radial, z],
        [-radial, 0.0, z],
        [0.0, -radial, z],
    ];
    let symmetry_fields = fine.evaluate_pressure_field(&symmetry_points);
    let axisymmetry_error = symmetry_fields
        .iter()
        .skip(1)
        .map(|field| relative_error(field.p_total, symmetry_fields[0].p_total))
        .fold(0.0, f64::max);

    assert!(fine_l2_error <= 0.01);
    assert!(fine_pointwise_error <= 0.02);
    assert!(axisymmetry_error <= 0.01);

    ContinuousQuadraticQuad4MieMetrics {
        coarse_l2_error,
        coarse_pointwise_error,
        fine_l2_error,
        fine_pointwise_error,
        axisymmetry_error,
        coarse_dofs,
        fine_dofs,
        coarse_panels: coarse.elements.len(),
        fine_panels: fine.elements.len(),
    }
}

#[test]
fn continuous_quadratic_quad4_sphere_converges_to_mie() {
    let metrics = run_continuous_quadratic_quad4_mie_convergence();
    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.continuous-quadratic-quad4-mie-convergence.v1\",\"status\":\"passed\",\"coarse_l2_error\":{:.6e},\"coarse_pointwise_error\":{:.6e},\"fine_l2_error\":{:.6e},\"fine_pointwise_error\":{:.6e},\"axisymmetry_error\":{:.6e},\"coarse_dofs\":{},\"fine_dofs\":{},\"coarse_panels\":{},\"fine_panels\":{}}}",
        metrics.coarse_l2_error,
        metrics.coarse_pointwise_error,
        metrics.fine_l2_error,
        metrics.fine_pointwise_error,
        metrics.axisymmetry_error,
        metrics.coarse_dofs,
        metrics.fine_dofs,
        metrics.coarse_panels,
        metrics.fine_panels,
    );
}
