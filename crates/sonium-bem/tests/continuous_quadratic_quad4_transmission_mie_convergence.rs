//! Independent Mie convergence for continuous quadratic Quad4 transmission.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::solver::transmission::solve_incident_transmission;
use sonium_bem::core::types::PhysicsParams;
use sonium_wave::analytical::FluidSphereTransmission3d;
use std::f64::consts::PI;

const RADIUS: f64 = 0.2;
const FREQUENCY: f64 = 420.0;
const EXTERIOR_SPEED: f64 = 343.0;
const EXTERIOR_DENSITY: f64 = 1.21;
const INTERIOR_SPEED: f64 = 1_480.0;
const INTERIOR_DENSITY: f64 = 1_000.0;
const EVALUATION_RADIUS: f64 = 0.8;

#[derive(Debug, Clone, Copy)]
pub struct ContinuousQuadraticQuad4TransmissionMetrics {
    pub coarse_error: f64,
    pub fine_error: f64,
    pub coarse_residual: f64,
    pub fine_residual: f64,
    pub coarse_solution_error: f64,
    pub fine_solution_error: f64,
    pub coarse_dofs: usize,
    pub fine_dofs: usize,
    pub coarse_panels: usize,
    pub fine_panels: usize,
}

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::new(FREQUENCY, EXTERIOR_SPEED, EXTERIOR_DENSITY, false),
        PhysicsParams::new(FREQUENCY, INTERIOR_SPEED, INTERIOR_DENSITY, true),
    )
}

fn relative_error(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn solve_level(face_resolution: usize, reference: &[Complex64]) -> (f64, f64, f64, usize, usize) {
    let mesh = generate_cubed_sphere_quad_mesh(RADIUS, face_resolution);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("continuous Quad4 transmission assembly");
    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, 1.0e-8)
        .expect("continuous Quad4 transmission solve");
    let points = [
        [
            EVALUATION_RADIUS * 0.2_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 0.2_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 0.55_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 0.55_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 0.9_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 0.9_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 1.25_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 1.25_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 1.6_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 1.6_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 2.0_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 2.0_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 2.4_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 2.4_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 2.8_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 2.8_f64.cos(),
        ],
    ];
    let fields = solution
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .expect("continuous Quad4 exterior field");
    let actual = fields.iter().map(|field| field.p_total).collect::<Vec<_>>();
    let field_error = relative_error(&actual, reference);

    let rhs = system
        .incident_rhs(&incident)
        .expect("continuous Quad4 incident RHS");
    let direct = sonium_solvers::direct::lu_solve(system.matrix(), &rhs)
        .expect("independent continuous Quad4 direct solve");
    let solution_error = relative_error(solution.values(), &direct);
    (
        field_error,
        solution.relative_residual,
        solution_error,
        system.interface_dofs(),
        mesh.elements.len(),
    )
}

pub fn run_continuous_quadratic_quad4_transmission_mie_convergence()
-> ContinuousQuadraticQuad4TransmissionMetrics {
    let exterior_wave_number = 2.0 * PI * FREQUENCY / EXTERIOR_SPEED;
    let interior_wave_number = 2.0 * PI * FREQUENCY / INTERIOR_SPEED;
    let mie = FluidSphereTransmission3d::try_new(
        RADIUS,
        exterior_wave_number,
        interior_wave_number,
        EXTERIOR_DENSITY,
        INTERIOR_DENSITY,
        32,
    )
    .expect("fluid-sphere Mie reference");
    let angles = [0.2, 0.55, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8];
    let reference = angles
        .iter()
        .map(|&theta| {
            mie.exterior_pressure(EVALUATION_RADIUS, theta)
                .expect("Mie exterior reference")
        })
        .collect::<Vec<_>>();

    let (coarse_error, coarse_residual, coarse_solution_error, coarse_dofs, coarse_panels) =
        solve_level(1, &reference);
    let (fine_error, fine_residual, fine_solution_error, fine_dofs, fine_panels) =
        solve_level(2, &reference);
    assert!(fine_error < coarse_error);
    assert!(fine_error <= 0.2);
    assert!(fine_residual <= 1.0e-8);
    assert!(fine_solution_error <= 1.0e-8);
    assert!(fine_dofs > coarse_dofs);
    assert!(fine_panels > coarse_panels);

    ContinuousQuadraticQuad4TransmissionMetrics {
        coarse_error,
        fine_error,
        coarse_residual,
        fine_residual,
        coarse_solution_error,
        fine_solution_error,
        coarse_dofs,
        fine_dofs,
        coarse_panels,
        fine_panels,
    }
}

#[test]
fn continuous_quadratic_quad4_transmission_converges_to_mie() {
    let metrics = run_continuous_quadratic_quad4_transmission_mie_convergence();
    assert!(metrics.fine_error <= 0.2);
    assert!(metrics.fine_residual <= 1.0e-8);
    assert!(metrics.fine_solution_error <= 1.0e-8);
}
