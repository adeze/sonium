//! Independent physical convergence reference for the affine-Tri3 Galerkin
//! Müller transmission path.
//!
//! The operator/solve contract is useful on its own, but a transmission
//! implementation also needs an independent field reference.  This fixture
//! solves the same lossless fluid sphere on two conforming icosphere levels
//! and compares checked exterior pressure samples with the independent Mie
//! series from `sonium-wave`.

use num_complex::Complex64;
use sonium_bem::core::assembly::assemble_galerkin_muller_transmission_with_global_dofs;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{BasisOrder, PhysicsParams};
use sonium_wave::analytical::FluidSphereTransmission3d;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.2;
const EXTERIOR_SOUND_SPEED_M_S: f64 = 343.0;
const EXTERIOR_DENSITY_KG_M3: f64 = 1.21;
const INTERIOR_SOUND_SPEED_M_S: f64 = 240.0;
const INTERIOR_DENSITY_KG_M3: f64 = 2.4;
const EXTERIOR_KA: f64 = 0.7;
const EVALUATION_RADIUS_M: f64 = 1.0;
const MIE_TERMS: usize = 32;
const QUADRATURE_ORDER: usize = 3;
const SOLVE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy)]
pub struct GalerkinMullerMieMetrics {
    pub coarse_l2_error: f64,
    pub coarse_pointwise_error: f64,
    pub fine_l2_error: f64,
    pub fine_pointwise_error: f64,
    pub coarse_residual: f64,
    pub fine_residual: f64,
    pub coarse_panels: usize,
    pub fine_panels: usize,
    pub coarse_dofs: usize,
    pub fine_dofs: usize,
    pub maximum_fine_field_magnitude: f64,
}

struct LevelResult {
    field: Vec<Complex64>,
    residual: f64,
    panels: usize,
    dofs: usize,
}

fn frequency_hz() -> f64 {
    EXTERIOR_KA * EXTERIOR_SOUND_SPEED_M_S / (2.0 * PI * RADIUS_M)
}

fn evaluation_angles() -> [f64; 8] {
    [0.2, 0.55, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8]
}

fn point(radius: f64, theta: f64) -> [f64; 3] {
    [radius * theta.sin(), 0.0, radius * theta.cos()]
}

fn conforming_linear_mesh(
    subdivisions: usize,
) -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>, usize) {
    let mut mesh = generate_icosphere_mesh(RADIUS_M, subdivisions);
    let mut unique_nodes = mesh
        .elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();

    for element in &mut mesh.elements {
        element.dof_addresses = element
            .connectivity
            .iter()
            .map(|node| {
                unique_nodes
                    .binary_search(node)
                    .expect("connectivity node must be in the mesh")
            })
            .collect();
    }

    (mesh.elements, mesh.nodes, unique_nodes.len())
}

fn relative_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

fn field_errors(actual: &[Complex64], reference: &[Complex64]) -> (f64, f64) {
    let reference_norm = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let l2_error = actual
        .iter()
        .zip(reference)
        .map(|(&value, &expected)| (value - expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm;
    let pointwise_error = actual
        .iter()
        .zip(reference)
        .map(|(&value, &expected)| relative_error(value, expected))
        .fold(0.0, f64::max);
    (l2_error, pointwise_error)
}

fn solve_level(
    subdivisions: usize,
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
) -> LevelResult {
    let (elements, nodes, dofs) = conforming_linear_mesh(subdivisions);
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        exterior,
        interior,
        BasisOrder::Linear,
        QUADRATURE_ORDER,
        dofs,
    )
    .expect("Galerkin Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let rhs = operators
        .incident_rhs_from_field(&elements, &nodes, &incident, exterior, QUADRATURE_ORDER)
        .expect("projected incident Galerkin RHS");
    let solution = operators
        .solve(&rhs, SOLVE_TOLERANCE)
        .expect("checked Galerkin Müller solve");
    let points = evaluation_angles()
        .into_iter()
        .map(|theta| point(EVALUATION_RADIUS_M, theta))
        .collect::<Vec<_>>();
    let field = operators
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &solution,
            &points,
            &incident,
            exterior,
            QUADRATURE_ORDER,
        )
        .expect("checked Galerkin exterior field");
    let residual_norm = (0..operators.total_dofs())
        .map(|row| {
            let action = (0..operators.total_dofs())
                .map(|column| operators.matrix()[[row, column]] * solution[column])
                .sum::<Complex64>();
            (action - rhs[row]).norm_sqr()
        })
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();

    LevelResult {
        field: field.into_iter().map(|value| value.p_total).collect(),
        residual: residual_norm / rhs_norm.max(f64::MIN_POSITIVE),
        panels: elements.len(),
        dofs,
    }
}

pub fn run_galerkin_muller_mie_convergence() -> GalerkinMullerMieMetrics {
    let frequency = frequency_hz();
    let exterior_wave_number = EXTERIOR_KA / RADIUS_M;
    let interior_wave_number = 2.0 * PI * frequency / INTERIOR_SOUND_SPEED_M_S;
    let mie = FluidSphereTransmission3d::try_new(
        RADIUS_M,
        exterior_wave_number,
        interior_wave_number,
        EXTERIOR_DENSITY_KG_M3,
        INTERIOR_DENSITY_KG_M3,
        MIE_TERMS,
    )
    .expect("valid fluid-sphere Mie reference");
    let exterior = PhysicsParams::try_new(
        frequency,
        EXTERIOR_SOUND_SPEED_M_S,
        EXTERIOR_DENSITY_KG_M3,
        false,
    )
    .expect("valid exterior physics");
    let interior = PhysicsParams::try_new(
        frequency,
        INTERIOR_SOUND_SPEED_M_S,
        INTERIOR_DENSITY_KG_M3,
        false,
    )
    .expect("valid interior physics");

    let angles = evaluation_angles();
    let reference = angles
        .into_iter()
        .map(|theta| {
            mie.exterior_pressure(EVALUATION_RADIUS_M, theta)
                .expect("valid exterior Mie sample")
        })
        .collect::<Vec<_>>();
    let coarse = solve_level(0, &exterior, &interior);
    let fine = solve_level(1, &exterior, &interior);
    let (coarse_l2_error, coarse_pointwise_error) = field_errors(&coarse.field, &reference);
    let (fine_l2_error, fine_pointwise_error) = field_errors(&fine.field, &reference);

    GalerkinMullerMieMetrics {
        coarse_l2_error,
        coarse_pointwise_error,
        fine_l2_error,
        fine_pointwise_error,
        coarse_residual: coarse.residual,
        fine_residual: fine.residual,
        coarse_panels: coarse.panels,
        fine_panels: fine.panels,
        coarse_dofs: coarse.dofs,
        fine_dofs: fine.dofs,
        maximum_fine_field_magnitude: fine
            .field
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max),
    }
}

#[test]
fn galerkin_muller_converges_to_independent_mie_field() {
    let metrics = run_galerkin_muller_mie_convergence();
    assert!(metrics.coarse_l2_error.is_finite());
    assert!(metrics.fine_l2_error.is_finite());
    assert!(metrics.coarse_pointwise_error.is_finite());
    assert!(metrics.fine_pointwise_error.is_finite());
    assert!(metrics.fine_residual <= SOLVE_TOLERANCE);
    assert!(metrics.fine_panels > metrics.coarse_panels);
    assert!(metrics.fine_dofs > metrics.coarse_dofs);
    assert!(metrics.maximum_fine_field_magnitude > 0.0);
    assert!(
        metrics.fine_l2_error <= metrics.coarse_l2_error * 1.05,
        "Galerkin Mie L2 error did not refine: coarse={:.3e}, fine={:.3e}",
        metrics.coarse_l2_error,
        metrics.fine_l2_error
    );
    assert!(
        metrics.fine_pointwise_error <= metrics.coarse_pointwise_error * 1.10,
        "Galerkin Mie pointwise error did not refine: coarse={:.3e}, fine={:.3e}",
        metrics.coarse_pointwise_error,
        metrics.fine_pointwise_error
    );
}
