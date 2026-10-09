//! Bounded higher-order and formulation-equivalence evidence for the CPU BEM path.
//!
//! The production linear basis is discontinuous per Tri3 panel, so this test
//! treats it as a higher-order *panel* contract rather than claiming a global
//! continuous finite-element space.  Two resolutions are compared with an
//! independent spherical Mie field.  The fine linear solve is also repeated
//! with the conventional formulation to check that the higher-order path does
//! not change the physical field when Burton--Miller stabilization is toggled.

use num_complex::Complex64;
use sonium_bem::core::postprocess::compute_total_field_checked;
use sonium_bem::core::{BasisOrder, BemProblem, BemSolution, BemSolver, IncidentField};
use sonium_wave::analytical::sphere_scattering_3d;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.1;
const SPEED_OF_SOUND_M_PER_S: f64 = 343.0;
const DENSITY_KG_PER_M3: f64 = 1.21;
const KA: f64 = 0.35;
const COARSE_N_THETA: usize = 4;
const COARSE_N_PHI: usize = 8;
const FINE_N_THETA: usize = 5;
const FINE_N_PHI: usize = 10;
const EVALUATION_RADIUS_M: f64 = 1.2;
const MIE_TERMS: usize = 32;

// These limits are deliberately tied to this small, low-ka reference slice;
// they are not universal BEM acceptance limits.
const MAX_FINE_LINEAR_L2_ERROR: f64 = 0.30;
const MAX_FINE_LINEAR_POINTWISE_ERROR: f64 = 0.45;
const MAX_FINE_FORMULATION_RELATIVE_ERROR: f64 = 0.20;
const MAX_AXISYMMETRY_RELATIVE_ERROR: f64 = 0.20;

fn frequency_hz() -> f64 {
    KA * SPEED_OF_SOUND_M_PER_S / (2.0 * PI * RADIUS_M)
}

fn xem_pressure_exchange() -> sonium_xem::MultiphysicsProblem {
    let mut exchange = sonium_xem::MultiphysicsProblem {
        id: "bem-linear-reference".into(),
        domains: vec![sonium_xem::Domain {
            id: "acoustics".into(),
            label: "Acoustics".into(),
            kind: sonium_xem::DomainKind::Acoustics,
        }],
        fields: vec![sonium_xem::FieldSpec {
            id: "pressure".into(),
            name: "surface pressure".into(),
            domain_id: "acoustics".into(),
            kind: sonium_xem::FieldKind::ComplexScalar,
            units: Some("Pa".into()),
            layout: sonium_xem::TensorLayout {
                scalar_type: sonium_xem::ScalarType::ComplexF64,
                shape: vec![1],
                strides: vec![1],
                ..Default::default()
            },
        }],
        ..Default::default()
    };
    // The Mie reference uses the canonical outgoing spatial factor
    // `exp(+ikr)`, which is the XEM `exp(-iωt)` convention. Carrying it
    // through XEM ensures the test exercises exchange admission rather than
    // bypassing it.
    exchange.conventions.harmonic_time = sonium_xem::HarmonicConvention::ExpMinusIwt;
    exchange
}

fn solve(
    n_theta: usize,
    n_phi: usize,
    basis_order: BasisOrder,
    use_burton_miller: bool,
) -> BemSolution {
    let base = BemProblem::try_rigid_sphere_scattering_custom(
        RADIUS_M,
        frequency_hz(),
        SPEED_OF_SOUND_M_PER_S,
        DENSITY_KG_PER_M3,
        n_theta,
        n_phi,
    )
    .expect("checked sphere problem admission")
    .with_incident_field(IncidentField::plane_wave([0.0, 0.0, 1.0], 1.0));
    let problem = BemProblem::from_xem(
        &xem_pressure_exchange(),
        "pressure",
        base.mesh,
        base.physics,
        base.incident_field,
        base.bc_type,
        use_burton_miller,
    )
    .expect("checked XEM/BEM problem admission");

    BemSolver::new()
        .with_basis_order(basis_order)
        .with_max_iterations(500)
        .with_tolerance(1.0e-8)
        .solve(&problem)
        .expect("production BEM solve")
}

fn evaluation_points() -> Vec<[f64; 3]> {
    [0.25_f64, 0.7, 1.15, 1.6, 2.05, 2.5, 2.9]
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
    let theta: Vec<f64> = [0.25_f64, 0.7, 1.15, 1.6, 2.05, 2.5, 2.9]
        .into_iter()
        .collect();
    sphere_scattering_3d(
        KA / RADIUS_M,
        RADIUS_M,
        MIE_TERMS,
        vec![EVALUATION_RADIUS_M],
        theta,
    )
    .pressure
}

fn relative_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

fn field_errors(solution: &BemSolution, reference: &[Complex64]) -> (f64, f64) {
    let points = evaluation_points();
    let actual: Vec<Complex64> = compute_total_field_checked(
        &points,
        &solution.elements,
        &solution.nodes,
        &solution.surface_pressure,
        None,
        &solution.incident_field,
        &solution.physics,
        solution.basis_order,
    )
    .expect("checked production linear field evaluation")
    .into_iter()
    .map(|field| field.p_total)
    .collect();
    assert_eq!(actual.len(), reference.len());
    assert!(
        actual
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        reference
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let reference_norm = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let l2_error = actual
        .iter()
        .zip(reference.iter())
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm.max(f64::MIN_POSITIVE);
    let pointwise_error = actual
        .iter()
        .zip(reference.iter())
        .map(|(value, expected)| relative_error(*value, *expected))
        .fold(0.0, f64::max);
    (l2_error, pointwise_error)
}

fn max_field_difference(first: &BemSolution, second: &BemSolution) -> f64 {
    let points = evaluation_points();
    first
        .evaluate_pressure_field(&points)
        .into_iter()
        .zip(second.evaluate_pressure_field(&points))
        .map(|(left, right)| relative_error(left.p_total, right.p_total))
        .fold(0.0, f64::max)
}

pub fn linear_panels_converge_to_mie_and_preserve_formulation_and_axisymmetry() {
    let reference = mie_reference();
    let coarse_constant = solve(COARSE_N_THETA, COARSE_N_PHI, BasisOrder::Constant, true);
    let coarse_linear = solve(COARSE_N_THETA, COARSE_N_PHI, BasisOrder::Linear, true);
    let fine_linear = solve(FINE_N_THETA, FINE_N_PHI, BasisOrder::Linear, true);
    let fine_linear_regular = solve(FINE_N_THETA, FINE_N_PHI, BasisOrder::Linear, false);

    let expected_coarse_linear_dofs: usize = coarse_linear
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .map(|element| element.connectivity.len())
        .sum();
    let expected_fine_linear_dofs: usize = fine_linear
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .map(|element| element.connectivity.len())
        .sum();
    assert_eq!(coarse_linear.num_dofs(), expected_coarse_linear_dofs);
    assert_eq!(fine_linear.num_dofs(), expected_fine_linear_dofs);
    assert!(fine_linear.num_dofs() > coarse_linear.num_dofs());
    assert!(
        fine_linear
            .surface_pressure
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let (coarse_constant_l2, coarse_constant_pointwise) =
        field_errors(&coarse_constant, &reference);
    let (coarse_linear_l2, coarse_linear_pointwise) = field_errors(&coarse_linear, &reference);
    let (fine_linear_l2, fine_linear_pointwise) = field_errors(&fine_linear, &reference);
    let formulation_error = max_field_difference(&fine_linear, &fine_linear_regular);

    // The exact sphere problem is invariant under azimuthal rotations for a
    // z-directed incident plane wave.  The discretised mesh is not forced to
    // share that symmetry, so this is a bounded field-level discretisation
    // invariant rather than an assertion about every surface coefficient.
    let symmetry_points = [
        [0.42, 0.0, 0.84],
        [0.0, 0.42, 0.84],
        [-0.42, 0.0, 0.84],
        [0.0, -0.42, 0.84],
    ];
    let symmetry_fields = fine_linear.evaluate_pressure_field(&symmetry_points);
    let axisymmetry_error = symmetry_fields
        .iter()
        .skip(1)
        .map(|field| relative_error(field.p_total, symmetry_fields[0].p_total))
        .fold(0.0, f64::max);

    assert!(fine_linear_l2.is_finite() && fine_linear_l2 <= MAX_FINE_LINEAR_L2_ERROR);
    assert!(
        fine_linear_pointwise.is_finite()
            && fine_linear_pointwise <= MAX_FINE_LINEAR_POINTWISE_ERROR
    );
    assert!(fine_linear_l2 < coarse_linear_l2);
    assert!(fine_linear_pointwise < coarse_linear_pointwise);
    assert!(
        formulation_error.is_finite() && formulation_error <= MAX_FINE_FORMULATION_RELATIVE_ERROR
    );
    assert!(axisymmetry_error.is_finite() && axisymmetry_error <= MAX_AXISYMMETRY_RELATIVE_ERROR);

    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.higher-order-formulation-reference.v1\",\"status\":\"passed\",\"coarse_constant_l2\":{coarse_constant_l2:.6e},\"coarse_constant_pointwise\":{coarse_constant_pointwise:.6e},\"coarse_linear_l2\":{coarse_linear_l2:.6e},\"coarse_linear_pointwise\":{coarse_linear_pointwise:.6e},\"fine_linear_l2\":{fine_linear_l2:.6e},\"fine_linear_pointwise\":{fine_linear_pointwise:.6e},\"fine_linear_dofs\":{},\"formulation_relative_error\":{formulation_error:.6e},\"axisymmetry_relative_error\":{axisymmetry_error:.6e}}}",
        fine_linear.num_dofs()
    );
}

pub fn quadratic_panels_run_against_independent_mie_reference() {
    let reference = mie_reference();
    let solution = solve(COARSE_N_THETA, COARSE_N_PHI, BasisOrder::Quadratic, true);
    let (l2_error, pointwise_error) = field_errors(&solution, &reference);

    let panel_count = solution
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    assert_eq!(
        solution.num_dofs(),
        panel_count * 6,
        "quadratic Tri3 panels must allocate six independent field DOFs per panel"
    );
    assert!(l2_error.is_finite() && pointwise_error.is_finite());
    assert!(
        l2_error < 0.80 && pointwise_error < 1.20,
        "quadratic panel Mie errors too large: l2={l2_error}, pointwise={pointwise_error}"
    );
}

#[test]
fn higher_order_formulation_reference_test() {
    linear_panels_converge_to_mie_and_preserve_formulation_and_axisymmetry();
    quadratic_panels_run_against_independent_mie_reference();
}
