//! Checked forward/adjoint field parity for compressed transmission solves.

use num_complex::Complex64;
use sonium_bem::core::assembly::{
    build_transmission_tbem_system_with_basis,
    build_transmission_tbem_system_with_continuous_quadratic,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::transmission::{
    solve_incident_transmission, solve_transmission_adjoint_system,
};
use sonium_bem::core::types::PhysicsParams;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::new(42.0, 343.0, 1.21, false),
        PhysicsParams::new(42.0, 343.0, 1.21, true),
    )
}

fn relative_field_error(actual: &[Complex64], reference: &[Complex64]) -> f64 {
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
        .max(1.0e-15);
    numerator / denominator
}

#[test]
fn compressed_transmission_forward_and_adjoint_fields_match_dense_reference() {
    // Keep this regression packet small enough for deterministic CPU-only CI;
    // the higher-order basis still exercises both compressed trace blocks.
    let mesh = generate_icosphere_mesh(0.2, 0);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("continuous quadratic transmission system");
    let compressed = system.compress(16, 1.0e-3, 32).unwrap();
    let dimension = system.total_dofs();
    let config = GmresConfig::<f64> {
        max_iterations: 120,
        restart: 32,
        tolerance: 1.0e-10,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    };
    // Use a Jacobi near-field ILU built from the assembled diagonal.  An
    // identity preconditioner leaves this indefinite transmission operator
    // too poorly scaled for a bounded field-parity test, while factoring the
    // full matrix is needlessly fragile for this deliberately compressed
    // approximation.
    let mut nearfield = system.matrix().clone();
    for index in 0..dimension {
        nearfield[[index, index]] += Complex64::new(1.0, 0.0);
    }
    let mut adjoint_nearfield = DenseMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            adjoint_nearfield[[row, column]] = system.matrix()[[column, row]].conj();
        }
        adjoint_nearfield[[row, row]] += Complex64::new(1.0, 0.0);
    }
    let incident = IncidentField::plane_wave_z();
    let dense_forward = solve_incident_transmission(&system, &incident, 1.0e-8)
        .expect("dense forward transmission solve");
    let forward_rhs = system
        .incident_rhs(&incident)
        .expect("incident transmission RHS");
    let compressed_forward = compressed
        .solve_with_ilu(&forward_rhs, &nearfield, &config, 1.0e-8)
        .expect("compressed forward transmission solve");
    assert!(compressed_forward.relative_residual() <= 1.0e-8);

    let points = [[0.0, 0.0, 0.8], [0.2, 0.0, 0.7]];
    let dense_forward_field = dense_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .expect("dense forward field");
    let compressed_forward_field = compressed_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .expect("compressed forward field");
    let dense_forward_pressure = dense_forward_field
        .iter()
        .map(|field| field.p_total)
        .collect::<Vec<_>>();
    let compressed_forward_pressure = compressed_forward_field
        .iter()
        .map(|field| field.p_total)
        .collect::<Vec<_>>();
    assert!(
        relative_field_error(&compressed_forward_pressure, &dense_forward_pressure) <= 0.2,
        "compressed forward field parity error {:.3e}",
        relative_field_error(&compressed_forward_pressure, &dense_forward_pressure)
    );
    assert!(
        compressed_forward
            .evaluate_exterior_field_checked(&system, &[[f64::NAN, 0.0, 0.8]], &incident)
            .is_err()
    );

    let adjoint_rhs = (0..dimension)
        .map(|index| Complex64::new(0.17 + index as f64 / 31.0, -0.09 + index as f64 / 43.0))
        .collect::<Vec<_>>();
    let dense_adjoint = solve_transmission_adjoint_system(&system, &adjoint_rhs, 1.0e-8)
        .expect("dense adjoint transmission solve");
    let compressed_adjoint = compressed
        .solve_hermitian_with_ilu(&adjoint_rhs, &adjoint_nearfield, &config, 2.0e-6)
        .expect("compressed adjoint transmission solve");
    assert!(compressed_adjoint.relative_residual() <= 2.0e-6);
    let dense_adjoint_field = dense_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .expect("dense adjoint field");
    let compressed_adjoint_field = compressed_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .expect("compressed adjoint field");
    assert!(
        relative_field_error(&compressed_adjoint_field, &dense_adjoint_field) <= 0.2,
        "compressed adjoint field parity error {:.3e}",
        relative_field_error(&compressed_adjoint_field, &dense_adjoint_field)
    );
    assert!(
        compressed_adjoint
            .evaluate_exterior_adjoint_pressure_checked(&system, &[[f64::NAN, 0.0, 0.8]])
            .is_err()
    );
}

#[test]
fn kernel_native_panel_linear_forward_and_adjoint_fields_match_dense_reference() {
    // Keep this production-kernel path intentionally small: panel-local linear
    // traces still exercise the basis-weighted on-demand ACA entries without
    // starting a simulator-scale compression campaign.
    let mesh = generate_icosphere_mesh(0.2, 0);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_basis(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
        sonium_bem::core::types::BasisOrder::Linear,
    )
    .expect("panel-local linear transmission system");
    let compressed = system
        .compress_kernel_native(16, 1.0e-3, 32)
        .expect("kernel-native transmission compression");
    let dimension = system.total_dofs();
    let config = GmresConfig::<f64> {
        max_iterations: 120,
        restart: 32,
        tolerance: 1.0e-10,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    };
    let mut nearfield = system.matrix().clone();
    for index in 0..dimension {
        nearfield[[index, index]] += Complex64::new(1.0, 0.0);
    }
    let mut adjoint_nearfield = DenseMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            adjoint_nearfield[[row, column]] = system.matrix()[[column, row]].conj();
        }
        adjoint_nearfield[[row, row]] += Complex64::new(1.0, 0.0);
    }
    let incident = IncidentField::plane_wave_z();
    let dense_forward = solve_incident_transmission(&system, &incident, 1.0e-8)
        .expect("dense panel-local forward transmission solve");
    let rhs = system
        .incident_rhs(&incident)
        .expect("panel-local incident RHS");
    let compressed_forward = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .expect("kernel-native panel-local forward solve");
    let points = [[0.0, 0.0, 0.8], [0.2, 0.0, 0.7]];
    let dense_forward_field = dense_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .expect("dense panel-local forward field");
    let compressed_forward_field = compressed_forward
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .expect("kernel-native panel-local forward field");
    let dense_forward_pressure: Vec<_> = dense_forward_field
        .iter()
        .map(|field| field.p_total)
        .collect();
    let compressed_forward_pressure: Vec<_> = compressed_forward_field
        .iter()
        .map(|field| field.p_total)
        .collect();
    assert!(
        relative_field_error(&compressed_forward_pressure, &dense_forward_pressure) <= 0.2,
        "kernel-native panel-local forward field parity error {:.3e}",
        relative_field_error(&compressed_forward_pressure, &dense_forward_pressure)
    );

    let adjoint_rhs: Vec<_> = (0..dimension)
        .map(|index| Complex64::new(0.13 + index as f64 / 37.0, -0.07 + index as f64 / 41.0))
        .collect();
    let dense_adjoint = solve_transmission_adjoint_system(&system, &adjoint_rhs, 1.0e-8)
        .expect("dense panel-local adjoint transmission solve");
    let compressed_adjoint = compressed
        .solve_hermitian_with_ilu(&adjoint_rhs, &adjoint_nearfield, &config, 2.0e-6)
        .expect("kernel-native panel-local adjoint solve");
    assert!(compressed_adjoint.relative_residual() <= 2.0e-6);
    let dense_adjoint_field = dense_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .expect("dense panel-local adjoint field");
    let compressed_adjoint_field = compressed_adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &points)
        .expect("kernel-native panel-local adjoint field");
    assert!(
        relative_field_error(&compressed_adjoint_field, &dense_adjoint_field) <= 0.2,
        "kernel-native panel-local adjoint field parity error {:.3e}",
        relative_field_error(&compressed_adjoint_field, &dense_adjoint_field)
    );
}
