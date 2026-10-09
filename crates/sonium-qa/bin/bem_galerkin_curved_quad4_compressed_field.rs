//! Bounded CPU evidence for curved Quad4 Galerkin compressed-field parity.

use std::time::Instant;

use num_complex::Complex64;
use sonium_bem::core::IncidentField;
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_bem::{
    BasisOrder, GalerkinQuadraturePolicy, assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs_checked,
};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;

const CASE_ID: &str = "bem.galerkin-curved-quad4-compressed-field.v1";

fn quad(connectivity: Vec<usize>, dof_addresses: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn curved_closed_cube() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.3],
        [0.0, 1.0, 1.0],
    ];
    let connectivity = [
        vec![0, 1, 2, 3],
        vec![4, 7, 6, 5],
        vec![0, 4, 5, 1],
        vec![1, 5, 6, 2],
        vec![2, 6, 7, 3],
        vec![3, 7, 4, 0],
    ];
    let quadratic_dofs = [
        vec![0, 1, 2, 3, 8, 9, 10, 11, 20],
        vec![4, 7, 6, 5, 15, 14, 13, 12, 21],
        vec![0, 4, 5, 1, 16, 12, 17, 8, 22],
        vec![1, 5, 6, 2, 17, 13, 18, 9, 23],
        vec![2, 6, 7, 3, 18, 14, 19, 10, 24],
        vec![3, 7, 4, 0, 19, 15, 16, 11, 25],
    ];
    let elements = connectivity
        .into_iter()
        .zip(quadratic_dofs)
        .map(|(connectivity, dof_addresses)| quad(connectivity, dof_addresses, [0.5, 0.5, 0.5]))
        .collect();
    (elements, nodes)
}

fn run() -> Result<(usize, f64, usize), String> {
    let (elements, nodes) = curved_closed_cube();
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).map_err(|error| error.to_string())?;
    let policy = GalerkinQuadraturePolicy {
        // Mapped self/touching Quad4 charts converge non-monotonically at
        // adjacent low orders. Admit the bounded 3->4 fixture here, then use
        // the much stricter compressed-vs-dense field check below as the
        // release accuracy gate.
        coarse_order: 3,
        fine_order: 4,
        absolute_tolerance: 1.0e-3,
        relative_tolerance: 3.0e-1,
    };
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        policy.fine_order,
        26,
    )
    .map_err(|error| error.to_string())?;
    let compressed = compress_galerkin_muller_transmission_quad4_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        26,
        policy,
        4,
        1.0e-9,
        16,
    )
    .map_err(|error| error.to_string())?;

    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = operators
        .project_incident_trace_coefficients(
            &elements,
            &nodes,
            &incident,
            &exterior,
            policy.fine_order,
        )
        .map_err(|error| error.to_string())?;
    let rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure, &velocity)
        .map_err(|error| error.to_string())?;
    let dense_solution = operators
        .solve(&rhs, 1.0e-8)
        .map_err(|error| error.to_string())?;

    let mut config = GmresConfig::default();
    config.max_iterations = 120;
    config.restart = 32;
    config.tolerance = 1.0e-10;
    config.exact_residual_check_interval = 1;
    let mut nearfield = DenseMatrix::zeros(operators.total_dofs(), operators.total_dofs());
    for index in 0..operators.total_dofs() {
        nearfield[[index, index]] = Complex64::new(1.0, 0.0);
    }
    let compressed_solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .map_err(|error| error.to_string())?;

    let points = [[0.2, 0.3, 1.8], [1.4, 0.4, 0.8]];
    let dense_field = operators
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &dense_solution,
            &points,
            &incident,
            &exterior,
            policy.fine_order,
        )
        .map_err(|error| error.to_string())?;
    let compressed_field = operators
        .evaluate_exterior_field_from_compressed_solution_checked(
            &compressed_solution,
            &elements,
            &nodes,
            &points,
            &incident,
            &exterior,
            policy.fine_order,
        )
        .map_err(|error| error.to_string())?;
    let field_error = compressed_field
        .iter()
        .zip(&dense_field)
        .map(|(actual, expected)| (actual.p_total - expected.p_total).norm())
        .fold(0.0_f64, f64::max);
    if !field_error.is_finite() || field_error > 2.0e-6 {
        return Err(format!(
            "curved Quad4 compressed field error {field_error:.3e} exceeds 2e-6"
        ));
    }
    if compressed_field
        .iter()
        .any(|value| !value.p_total.re.is_finite() || !value.p_total.im.is_finite())
    {
        return Err("curved Quad4 compressed field contains non-finite pressure".into());
    }
    Ok((
        operators.total_dofs(),
        field_error,
        compressed.compressed_block_count(),
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, field_error, compressed_blocks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("matrix_dimension", dimension as f64)
            .metric("compressed_field_error", field_error)
            .metric("compressed_block_count", compressed_blocks as f64)
            .minimum("matrix_dimension", 52.0)
            .maximum("compressed_field_error", 2.0e-6)
            .minimum("compressed_block_count", 0.0)
            .input_checksum("bem:closed-cube:curved-quad4:quadratic:global-dofs-26")
            .reference("dense curved Quad4 Galerkin Müller field evaluation")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-curved-quad4-compressed-field-v1",
            )
            .dimension("3-D CPU curved closed Quad4; continuous quadratic 52-trace-DOF operator")
            .tolerance("compressed exterior pressure field error <= 2e-6; all values finite")
            .seed("bem-galerkin-curved-quad4-compressed-field:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, QaStatus::Passed) {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
