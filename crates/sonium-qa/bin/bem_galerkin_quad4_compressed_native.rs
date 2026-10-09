//! Bounded CPU evidence for on-demand curved-Quad4 Galerkin compression.

use num_complex::Complex64;
use sonium_bem::core::IncidentField;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, GalerkinQuadraturePolicy, PhysicsParams,
    assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs_checked,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-compressed-native.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    linear_error: f64,
    quadratic_error: f64,
    linear_blocks: usize,
    quadratic_blocks: usize,
    exact_residual: f64,
}

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

fn closed_cube() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
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
    let linear_dofs = [
        vec![0, 1, 2, 3],
        vec![4, 7, 6, 5],
        vec![0, 4, 5, 1],
        vec![1, 5, 6, 2],
        vec![2, 6, 7, 3],
        vec![3, 7, 4, 0],
    ];
    let elements = connectivity
        .into_iter()
        .zip(linear_dofs)
        .map(|(connectivity, dofs)| quad(connectivity, dofs, [0.5, 0.5, 0.5]))
        .collect();
    (elements, nodes)
}

fn action_error(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    basis: BasisOrder,
    global_dof_count: usize,
) -> Result<(f64, usize), String> {
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        2,
        global_dof_count,
    )
    .map_err(|error| format!("dense Quad4 assembly failed: {error}"))?;
    let compressed = compress_galerkin_muller_transmission_quad4_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        basis,
        2,
        global_dof_count,
        4,
        1.0e-10,
        64,
    )
    .map_err(|error| format!("kernel-native Quad4 compression failed: {error}"))?;
    let trial: Vec<Complex64> = (0..compressed.total_dofs())
        .map(|index| Complex64::new(0.07 * (index as f64 + 1.0), -0.03))
        .collect();
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let dense_action = (0..dense.matrix().nrows())
        .map(|row| {
            (0..dense.matrix().ncols())
                .map(|column| dense.matrix()[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let scale = dense_action
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let error = compressed_action
        .iter()
        .zip(dense_action.iter())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max)
        / scale;
    if !error.is_finite() || error > 2.0e-7 {
        return Err(format!("Quad4 compressed action error {error:e}"));
    }
    Ok((error, compressed.compressed_block_count()))
}

fn exact_source_residual(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
) -> Result<f64, String> {
    let policy = GalerkinQuadraturePolicy {
        coarse_order: 1,
        fine_order: 2,
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
    };
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        exterior,
        interior,
        BasisOrder::Quadratic,
        policy.fine_order,
        26,
    )
    .map_err(|error| format!("exact-source dense assembly failed: {error}"))?;
    let compressed = compress_galerkin_muller_transmission_quad4_with_global_dofs_checked(
        elements,
        nodes,
        exterior,
        interior,
        BasisOrder::Quadratic,
        26,
        policy,
        4,
        1.0e-1,
        1,
    )
    .map_err(|error| format!("checked exact-source compression failed: {error}"))?;
    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = dense
        .project_incident_trace_coefficients(elements, nodes, &incident, exterior, 2)
        .map_err(|error| format!("incident projection failed: {error}"))?;
    let rhs = dense
        .incident_rhs_from_trace_coefficients(&pressure, &velocity)
        .map_err(|error| format!("incident RHS failed: {error}"))?;
    let mut config = GmresConfig::default();
    config.max_iterations = 40;
    config.restart = 8;
    config.tolerance = 1.0e-10;
    config.exact_residual_check_interval = 1;
    let mut nearfield = DenseMatrix::zeros(52, 52);
    for index in 0..52 {
        nearfield[[index, index]] = Complex64::new(1.0, 0.0);
    }
    let solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .map_err(|error| format!("exact-source-certified solve failed: {error}"))?;
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for row in 0..dense.total_dofs() {
        let action = (0..dense.total_dofs())
            .map(|column| dense.matrix()[[row, column]] * solution.values()[column])
            .sum::<Complex64>();
        residual_squared += (action - rhs[row]).norm_sqr();
        rhs_squared += rhs[row].norm_sqr();
    }
    let residual = residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE);
    if !residual.is_finite() || residual > 1.0e-8 {
        return Err(format!("exact source residual {residual:e} exceeds 1e-8"));
    }
    Ok(residual)
}

fn run() -> Result<Metrics, String> {
    let (linear_elements, nodes) = closed_cube();
    let quadratic_dofs = [
        vec![0, 1, 2, 3, 8, 9, 10, 11, 20],
        vec![4, 7, 6, 5, 15, 14, 13, 12, 21],
        vec![0, 4, 5, 1, 16, 12, 17, 8, 22],
        vec![1, 5, 6, 2, 17, 13, 18, 9, 23],
        vec![2, 6, 7, 3, 18, 14, 19, 10, 24],
        vec![3, 7, 4, 0, 19, 15, 16, 11, 25],
    ];
    let mut quadratic_elements = linear_elements.clone();
    for (element, dofs) in quadratic_elements.iter_mut().zip(quadratic_dofs) {
        element.dof_addresses = dofs;
    }
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("exterior physics: {error}"))?;
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("interior physics: {error}"))?;
    let (linear_error, linear_blocks) = action_error(
        &linear_elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
    )?;
    let (quadratic_error, quadratic_blocks) = action_error(
        &quadratic_elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        26,
    )?;
    let exact_residual = exact_source_residual(&quadratic_elements, &nodes, &exterior, &interior)?;
    Ok(Metrics {
        linear_error,
        quadratic_error,
        linear_blocks,
        quadratic_blocks,
        exact_residual,
    })
}

fn main() {
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(run));
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("linear_action_error", metrics.linear_error)
            .metric("quadratic_action_error", metrics.quadratic_error)
            .metric("linear_compressed_blocks", metrics.linear_blocks as f64)
            .metric("quadratic_compressed_blocks", metrics.quadratic_blocks as f64)
            .metric("quadratic_exact_source_residual", metrics.exact_residual)
            .dimension("3-D closed conforming curved-Quad4 two-trace Galerkin")
            .formulation("on-demand isoparametric self/touching/separated weak-form pair entries with block ACA")
            .reference_checksum("analytic/deterministic-fixture:bem-galerkin-quad4-compressed-native-v1")
            .seed("bem-galerkin-quad4-compressed-native:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D closed conforming curved-Quad4 two-trace Galerkin")
            .seed("bem-galerkin-quad4-compressed-native:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(_) => QaResult::failed(CASE_ID, "Quad4 compressed Galerkin QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension("3-D closed conforming curved-Quad4 two-trace Galerkin")
            .seed("bem-galerkin-quad4-compressed-native:cpu:v1")
            .elapsed_ms(elapsed_ms),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
}
