//! Checked higher-order adjoint dual-basis field-map contract.

use num_complex::Complex64;
use sonium_bem::core::postprocess::map_dual_basis_coefficients_checked;
use sonium_bem::core::types::{
    BasisOrder, BoundaryCondition, Element, ElementProperty, PhysicsParams,
};
use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-dual-basis-field.v1";

fn tri_element(dof_addresses: Vec<usize>) -> Element {
    Element {
        connectivity: vec![0, 1, 2],
        element_type: sonium_bem::core::types::ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0; 3]; 3],
        center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn quad_element(dof_addresses: Vec<usize>) -> Element {
    Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: sonium_bem::core::types::ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0; 3]; 4],
        center: [0.0, 0.0, 0.0],
        area: 4.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn run() -> Result<(f64, f64, bool), String> {
    let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
    let tri_nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let tri_dual = [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0 / 6.0, 0.0),
        Complex64::new(1.0 / 6.0, 0.0),
        Complex64::new(1.0 / 6.0, 0.0),
    ];
    let tri_primal = map_dual_basis_coefficients_checked(
        &tri_dual,
        &[tri_element((0..6).collect())],
        &tri_nodes,
        &physics,
        BasisOrder::Quadratic,
    )
    .map_err(|error| error.to_string())?;
    let tri_error = tri_primal
        .iter()
        .map(|value| (value - Complex64::new(1.0, 0.0)).norm())
        .fold(0.0, f64::max);

    let quad_nodes = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let quad_dual: Vec<_> = (0..9)
        .map(|index| Complex64::new(0.25 + index as f64 * 0.03125, -0.1))
        .collect();
    let quad_primal = map_dual_basis_coefficients_checked(
        &quad_dual,
        &[quad_element((0..9).collect())],
        &quad_nodes,
        &physics,
        BasisOrder::Quadratic,
    )
    .map_err(|error| error.to_string())?;
    let quad_finite = quad_primal
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite());

    let mut degenerate_nodes = tri_nodes.clone();
    degenerate_nodes[2] = degenerate_nodes[0];
    let rejected_degenerate = matches!(
        map_dual_basis_coefficients_checked(
            &tri_dual,
            &[tri_element((0..6).collect())],
            &degenerate_nodes,
            &physics,
            BasisOrder::Quadratic,
        ),
        Err(sonium_bem::core::postprocess::FieldEvaluationError::DualBasisSolve { .. })
    );
    if !tri_error.is_finite() || tri_error > 1.0e-12 || !quad_finite || !rejected_degenerate {
        return Err(format!(
            "dual-basis contract failed: tri_error={tri_error:e}, quad_finite={quad_finite}, rejected_degenerate={rejected_degenerate}"
        ));
    }
    Ok((tri_error, quad_primal.len() as f64, rejected_degenerate))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((tri_error, quad_dofs, rejected_degenerate))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("tri_constant_trace_max_error", tri_error)
            .metric("quad_dofs", quad_dofs)
            .metric("degenerate_geometry_rejections", rejected_degenerate as u64 as f64)
            .maximum("tri_constant_trace_max_error", 1.0e-12)
            .minimum("quad_dofs", 9.0)
            .minimum("degenerate_geometry_rejections", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-higher-order-dual-basis-field-v1")
            .reference("independent quadratic surface L2 mass-map moments and typed degenerate-geometry rejection")
            .formulation("checked CPU higher-order Tri3/Quad4 dual-to-primal adjoint field map")
            .tolerance("Tri3 constant-trace recovery <= 1e-12; all Quad4 outputs finite; degenerate geometry rejected")
            .seed("bem-higher-order-dual-basis-field:cpu:v1"),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error),
        Err(_) => QaResult::failed(CASE_ID, "dual-basis field contract panicked"),
    }
    .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
    println!("{}", result.to_json_line().expect("serialize QA result"));
}
