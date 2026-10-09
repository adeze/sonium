//! Release-facing regression for higher-order BEM exterior field evaluation.
//!
//! This exercises the production checked field path with independent
//! quadratic pressure and normal-velocity DOFs. The formulation/reference
//! packet covers physical convergence separately; this case protects the
//! basis-layout contract and the quadratic velocity term.

use num_complex::Complex64;
use sonium_bem::core::postprocess::pressure::{
    FieldEvaluationError, compute_scattered_field_checked,
};
use sonium_bem::core::types::{
    BasisOrder, BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-field-evaluation-reference.v1";

fn run() {
    let physics = PhysicsParams::try_new(80.0, 343.0, 1.21, false).expect("valid physics");
    let nodes = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let element = Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.0, 0.0, 0.0],
        area: 4.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: (0..9).collect(),
    };
    let eval_points = [[0.23, -0.17, 2.0], [-0.41, 0.29, 1.7]];
    let zeros = vec![Complex64::new(0.0, 0.0); 9];
    let mut pressure = zeros.clone();
    pressure[8] = Complex64::new(1.0, 0.35);
    let mut velocity = zeros.clone();
    velocity[8] = Complex64::new(0.4, -0.2);

    let pressure_field = compute_scattered_field_checked(
        &eval_points,
        std::slice::from_ref(&element),
        &nodes,
        &pressure,
        Some(&zeros),
        &physics,
        BasisOrder::Quadratic,
    )
    .expect("quadratic pressure field must evaluate");
    assert!(pressure_field.iter().any(|value| value.norm() > 1.0e-12));

    let velocity_field = compute_scattered_field_checked(
        &eval_points,
        std::slice::from_ref(&element),
        &nodes,
        &zeros,
        Some(&velocity),
        &physics,
        BasisOrder::Quadratic,
    )
    .expect("quadratic velocity field must evaluate");
    assert!(velocity_field.iter().any(|value| value.norm() > 1.0e-12));

    let error = compute_scattered_field_checked(
        &eval_points,
        std::slice::from_ref(&element),
        &nodes,
        &pressure,
        Some(&zeros[..1]),
        &physics,
        BasisOrder::Quadratic,
    )
    .expect_err("quadratic velocity layout must be checked");
    assert!(matches!(
        error,
        FieldEvaluationError::SurfaceVelocityCount {
            expected: 9,
            actual: 1,
        }
    ));
}

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(run);
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let result = match outcome {
        Ok(()) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("quadratic_pressure_field_nonzero", 1.0)
            .minimum("quadratic_pressure_field_nonzero", 1.0)
            .metric("quadratic_velocity_field_nonzero", 1.0)
            .minimum("quadratic_velocity_field_nonzero", 1.0)
            .metric("quadratic_velocity_layout_rejected", 1.0)
            .minimum("quadratic_velocity_layout_rejected", 1.0)
            .input_checksum("bem:quad4:q2:pressure-and-velocity-field-evaluation")
            .reference("quadratic pressure/velocity DOF interpolation and checked buffer-layout rejection")
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-field-evaluation-v1")
            .dimension("single Quad4 panel; nine quadratic pressure and velocity DOFs")
            .formulation("production CPU checked Quad4 quadratic pressure and normal-velocity field evaluation")
            .tolerance("nonzero pressure/velocity responses; expected velocity length 9 rejected for actual length 1")
            .seed("bem-higher-order-field-evaluation-reference:cpu:v1"),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("higher-order field evaluation panic: {payload:?}"),
        )
        .requested_backend("cpu")
        .actual_backend("cpu"),
    };
    let result = result.elapsed_ms(elapsed_ms);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
