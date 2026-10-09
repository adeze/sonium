//! Structured near-singular regular-quadrature convergence contract.

use sonium_bem::core::integration::{quasi_singular_integration, regular_integration_fixed_order};
use sonium_bem::core::types::{ElementType, PhysicsParams};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.near-singular-quadrature-reference.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    reference_norm: f64,
    error_order_4: f64,
    error_order_8: f64,
    error_order_16: f64,
    error_order_32: f64,
    dispatch_error: f64,
    monotone: f64,
}

fn run() -> Result<Metrics, String> {
    let coords =
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let source = [0.25, 0.25, 0.1];
    let normal = [0.0, 0.0, 1.0];
    let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);
    let reference =
        regular_integration_fixed_order(&source, &normal, &coords, ElementType::Tri3, &physics, 64)
            .g_integral;
    let error = |order: usize| {
        (regular_integration_fixed_order(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            order,
        )
        .g_integral
            - reference)
            .norm()
    };
    let error_order_4 = error(4);
    let error_order_8 = error(8);
    let error_order_16 = error(16);
    let error_order_32 = error(32);
    let dispatched = quasi_singular_integration(
        &source,
        &normal,
        &coords,
        ElementType::Tri3,
        &[1.0 / 3.0, 1.0 / 3.0, 0.0],
        0.5,
        &physics,
    );
    let dispatch_error = (dispatched.g_integral - reference).norm();
    let monotone = (error_order_8 < error_order_4
        && error_order_16 < error_order_8
        && error_order_32 < error_order_16) as u8 as f64;
    if !reference.re.is_finite()
        || !reference.im.is_finite()
        || !dispatch_error.is_finite()
        || monotone != 1.0
        || error_order_32 >= 1.0e-10
    {
        return Err(format!(
            "near-singular quadrature failed: errors=[{error_order_4:e}, {error_order_8:e}, {error_order_16:e}, {error_order_32:e}], monotone={monotone}"
        ));
    }
    Ok(Metrics {
        reference_norm: reference.norm(),
        error_order_4,
        error_order_8,
        error_order_16,
        error_order_32,
        dispatch_error,
        monotone,
    })
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(metrics)) => QaResult::passed(CASE_ID)
            .metric("reference_norm", metrics.reference_norm)
            .metric("error_order_4", metrics.error_order_4)
            .metric("error_order_8", metrics.error_order_8)
            .metric("error_order_16", metrics.error_order_16)
            .metric("error_order_32", metrics.error_order_32)
            .metric("dispatch_error", metrics.dispatch_error)
            .metric("monotone", metrics.monotone)
            .maximum("error_order_32", 1.0e-10)
            .minimum("monotone", 1.0)
            .input_checksum(
                "analytic/deterministic-fixture:bem-near-singular-quadrature-reference-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-near-singular-quadrature-reference-v1",
            )
            .dimension("Tri3 near-singular panel; source offset 0.1; orders 4/8/16/32/64")
            .reference("independent order-64 over-integrated regular-panel reference")
            .seed("bem-near-singular-quadrature:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(_) => QaResult::failed(CASE_ID, "panic during near-singular quadrature validation")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    let passed = matches!(&result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        std::process::exit(1);
    }
}
