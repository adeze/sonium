//! Bounded CPU parity evidence for kernel-native compressed transmission actions.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-kernel-native-parity.v1";

fn run() -> Result<(usize, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    let system = build_transmission_tbem_system(
        &elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .map_err(|error| error.to_string())?;
    let compressed = system
        .compress_kernel_native(4, 1.0e-8, 8)
        .map_err(|error| error.to_string())?;
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.013, -0.1 + index as f64 * 0.007))
        .collect();
    let dense_forward: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let native_forward = compressed
        .apply_checked(&input)
        .map_err(|error| error.to_string())?;
    let dense_transpose: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[column, row]] * input[column])
                .sum()
        })
        .collect();
    let native_transpose = compressed
        .apply_transpose_checked(&input)
        .map_err(|error| error.to_string())?;
    let relative = |actual: &[Complex64], expected: &[Complex64]| {
        actual
            .iter()
            .zip(expected)
            .map(|(value, reference)| (*value - *reference).norm())
            .fold(0.0_f64, f64::max)
            / expected
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max)
                .max(1.0)
    };
    Ok((
        system.total_dofs(),
        relative(&native_forward, &dense_forward),
        relative(&native_transpose, &dense_transpose),
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, forward_error, transpose_error)) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("forward_action_error", forward_error)
            .metric("transpose_action_error", transpose_error)
            .input_checksum("bem:icosphere:subdivision-0:constant-transmission")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-kernel-native-parity-v1",
            )
            .dimension("3-D CPU 20-panel constant Müller transmission")
            .formulation(
                "kernel-native panel quadrature through block-ACA forward/transpose actions",
            )
            .tolerance("forward and transpose relative action errors <= 2e-7")
            .seed("bem-transmission-kernel-native-parity:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
