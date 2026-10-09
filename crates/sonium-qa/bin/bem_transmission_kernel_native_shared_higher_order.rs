//! Bounded CPU parity evidence for shared higher-order kernel-native transmission.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::{TransmissionMaterialParameter, TransmissionSide};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-kernel-native-shared-higher-order-parity.v1";
const MAX_ERROR: f64 = 2.0e-7;

fn run() -> Result<(usize, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("exterior physics: {error}"))?;
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("interior physics: {error}"))?;
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .map_err(|error| format!("continuous quadratic transmission assembly: {error}"))?;

    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.08 + index as f64 * 0.002, -0.03 + index as f64 * 0.001))
        .collect();
    let dense = system.matrix();
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| dense[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let scale = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);

    // One bounded dense ACA tile isolates the kernel-native entry contract
    // from rank-truncation error while still exercising the compressed path.
    let compressed = system
        .compress_kernel_native(system.total_dofs(), 1.0e-8, 8)
        .map_err(|error| format!("shared higher-order kernel-native compression: {error}"))?;
    let actual = compressed
        .apply_checked(&input)
        .map_err(|error| format!("shared higher-order compressed action: {error}"))?;
    let value_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max)
        / scale;

    let dense_derivative = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .map_err(|error| format!("shared higher-order dense derivative: {error}"))?;
    let derivative_expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| dense_derivative[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let derivative_scale = derivative_expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let compressed_derivative = system
        .compress_kernel_native_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            system.total_dofs(),
            1.0e-8,
            8,
        )
        .map_err(|error| format!("shared higher-order kernel-native derivative: {error}"))?;
    let derivative_actual = compressed_derivative
        .apply_checked(&input)
        .map_err(|error| format!("shared higher-order derivative action: {error}"))?;
    let derivative_error = derivative_actual
        .iter()
        .zip(derivative_expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max)
        / derivative_scale;

    if !value_error.is_finite()
        || !derivative_error.is_finite()
        || value_error > MAX_ERROR
        || derivative_error > MAX_ERROR
    {
        return Err(format!(
            "shared higher-order parity errors value={value_error:.3e}, derivative={derivative_error:.3e}, tolerance={MAX_ERROR:.3e}"
        ));
    }
    Ok((system.interface_dofs(), value_error, derivative_error))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((global_dofs, value_error, derivative_error)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("global_dofs", global_dofs as f64)
                .metric("value_relative_error", value_error)
                .metric("wavenumber_derivative_relative_error", derivative_error)
                .maximum("value_relative_error", MAX_ERROR)
                .maximum("wavenumber_derivative_relative_error", MAX_ERROR)
                .reference("dense continuous-quadratic transmission matrix and analytic wavenumber derivative")
                .dimension("3-D radius 0.2 m; 20-panel icosphere; globally shared quadratic trace DOFs")
                .formulation("CPU kernel-native panel quadrature with conforming shared-DOF accumulation")
                .tolerance("value and wavenumber-derivative compressed actions <= 2e-7 relative error")
                .seed("bem-transmission-kernel-native-shared-higher-order-parity:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("dense continuous-quadratic transmission matrix and analytic wavenumber derivative")
                .dimension("3-D radius 0.2 m; 20-panel icosphere; globally shared quadratic trace DOFs")
                .formulation("CPU kernel-native panel quadrature with conforming shared-DOF accumulation")
                .tolerance("value and wavenumber-derivative compressed actions <= 2e-7 relative error")
                .seed("bem-transmission-kernel-native-shared-higher-order-parity:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("serialize QA result")
            );
            eprintln!("BEM SHARED HIGHER-ORDER KERNEL-NATIVE PARITY FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
