//! Release-facing compressed transmission material-derivative action contract.

use num_complex::Complex64;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::{TransmissionMaterialParameter, TransmissionSide, build_transmission_tbem_system};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.compressed-transmission-material-derivative.v1";

fn run() -> Result<(f64, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).map_err(|error| error.to_string())?;
    let system = build_transmission_tbem_system(&elements, &mesh.nodes, exterior, interior)
        .map_err(|error| error.to_string())?;
    let dense = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .map_err(|error| error.to_string())?;
    let compressed = system
        .compress_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            4,
            1.0e-8,
            8,
        )
        .map_err(|error| error.to_string())?;
    let kernel_native = system
        .compress_kernel_native_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            4,
            1.0e-8,
            8,
        )
        .map_err(|error| error.to_string())?;
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.013, -0.1 + index as f64 * 0.007))
        .collect();
    let actual = compressed
        .apply_checked(&input)
        .map_err(|error| error.to_string())?;
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| dense[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    let scale = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let kernel_actual = kernel_native
        .apply_checked(&input)
        .map_err(|error| error.to_string())?;
    let kernel_error = kernel_actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    Ok((
        error / scale,
        kernel_error / scale,
        compressed.compressed_block_count() as f64,
        kernel_native.compressed_block_count() as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok((action_error, kernel_action_error, compressed_blocks, kernel_blocks))) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("relative_action_error", action_error)
                .metric("kernel_native_relative_action_error", kernel_action_error)
                .metric("compressed_block_count", compressed_blocks)
                .metric("kernel_native_compressed_block_count", kernel_blocks)
                .maximum("relative_action_error", 2.0e-7)
                .maximum("kernel_native_relative_action_error", 2.0e-7)
                .input_checksum(
                    "analytic/deterministic-fixture:bem-compressed-transmission-material-derivative-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-compressed-transmission-material-derivative-v1",
                )
                .dimension("3D closed icosphere, constant-panel two-medium interface")
                .formulation(
                    "dense and kernel-native material derivatives compressed with deterministic block ACA",
                )
                .tolerance("both compressed derivative actions match their dense source")
                .seed("bem-compressed-transmission-material-derivative:icosphere-level-0:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record =
                QaResult::failed(CASE_ID, error).elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            let record = QaResult::failed(CASE_ID, "panic during QA execution")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
