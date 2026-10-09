//! Release-facing compressed affine-Tri3 Galerkin transmission derivative.

use num_complex::Complex64;
use sonium_bem::core::assembly::compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{
    BasisOrder, GalerkinQuadraturePolicy, PhysicsParams, TransmissionMaterialParameter,
    TransmissionSide, assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-compressed-transmission-material-derivative.v1";

fn run() -> Result<(f64, f64), String> {
    let mut mesh = generate_icosphere_mesh(0.2, 0);
    for element in &mut mesh.elements {
        element.dof_addresses = element.connectivity.clone();
    }
    let global_dof_count = mesh.nodes.len();
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).map_err(|error| error.to_string())?;
    let policy = GalerkinQuadraturePolicy {
        coarse_order: 2,
        fine_order: 3,
        absolute_tolerance: 30.0,
        relative_tolerance: 5.0e-2,
    };
    let compressed =
        compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked(
            &mesh.elements,
            &mesh.nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            global_dof_count,
            policy,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            4,
            1.0e-8,
            8,
        )
        .map_err(|error| error.to_string())?;
    let dense = assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
        &mesh.elements,
        &mesh.nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        policy.fine_order,
        global_dof_count,
        TransmissionSide::Exterior,
        TransmissionMaterialParameter::Wavenumber,
    )
    .map_err(|error| error.to_string())?;
    let input: Vec<Complex64> = (0..2 * global_dof_count)
        .map(|index| Complex64::new(0.2 + index as f64 * 0.01, -0.1 + index as f64 * 0.003))
        .collect();
    let actual = compressed
        .apply_checked(&input)
        .map_err(|error| error.to_string())?;
    let expected: Vec<Complex64> = (0..2 * global_dof_count)
        .map(|row| {
            (0..2 * global_dof_count)
                .map(|column| dense[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let absolute_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    let scale = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    Ok((
        absolute_error / scale,
        compressed.compressed_block_count() as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok((relative_error, compressed_blocks))) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("relative_action_error", relative_error)
                .metric("compressed_block_count", compressed_blocks)
                .maximum("relative_action_error", 2.0e-7)
                .input_checksum(
                    "analytic/deterministic-fixture:bem-galerkin-compressed-transmission-material-derivative-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-galerkin-compressed-transmission-material-derivative-v1",
                )
                .dimension("3D closed icosphere, affine-Tri3 shared-DOF two-medium interface")
                .formulation("dense Galerkin material derivative compressed with deterministic block ACA")
                .tolerance("compressed action matches the fine-order dense derivative")
                .seed("bem-galerkin-compressed-transmission-material-derivative:icosphere-level-0:v1")
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
