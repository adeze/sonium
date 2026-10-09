//! Release-facing 3-D dense Müller transmission adjoint contract.

use num_complex::Complex64;
use sonium_bem::build_transmission_tbem_system_from_xem;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::{
    solve_transmission_adjoint_many, solve_transmission_adjoint_system,
};
use sonium_qa::QaResult;
use sonium_xem::AcousticMedium;
use std::panic::catch_unwind;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-adjoint.v1";
const FREQUENCY_HZ: f64 = 420.0;
const RESIDUAL_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    interface_dofs: f64,
    adjoint_residual: f64,
    batch_residual: f64,
    inner_product_error: f64,
    batch_columns: f64,
}

fn run() -> Result<Metrics, String> {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21)
        .map_err(|error| format!("incident medium: {error}"))?;
    let transmitted = AcousticMedium::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0)
        .map_err(|error| format!("transmitted medium: {error}"))?;
    let system =
        build_transmission_tbem_system_from_xem(&mesh.elements, &mesh.nodes, incident, transmitted)
            .map_err(|error| format!("transmission assembly: {error}"))?;
    let dimension = system.total_dofs();
    let trial = (0..dimension)
        .map(|index| Complex64::new(0.15 + index as f64 / 19.0, -0.07 + index as f64 / 23.0))
        .collect::<Vec<_>>();
    let rhs = (0..dimension)
        .map(|index| Complex64::new(0.4 + index as f64 / 17.0, 0.03 + index as f64 / 29.0))
        .collect::<Vec<_>>();
    let adjoint = solve_transmission_adjoint_system(&system, &rhs, RESIDUAL_TOLERANCE)
        .map_err(|error| format!("adjoint solve: {error}"))?;
    let forward_action = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[row, column]] * trial[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let lhs = forward_action
        .iter()
        .zip(adjoint.values())
        .map(|(left, right)| left.conj() * right)
        .sum::<Complex64>();
    let rhs_pairing = trial
        .iter()
        .zip(&rhs)
        .map(|(left, right)| left.conj() * right)
        .sum::<Complex64>();
    let inner_product_error = (lhs - rhs_pairing).norm();
    if inner_product_error > 1.0e-9 {
        return Err(format!(
            "Hermitian inner-product error {inner_product_error:e}"
        ));
    }

    let second_rhs = rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + Complex64::new(index as f64 / 41.0, 0.05))
        .collect::<Vec<_>>();
    let batch = solve_transmission_adjoint_many(&system, &[rhs, second_rhs], RESIDUAL_TOLERANCE)
        .map_err(|error| format!("adjoint batch: {error}"))?;
    let batch_residual = batch
        .iter()
        .map(|solution| solution.relative_residual)
        .fold(0.0, f64::max);

    Ok(Metrics {
        interface_dofs: system.interface_dofs() as f64,
        adjoint_residual: adjoint.relative_residual,
        batch_residual,
        inner_product_error,
        batch_columns: batch.len() as f64,
    })
}

fn main() {
    let started = Instant::now();
    match catch_unwind(run) {
        Ok(Ok(metrics)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("interface_dofs", metrics.interface_dofs)
                .metric("adjoint_residual", metrics.adjoint_residual)
                .metric("batch_residual", metrics.batch_residual)
                .metric("inner_product_error", metrics.inner_product_error)
                .metric("batch_columns", metrics.batch_columns)
                .minimum("interface_dofs", 1.0)
                .maximum("adjoint_residual", RESIDUAL_TOLERANCE)
                .maximum("batch_residual", RESIDUAL_TOLERANCE)
                .maximum("inner_product_error", 1.0e-9)
                .minimum("batch_columns", 2.0)
                .input_checksum("analytic/deterministic-fixture:bem-transmission-adjoint-v1")
                .reference_checksum("analytic/deterministic-fixture:bem-transmission-adjoint-v1")
                .reference("independent Hermitian inner-product identity and true residual")
                .dimension("3D closed icosphere, constant-panel two-medium interface")
                .formulation("dense constant-panel Müller Hermitian-adjoint solve")
                .tolerance("Aᴴ residual and complex inner-product parity")
                .seed("bem-transmission-adjoint:icosphere-level-1:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            let record =
                QaResult::failed(CASE_ID, "panic while running transmission adjoint contract")
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", record.to_json_line().unwrap());
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
