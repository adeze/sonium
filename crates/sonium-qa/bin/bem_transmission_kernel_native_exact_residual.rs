//! Bounded CPU exact-source residual gate for kernel-native transmission ACA.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_transmission_tbem_system;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-kernel-native-exact-residual.v1";
const MAX_RESIDUAL: f64 = 1.0e-9;

fn relative_residual(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    values: &[Complex64],
    rhs: &[Complex64],
    hermitian: bool,
) -> f64 {
    let mut action = vec![Complex64::default(); matrix.nrows()];
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            let coefficient = if hermitian {
                matrix[[column, row]].conj()
            } else {
                matrix[[row, column]]
            };
            action[row] += coefficient * values[column];
        }
    }
    let residual = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    residual / rhs_norm.max(f64::MIN_POSITIVE)
}

fn run() -> Result<(usize, f64, f64, usize, usize), String> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false)
        .map_err(|error| format!("exterior physics: {error}"))?;
    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true)
        .map_err(|error| format!("interior physics: {error}"))?;
    let system = build_transmission_tbem_system(&mesh.elements, &mesh.nodes, exterior, interior)
        .map_err(|error| format!("transmission assembly: {error}"))?;
    let compressed = system
        .compress_kernel_native(16, 0.5, 4)
        .map_err(|error| format!("kernel-native compression: {error}"))?;
    let rhs = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.001, -0.1))
        .collect::<Vec<_>>();
    let mut config = GmresConfig::default();
    config.max_iterations = 80;
    config.restart = 16;
    config.tolerance = MAX_RESIDUAL;
    config.exact_residual_check_interval = 1;
    let solution = compressed
        .solve_with_ilu(&rhs, system.matrix(), &config, MAX_RESIDUAL)
        .map_err(|error| format!("exact-source-certified solve: {error}"))?;
    let residual = relative_residual(system.matrix(), solution.values(), &rhs, false);
    if !residual.is_finite() || residual > MAX_RESIDUAL {
        return Err(format!(
            "exact residual {residual:.3e} exceeds {MAX_RESIDUAL:.3e}"
        ));
    }
    let mut hermitian_nearfield = DenseMatrix::zeros(system.total_dofs(), system.total_dofs());
    for row in 0..system.total_dofs() {
        for column in 0..system.total_dofs() {
            hermitian_nearfield[[row, column]] = system.matrix()[[column, row]].conj();
        }
    }
    let adjoint_rhs = (0..system.total_dofs())
        .map(|index| Complex64::new(-0.15 + index as f64 * 0.002, 0.07 - index as f64 * 0.001))
        .collect::<Vec<_>>();
    let adjoint_solution = compressed
        .solve_hermitian_with_ilu(&adjoint_rhs, &hermitian_nearfield, &config, MAX_RESIDUAL)
        .map_err(|error| format!("exact-source-certified Hermitian solve: {error}"))?;
    let adjoint_residual = relative_residual(
        system.matrix(),
        adjoint_solution.values(),
        &adjoint_rhs,
        true,
    );
    if !adjoint_residual.is_finite() || adjoint_residual > MAX_RESIDUAL {
        return Err(format!(
            "exact Hermitian residual {adjoint_residual:.3e} exceeds {MAX_RESIDUAL:.3e}"
        ));
    }
    Ok((
        system.interface_dofs(),
        residual,
        adjoint_residual,
        compressed.compressed_block_count(),
        compressed.dense_block_count(),
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((interface_dofs, residual, adjoint_residual, compressed_blocks, dense_blocks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("interface_dofs", interface_dofs as f64)
                .metric("exact_source_forward_relative_residual", residual)
                .metric("exact_source_hermitian_relative_residual", adjoint_residual)
                .metric("compressed_blocks", compressed_blocks as f64)
                .metric("dense_blocks", dense_blocks as f64)
                .minimum("interface_dofs", 1.0)
                .maximum("exact_source_forward_relative_residual", MAX_RESIDUAL)
                .maximum("exact_source_hermitian_relative_residual", MAX_RESIDUAL)
                .reference("dense transmission matrix retained by kernel-native owner")
                .dimension("3-D constant-panel 20-panel icosphere; two-trace ACA")
                .formulation(
                    "kernel-native transmission ACA with exact-source residual certification",
                )
                .tolerance("exact-source relative residual <= 1e-9")
                .seed("bem-transmission-kernel-native-exact-residual:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("dense transmission matrix retained by kernel-native owner")
            .dimension("3-D constant-panel 20-panel icosphere; two-trace ACA")
            .formulation("kernel-native transmission ACA with exact-source residual certification")
            .tolerance("exact-source relative residual <= 1e-9")
            .seed("bem-transmission-kernel-native-exact-residual:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
