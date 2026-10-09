//! Production BEM assembly uploaded to the checked compute CSR backend.
//!
//! A small constant-panel TBEM system is assembled first, then the compute
//! backend owns both forward and Hermitian applications.  The checks compare
//! device results with an independent dense row accumulation and exercise the
//! two-column column-major path as well as the scalar path.

use num_complex::Complex64;
use sonium_bem::core::assembly::build_tbem_system;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, PhysicsParams};
use sonium_compute::{
    AdjointDeviceCsr, Backend, CpuBackend, DeviceAdjointOperator, DeviceOperator,
    GmresBackendConfig, HostCsrRef, IdentityDevicePreconditioner, gmres_backend_preconditioned,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.bem-adjoint.v1";
const APPLY_TOLERANCE: f64 = 2.0e-12;
const INNER_PRODUCT_TOLERANCE: f64 = 2.0e-12;
const BLOCK_TOLERANCE: f64 = 2.0e-12;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    dofs: usize,
    nnz: usize,
    forward_error: f64,
    adjoint_error: f64,
    inner_product_error: f64,
    block_error: f64,
    solve_error: f64,
    solve_residual: f64,
}

fn dotc(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    x.iter().zip(y).map(|(&x, &y)| x.conj() * y).sum()
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual - expected).norm() / expected.norm().max(1.0e-30))
        .fold(0.0, f64::max)
}

fn dense_forward(data: &[Complex64], dim: usize, x: &[Complex64]) -> Vec<Complex64> {
    (0..dim)
        .map(|row| (0..dim).map(|col| data[row * dim + col] * x[col]).sum())
        .collect()
}

fn dense_adjoint(data: &[Complex64], dim: usize, x: &[Complex64]) -> Vec<Complex64> {
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|col| data[col * dim + row].conj() * x[col])
                .sum()
        })
        .collect()
}

fn run() -> Result<Metrics, String> {
    let mesh = generate_icosphere_mesh(0.1, 0);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        let drive = 1.0e-3 * (1.0 + (index as f64 * 0.37).sin());
        element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(drive, 0.0)]);
        element.dof_addresses = vec![index];
    }
    let physics = PhysicsParams::new(500.0, 343.0, 1.21, false);
    let system = build_tbem_system(&elements, &mesh.nodes, &physics);
    let dim = system.num_dofs;
    if dim == 0 || system.matrix.nrows() != dim || system.matrix.ncols() != dim {
        return Err(format!(
            "BEM assembly produced invalid dimensions: dofs={dim}, rows={}, cols={}",
            system.matrix.nrows(),
            system.matrix.ncols()
        ));
    }

    let mut row_ptrs = Vec::with_capacity(dim + 1);
    let mut col_indices = Vec::with_capacity(dim * dim);
    let mut values = Vec::with_capacity(dim * dim);
    row_ptrs.push(0);
    for row in 0..dim {
        for col in 0..dim {
            col_indices.push(col);
            values.push(system.matrix.as_slice()[row * dim + col]);
        }
        row_ptrs.push(values.len());
    }
    let host = HostCsrRef::try_new(dim, dim, &row_ptrs, &col_indices, &values)
        .map_err(|error| format!("BEM dense-to-CSR conversion failed: {error}"))?;
    let backend = CpuBackend::new();
    let operator = AdjointDeviceCsr::try_from_host_csr_ref(&backend, host)
        .map_err(|error| format!("BEM compute CSR upload failed: {error}"))?;

    let x_host: Vec<_> = (0..dim)
        .map(|index| Complex64::new((index as f64 * 0.23).sin(), (index as f64 * 0.17).cos()))
        .collect();
    let y_host: Vec<_> = (0..dim)
        .map(|index| Complex64::new((index as f64 * 0.31).cos(), (index as f64 * 0.11).sin()))
        .collect();
    let mut x = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("x allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("y allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&x_host, &mut x)
        .map_err(|error| format!("x upload failed: {error}"))?;
    backend
        .try_copy_h2d(&y_host, &mut y)
        .map_err(|error| format!("y upload failed: {error}"))?;
    let mut ax = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Ax allocation failed: {error}"))?;
    let mut ahy = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("Ahy allocation failed: {error}"))?;
    operator
        .try_apply(&backend, &x, &mut ax)
        .map_err(|error| format!("forward BEM application failed: {error}"))?;
    operator
        .try_apply_hermitian(&backend, &y, &mut ahy)
        .map_err(|error| format!("Hermitian BEM application failed: {error}"))?;
    let mut ax_host = vec![Complex64::default(); dim];
    let mut ahy_host = vec![Complex64::default(); dim];
    backend.copy_d2h(&ax, &mut ax_host);
    backend.copy_d2h(&ahy, &mut ahy_host);
    let expected_forward = dense_forward(system.matrix.as_slice(), dim, &x_host);
    let expected_adjoint = dense_adjoint(system.matrix.as_slice(), dim, &y_host);
    let forward_error = relative_error(&ax_host, &expected_forward);
    let adjoint_error = relative_error(&ahy_host, &expected_adjoint);
    let inner_product_error = (dotc(&ax_host, &y_host) - dotc(&x_host, &ahy_host)).norm()
        / dotc(&ax_host, &y_host).norm().max(1.0e-30);

    let block_input_host = [x_host.as_slice(), y_host.as_slice()].concat();
    let mut block_input = backend
        .raw_alloc::<Complex64>(2 * dim)
        .map_err(|error| format!("block input allocation failed: {error}"))?;
    let mut block_output = backend
        .raw_alloc::<Complex64>(2 * dim)
        .map_err(|error| format!("block output allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&block_input_host, &mut block_input)
        .map_err(|error| format!("block input upload failed: {error}"))?;
    operator
        .try_apply_hermitian_colmajor(&backend, 2, &block_input, &mut block_output)
        .map_err(|error| format!("Hermitian BEM block application failed: {error}"))?;
    let mut block_output_host = vec![Complex64::default(); 2 * dim];
    backend.copy_d2h(&block_output, &mut block_output_host);
    let expected_block = [
        dense_adjoint(system.matrix.as_slice(), dim, &x_host),
        dense_adjoint(system.matrix.as_slice(), dim, &y_host),
    ]
    .concat();
    let block_error = relative_error(&block_output_host, &expected_block);

    // Exercise the production backend GMRES consumer as well as raw operator
    // application: construct b = A*x_known so the independent solution is
    // known without introducing a second solver implementation.
    let mut rhs = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("GMRES RHS allocation failed: {error}"))?;
    let mut solution = backend
        .raw_alloc::<Complex64>(dim)
        .map_err(|error| format!("GMRES solution allocation failed: {error}"))?;
    backend
        .try_copy_h2d(&expected_forward, &mut rhs)
        .map_err(|error| format!("GMRES RHS upload failed: {error}"))?;
    backend
        .try_copy_h2d(&vec![Complex64::default(); dim], &mut solution)
        .map_err(|error| format!("GMRES initial solution upload failed: {error}"))?;
    let gmres_status = gmres_backend_preconditioned(
        &backend,
        &operator,
        &IdentityDevicePreconditioner,
        &rhs,
        None,
        &mut solution,
        &GmresBackendConfig::<Complex64> {
            max_iterations: 100,
            tolerance: 1.0e-10,
        },
    )
    .map_err(|error| format!("BEM compute GMRES failed: {error}"))?;
    let mut solution_host = vec![Complex64::default(); dim];
    backend.copy_d2h(&solution, &mut solution_host);
    let solve_error = relative_error(&solution_host, &x_host);
    let solve_residual = gmres_status.final_residual;

    if !forward_error.is_finite()
        || !adjoint_error.is_finite()
        || !inner_product_error.is_finite()
        || !block_error.is_finite()
        || !solve_error.is_finite()
        || !solve_residual.is_finite()
        || forward_error > APPLY_TOLERANCE
        || adjoint_error > APPLY_TOLERANCE
        || inner_product_error > INNER_PRODUCT_TOLERANCE
        || block_error > BLOCK_TOLERANCE
        || !gmres_status.converged
        || solve_error > 1.0e-8
        || solve_residual > 1.0e-10
    {
        return Err(format!(
            "BEM compute parity failed: forward={forward_error:.3e}, adjoint={adjoint_error:.3e}, inner={inner_product_error:.3e}, block={block_error:.3e}, solve={solve_error:.3e}, residual={solve_residual:.3e}, converged={}",
            gmres_status.converged
        ));
    }

    Ok(Metrics {
        dofs: dim,
        nnz: values.len(),
        forward_error,
        adjoint_error,
        inner_product_error,
        block_error,
        solve_error,
        solve_residual,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("dofs", metrics.dofs as f64)
                .metric("nnz", metrics.nnz as f64)
                .metric("forward_error", metrics.forward_error)
                .metric("adjoint_error", metrics.adjoint_error)
                .metric("inner_product_error", metrics.inner_product_error)
                .metric("block_error", metrics.block_error)
                .metric("solve_error", metrics.solve_error)
                .metric("solve_residual", metrics.solve_residual)
                .maximum("forward_error", APPLY_TOLERANCE)
                .maximum("adjoint_error", APPLY_TOLERANCE)
                .maximum("inner_product_error", INNER_PRODUCT_TOLERANCE)
                .maximum("block_error", BLOCK_TOLERANCE)
                .maximum("solve_error", 1.0e-8)
                .maximum("solve_residual", 1.0e-10)
                .input_checksum("production-bem:icosphere-subdivision-0:tbem-v1")
                .reference("independent dense row accumulation and Hermitian identity")
                .reference_checksum("analytic/deterministic-fixture:compute-bem-adjoint-v1")
                .dimension("constant-panel TBEM assembled then uploaded as dense CSR")
                .formulation("CPU DeviceCsr forward/Hermitian scalar and column-major block")
                .tolerance("forward/adjoint <= 2e-12")
                .seed("compute-bem-adjoint:icosphere0:k500:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute BEM adjoint result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute BEM adjoint failure result must serialize")
            );
            eprintln!("COMPUTE BEM ADJOINT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_bem_matrix_reaches_compute_adjoint_path() {
        let metrics = run().expect("BEM compute parity");
        assert!(metrics.dofs >= 1);
        assert_eq!(metrics.nnz, metrics.dofs * metrics.dofs);
        assert!(metrics.forward_error <= APPLY_TOLERANCE);
        assert!(metrics.adjoint_error <= APPLY_TOLERANCE);
        assert!(metrics.inner_product_error <= INNER_PRODUCT_TOLERANCE);
        assert!(metrics.block_error <= BLOCK_TOLERANCE);
        assert!(metrics.solve_error <= 1.0e-8);
        assert!(metrics.solve_residual <= 1.0e-10);
    }
}
