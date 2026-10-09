//! Bounded W2-03 deterministic compute timing/memory bundle.

use num_complex::Complex64;
use sonium_bem::core::{BemProblem, BemSolver};
use sonium_compute::{
    Backend, BlockGmresConfig, CpuBackend, DeviceCsr, IdentityDevicePreconditioner,
    gmres_backend_block_preconditioned,
};
use sonium_fem::assembly::HelmholtzProblem;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::unit_square_triangles;
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "compute.reproducible-workload-bundle.v1";

#[derive(Debug, Clone)]
struct Run {
    output: [f64; 4],
    max_residual: f64,
    max_solution_error: f64,
    memory_bytes: u64,
    elapsed_ms: u64,
}

#[derive(Debug, Clone)]
struct FemBemRun {
    fem_output: Vec<Complex64>,
    bem_output: Vec<Complex64>,
    memory_bytes: u64,
    elapsed_ms: u64,
}

fn run_once(backend: &CpuBackend) -> Result<Run, String> {
    let started = Instant::now();
    let matrix = DeviceCsr::try_from_host(
        backend,
        2,
        2,
        &[0, 2, 4],
        &[0, 1, 0, 1],
        &[4.0_f64, 1.0, 1.0, 3.0],
    )
    .map_err(|error| format!("fixture upload failed: {error}"))?;
    let mut rhs = backend.raw_alloc::<f64>(4).map_err(|e| e.to_string())?;
    backend
        .try_copy_h2d(&[9.0, 8.0, 3.0, 7.0], &mut rhs)
        .map_err(|e| e.to_string())?;
    let mut solution = backend.raw_alloc::<f64>(4).map_err(|e| e.to_string())?;
    let status = gmres_backend_block_preconditioned(
        backend,
        &matrix,
        &IdentityDevicePreconditioner,
        &rhs,
        None,
        &mut solution,
        2,
        &BlockGmresConfig {
            max_block_iterations: 4,
            tolerance: 1.0e-10,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut output = [0.0; 4];
    backend
        .try_copy_d2h(&solution, &mut output)
        .map_err(|e| e.to_string())?;
    let expected = [19.0 / 11.0, 23.0 / 11.0, 2.0 / 11.0, 25.0 / 11.0];
    let max_residual = status
        .rhs
        .iter()
        .map(|rhs| rhs.final_residual)
        .fold(0.0, f64::max);
    let max_solution_error = output
        .iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max);
    if !status.converged || !status.rhs.iter().all(|rhs| rhs.converged) {
        return Err(format!(
            "reproducible workload did not converge: {status:?}"
        ));
    }
    // Explicitly account for host/device buffers and CSR metadata. This is a
    // deterministic workload footprint, not a claim about allocator RSS.
    let memory_bytes =
        (3 * 4 * std::mem::size_of::<i32>() + 4 * std::mem::size_of::<f64>() * 4) as u64;
    Ok(Run {
        output,
        max_residual,
        max_solution_error,
        memory_bytes,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn run_fem_bem_once() -> Result<FemBemRun, String> {
    let started = Instant::now();
    let fem_mesh = unit_square_triangles(1);
    let fem = HelmholtzProblem::try_assemble(
        &fem_mesh,
        PolynomialDegree::P1,
        Complex64::new(1.25, 0.0),
        |x, y, _| Complex64::new(1.0 + x - 0.5 * y, 0.0),
    )
    .map_err(|error| format!("reproducible FEM assembly failed: {error}"))?;
    let mut fem_output = fem.matrix.values.clone();
    fem_output.extend_from_slice(&fem.rhs);

    let bem_problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let bem_solution = BemSolver::new()
        .solve(&bem_problem)
        .map_err(|error| format!("reproducible BEM solve failed: {error}"))?;
    let bem_output = bem_solution.surface_pressure;
    if fem_output
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || bem_output
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("reproducible FEM/BEM workload returned non-finite values".into());
    }
    let memory_bytes =
        ((fem_output.len() + bem_output.len()) * std::mem::size_of::<Complex64>()) as u64;
    Ok(FemBemRun {
        fem_output,
        bem_output,
        memory_bytes,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn main() {
    let started = Instant::now();
    let backend = CpuBackend::new();
    match (
        run_once(&backend),
        run_once(&backend),
        run_fem_bem_once(),
        run_fem_bem_once(),
    ) {
        (Ok(first), Ok(second), Ok(fem_bem_first), Ok(fem_bem_second)) => {
            let replay_equal = first.output == second.output
                && first.max_residual.to_bits() == second.max_residual.to_bits()
                && first.max_solution_error.to_bits() == second.max_solution_error.to_bits()
                && first.memory_bytes == second.memory_bytes;
            let fem_bem_replay_equal = fem_bem_first.fem_output == fem_bem_second.fem_output
                && fem_bem_first.bem_output == fem_bem_second.bem_output
                && fem_bem_first.memory_bytes == fem_bem_second.memory_bytes;
            if !replay_equal || !fem_bem_replay_equal {
                let error = "deterministic compute/FEM/BEM replay changed output/residual/memory";
                println!(
                    "QA_RESULT: {}",
                    QaResult::failed(CASE_ID, error)
                        .requested_backend("cpu")
                        .actual_backend("cpu")
                        .to_json_line()
                        .expect("replay failure result must serialize")
                );
                println!("FAILED");
                std::process::exit(1);
            }
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_final_residual", first.max_residual)
                .metric("max_solution_error", first.max_solution_error)
                .metric("memory_bytes", first.memory_bytes as f64)
                .metric("replay_equal", 1.0)
                .metric("sample_count", 2.0)
                .metric("fem_dof_count", fem_bem_first.fem_output.len() as f64)
                .metric("bem_pressure_dof_count", fem_bem_first.bem_output.len() as f64)
                .metric("fem_bem_memory_bytes", fem_bem_first.memory_bytes as f64)
                .metric("fem_bem_replay_equal", 1.0)
                .maximum("max_final_residual", 1.0e-8)
                .maximum("max_solution_error", 1.0e-8)
                .minimum("memory_bytes", 1.0)
                .minimum("replay_equal", 1.0)
                .minimum("sample_count", 2.0)
                .minimum("fem_dof_count", 1.0)
                .minimum("bem_pressure_dof_count", 1.0)
                .minimum("fem_bem_memory_bytes", 1.0)
                .minimum("fem_bem_replay_equal", 1.0)
                .input_checksum("analytic/deterministic-fixture:csr-block-gmres-replay-v1")
                .reference("analytical dense-system solution plus bit-stable CPU, FEM, and BEM replays")
                .reference_checksum("analytic/deterministic-fixture:compute-fem-bem-replay-v1")
                .dimension("2x2 CSR, two RHS plus 2D P1 FEM and 3D rigid-sphere BEM fixtures")
                .formulation("deterministic CPU sparse, FEM assembly, and BEM solve timing/memory/provenance bundle")
                .tolerance("residual and solution error <= 1e-8; output/residual/footprint replay identical")
                .seed("compute-reproducible-workload-v1")
                .memory_bytes(first.memory_bytes)
                .timing_ms("replay_1", first.elapsed_ms)
                .timing_ms("replay_2", second.elapsed_ms)
                .timing_ms("fem_bem_replay_1", fem_bem_first.elapsed_ms)
                .timing_ms("fem_bem_replay_2", fem_bem_second.elapsed_ms)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compute result must serialize")
            );
            println!("SUCCESS");
        }
        (Err(error), _, _, _)
        | (_, Err(error), _, _)
        | (_, _, Err(error), _)
        | (_, _, _, Err(error)) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("compute failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
