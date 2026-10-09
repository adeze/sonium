//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

//! CFD linear-solve backend parity QA.
//!
//! Validates the GPU-offload track that the Picard-loop linear solve goes
//! through:
//!
//! - `gmres_backend_preconditioned` (the device-resident GMRES) +
//!   `PolynomialJacobiBackend` (the GPU-friendly preconditioner) on
//!   `CpuBackend` produces the same solution as the legacy
//!   `gmres_preconditioned_with_guess` + `IluPreconditioner` path.
//! - Polynomial degree `K = 1, ω = 1` is bit-equivalent to point-Jacobi.
//! - Polynomial degree `K = 3` reduces GMRES iteration count vs `K = 1`
//!   on a moderately-conditioned problem.
//!
//! Under `--features cuda` the CUDA backend is added to the parity matrix
//! (skipped at runtime if no driver is present); under `--features metal`,
//! the Metal backend is added likewise.
//!
//! # Reading the output
//!
//! Each named subtest prints its iteration count and final residual.
//! At the end either `SUCCESS` or `FAILED` is printed. Non-zero exit
//! status on `FAILED`.

use sonium_compute::gmres::{
    BackendCsr, BackendPreconditioner, GmresBackendConfig, IdentityBackendPreconditioner,
    gmres_backend_preconditioned,
};
use sonium_compute::{
    Backend, ChebyshevBackend, CpuBackend, PointJacobiBackend, PolynomialJacobiBackend,
};
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres_preconditioned_with_guess};

/// 1-D Laplacian-shape diagonally-dominant tridiagonal `4·I − tridiag(−1, 0, −1)`,
/// size `n × n`. Diagonally dominant by 2 → both ILU and Jacobi converge
/// quickly. Same fixture as the in-tree unit tests so the QA exercises a
/// known-good problem.
fn laplacian_4_diag_csr(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut row_ptr = vec![0_i32; n + 1];
    let mut col_idx: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(-1.0);
        }
        col_idx.push(i as i32);
        values.push(4.0);
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(-1.0);
        }
        row_ptr[i + 1] = values.len() as i32;
    }
    (row_ptr, col_idx, values)
}

/// Convert i32 host CSR back to `sonium_solvers::CsrMatrix<f64>` (which
/// uses `usize` indices) so we can drive the legacy CPU GMRES.
fn csr_i32_to_solver(
    row_ptr_i32: &[i32],
    col_idx_i32: &[i32],
    values: &[f64],
    n: usize,
) -> CsrMatrix<f64> {
    let row_ptrs: Vec<usize> = row_ptr_i32.iter().map(|&v| v as usize).collect();
    let col_indices: Vec<usize> = col_idx_i32.iter().map(|&v| v as usize).collect();
    CsrMatrix::<f64>::from_raw_parts(n, n, row_ptrs, col_indices, values.to_vec())
}

fn make_buf<T: sonium_compute::scalar::BufferElement>(
    backend: &CpuBackend,
    host: &[T],
) -> sonium_compute::cpu::HostBuffer<T> {
    let mut b = backend.raw_alloc::<T>(host.len()).unwrap();
    backend.copy_h2d(host, &mut b);
    b
}

/// Solve `A x = b` via the legacy CPU path: sonium-solvers GMRES + ILU.
fn solve_legacy_cpu(csr: &CsrMatrix<f64>, b: &[f64]) -> (Vec<f64>, usize, f64) {
    let precond = IluPreconditioner::from_csr(csr);
    let cfg = GmresConfig {
        max_iterations: 200,
        tolerance: 1.0e-10,
        ..Default::default()
    };
    let result = gmres_preconditioned_with_guess(csr, &precond, b, None, &cfg);
    (result.x, result.iterations, result.residual)
}

/// Solve `A x = b` via the backend GMRES with the given preconditioner.
fn solve_backend<B: Backend, P: BackendPreconditioner<B>>(
    backend: &B,
    a: &BackendCsr<B>,
    b_host: &[f64],
    precond: &P,
) -> (Vec<f64>, usize, f64) {
    let n = a.n;
    let b_dev = {
        let mut buf = backend.raw_alloc::<f64>(n).unwrap();
        backend.copy_h2d(b_host, &mut buf);
        buf
    };
    let mut out = backend.raw_alloc::<f64>(n).unwrap();
    let status = gmres_backend_preconditioned(
        backend,
        a,
        precond,
        &b_dev,
        None,
        &mut out,
        &GmresBackendConfig {
            max_iterations: 200,
            tolerance: 1.0e-10,
        },
    )
    .expect("backend GMRES");
    let mut x_host = vec![0.0_f64; n];
    backend.copy_d2h(&out, &mut x_host);
    (x_host, status.iterations, status.final_residual)
}

/// Element-wise compare `expected` (CPU baseline) and `got`. Returns
/// `Err(message)` if any pair differs by more than `tol`.
fn check_parity(label: &str, expected: &[f64], got: &[f64], tol: f64) -> Result<(), String> {
    if expected.len() != got.len() {
        return Err(format!(
            "{label}: length mismatch ({} vs {})",
            expected.len(),
            got.len()
        ));
    }
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        let diff = (a - b).abs();
        if !diff.is_finite() || diff > tol {
            return Err(format!(
                "{label}: row {i}: expected {a}, got {b}, diff {diff:.3e} > tol {tol:.3e}"
            ));
        }
    }
    Ok(())
}

/// Subtest 1: backend (CpuBackend) + identity preconditioner matches
/// legacy CPU GMRES + identity preconditioner. Pure dispatcher correctness.
fn subtest_identity_parity(n: usize) -> Result<(), String> {
    let backend = CpuBackend::new();
    let (rp, ci, va) = laplacian_4_diag_csr(n);
    let solver_csr = csr_i32_to_solver(&rp, &ci, &va, n);
    let device_csr = BackendCsr::try_from_host(&backend, n, n, &rp, &ci, &va)
        .map_err(|error| format!("fixture upload failed: {error}"))?;
    let b_host: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin() + 1.0).collect();

    // Legacy with ILU is fine — we're checking the *backend* against a
    // well-conditioned reference, so ILU vs identity preconditioner only
    // changes iter count, not the converged solution.
    let (cpu_x, cpu_iters, cpu_resid) = solve_legacy_cpu(&solver_csr, &b_host);

    let (be_x, be_iters, be_resid) = solve_backend(
        &backend,
        &device_csr,
        &b_host,
        &IdentityBackendPreconditioner,
    );

    println!(
        "  identity:    cpu iters={cpu_iters:3} resid={cpu_resid:.3e}   backend iters={be_iters:3} resid={be_resid:.3e}"
    );
    check_parity("identity", &cpu_x, &be_x, 1.0e-7)
}

/// Subtest 2: PointJacobi (degree-1 polynomial with ω = 1) matches
/// PolynomialJacobi(degree=1, ω=1.0) bit-for-bit and converges to the
/// same solution as the legacy CPU path.
fn subtest_point_jacobi_vs_polynomial_degree_one(n: usize) -> Result<(), String> {
    let backend = CpuBackend::new();
    let (rp, ci, va) = laplacian_4_diag_csr(n);
    let solver_csr = csr_i32_to_solver(&rp, &ci, &va, n);
    let device_csr = BackendCsr::try_from_host(&backend, n, n, &rp, &ci, &va)
        .map_err(|error| format!("fixture upload failed: {error}"))?;
    let b_host: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.07).cos() + 0.5).collect();

    let (cpu_x, _, _) = solve_legacy_cpu(&solver_csr, &b_host);

    // Apply each preconditioner to the same residual and compare.
    let pj = PointJacobiBackend::from_csr_host(&backend, n, &rp, &ci, &va);
    let poly1 =
        PolynomialJacobiBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 1, 1.0);

    let r = make_buf(&backend, &b_host);
    let mut z_pj = backend.raw_alloc::<f64>(n).unwrap();
    let mut z_poly1 = backend.raw_alloc::<f64>(n).unwrap();
    // The `BackendPreconditioner::apply` trait signature is
    // `apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>)`.
    // The operator (`device_csr` / its prebake) is passed at construction,
    // not at apply-time — passing it again here was a stale call signature.
    pj.apply(&backend, &r, &mut z_pj);
    poly1.apply(&backend, &r, &mut z_poly1);
    let mut h_pj = vec![0.0_f64; n];
    let mut h_poly1 = vec![0.0_f64; n];
    backend.copy_d2h(&z_pj, &mut h_pj);
    backend.copy_d2h(&z_poly1, &mut h_poly1);
    check_parity("apply: poly(1) vs point", &h_pj, &h_poly1, 1.0e-15)?;

    // GMRES with each must converge to the same answer.
    let (be_pj_x, pj_iters, pj_resid) = solve_backend(&backend, &device_csr, &b_host, &pj);
    let (be_p1_x, p1_iters, p1_resid) = solve_backend(&backend, &device_csr, &b_host, &poly1);
    println!(
        "  jacobi-1:    point iters={pj_iters:3} resid={pj_resid:.3e}   poly-1 iters={p1_iters:3} resid={p1_resid:.3e}"
    );

    if pj_iters != p1_iters {
        return Err(format!(
            "iteration counts differ: point={pj_iters} poly-1={p1_iters} (must be equal — same operator)"
        ));
    }
    check_parity("solution: poly(1) vs point", &be_pj_x, &be_p1_x, 1.0e-12)?;
    check_parity("solution: poly(1) vs cpu-ilu", &cpu_x, &be_p1_x, 1.0e-7)
}

/// Subtest: Chebyshev(K=4) converges to the same solution as CPU+ILU.
/// Iteration-count comparison vs PolynomialJacobi is logged for human
/// inspection but not asserted — the relative ordering depends on how
/// well the default `eigenvalue_ratio = 30` matches the actual spectrum
/// of `D⁻¹A`, which varies by problem.
fn subtest_chebyshev(n: usize) -> Result<(), String> {
    let backend = CpuBackend::new();
    let (rp, ci, va) = laplacian_4_diag_csr(n);
    let solver_csr = csr_i32_to_solver(&rp, &ci, &va, n);
    let device_csr = BackendCsr::try_from_host(&backend, n, n, &rp, &ci, &va)
        .map_err(|error| format!("fixture upload failed: {error}"))?;
    let b_host: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.17).cos() + 0.5).collect();

    let (cpu_x, _, _) = solve_legacy_cpu(&solver_csr, &b_host);
    let cheb = ChebyshevBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 4, 30.0);
    let poly = PolynomialJacobiBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 4, 1.0);

    let (be_cheb_x, cheb_iters, cheb_resid) = solve_backend(&backend, &device_csr, &b_host, &cheb);
    let (_, poly_iters, _) = solve_backend(&backend, &device_csr, &b_host, &poly);
    println!(
        "  cheb-K=4:    cheb iters={cheb_iters:3} resid={cheb_resid:.3e}   poly-4 iters={poly_iters:3} (info)"
    );

    check_parity("solution: cheb(4) vs cpu-ilu", &cpu_x, &be_cheb_x, 1.0e-6)
}

/// Subtest 3: PolynomialJacobi(K=3) converges to the same solution as
/// CPU+ILU AND uses ≤ as many GMRES outer iters as PolynomialJacobi(K=1)
/// (preconditioner-strength regression test).
fn subtest_polynomial_degree_three(n: usize) -> Result<(), String> {
    let backend = CpuBackend::new();
    let (rp, ci, va) = laplacian_4_diag_csr(n);
    let solver_csr = csr_i32_to_solver(&rp, &ci, &va, n);
    let device_csr = BackendCsr::try_from_host(&backend, n, n, &rp, &ci, &va)
        .map_err(|error| format!("fixture upload failed: {error}"))?;
    let b_host: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.13).sin() + 1.0).collect();

    let (cpu_x, _, _) = solve_legacy_cpu(&solver_csr, &b_host);
    let poly1 =
        PolynomialJacobiBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 1, 1.0);
    let poly3 =
        PolynomialJacobiBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 3, 1.0);

    let (_, p1_iters, _) = solve_backend(&backend, &device_csr, &b_host, &poly1);
    let (be_p3_x, p3_iters, p3_resid) = solve_backend(&backend, &device_csr, &b_host, &poly3);
    println!(
        "  poly-K:      poly-1 iters={p1_iters:3}  poly-3 iters={p3_iters:3} resid={p3_resid:.3e}"
    );

    if p3_iters > p1_iters {
        return Err(format!(
            "poly-3 ({p3_iters}) took more outer GMRES iters than poly-1 ({p1_iters}) — regression"
        ));
    }
    check_parity("solution: poly(3) vs cpu-ilu", &cpu_x, &be_p3_x, 1.0e-7)
}

/// Subtest 4 (cfg-gated): CUDA backend reproduces the CPU baseline.
#[cfg(feature = "cuda")]
fn subtest_cuda(n: usize) -> Result<(), String> {
    let backend = match sonium_compute::cuda::CudaBackend::new_default() {
        Ok(b) => b,
        Err(err) => {
            println!("  cuda:        skipped ({err:?})");
            return Ok(());
        }
    };
    let (rp, ci, va) = laplacian_4_diag_csr(n);
    let solver_csr = csr_i32_to_solver(&rp, &ci, &va, n);
    let device_csr = BackendCsr::try_from_host(&backend, n, n, &rp, &ci, &va)
        .map_err(|error| format!("fixture upload failed: {error}"))?;
    let b_host: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.05).sin() + 1.0).collect();
    let (cpu_x, _, _) = solve_legacy_cpu(&solver_csr, &b_host);
    let poly = PolynomialJacobiBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 2, 1.0);
    let (cuda_x, iters, resid) = solve_backend(&backend, &device_csr, &b_host, &poly);
    println!("  cuda:        iters={iters:3} resid={resid:.3e}");
    check_parity("solution: cuda vs cpu-ilu", &cpu_x, &cuda_x, 1.0e-6)
}

/// Subtest 4 (cfg-gated): Metal backend reproduces the CPU baseline.
#[cfg(feature = "metal")]
fn subtest_metal(n: usize) -> Result<(), String> {
    let backend = match sonium_compute::metal::MetalBackend::new() {
        Ok(b) => b,
        Err(err) => {
            println!("  metal:       skipped ({err:?})");
            return Ok(());
        }
    };
    let (rp, ci, va) = laplacian_4_diag_csr(n);
    let solver_csr = csr_i32_to_solver(&rp, &ci, &va, n);
    let device_csr = BackendCsr::try_from_host(&backend, n, n, &rp, &ci, &va)
        .map_err(|error| format!("fixture upload failed: {error}"))?;
    let b_host: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.05).sin() + 1.0).collect();
    let (cpu_x, _, _) = solve_legacy_cpu(&solver_csr, &b_host);
    let poly = PolynomialJacobiBackend::from_csr_host(&backend, &device_csr, &rp, &ci, &va, 2, 1.0);
    let (metal_x, iters, resid) = solve_backend(&backend, &device_csr, &b_host, &poly);
    println!("  metal:       iters={iters:3} resid={resid:.3e}");
    check_parity("solution: metal vs cpu-ilu", &cpu_x, &metal_x, 1.0e-6)
}

fn run() -> Result<(), String> {
    for &n in &[64_usize, 256, 1024] {
        println!("[n = {n}]");
        subtest_identity_parity(n)?;
        subtest_point_jacobi_vs_polynomial_degree_one(n)?;
        subtest_polynomial_degree_three(n)?;
        subtest_chebyshev(n)?;

        #[cfg(feature = "cuda")]
        subtest_cuda(n)?;

        #[cfg(feature = "metal")]
        subtest_metal(n)?;
    }
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(err) => {
            eprintln!("PARITY CHECK FAILED: {err}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
