use sonium_compute::cpu::CpuBackend;
use sonium_compute::gmres::BackendCsr;
use sonium_compute::gmres_ir::{BackendCsrF32, GmresIrConfig, gmres_ir_backend_preconditioned};
use sonium_compute::metal::MetalBackend;
use sonium_compute::{Backend, ChebyshevBackendF32, PolynomialJacobiBackendF32};

fn run<B: Backend>(label: &str, backend: &B, n: usize) {
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_idx: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(-1.0);
        }
        col_idx.push(i as i32);
        values.push(2.0);
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(-1.0);
        }
        row_ptr[i + 1] = values.len() as i32;
    }
    let csr_f64 = BackendCsr::try_from_host(backend, n, n, &row_ptr, &col_idx, &values)
        .expect("smoke fixture CSR upload must succeed");
    let csr_f32 =
        BackendCsrF32::try_from_host_cast::<f64>(backend, n, n, &row_ptr, &col_idx, &values)
            .expect("smoke fixture f32 CSR upload must succeed");
    let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let b_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.0123).sin() + 1.0).collect();
    let mut b_dev = backend.raw_alloc::<f64>(n).unwrap();
    backend.copy_h2d(&b_host, &mut b_dev);

    // Polyjac K=2
    {
        let pc = PolynomialJacobiBackendF32::from_csr_f32(
            backend,
            &csr_f32,
            &row_ptr,
            &col_idx,
            &values_f32,
            2,
            1.0,
        );
        let mut out = backend.raw_alloc::<f64>(n).unwrap();
        let start = std::time::Instant::now();
        let s = gmres_ir_backend_preconditioned(
            backend,
            &csr_f64,
            &csr_f32,
            &pc,
            &b_dev,
            None,
            &mut out,
            &GmresIrConfig {
                max_outer_iterations: 60,
                outer_tolerance: 1.0e-8,
                max_inner_iterations: 80,
                inner_tolerance: 1.0e-3,
                ..Default::default()
            },
        )
        .unwrap();
        let elapsed = start.elapsed();
        println!(
            "{label} polyjac K=2: converged={} outer={} inner={} resid={:.3e} elapsed={:?}",
            s.converged,
            s.outer_iterations,
            s.total_inner_iterations,
            s.final_outer_residual,
            elapsed
        );
    }
    // Cheb K=5
    {
        let pc = ChebyshevBackendF32::from_csr_f32(
            backend,
            &csr_f32,
            &row_ptr,
            &col_idx,
            &values_f32,
            5,
            30.0,
        );
        let mut out = backend.raw_alloc::<f64>(n).unwrap();
        let start = std::time::Instant::now();
        let s = gmres_ir_backend_preconditioned(
            backend,
            &csr_f64,
            &csr_f32,
            &pc,
            &b_dev,
            None,
            &mut out,
            &GmresIrConfig {
                max_outer_iterations: 60,
                outer_tolerance: 1.0e-8,
                max_inner_iterations: 80,
                inner_tolerance: 1.0e-3,
                ..Default::default()
            },
        )
        .unwrap();
        let elapsed = start.elapsed();
        println!(
            "{label} cheb K=5:    converged={} outer={} inner={} resid={:.3e} elapsed={:?}",
            s.converged,
            s.outer_iterations,
            s.total_inner_iterations,
            s.final_outer_residual,
            elapsed
        );
    }
}

fn main() {
    for &n in &[961usize, 10000, 49729] {
        println!("\n[n = {n}]");
        let cpu = CpuBackend::new();
        run("cpu_backend", &cpu, n);
        if let Ok(metal) = MetalBackend::new() {
            run("metal      ", &metal, n);
        }
    }
}
