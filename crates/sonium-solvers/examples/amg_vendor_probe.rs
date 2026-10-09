//! AMG vendor-parity probe: CG + AmgPreconditioner on 3D Poisson 7-point,
//! times setup + solve, dumps CSR + rhs + solution to `/tmp/amg_n{n}.bin`
//! for the `/tmp/petsc_amg_compare.py` PETSc (BoomerAMG/GAMG) check.
//!
//! Run: `cargo run --release -p sonium-solvers --example amg_vendor_probe`

use sonium_solvers::{
    AmgCoarsening, AmgConfig, AmgInterpolation, AmgPreconditioner, AmgSmoother, CgConfig,
    CsrMatrix, Preconditioner, pcg,
};

/// hypre-style interpolation table: entries/row, weight range, row-sum
/// range per level, for direct comparison against BoomerAMG's
/// `print_statistics` output.
fn print_p_stats(name: &str, amg: &AmgPreconditioner<f64>) {
    println!("{name} P stats: lev rows x cols min max avgW minW maxW minSum maxSum");
    let mut lev = 0;
    while let Some(p) = amg.level_prolongation(lev) {
        let n = p.num_rows;
        let mut min_e = usize::MAX;
        let mut max_e = 0usize;
        let mut tot_e = 0usize;
        let mut min_w = f64::INFINITY;
        let mut max_w = f64::NEG_INFINITY;
        let mut min_s = f64::INFINITY;
        let mut max_s = f64::NEG_INFINITY;
        for i in 0..n {
            let mut cnt = 0;
            let mut sum = 0.0;
            for (_, v) in p.row_entries(i) {
                cnt += 1;
                sum += v;
                if v < min_w {
                    min_w = v;
                }
                if v > max_w {
                    max_w = v;
                }
            }
            min_e = min_e.min(cnt);
            max_e = max_e.max(cnt);
            tot_e += cnt;
            min_s = min_s.min(sum);
            max_s = max_s.max(sum);
        }
        println!(
            "  {lev} {n} x ? {min_e} {max_e} {:.1} {min_w:.3e} {max_w:.3e} {min_s:.3e} {max_s:.3e}",
            tot_e as f64 / n as f64
        );
        lev += 1;
    }
}

/// Stationary iteration x += M(b - Ax): prints the relative residual per
/// cycle, i.e. the V-cycle convergence factor independent of the outer CG.
fn richardson_factor(name: &str, matrix: &CsrMatrix<f64>, b: &[f64], config: AmgConfig) {
    let amg = AmgPreconditioner::from_csr(matrix, config);
    let bnorm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut x = vec![0.0; b.len()];
    let mut factors = Vec::new();
    for _ in 0..6 {
        let ax = matrix.matvec(&x);
        let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
        let rn: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt() / bnorm;
        factors.push(rn);
        let c = amg.apply(&r);
        for (xi, ci) in x.iter_mut().zip(c.iter()) {
            *xi += ci;
        }
    }
    let s: Vec<String> = factors.iter().map(|v| format!("{v:.2e}")).collect();
    println!("{name} richardson: {}", s.join(" "));
}
use std::time::Instant;

fn poisson_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut triplets = Vec::with_capacity(7 * nn);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = i * n * n + j * n + k;
                triplets.push((row, row, 6.0));
                if i > 0 {
                    triplets.push((row, (i - 1) * n * n + j * n + k, -1.0));
                }
                if i + 1 < n {
                    triplets.push((row, (i + 1) * n * n + j * n + k, -1.0));
                }
                if j > 0 {
                    triplets.push((row, i * n * n + (j - 1) * n + k, -1.0));
                }
                if j + 1 < n {
                    triplets.push((row, i * n * n + (j + 1) * n + k, -1.0));
                }
                if k > 0 {
                    triplets.push((row, i * n * n + j * n + k - 1, -1.0));
                }
                if k + 1 < n {
                    triplets.push((row, i * n * n + j * n + k + 1, -1.0));
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

fn run_case(name: &str, matrix: &CsrMatrix<f64>, b: &[f64], config: AmgConfig) -> Vec<f64> {
    let t0 = Instant::now();
    let amg = AmgPreconditioner::from_csr(matrix, config);
    let setup_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cg = CgConfig {
        max_iterations: 1000,
        tolerance: 1e-6,
        print_interval: 0,
    };
    // Warmup (caches, thread pool).
    let _ = pcg(matrix, &amg, b, &cg);
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..3 {
        let t1 = Instant::now();
        let sol = pcg(matrix, &amg, b, &cg);
        best = best.min(t1.elapsed().as_secs_f64() * 1000.0);
        out = Some(sol);
    }
    let sol = out.unwrap();
    let d = amg.diagnostics();
    println!(
        "{name}: setup {setup_ms:.1} ms, solve {best:.1} ms, iters {}, converged {}, levels {}, grid_c {:.2}, op_c {:.2}",
        sol.iterations, sol.converged, d.num_levels, d.grid_complexity, d.operator_complexity
    );
    println!("  dims {:?}", amg.level_dims());
    assert!(sol.converged, "{name} did not converge");
    sol.x
}

fn main() {
    // Per-level setup timings (strength/coarsen/galerkin) via the library's
    // log::debug spans. Only active under `--features cli` (env_logger is a
    // cli-gated optional dep): `RUST_LOG=debug cargo run -p sonium-solvers
    // --features cli --release --example amg_vendor_probe`.
    #[cfg(feature = "cli")]
    let _ = env_logger::try_init();
    for &n in &[16usize, 24, 32] {
        let matrix = poisson_3d(n);
        let nn = n * n * n;
        let b = vec![1.0; nn];
        println!("n={n} dofs={nn} nnz={}", matrix.values.len());
        let _ = run_case("default   ", &matrix, &b, AmgConfig::default());
        // Smoother ladder on the Falgout + Standard hierarchy: L1-Jacobi,
        // full-point SGS, CF-ordered SGS. (Historical sweeps retired:
        // no-truncation and theta 0.25-0.7 measured neutral on this problem.)
        for (tag, smoother) in [
            ("falgout+l1j", AmgSmoother::L1Jacobi),
            ("falgout+sgs", AmgSmoother::SymmetricGaussSeidel),
            ("falgout+cfs", AmgSmoother::CfSymmetricGaussSeidel),
        ] {
            let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
            cfg.coarsening = AmgCoarsening::Falgout;
            cfg.interpolation = AmgInterpolation::Standard;
            cfg.smoother = smoother;
            let _ = run_case(tag, &matrix, &b, cfg);
        }
        let x = run_case(
            "hypre-mirror",
            &matrix,
            &b,
            AmgConfig::for_hypre_boomeramg_comparison(),
        );
        // Interpolation ladder: direct-only Standard vs indirect Extended
        // variants (hypre "modified classical" includes distance-2 terms).
        for (tag, interp) in [
            ("ext  ", AmgInterpolation::Extended),
            ("ext+i", AmgInterpolation::ExtendedI),
        ] {
            let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
            cfg.interpolation = interp;
            let _ = run_case(tag, &matrix, &b, cfg);
        }
        // Galerkin drop sweep on the mirror config.
        for (tag, tol) in [("drop1e-7", 1e-7), ("drop0  ", 0.0)] {
            let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
            cfg.galerkin_drop_tol = tol;
            let _ = run_case(tag, &matrix, &b, cfg);
        }
        // P-truncation sweep on extended+i: BoomerAMG-style relative-size
        // dropping (small distance-2 stencil removed on fine levels, kept
        // where relatively large on coarse levels). Count cap lifted to 64
        // so only the relative threshold binds.
        for (tag, trunc) in [("trunc0.1", 0.1), ("trunc0.3", 0.3)] {
            let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
            cfg.interpolation = AmgInterpolation::ExtendedI;
            cfg.trunc_factor = trunc;
            cfg.max_interp_elements = 64;
            let _ = run_case(tag, &matrix, &b, cfg);
        }
        // Full-point SGS on the winning truncated extended+i config: SGS
        // gains ~1 PCG iteration over CF-SGS on every hierarchy so far.
        {
            let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
            cfg.interpolation = AmgInterpolation::ExtendedI;
            cfg.trunc_factor = 0.3;
            cfg.max_interp_elements = 64;
            cfg.smoother = AmgSmoother::SymmetricGaussSeidel;
            let _ = run_case("trunc0.3sgs", &matrix, &b, cfg);
        }
        // Retired: Chebyshev smoother on the truncated extended+i hierarchy
        // measured 14-17 iters (vs 4-5 for GS variants) — the default
        // Chebyshev eigenvalue bounds do not suit the classical hierarchy.
        if n == 32 {
            let mirror =
                AmgPreconditioner::from_csr(&matrix, AmgConfig::for_hypre_boomeramg_comparison());
            print_p_stats("hypre-mirror", &mirror);
            // Extended+i P table: BoomerAMG's default extended+i reports
            // op_c ~3.65 here; compare our row density/row sums against it.
            let mut ext_cfg = AmgConfig::for_hypre_boomeramg_comparison();
            ext_cfg.interpolation = AmgInterpolation::ExtendedI;
            let ext = AmgPreconditioner::from_csr(&matrix, ext_cfg);
            print_p_stats("ext+i", &ext);
            // Truncated extended+i P table (the trunc0.3 operating point).
            let mut trunc_cfg = AmgConfig::for_hypre_boomeramg_comparison();
            trunc_cfg.interpolation = AmgInterpolation::ExtendedI;
            trunc_cfg.trunc_factor = 0.3;
            trunc_cfg.max_interp_elements = 64;
            let trunc = AmgPreconditioner::from_csr(&matrix, trunc_cfg);
            print_p_stats("trunc0.3", &trunc);
            for (tag, smoother) in [
                ("sgs", AmgSmoother::SymmetricGaussSeidel),
                ("cfs", AmgSmoother::CfSymmetricGaussSeidel),
            ] {
                let mut cfg = AmgConfig::for_hypre_boomeramg_comparison();
                cfg.coarsening = AmgCoarsening::Falgout;
                cfg.interpolation = AmgInterpolation::Standard;
                cfg.smoother = smoother;
                richardson_factor(tag, &matrix, &b, cfg.clone());
                // Two-level (one coarse correction): the fine level has a
                // C/F splitting so the CF arm actually runs; isolates sweep
                // bugs from deep-hierarchy bugs.
                let mut s1 = cfg.clone();
                s1.max_levels = 2;
                richardson_factor(&format!("{tag}-L2"), &matrix, &b, s1);
            }
        }
        // Dump: u64 n_dofs, u64 nnz, row_ptrs u64[], cols u64[], vals f64[],
        // rhs f64[], x f64[].
        let mut buf = Vec::new();
        buf.extend_from_slice(&(nn as u64).to_le_bytes());
        buf.extend_from_slice(&(matrix.values.len() as u64).to_le_bytes());
        for &v in &matrix.row_ptrs {
            buf.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for &v in &matrix.col_indices {
            buf.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for &v in &matrix.values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &b {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &x {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("/tmp/amg_n{n}.bin"), &buf).expect("dump");
    }
}
