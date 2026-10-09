use sonium_solvers::*;

fn main() {
    for &n in &[32, 100, 316] {
        let nn = n * n;
        let mut triplets = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let row = i * n + j;
                triplets.push((row, row, 4.0));
                if i > 0 {
                    triplets.push((row, (i - 1) * n + j, -1.0));
                }
                if i + 1 < n {
                    triplets.push((row, (i + 1) * n + j, -1.0));
                }
                if j > 0 {
                    triplets.push((row, i * n + j - 1, -1.0));
                }
                if j + 1 < n {
                    triplets.push((row, i * n + j + 1, -1.0));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(nn, nn, triplets);
        let rhs = vec![1.0_f64; nn];
        let config = GmresConfig {
            max_iterations: 500,
            restart: 30,
            tolerance: 1e-6,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            ..Default::default()
        };

        // SA-AMG (Jacobi-smoothed, A^T*A strength)
        let amg_sa = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_aggregation());
        let d = amg_sa.diagnostics();
        let t = std::time::Instant::now();
        let sol = gmres_preconditioned(&matrix, &amg_sa, &rhs, &config);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "SA   n={nn}: levels={}, op_cx={:.2}, setup={:.1}ms, iters={}, solve={:.1}ms, total={:.1}ms",
            d.num_levels,
            d.operator_complexity,
            amg_sa.setup_time_ms(),
            sol.iterations,
            ms,
            amg_sa.setup_time_ms() + ms
        );

        // Energy-Minimizing AMG
        let amg_em = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_energy_minimizing());
        let d = amg_em.diagnostics();
        let t = std::time::Instant::now();
        let sol = gmres_preconditioned(&matrix, &amg_em, &rhs, &config);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "EM   n={nn}: levels={}, op_cx={:.2}, setup={:.1}ms, iters={}, solve={:.1}ms, total={:.1}ms",
            d.num_levels,
            d.operator_complexity,
            amg_em.setup_time_ms(),
            sol.iterations,
            ms,
            amg_em.setup_time_ms() + ms
        );

        // GNN-driven coarsening (requires the `experimental-gnn` feature)
        #[cfg(feature = "experimental-gnn")]
        {
            let amg_gnn = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_gnn());
            let d = amg_gnn.diagnostics();
            let t = std::time::Instant::now();
            let sol = gmres_preconditioned(&matrix, &amg_gnn, &rhs, &config);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "GNN  n={nn}: levels={}, op_cx={:.2}, setup={:.1}ms, iters={}, solve={:.1}ms, total={:.1}ms",
                d.num_levels,
                d.operator_complexity,
                amg_gnn.setup_time_ms(),
                sol.iterations,
                ms,
                amg_gnn.setup_time_ms() + ms
            );
        }

        eprintln!();
    }
}
