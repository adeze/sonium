//! Checked TR-BDF2 tests: stiff order, Robertson, index-1 DAE, SDOF.

// Rust guideline compliant 2026-02-21

use sonium_solvers_time_integration::{TrBdf2Config, trbdf2_integrate};

fn fixed_config(dt: f64) -> TrBdf2Config {
    TrBdf2Config {
        dt_initial: dt,
        fixed_step: true,
        ..TrBdf2Config::default()
    }
}

fn observed_order(errors: &[f64]) -> f64 {
    // Successive halving: slope from least-squares fit of log2(error).
    let logs: Vec<f64> = errors.iter().map(|error| error.log2()).collect();
    let n = logs.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = logs.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, &value) in logs.iter().enumerate() {
        numerator += (index as f64 - mean_x) * (value - mean_y);
        denominator += (index as f64 - mean_x).powi(2);
    }
    -numerator / denominator
}

#[test]
fn prothero_robinson_is_second_order() {
    // y' = lambda (y - sin t) + cos t, exact y = sin t, lambda stiff.
    let lambda = -1.0e3;
    let run = |dt: f64| {
        let result = trbdf2_integrate(
            |t, y| vec![lambda * (y[0] - t.sin()) + t.cos()],
            |_, _| vec![vec![lambda]],
            None,
            &[0.0],
            0.0,
            1.0,
            &fixed_config(dt),
        )
        .expect("valid stiff run");
        (result.states.last().expect("non-empty")[0] - 1.0f64.sin()).abs()
    };
    let errors = [0.1, 0.05, 0.025].map(run);
    let order = observed_order(&errors);
    assert!(
        (1.8..=2.2).contains(&order),
        "order = {order}, errors = {errors:?}"
    );
    assert!(errors[2] < 1.0e-6, "fine error = {}", errors[2]);
}

#[test]
fn robertson_adaptive_conserves_mass_and_completes() {
    let rhs = |_: f64, y: &[f64]| {
        vec![
            -0.04 * y[0] + 1.0e4 * y[1] * y[2],
            0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
            3.0e7 * y[1] * y[1],
        ]
    };
    let jacobian = |_: f64, y: &[f64]| {
        vec![
            vec![-0.04, 1.0e4 * y[2], 1.0e4 * y[1]],
            vec![0.04, -1.0e4 * y[2] - 6.0e7 * y[1], -1.0e4 * y[1]],
            vec![0.0, 6.0e7 * y[1], 0.0],
        ]
    };
    let config = TrBdf2Config {
        dt_initial: 1.0e-6,
        rel_tol: 1.0e-6,
        abs_tol: 1.0e-10,
        ..TrBdf2Config::default()
    };
    let result = trbdf2_integrate(rhs, jacobian, None, &[1.0, 0.0, 0.0], 0.0, 40.0, &config)
        .expect("Robertson must complete");
    for state in &result.states {
        let total: f64 = state.iter().sum();
        assert!((total - 1.0).abs() < 1.0e-8, "mass = {total}");
        assert!(state.iter().all(|value| *value >= -1.0e-12));
    }
    let last = result.states.last().expect("non-empty");
    assert!(
        (0.70..=0.73).contains(&last[0]),
        "y1(40) = {} (reference ≈ 0.7158)",
        last[0]
    );
    assert!(last[1] < 1.0e-4, "y2(40) = {}", last[1]);
    // Stiffness lets the controller span many decades of step size.
    let dts: Vec<f64> = result
        .history
        .iter()
        .filter(|record| !record.rejected)
        .map(|record| record.dt)
        .collect();
    let min_dt = dts.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_dt = dts.iter().cloned().fold(0.0, f64::max);
    assert!(max_dt / min_dt > 1.0e3, "min {min_dt}, max {max_dt}");
    assert!(result.steps_rejected < result.steps_accepted);
}

#[test]
fn index1_dae_converges_second_order() {
    // u' = v, 0 = u - sin t; exact u = sin t, v = cos t.
    let mass = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
    let run = |dt: f64| {
        let result = trbdf2_integrate(
            |t, y| vec![y[1], y[0] - t.sin()],
            |_, _| vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            Some(&mass),
            &[0.0, 1.0],
            0.0,
            1.0,
            &fixed_config(dt),
        )
        .expect("valid DAE run");
        let last = result.states.last().expect("non-empty");
        let exact = [1.0f64.sin(), 1.0f64.cos()];
        [(last[0] - exact[0]).abs(), (last[1] - exact[1]).abs()]
    };
    let grids = [0.1, 0.05, 0.025];
    let errors: Vec<[f64; 2]> = grids.map(run).to_vec();
    // u is fixed algebraically by the constraint (exact to Newton tolerance);
    // the convergence order lives on the differentiated variable v.
    assert!(errors.iter().all(|pair| pair[0] < 1.0e-9));
    let series: Vec<f64> = errors.iter().map(|pair| pair[1]).collect();
    let order = observed_order(&series);
    assert!(
        order >= 1.8,
        "v-component: order = {order}, errors = {series:?}"
    );
    // The differentiated variable carries a larger error constant than the
    // constrained one; the slope above is the binding gate.
    assert!(errors[2][1] < 5.0e-4, "fine v-error = {}", errors[2][1]);
}

#[test]
fn damped_structural_sdof_matches_analytic() {
    // m u'' + c u' + k u = 0 in first-order form, underdamped analytic.
    let mass = 1.0f64;
    let damping = 0.2f64;
    let stiffness = 4.0f64;
    let omega = (stiffness / mass - (damping / (2.0 * mass)).powi(2)).sqrt();
    let config = TrBdf2Config {
        dt_initial: 1.0e-2,
        rel_tol: 1.0e-8,
        abs_tol: 1.0e-10,
        ..TrBdf2Config::default()
    };
    let result = trbdf2_integrate(
        |_, y| vec![y[1], -(damping * y[1] + stiffness * y[0]) / mass],
        |_, _| vec![vec![0.0, 1.0], vec![-stiffness / mass, -damping / mass]],
        None,
        &[1.0, 0.0],
        0.0,
        5.0,
        &config,
    )
    .expect("valid SDOF run");
    let last = result.states.last().expect("non-empty");
    let decay = (-damping / (2.0 * mass) * 5.0).exp();
    let expected_u = decay * (omega * 5.0).cos()
        + decay * (damping / (2.0 * mass * omega)) * (omega * 5.0).sin();
    // Global error after ~5 s of adaptive stepping at rel_tol = 1e-8.
    assert!(
        (last[0] - expected_u).abs() < 1.0e-5,
        "u = {}, expected {expected_u}",
        last[0]
    );
}

#[test]
fn estimator_shows_third_order_local_behavior() {
    // The companion residual is a local (per-step) quantity: halving the
    // step must shrink the mean estimate by ~8x on a smooth problem.
    let mean_estimate = |dt: f64| {
        let result = trbdf2_integrate(
            |t, y| vec![-y[0] + t.cos()],
            |_, _| vec![vec![-1.0]],
            None,
            &[1.0],
            0.0,
            2.0,
            &fixed_config(dt),
        )
        .expect("valid run");
        result
            .history
            .iter()
            .map(|record| record.error_estimate)
            .sum::<f64>()
            / result.history.len() as f64
    };
    let coarse = mean_estimate(0.04);
    let fine = mean_estimate(0.02);
    let ratio = coarse / fine;
    assert!(
        (4.0..=16.0).contains(&ratio),
        "estimator ratio = {ratio} (coarse {coarse}, fine {fine})"
    );
}

#[test]
fn invalid_stiff_inputs_are_rejected() {
    let rhs = |_: f64, y: &[f64]| vec![-y[0]];
    let jacobian = |_: f64, _: &[f64]| vec![vec![-1.0]];
    let config = TrBdf2Config::default();
    assert!(trbdf2_integrate(rhs, jacobian, None, &[1.0], 1.0, 0.0, &config).is_err());
    assert!(trbdf2_integrate(rhs, jacobian, None, &[], 0.0, 1.0, &config).is_err());
    assert!(trbdf2_integrate(rhs, jacobian, None, &[f64::NAN], 0.0, 1.0, &config).is_err());
    let bad_mass = vec![vec![1.0, 0.0]];
    assert!(trbdf2_integrate(rhs, jacobian, Some(&bad_mass), &[1.0], 0.0, 1.0, &config).is_err());
    let bad_config = TrBdf2Config {
        max_steps: 1,
        ..TrBdf2Config::default()
    };
    assert!(trbdf2_integrate(rhs, jacobian, None, &[1.0], 0.0, 10.0, &bad_config).is_err());
    let wrong_dim_rhs = |_: f64, _: &[f64]| vec![0.0, 0.0];
    assert!(trbdf2_integrate(wrong_dim_rhs, jacobian, None, &[1.0], 0.0, 1.0, &config).is_err());
}
