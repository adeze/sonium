use sonium_solvers_core::traits::{AccuracyStatus, SolverStatus};
use sonium_solvers_optimization::{
    BacktrackingConfig, HagerZhangConfig, StrongWolfeConfig, backtracking, hager_zhang_line_search,
    strong_wolfe,
};
use std::cell::Cell;

#[test]
fn backtracking_rejects_non_finite_trial_objective_immediately() {
    let calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        let call = calls.get();
        calls.set(call + 1);
        if call == 0 { f64::NAN } else { 0.0 }
    };
    let result = backtracking(
        &objective,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &BacktrackingConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(calls.get(), 1);
}

#[test]
fn strong_wolfe_rejects_non_finite_trial_objective_before_gradient() {
    let objective_calls = Cell::new(0usize);
    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        let call = objective_calls.get();
        objective_calls.set(call + 1);
        if call == 0 { f64::INFINITY } else { 0.0 }
    };
    let gradient = |_: &[f64]| {
        gradient_calls.set(gradient_calls.get() + 1);
        vec![1.0]
    };
    let result = strong_wolfe(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &StrongWolfeConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(objective_calls.get(), 1);
    assert_eq!(gradient_calls.get(), 0);
}

#[test]
fn strong_wolfe_rejects_non_finite_trial_gradient_immediately() {
    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| 0.0;
    let gradient = |_: &[f64]| {
        gradient_calls.set(gradient_calls.get() + 1);
        vec![f64::NAN]
    };
    let result = strong_wolfe(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &StrongWolfeConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(gradient_calls.get(), 1);
}

#[test]
fn hager_zhang_rejects_non_finite_trial_gradient_immediately() {
    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| 0.0;
    let gradient = |_: &[f64]| {
        let call = gradient_calls.get();
        gradient_calls.set(call + 1);
        if call == 0 {
            vec![f64::NAN]
        } else {
            vec![-1.0]
        }
    };
    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &HagerZhangConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(gradient_calls.get(), 1);
}

#[test]
fn hager_zhang_rejects_non_finite_trial_objective_immediately() {
    let objective_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        objective_calls.set(objective_calls.get() + 1);
        f64::NAN
    };
    let gradient = |_: &[f64]| vec![-1.0];
    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0],
        1.0,
        &[1.0],
        &HagerZhangConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.accuracy_status(), AccuracyStatus::NotChecked);
    assert_eq!(objective_calls.get(), 1);
}

#[test]
fn backtracking_rejects_dimension_mismatch_before_callbacks() {
    let objective_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        objective_calls.set(objective_calls.get() + 1);
        0.0
    };
    let result = backtracking(
        &objective,
        &[1.0],
        &[-1.0, -1.0],
        1.0,
        &[1.0],
        &BacktrackingConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.fun_evals, 0);
    assert_eq!(objective_calls.get(), 0);
}

#[test]
fn strong_wolfe_rejects_dimension_mismatch_before_callbacks() {
    let objective_calls = Cell::new(0usize);
    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        objective_calls.set(objective_calls.get() + 1);
        0.0
    };
    let gradient = |_: &[f64]| {
        gradient_calls.set(gradient_calls.get() + 1);
        vec![0.0]
    };
    let result = strong_wolfe(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0, -1.0],
        1.0,
        &[1.0],
        &StrongWolfeConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.fun_evals, 0);
    assert_eq!(result.grad_evals, 0);
    assert_eq!(objective_calls.get(), 0);
    assert_eq!(gradient_calls.get(), 0);
}

#[test]
fn hager_zhang_rejects_dimension_mismatch_before_callbacks() {
    let objective_calls = Cell::new(0usize);
    let gradient_calls = Cell::new(0usize);
    let objective = |_: &[f64]| {
        objective_calls.set(objective_calls.get() + 1);
        0.0
    };
    let gradient = |_: &[f64]| {
        gradient_calls.set(gradient_calls.get() + 1);
        vec![0.0]
    };
    let result = hager_zhang_line_search(
        &objective,
        &gradient,
        &[1.0],
        &[-1.0, -1.0],
        1.0,
        &[1.0],
        &HagerZhangConfig::default(),
    );

    assert_eq!(result.status(), SolverStatus::NumericalError);
    assert_eq!(result.fun_evals, 0);
    assert_eq!(result.grad_evals, 0);
    assert_eq!(objective_calls.get(), 0);
    assert_eq!(gradient_calls.get(), 0);
}
