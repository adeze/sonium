use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_time_integration::*;

fn state() -> NewmarkState<f64> {
    NewmarkState::new(vec![0.0], vec![0.0], vec![0.0])
}

#[test]
fn default_adjoint_budget_rejects_before_schedule_or_callbacks() {
    let matrix = CsrMatrix::<f64>::identity(1);
    let config = NewmarkConfig::average_acceleration(0.1, usize::MAX - 1);
    let error = newmark_adjoint_checkpointed(
        None::<&CsrMatrix<f64>>,
        &matrix,
        state(),
        &[1.0],
        &[0.0],
        &[0.0],
        |_, _| panic!("forward callback must not run"),
        |_, _, _| panic!("adjoint callback must not run"),
        &config,
        1,
    )
    .unwrap_err();
    assert!(error.to_string().contains("memory admission"));
    let config = GeneralizedAlphaConfig::from_spectral_radius(0.5, 0.1, usize::MAX - 1);
    let error = generalized_alpha_adjoint_checkpointed(
        &matrix,
        None::<&CsrMatrix<f64>>,
        &matrix,
        state(),
        &[1.0],
        &[0.0],
        &[0.0],
        |_, _| panic!("forward callback must not run"),
        |_, _, _| panic!("adjoint callback must not run"),
        &config,
        1,
    )
    .unwrap_err();
    assert!(error.to_string().contains("memory admission"));
}

#[test]
fn explicit_budget_has_a_checked_boundary_for_both_schemes() {
    let matrix = CsrMatrix::<f64>::identity(1);
    let bytes = checkpointed_adjoint_memory_bytes::<f64>(1, 1, 1) as usize;
    for budget in [0, bytes - 1, bytes] {
        let newmark = newmark_adjoint_checkpointed_with_memory_budget(
            None::<&CsrMatrix<f64>>,
            &matrix,
            state(),
            &[1.0],
            &[0.0],
            &[0.0],
            |_, state| Ok(state.clone()),
            |_, _, rhs| Ok(rhs.to_vec()),
            &NewmarkConfig::average_acceleration(0.1, 1),
            1,
            budget,
        );
        let alpha = generalized_alpha_adjoint_checkpointed_with_memory_budget(
            &matrix,
            None::<&CsrMatrix<f64>>,
            &matrix,
            state(),
            &[1.0],
            &[0.0],
            &[0.0],
            |_, state| Ok(state.clone()),
            |_, _, rhs| Ok(rhs.to_vec()),
            &GeneralizedAlphaConfig::from_spectral_radius(0.5, 0.1, 1),
            1,
            budget,
        );
        assert_eq!(newmark.is_ok(), budget == bytes);
        assert_eq!(alpha.is_ok(), budget == bytes);
    }
}

#[test]
fn estimates_include_output_schedule_and_checkpoint_storage_without_wrapping() {
    assert!(checkpointed_adjoint_memory_bytes::<f64>(0, usize::MAX, 1) > usize::MAX as u128);
    assert!(
        checkpointed_adjoint_memory_bytes::<f64>(usize::MAX, usize::MAX, usize::MAX)
            > usize::MAX as u128
    );
    let small = checkpointed_adjoint_memory_bytes::<f64>(100, 100, 1);
    assert!(small >= 100 * 100 * 8);
    assert!(checkpointed_adjoint_memory_bytes::<f64>(100, 100, 2) > small);
    assert!(checkpointed_adjoint_memory_bytes::<f64>(100, 200, 1) > small);
}
