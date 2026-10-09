use sonium_solvers_optimization::util::finite_diff::{
    jacobian_central_auto, try_jacobian_central, try_jacobian_central_parallel,
};

#[test]
fn invalid_stencils_are_rejected_before_callbacks_serial_and_parallel() {
    for n in [1, 16] {
        let never = |_: &[f64]| -> Vec<f64> { panic!("invalid stencil reached callback") };
        let x = vec![2.0; n];
        for h in [0.0, -1e-6, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
            assert!(try_jacobian_central(&never, &x, 1, h).is_err());
            assert!(try_jacobian_central_parallel(&never, &x, 1, h).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            let x = vec![value; n];
            assert!(try_jacobian_central(&never, &x, 1, 0.5).is_err());
            assert!(try_jacobian_central_parallel(&never, &x, 1, 0.5).is_err());
        }
    }
}

#[test]
fn callback_and_generated_nonfinite_values_are_errors() {
    for n in [1, 16] {
        let x = vec![0.0; n];
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let f = |_: &[f64]| vec![value];
            assert!(try_jacobian_central(&f, &x, 1, 1e-4).is_err());
            assert!(try_jacobian_central_parallel(&f, &x, 1, 1e-4).is_err());
        }
        let overflow = |x: &[f64]| {
            vec![if x.iter().sum::<f64>() > 0.0 {
                f64::MAX
            } else {
                -f64::MAX
            }]
        };
        assert!(try_jacobian_central(&overflow, &x, 1, 1e-4).is_err());
        assert!(try_jacobian_central_parallel(&overflow, &x, 1, 1e-4).is_err());
    }
}

#[test]
fn overflowing_dimensions_are_errors_without_allocation() {
    let never = |_: &[f64]| -> Vec<f64> { panic!("invalid size reached callback") };
    assert!(try_jacobian_central(&never, &[1.0; 16], usize::MAX, 1e-4).is_err());
    assert!(try_jacobian_central_parallel(&never, &[1.0; 16], usize::MAX, 1e-4).is_err());
}

#[test]
fn automatic_step_uses_scalar_precision() {
    let jac = jacobian_central_auto(&|x: &[f32]| vec![x[0] * x[0]], &[2.0_f32], 1);
    assert!((jac[[0, 0]] - 4.0).abs() < 1e-4);
}
