use sonium_solvers_core::traits::{Preconditioner, SolverError};
use sonium_solvers_preconditioners::{AmgConfig, AmgPreconditioner};
use sonium_solvers_sparse::CsrMatrix;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn amg_rejects_wrong_shapes_without_identity_fallback_or_output_mutation() {
    let matrix = CsrMatrix::<f64>::identity(4);
    let amg = AmgPreconditioner::try_from_csr(&matrix, AmgConfig::default()).unwrap();
    for size in [0, 3, 5] {
        let rhs = vec![2.0; size];
        assert!(matches!(
            amg.try_apply(&rhs),
            Err(SolverError::DimensionMismatch { expected: 4, got }) if got == size
        ));
        assert!(catch_unwind(AssertUnwindSafe(|| amg.apply(&rhs))).is_err());
        // Previously this equal-size output admitted the erroneous identity path.
        for output_size in [size, 4] {
            let mut output = vec![7.0; output_size];
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    amg.apply_into(&rhs, &mut output);
                }))
                .is_err()
            );
            assert_eq!(output, vec![7.0; output_size]);
        }
        let mut output = vec![7.0; size];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                amg.apply_into(&[2.0; 4], &mut output);
            }))
            .is_err()
        );
        assert_eq!(output, vec![7.0; size]);
    }
    assert_eq!(amg.apply(&[2.0; 4]), vec![2.0; 4]);
}
