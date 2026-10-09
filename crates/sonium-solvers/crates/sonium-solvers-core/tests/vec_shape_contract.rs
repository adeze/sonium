use sonium_solvers_core::vec_ops::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn mismatched_lengths_never_silently_truncate_in_release() {
    for (a, b) in [(vec![1.0, 2.0], vec![3.0]), (vec![1.0], vec![2.0, 3.0])] {
        assert!(catch_unwind(|| a.dot(&b)).is_err());
        assert!(catch_unwind(|| vec_add(&a, &b)).is_err());
        assert!(catch_unwind(|| vec_sub(&a, &b)).is_err());
        assert!(catch_unwind(|| vec_add_scaled(&a, &b, 2.0)).is_err());
        let mut destination = a.clone();
        assert!(catch_unwind(AssertUnwindSafe(|| vec_add_assign(&mut destination, &b))).is_err());
        assert_eq!(destination, a);
        assert!(catch_unwind(AssertUnwindSafe(|| vec_sub_assign(&mut destination, &b))).is_err());
        assert_eq!(destination, a);
        assert!(catch_unwind(AssertUnwindSafe(|| vec_assign(&mut destination, &b))).is_err());
        assert_eq!(destination, a);
    }
}
