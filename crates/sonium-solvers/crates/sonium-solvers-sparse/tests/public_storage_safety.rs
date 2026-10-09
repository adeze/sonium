use sonium_solvers_core::{LinearOperator, dense::DenseMatrix};
use sonium_solvers_sparse::{BsrMatrix, CsrMatrix};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn rejects(f: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
}

#[test]
fn numa_safe_actions_reject_corrupt_partition_storage() {
    use sonium_solvers_sparse::numa::NumaSpmvPolicy;
    for corruption in 0..5 {
        let mut a = CsrMatrix::<f64>::identity(4).to_numa_partitioned(2);
        match corruption {
            0 => a.partitions[1].row_start = 0, // overlapping SIMD writers
            1 => a.partitions[0].row_start = usize::MAX,
            2 => a.partitions[0].col_indices[0] = usize::MAX,
            3 => a.partitions[0].row_ptrs[1] = usize::MAX,
            _ => {
                a.partitions[0].values.pop();
            }
        }
        let x = [1.0; 4];
        assert!(a.try_apply(&x).is_err());
        rejects(|| {
            a.matvec(&x);
        });
        for policy in [
            NumaSpmvPolicy::SHARED_UNPINNED,
            NumaSpmvPolicy::REPLICATED_PINNED,
        ] {
            rejects(|| {
                a.matvec_with_policy(&x, policy);
            });
            rejects(|| {
                a.matvec_transpose_with_policy(&x, policy);
            });
            rejects(|| {
                a.matvec_hermitian_with_policy(&x, policy);
            });
        }
    }
}

#[test]
fn all_bsr_safe_actions_reject_corrupt_public_storage() {
    let valid = BsrMatrix::<f64, 2>::from_csr(&CsrMatrix::identity(2));
    let mut cases = Vec::new();
    let mut bad = valid.clone();
    bad.block_col_indices[0] = 1;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.block_values.pop();
    cases.push(bad);
    let mut bad = valid.clone();
    bad.block_row_ptrs[1] = 2;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.block_row_ptrs[0] = 1;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.num_block_rows = usize::MAX;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.num_block_cols = usize::MAX;
    cases.push(bad);
    for a in cases {
        let x = [1.0, 2.0];
        assert!(a.try_apply(&x).is_err());
        assert!(a.try_apply_transpose(&x).is_err());
        assert!(a.try_apply_hermitian(&x).is_err());
        rejects(|| {
            a.matvec(&x);
        });
        rejects(|| {
            a.matvec_parallel(&x);
        });
        rejects(|| {
            a.apply(&x);
        });
        rejects(|| {
            a.apply_into(&x, &mut [0.0; 2]);
        });
        rejects(|| {
            a.apply_transpose(&x);
        });
        rejects(|| {
            a.apply_transpose_into(&x, &mut [0.0; 2]);
        });
        rejects(|| {
            a.apply_hermitian(&x);
        });
    }
}

#[test]
fn zero_block_size_is_rejected_before_parallel_dispatch() {
    let a = BsrMatrix::<f64, 0> {
        num_block_rows: 0,
        num_block_cols: 0,
        block_row_ptrs: vec![0],
        block_col_indices: vec![],
        block_values: vec![],
    };
    assert!(a.try_apply(&[]).is_err());
    rejects(|| {
        a.matvec(&[]);
    });
    rejects(|| {
        a.matvec_parallel(&[]);
    });
    rejects(|| {
        a.apply_transpose(&[]);
    });
    rejects(|| {
        a.apply_hermitian(&[]);
    });
}

#[test]
fn all_csr_safe_actions_reject_corrupt_public_storage() {
    let valid = CsrMatrix::<f64>::identity(2);
    let mut cases = Vec::new();
    let mut bad = valid.clone();
    bad.col_indices[0] = 2;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.values.pop();
    cases.push(bad);
    let mut bad = valid.clone();
    bad.row_ptrs[1] = 3;
    cases.push(bad);
    let mut bad = valid.clone();
    bad.row_ptrs[0] = 1;
    cases.push(bad);
    for a in cases {
        let x = [1.0, 2.0];
        assert!(a.try_apply(&x).is_err());
        assert!(a.try_apply_transpose(&x).is_err());
        assert!(a.try_apply_hermitian(&x).is_err());
        rejects(|| {
            a.matvec(&x);
        });
        rejects(|| {
            a.matvec_into_buf(&x, &mut [0.0; 2]);
        });
        rejects(|| {
            a.matvec_add(&x, &mut [0.0; 2]);
        });
        rejects(|| {
            a.apply(&x);
        });
        rejects(|| {
            a.apply_into(&x, &mut [0.0; 2]);
        });
        rejects(|| {
            a.matvec_transpose(&x);
        });
        rejects(|| {
            a.matvec_transpose_into_buf(&x, &mut [0.0; 2]);
        });
        rejects(|| {
            a.matvec_hermitian(&x);
        });
        rejects(|| {
            a.matvec_hermitian_into_buf(&x, &mut [0.0; 2]);
        });
        let block = DenseMatrix::from_row_major_vec(2, 1, x.to_vec());
        rejects(|| {
            a.spmm_into(&block, &mut DenseMatrix::zeros(2, 1));
        });
        rejects(|| {
            a.apply_block(&block, &mut DenseMatrix::zeros(2, 1));
        });
    }
}
