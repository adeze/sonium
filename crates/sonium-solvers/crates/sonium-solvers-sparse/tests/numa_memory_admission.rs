use sonium_solvers_core::traits::SolverErrorKind;
use sonium_solvers_sparse::{CsrMatrix, NumaPartitionedCsr, NumaSpmvPolicy};
use sonium_solvers_sys::topology::MachineTopology;

fn matrix() -> CsrMatrix<f64> {
    CsrMatrix::from_raw_parts(
        7,
        7,
        vec![0, 1, 2, 3, 13, 14, 15, 16],
        (0..16).map(|index| index % 7).collect(),
        vec![1.0; 16],
    )
}

#[test]
fn memory_estimate_is_deterministic_and_budget_is_admitted_before_allocation() {
    let csr = matrix();
    let required = NumaPartitionedCsr::estimate_storage_bytes(&csr, 3).expect("estimate");
    assert!(required > 0);
    let admitted = NumaPartitionedCsr::try_from_csr_with_memory_budget(&csr, 3, required)
        .expect("exact budget should admit");
    assert_eq!(admitted.nnz(), csr.nnz());
    assert_eq!(
        NumaPartitionedCsr::estimate_storage_bytes(&csr, 3).expect("repeat estimate"),
        required
    );
}

#[test]
fn memory_budget_rejects_without_constructing_partitions() {
    let csr = matrix();
    let required = NumaPartitionedCsr::estimate_storage_bytes(&csr, 3).expect("estimate");
    let error = NumaPartitionedCsr::try_from_csr_with_memory_budget(&csr, 3, required - 1)
        .expect_err("one byte short must reject");
    assert_eq!(error.kind(), SolverErrorKind::OperationFailed);
    assert!(error.to_string().contains("exceeds budget"));
}

#[test]
fn memory_estimate_rejects_zero_partitions() {
    let error =
        NumaPartitionedCsr::estimate_storage_bytes(&matrix(), 0).expect_err("zero partitions");
    assert_eq!(error.kind(), SolverErrorKind::OperationFailed);
    assert!(error.to_string().contains("partition count"));
}

#[test]
fn policy_admission_rejects_malformed_csr_before_placement() {
    let mut malformed = matrix();
    let terminal = malformed.row_ptrs.len() - 1;
    malformed.row_ptrs[terminal] += 1;
    let placement = MachineTopology::detect().placement_plan(3);

    let error = NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
        &malformed,
        &placement,
        NumaSpmvPolicy::REPLICATED_PINNED,
        usize::MAX,
    )
    .expect_err("malformed CSR must fail before placement allocation");
    assert_eq!(error.kind(), SolverErrorKind::OperationFailed);
    assert!(error.to_string().contains("terminal row pointer"));
}

#[test]
fn placement_memory_budget_is_checked_before_first_touch() {
    let csr = matrix();
    let placement = MachineTopology::detect().placement_plan(3);
    let required = NumaPartitionedCsr::estimate_storage_bytes(&csr, 3).expect("estimate");

    let admitted = NumaPartitionedCsr::try_from_csr_with_placement_and_memory_budget(
        &csr, &placement, required,
    )
    .expect("exact placement budget should admit");
    assert_eq!(admitted.nnz(), csr.nnz());

    let error = NumaPartitionedCsr::try_from_csr_with_placement_and_memory_budget(
        &csr,
        &placement,
        required - 1,
    )
    .expect_err("one byte short must reject before first-touch allocation");
    assert_eq!(error.kind(), SolverErrorKind::OperationFailed);
    assert!(error.to_string().contains("exceeds budget"));
}

#[test]
fn repeated_admission_success_and_failure_are_cleanly_reusable() {
    let csr = matrix();
    let placement = MachineTopology::detect().placement_plan(3);
    let required = NumaPartitionedCsr::estimate_storage_bytes(&csr, 3).expect("estimate");

    for _ in 0..3 {
        let error = NumaPartitionedCsr::try_from_csr_with_placement_and_memory_budget(
            &csr,
            &placement,
            required - 1,
        )
        .expect_err("short budget must fail before construction");
        assert_eq!(error.kind(), SolverErrorKind::OperationFailed);
        assert!(error.to_string().contains("exceeds budget"));

        let admitted = NumaPartitionedCsr::try_from_csr_with_placement_and_memory_budget(
            &csr, &placement, required,
        )
        .expect("exact budget should remain reusable after rejection");
        assert_eq!(admitted.nnz(), csr.nnz());
    }
}

#[test]
fn policy_admission_includes_replicated_input_before_first_touch() {
    let csr = matrix();
    let placement = MachineTopology::detect().placement_plan(3);
    let shared = NumaPartitionedCsr::estimate_storage_bytes_with_policy(
        &csr,
        3,
        NumaSpmvPolicy::SHARED_PINNED,
    )
    .expect("shared policy estimate");
    let replicated = NumaPartitionedCsr::estimate_storage_bytes_with_policy(
        &csr,
        3,
        NumaSpmvPolicy::REPLICATED_PINNED,
    )
    .expect("replicated policy estimate");
    assert!(replicated.total_bytes > shared.total_bytes);

    let admitted = NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
        &csr,
        &placement,
        NumaSpmvPolicy::REPLICATED_PINNED,
        replicated.total_bytes,
    )
    .expect("exact replicated policy budget should admit");
    assert_eq!(admitted.nnz(), csr.nnz());

    let error = NumaPartitionedCsr::try_from_csr_with_placement_and_policy_memory_budget(
        &csr,
        &placement,
        NumaSpmvPolicy::REPLICATED_PINNED,
        replicated.total_bytes - 1,
    )
    .expect_err("replicated input must be included in pre-allocation admission");
    assert_eq!(error.kind(), SolverErrorKind::OperationFailed);
    assert!(error.to_string().contains("exceeds budget"));
}

#[test]
fn repeated_construction_and_all_actions_preserve_csr_parity() {
    let csr = matrix();
    let placement = MachineTopology::detect().placement_plan(3);
    let x = vec![1.0; csr.num_cols];
    let expected_forward = csr.matvec(&x);
    let expected_transpose = csr.matvec_transpose(&x);
    let expected_hermitian = csr.matvec_hermitian(&x);

    for _ in 0..3 {
        let numa = NumaPartitionedCsr::try_from_csr_with_placement(&csr, &placement)
            .expect("valid placement construction");
        for policy in [
            NumaSpmvPolicy::SHARED_UNPINNED,
            NumaSpmvPolicy::SHARED_PINNED,
            NumaSpmvPolicy::REPLICATED_PINNED,
        ] {
            assert_eq!(
                numa.try_matvec_with_policy(&x, policy)
                    .expect("forward action"),
                expected_forward
            );
            assert_eq!(
                numa.try_matvec_transpose_with_policy(&x, policy)
                    .expect("transpose action"),
                expected_transpose
            );
            assert_eq!(
                numa.try_matvec_hermitian_with_policy(&x, policy)
                    .expect("Hermitian action"),
                expected_hermitian
            );
        }
    }
}
