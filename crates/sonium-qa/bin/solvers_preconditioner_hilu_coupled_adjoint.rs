use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::{
    AdjointCapability, CsrMatrix, HiluConfig, HiluPreconditioner, Preconditioner,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-hilu-coupled-adjoint.v1";

fn matrix(n: usize) -> CsrMatrix<Complex64> {
    let mut row_ptrs = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row > 0 {
            columns.push(row - 1);
            values.push(Complex64::new(-1.0, 0.3));
        }
        columns.push(row);
        values.push(Complex64::new(6.0 + 0.1 * (row % 3) as f64, 0.2));
        if row + 1 < n {
            columns.push(row + 1);
            values.push(Complex64::new(-0.7, -0.2));
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::try_from_raw_parts(n, n, row_ptrs, columns, values)
        .expect("HILU coupled fixture must be canonical")
}

fn vector(n: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let real = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            state = state.rotate_left(17);
            let imag = ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0;
            Complex64::new(real, imag)
        })
        .collect()
}

fn run() -> Result<(f64, f64, f64, f64), String> {
    let matrix = matrix(20);
    let config = HiluConfig {
        min_block_size: 4,
        max_levels: None,
    };
    let preconditioner =
        HiluPreconditioner::try_from_csr_coupled_with_config(&matrix, &config, 0.0)
            .map_err(|error| format!("coupled HILU setup failed: {error}"))?;
    if !preconditioner.is_coupled()
        || preconditioner.transpose_capability() != AdjointCapability::Exact
        || preconditioner.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("coupled HILU did not advertise exact adjoint capabilities".to_string());
    }

    let mut max_transpose = 0.0_f64;
    let mut max_hermitian = 0.0_f64;
    let mut max_in_place_hermitian = 0.0_f64;
    for seed in 0..16_u64 {
        let r = vector(matrix.num_rows, 0xA5A5_0000 + seed);
        let s = vector(matrix.num_rows, 0x5A5A_0000 + seed);
        let mr = preconditioner.apply(&r);
        let mt_s = preconditioner
            .try_apply_transpose(&s)
            .map_err(|error| format!("transpose action failed: {error}"))?;
        let mh_s = preconditioner
            .try_apply_hermitian(&s)
            .map_err(|error| format!("Hermitian action failed: {error}"))?;
        let mut mh_s_into = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
        preconditioner.apply_hermitian_into(&s, &mut mh_s_into);
        let transpose_lhs: Complex64 = mr.iter().zip(&s).map(|(a, b)| *a * *b).sum();
        let transpose_rhs: Complex64 = r.iter().zip(&mt_s).map(|(a, b)| *a * *b).sum();
        let hermitian_lhs: Complex64 = mr.iter().zip(&s).map(|(a, b)| a.conj() * *b).sum();
        let hermitian_rhs: Complex64 = r.iter().zip(&mh_s).map(|(a, b)| a.conj() * *b).sum();
        max_transpose = max_transpose.max((transpose_lhs - transpose_rhs).norm());
        max_hermitian = max_hermitian.max((hermitian_lhs - hermitian_rhs).norm());
        max_in_place_hermitian = max_in_place_hermitian.max(
            mh_s.iter()
                .zip(&mh_s_into)
                .map(|(expected, actual)| (*expected - *actual).norm())
                .fold(0.0, f64::max),
        );
    }
    Ok((max_transpose, max_hermitian, max_in_place_hermitian, 16.0))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((transpose_error, hermitian_error, in_place_error, cases)) => QaResult::passed(CASE_ID)
            .metric("max_transpose_error", transpose_error)
            .metric("max_hermitian_error", hermitian_error)
            .metric("max_in_place_hermitian_error", in_place_error)
            .metric("randomized_case_count", cases)
            .maximum("max_transpose_error", 1.0e-10)
            .maximum("max_hermitian_error", 1.0e-10)
            .maximum("max_in_place_hermitian_error", 1.0e-12)
            .minimum("randomized_case_count", 16.0)
            .input_checksum("analytic/deterministic-fixture:hilu-coupled-adjoint-v1")
            .reference_checksum("analytic/deterministic-fixture:hilu-coupled-adjoint-v1")
            .reference("complex tridiagonal coupled HILU Schur-complement adjoint")
            .dimension("20x20 CSR; min_block_size=4; 16 deterministic vector pairs")
            .tolerance("transpose/Hermitian errors <= 1e-10; in-place Hermitian parity <= 1e-12")
            .seed("solvers-preconditioner-hilu-coupled-adjoint:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .input_checksum("analytic/deterministic-fixture:hilu-coupled-adjoint-v1")
            .reference_checksum("analytic/deterministic-fixture:hilu-coupled-adjoint-v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
