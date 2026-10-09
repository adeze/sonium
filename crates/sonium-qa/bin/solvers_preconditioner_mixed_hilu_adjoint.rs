//! Independent CPU evidence for mixed-precision coupled-HILU adjoints.

use num_complex::{Complex32, Complex64};
use sonium_qa::QaResult;
use sonium_solvers::mixed_precision::MixedPrecisionPreconditioner;
use sonium_solvers::{
    AdjointCapability, CsrMatrix, HiluConfig, HiluPreconditioner, Preconditioner,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-mixed-hilu-adjoint.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:preconditioner-mixed-hilu-adjoint-v1";

fn matrix() -> CsrMatrix<Complex32> {
    let n = 20;
    let mut row_ptrs = vec![0usize];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row > 0 {
            columns.push(row - 1);
            values.push(Complex32::new(-1.0, 0.3));
        }
        columns.push(row);
        values.push(Complex32::new(6.0 + 0.1 * (row % 3) as f32, 0.2));
        if row + 1 < n {
            columns.push(row + 1);
            values.push(Complex32::new(-0.7, -0.2));
        }
        row_ptrs.push(columns.len());
    }
    CsrMatrix::try_from_raw_parts(n, n, row_ptrs, columns, values)
        .expect("mixed HILU fixture must be canonical")
}

fn vector(n: usize, mut state: u64) -> Vec<Complex64> {
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

fn bilinear_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b).map(|(left, right)| *left * *right).sum()
}

fn hermitian_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

fn run() -> Result<(f64, f64, f64), String> {
    let low = HiluPreconditioner::try_from_csr_coupled_with_config(
        &matrix(),
        &HiluConfig {
            min_block_size: 4,
            max_levels: None,
        },
        0.0,
    )
    .map_err(|error| format!("coupled f32 HILU setup failed: {error}"))?;
    if !low.is_coupled()
        || low.transpose_capability() != AdjointCapability::Exact
        || low.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("coupled f32 HILU did not expose exact adjoints".into());
    }
    let mixed = MixedPrecisionPreconditioner::<Complex64, Complex32>::new(Box::new(low));
    if mixed.transpose_capability() != AdjointCapability::Exact
        || mixed.hermitian_capability() != AdjointCapability::Exact
    {
        return Err("mixed coupled HILU did not preserve exact adjoint capabilities".into());
    }

    let mut max_transpose = 0.0_f64;
    let mut max_hermitian = 0.0_f64;
    for seed in 0..8_u64 {
        let r = vector(20, 0xA5A5_0000 + seed);
        let s = vector(20, 0x5A5A_0000 + seed);
        let mr = mixed.apply(&r);
        let mt_s = mixed
            .try_apply_transpose(&s)
            .map_err(|error| format!("mixed transpose failed: {error}"))?;
        let mh_s = mixed
            .try_apply_hermitian(&s)
            .map_err(|error| format!("mixed Hermitian failed: {error}"))?;
        max_transpose =
            max_transpose.max((bilinear_inner(&mr, &s) - bilinear_inner(&r, &mt_s)).norm());
        max_hermitian =
            max_hermitian.max((hermitian_inner(&mr, &s) - hermitian_inner(&r, &mh_s)).norm());
    }
    Ok((max_transpose, max_hermitian, 16.0))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((max_transpose, max_hermitian, checks)) => QaResult::passed(CASE_ID)
            .metric("max_transpose_error", max_transpose)
            .metric("max_hermitian_error", max_hermitian)
            .metric("identity_check_count", checks)
            .maximum("max_transpose_error", 3.0e-5)
            .maximum("max_hermitian_error", 3.0e-5)
            .minimum("identity_check_count", 16.0)
            .reference("mixed Complex64 wrapper around coupled Complex32 HILU")
            .input_checksum(CHECKSUM)
            .reference_checksum(CHECKSUM)
            .dimension("20x20 CSR; min_block_size=4; eight vector pairs")
            .tolerance("transpose/Hermitian errors <= 3e-5")
            .seed("solvers-preconditioner-mixed-hilu-adjoint:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error)
            .input_checksum(CHECKSUM)
            .reference_checksum(CHECKSUM)
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
