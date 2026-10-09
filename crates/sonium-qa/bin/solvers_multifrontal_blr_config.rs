//! Independent CPU checks for multifrontal scalar-family and configuration
//! admission contracts.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use sonium_qa::QaResult;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_direct::{
    BlrConfig, BlrError, LdltBlrConfig, LdltConfig, MultifrontalBlrFactorization,
    MultifrontalLdltFactorization, PanelKind,
};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.multifrontal-blr-config.v1";

fn scalar<T: ComplexField>(value: f64) -> T {
    T::from_real(T::Real::from_f64(value).expect("fixture value representable"))
}

fn fixture<T: ComplexField>() -> CsrMatrix<T> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            scalar(6.0),
            scalar(1.0),
            scalar(0.5),
            scalar(1.0),
            scalar(5.0),
            scalar(0.75),
            scalar(0.5),
            scalar(0.75),
            scalar(4.0),
        ],
    )
}

fn seeded_spd<T: ComplexField>(seed: u64) -> CsrMatrix<T> {
    let n = 6;
    let mut state = seed.wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = ((state >> 11) as f64) / ((1_u64 << 53) as f64);
        0.18 * (2.0 * unit - 1.0)
    };
    let mut values = vec![vec![0.0_f64; n]; n];
    for row in 0..n {
        for col in (row + 1)..n {
            let value = next();
            values[row][col] = value;
            values[col][row] = value;
        }
    }
    for row in 0..n {
        values[row][row] = 2.0 + values[row].iter().map(|value| value.abs()).sum::<f64>();
    }
    let mut triplets = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            triplets.push((row, col, scalar::<T>(values[row][col])));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn matvec<T: ComplexField>(matrix: &CsrMatrix<T>, vector: &[T]) -> Vec<T> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]).fold(T::zero(), |sum, index| {
                sum + matrix.values[index] * vector[matrix.col_indices[index]]
            })
        })
        .collect()
}

fn relative_residual<T: ComplexField>(matrix: &CsrMatrix<T>, solution: &[T], rhs: &[T]) -> f64 {
    let actual = matvec(matrix, solution);
    let residual_sq = actual
        .iter()
        .zip(rhs)
        .fold(T::Real::zero(), |sum, (lhs, expected)| {
            sum + (*lhs - *expected).norm_sqr()
        });
    let rhs_sq = rhs
        .iter()
        .fold(T::Real::zero(), |sum, value| sum + value.norm_sqr());
    (residual_sq.sqrt() / rhs_sq.sqrt().max(T::Real::one()))
        .to_f64()
        .expect("fixture residual representable")
}

fn check_matrix<T: ComplexField + FromPrimitive>(
    matrix: CsrMatrix<T>,
    tolerance: f64,
) -> Result<f64, String> {
    let expected: Vec<_> = (0..matrix.num_rows)
        .map(|index| match index % 3 {
            0 => scalar(1.0),
            1 => scalar(-2.0),
            _ => scalar(0.5),
        })
        .collect();
    let rhs = matvec(&matrix, &expected);

    let blr = MultifrontalBlrFactorization::<T>::factorize_with_config(
        &matrix,
        &BlrConfig {
            epsilon: 0.0,
            block_size: 2,
            min_block_size: 1,
            ..BlrConfig::default()
        },
    )
    .map_err(|error| format!("BLR factorization failed: {error}"))?;
    let blr_solution = blr
        .try_solve(&rhs)
        .map_err(|error| format!("BLR solve failed: {error}"))?;
    let blr_residual = relative_residual(&matrix, &blr_solution, &rhs);
    if blr_residual > tolerance {
        return Err(format!(
            "BLR residual {blr_residual:e} exceeds {tolerance:e}"
        ));
    }

    let ldlt = MultifrontalLdltFactorization::<T>::factorize_with_config(
        &matrix,
        &LdltConfig {
            min_block_size: 1,
            blr: Some(LdltBlrConfig {
                epsilon: 0.0,
                block_size: 2,
                panel_kind: PanelKind::Blr,
            }),
            ..LdltConfig::default()
        },
    )
    .map_err(|error| format!("LDLT factorization failed: {error}"))?;
    let ldlt_solution = ldlt
        .try_solve(&rhs)
        .map_err(|error| format!("LDLT solve failed: {error}"))?;
    let ldlt_residual = relative_residual(&matrix, &ldlt_solution, &rhs);
    if ldlt_residual > tolerance {
        return Err(format!(
            "LDLT residual {ldlt_residual:e} exceeds {tolerance:e}"
        ));
    }

    Ok(blr_residual.max(ldlt_residual))
}

fn complex32_fixture() -> CsrMatrix<Complex32> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            Complex32::new(6.0, 0.0),
            Complex32::new(1.0, 0.5),
            Complex32::new(0.5, -0.25),
            Complex32::new(1.0, -0.5),
            Complex32::new(5.0, 0.0),
            Complex32::new(0.75, 0.125),
            Complex32::new(0.5, 0.25),
            Complex32::new(0.75, -0.125),
            Complex32::new(4.0, 0.0),
        ],
    )
}

fn complex64_fixture() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![
            Complex64::new(6.0, 0.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(0.5, -0.25),
            Complex64::new(1.0, -0.5),
            Complex64::new(5.0, 0.0),
            Complex64::new(0.75, 0.125),
            Complex64::new(0.5, 0.25),
            Complex64::new(0.75, -0.125),
            Complex64::new(4.0, 0.0),
        ],
    )
}

fn run() -> Result<(usize, usize, f64, usize), String> {
    let identity = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f32, 2.0]);
    let mut overflow_rejections = 0;

    let epsilon_error = MultifrontalBlrFactorization::<f32>::factorize_with_config(
        &identity,
        &BlrConfig {
            epsilon: f64::MAX,
            ..BlrConfig::default()
        },
    )
    .err()
    .ok_or_else(|| "an epsilon that overflows f32 was accepted".to_string())?;
    if !matches!(
        epsilon_error,
        BlrError::InvalidConfiguration { field: "epsilon" }
    ) {
        return Err(format!(
            "BLR epsilon overflow returned wrong error: {epsilon_error}"
        ));
    }
    overflow_rejections += 1;

    let pivot_error = MultifrontalBlrFactorization::<f32>::factorize_with_config(
        &identity,
        &BlrConfig {
            pivot_tol_multiplier: f64::MAX,
            ..BlrConfig::default()
        },
    )
    .err()
    .ok_or_else(|| "a pivot tolerance that overflows f32 was accepted".to_string())?;
    if !matches!(
        pivot_error,
        BlrError::InvalidConfiguration {
            field: "pivot_tol_multiplier"
        }
    ) {
        return Err(format!(
            "BLR pivot overflow returned wrong error: {pivot_error}"
        ));
    }
    overflow_rejections += 1;

    let ldlt_pivot_error = MultifrontalLdltFactorization::<f32>::factorize_with_config(
        &identity,
        &LdltConfig {
            pivot_tol_multiplier: f64::MAX,
            ..LdltConfig::default()
        },
    )
    .err()
    .ok_or_else(|| "an LDLT pivot tolerance that overflows f32 was accepted".to_string())?;
    if !matches!(
        ldlt_pivot_error,
        BlrError::InvalidConfiguration {
            field: "pivot_tol_multiplier"
        }
    ) {
        return Err(format!(
            "LDLT pivot overflow returned wrong error: {ldlt_pivot_error}"
        ));
    }
    overflow_rejections += 1;

    let ldlt_epsilon_error = MultifrontalLdltFactorization::<f32>::factorize_with_config(
        &identity,
        &LdltConfig {
            blr: Some(LdltBlrConfig {
                epsilon: f64::MAX,
                ..LdltBlrConfig::default()
            }),
            ..LdltConfig::default()
        },
    )
    .err()
    .ok_or_else(|| "an LDLT BLR epsilon that overflows f32 was accepted".to_string())?;
    if !matches!(
        ldlt_epsilon_error,
        BlrError::InvalidConfiguration {
            field: "blr.epsilon"
        }
    ) {
        return Err(format!(
            "LDLT BLR epsilon overflow returned wrong error: {ldlt_epsilon_error}"
        ));
    }
    overflow_rejections += 1;

    let mut scalar_checks = 0;
    let mut maximum_residual = 0.0_f64;
    maximum_residual = maximum_residual.max(check_matrix(fixture::<f32>(), 3.0e-5)?);
    scalar_checks += 1;
    maximum_residual = maximum_residual.max(check_matrix(fixture::<f64>(), 3.0e-13)?);
    scalar_checks += 1;
    maximum_residual = maximum_residual.max(check_matrix(complex32_fixture(), 3.0e-5)?);
    scalar_checks += 1;
    maximum_residual = maximum_residual.max(check_matrix(complex64_fixture(), 3.0e-13)?);
    scalar_checks += 1;

    let mut matrix_bank_checks = 0;
    for seed in 0..8 {
        maximum_residual = maximum_residual.max(check_matrix(seeded_spd::<f64>(seed), 3.0e-13)?);
        matrix_bank_checks += 1;
        maximum_residual = maximum_residual.max(check_matrix(
            seeded_spd::<Complex64>(0xC0DE_0000 + seed),
            3.0e-13,
        )?);
        matrix_bank_checks += 1;
    }

    Ok((
        scalar_checks,
        matrix_bank_checks,
        maximum_residual,
        overflow_rejections,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((scalar_checks, matrix_bank_checks, maximum_residual, overflow_rejections)) => {
            let result = QaResult::passed(CASE_ID)
                .reference(
                    "independent four-family BLR/LDLT residual and configuration-boundary fixture",
                )
                .input_checksum("analytic/deterministic-fixture:solvers-multifrontal-blr-config-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-multifrontal-blr-config-v1",
                )
                .dimension("3x3 symmetric CSR; f32/f64/Complex32/Complex64 BLR and Hermitian LDLT")
                .metric("scalar_family_checks", scalar_checks as f64)
                .metric("matrix_bank_checks", matrix_bank_checks as f64)
                .metric("overflow_rejections", overflow_rejections as f64)
                .metric("maximum_relative_residual", maximum_residual)
                .minimum("scalar_family_checks", 4.0)
                .minimum("matrix_bank_checks", 16.0)
                .minimum("overflow_rejections", 4.0)
                .maximum("maximum_relative_residual", 3.0e-5)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .reference(
                    "independent four-family BLR/LDLT residual and configuration-boundary fixture",
                )
                .input_checksum("analytic/deterministic-fixture:solvers-multifrontal-blr-config-v1")
                .reference_checksum(
                    "analytic/deterministic-fixture:solvers-multifrontal-blr-config-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
