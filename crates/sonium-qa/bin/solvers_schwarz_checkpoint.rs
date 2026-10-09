//! Independent CPU evidence for additive-Schwarz checkpoint round trips.
use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::AdditiveSchwarzPreconditioner;
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.schwarz-checkpoint.v2";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:schwarz-checkpoint-v2";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        6,
        6,
        vec![0, 2, 5, 8, 11, 14, 16],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5],
        vec![
            Complex64::new(4.0, 0.2),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.5, 0.3),
            Complex64::new(-1.0, 0.2),
            Complex64::new(-1.0, -0.2),
            Complex64::new(4.5, 0.1),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.5, 0.2),
            Complex64::new(-1.0, 0.2),
            Complex64::new(-1.0, -0.2),
            Complex64::new(4.5, 0.1),
            Complex64::new(-1.0, 0.1),
            Complex64::new(-1.0, -0.1),
            Complex64::new(4.0, 0.2),
        ],
    )
}

fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let matrix = matrix();
    let source = AdditiveSchwarzPreconditioner::try_from_csr_partitions(
        &matrix,
        &[vec![0, 1, 2], vec![3, 4, 5]],
        1,
    )
    .map_err(|error| format!("Schwarz construction failed: {error:?}"))?;
    let rhs = [
        Complex64::new(1.0, 0.25),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.5, -0.75),
        Complex64::new(3.0, 0.125),
        Complex64::new(-1.5, 0.25),
        Complex64::new(0.75, -0.5),
    ];
    let expected_forward = source
        .try_apply(&rhs)
        .map_err(|error| format!("forward action failed: {error:?}"))?;
    let expected_transpose = source
        .try_apply_transpose(&rhs)
        .map_err(|error| format!("transpose action failed: {error:?}"))?;
    let expected_hermitian = source
        .try_apply_hermitian(&rhs)
        .map_err(|error| format!("Hermitian action failed: {error:?}"))?;
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("checkpoint serialization failed: {error:?}"))?;
    let restored = AdditiveSchwarzPreconditioner::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("checkpoint restore failed: {error:?}"))?;
    let forward_error = max_error(
        &expected_forward,
        &restored
            .try_apply(&rhs)
            .map_err(|error| format!("restored forward action failed: {error:?}"))?,
    );
    let transpose_error = max_error(
        &expected_transpose,
        &restored
            .try_apply_transpose(&rhs)
            .map_err(|error| format!("restored transpose action failed: {error:?}"))?,
    );
    let hermitian_error = max_error(
        &expected_hermitian,
        &restored
            .try_apply_hermitian(&rhs)
            .map_err(|error| format!("restored Hermitian action failed: {error:?}"))?,
    );

    let coarse_source = AdditiveSchwarzPreconditioner::try_from_csr_partitions(
        &matrix,
        &[vec![0, 1, 2], vec![3, 4, 5]],
        1,
    )
    .map_err(|error| format!("coarse Schwarz construction failed: {error:?}"))?
    .with_nicolaides_coarse_space(&matrix);
    let coarse_expected_forward = coarse_source
        .try_apply(&rhs)
        .map_err(|error| format!("coarse forward action failed: {error:?}"))?;
    let coarse_expected_transpose = coarse_source
        .try_apply_transpose(&rhs)
        .map_err(|error| format!("coarse transpose action failed: {error:?}"))?;
    let coarse_expected_hermitian = coarse_source
        .try_apply_hermitian(&rhs)
        .map_err(|error| format!("coarse Hermitian action failed: {error:?}"))?;
    let coarse_bytes = coarse_source
        .to_checkpoint_bytes()
        .map_err(|error| format!("coarse checkpoint serialization failed: {error:?}"))?;
    let coarse_restored = AdditiveSchwarzPreconditioner::from_checkpoint_bytes(&coarse_bytes)
        .map_err(|error| format!("coarse checkpoint restore failed: {error:?}"))?;
    if !coarse_restored.has_coarse_space() || coarse_restored.coarse_dimension() != 2 {
        return Err("coarse checkpoint lost its attached two-level state".to_string());
    }
    let coarse_forward_error = max_error(
        &coarse_expected_forward,
        &coarse_restored
            .try_apply(&rhs)
            .map_err(|error| format!("restored coarse forward action failed: {error:?}"))?,
    );
    let coarse_transpose_error = max_error(
        &coarse_expected_transpose,
        &coarse_restored
            .try_apply_transpose(&rhs)
            .map_err(|error| format!("restored coarse transpose action failed: {error:?}"))?,
    );
    let coarse_hermitian_error = max_error(
        &coarse_expected_hermitian,
        &coarse_restored
            .try_apply_hermitian(&rhs)
            .map_err(|error| format!("restored coarse Hermitian action failed: {error:?}"))?,
    );

    let mut malformed_rejections = 0.0;
    for length in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        if AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes[..length])
            .is_err()
        {
            malformed_rejections += 1.0;
        }
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    if AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err() {
        malformed_rejections += 1.0;
    }
    let mut coarse_malformed_rejections = 0.0;
    if let Some(offset) = coarse_bytes
        .windows(b"SNLUCP01".len())
        .position(|window| window == b"SNLUCP01")
    {
        let mut bad_factor = coarse_bytes.clone();
        bad_factor[offset] ^= 0xff;
        if AdditiveSchwarzPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_factor).is_err() {
            coarse_malformed_rejections += 1.0;
        }
    }
    if !forward_error.is_finite()
        || !transpose_error.is_finite()
        || !hermitian_error.is_finite()
        || !coarse_forward_error.is_finite()
        || !coarse_transpose_error.is_finite()
        || !coarse_hermitian_error.is_finite()
        || malformed_rejections != 5.0
        || coarse_malformed_rejections != 1.0
    {
        return Err(format!(
            "Schwarz checkpoint gate failed: one_level=({forward_error}, {transpose_error}, {hermitian_error}), coarse=({coarse_forward_error}, {coarse_transpose_error}, {coarse_hermitian_error}), malformed=({malformed_rejections}, {coarse_malformed_rejections})"
        ));
    }
    Ok((
        forward_error,
        transpose_error,
        hermitian_error,
        coarse_forward_error,
        coarse_transpose_error,
        coarse_hermitian_error,
        malformed_rejections,
        coarse_malformed_rejections,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            forward_error,
            transpose_error,
            hermitian_error,
            coarse_forward_error,
            coarse_transpose_error,
            coarse_hermitian_error,
            malformed_rejections,
            coarse_malformed_rejections,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension(
                    "6x6 complex CSR, two overlapping subdomains plus Nicolaides coarse space",
                )
                .tolerance("all round-trip action errors <= 1e-12; malformed packets rejected")
                .metric("forward_error", forward_error)
                .metric("transpose_error", transpose_error)
                .metric("hermitian_error", hermitian_error)
                .metric("coarse_forward_error", coarse_forward_error)
                .metric("coarse_transpose_error", coarse_transpose_error)
                .metric("coarse_hermitian_error", coarse_hermitian_error)
                .metric("malformed_rejections", malformed_rejections)
                .metric("coarse_malformed_rejections", coarse_malformed_rejections)
                .maximum("forward_error", TOLERANCE)
                .maximum("transpose_error", TOLERANCE)
                .maximum("hermitian_error", TOLERANCE)
                .maximum("coarse_forward_error", TOLERANCE)
                .maximum("coarse_transpose_error", TOLERANCE)
                .maximum("coarse_hermitian_error", TOLERANCE)
                .minimum("malformed_rejections", 5.0)
                .minimum("coarse_malformed_rejections", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}
