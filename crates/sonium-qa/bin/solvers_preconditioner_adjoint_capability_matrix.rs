//! Deterministic CPU capability and adjoint matrix for composed preconditioners.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{
    AdjointCapability, AmgConfig, AmgPreconditioner, AmgRestriction, AmgSmoother,
    BlockPreconditionerConfig, ChebyshevConfig, ChebyshevPreconditioner, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, SolverError,
};
use sonium_solvers_core::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-adjoint-capability-matrix.v1";
const CHECKSUM: &str = "analytic/deterministic-fixture:preconditioner-adjoint-capability-matrix-v1";
const TOLERANCE: f64 = 1.0e-10;

fn vector(len: usize, seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..len)
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

fn hermitian_error<P: Preconditioner<Complex64>>(
    preconditioner: &P,
    r: &[Complex64],
    s: &[Complex64],
) -> Result<f64, SolverError> {
    let mr = preconditioner.try_apply(r)?;
    let mh_s = preconditioner.try_apply_hermitian(s)?;
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| a.conj() * b).sum();
    let rhs: Complex64 = r.iter().zip(mh_s).map(|(a, b)| a.conj() * b).sum();
    Ok((lhs - rhs).norm())
}

fn transpose_error<P: Preconditioner<Complex64>>(
    preconditioner: &P,
    r: &[Complex64],
    s: &[Complex64],
) -> Result<f64, SolverError> {
    let mr = preconditioner.try_apply(r)?;
    let mt_s = preconditioner.try_apply_transpose(s)?;
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| *a * *b).sum();
    let rhs: Complex64 = r.iter().zip(mt_s).map(|(a, b)| *a * b).sum();
    Ok((lhs - rhs).norm())
}

fn check_exact<P: Preconditioner<Complex64>>(
    name: &str,
    preconditioner: &P,
    len: usize,
    seed: u64,
) -> Result<(f64, f64, f64), String> {
    if preconditioner.transpose_capability() != AdjointCapability::Exact {
        return Err(format!(
            "{name} did not advertise exact transpose capability"
        ));
    }
    if preconditioner.hermitian_capability() != AdjointCapability::Exact {
        return Err(format!(
            "{name} did not advertise exact Hermitian capability"
        ));
    }
    let r = vector(len, seed);
    let s = vector(len, seed.wrapping_add(0x9E37_79B9));
    let transpose = transpose_error(preconditioner, &r, &s)
        .map_err(|error| format!("{name} transpose action failed: {error}"))?;
    let hermitian = hermitian_error(preconditioner, &r, &s)
        .map_err(|error| format!("{name} Hermitian action failed: {error}"))?;
    if !transpose.is_finite() || !hermitian.is_finite() {
        return Err(format!("{name} produced a non-finite adjoint error"));
    }
    Ok((transpose, hermitian, 2.0))
}

fn check_hermitian_exact<P: Preconditioner<Complex64>>(
    name: &str,
    preconditioner: &P,
    len: usize,
    seed: u64,
) -> Result<(f64, f64), String> {
    if preconditioner.hermitian_capability() != AdjointCapability::Exact {
        return Err(format!(
            "{name} did not advertise exact Hermitian capability"
        ));
    }
    let r = vector(len, seed);
    let s = vector(len, seed.wrapping_add(0x9E37_79B9));
    let hermitian = hermitian_error(preconditioner, &r, &s)
        .map_err(|error| format!("{name} Hermitian action failed: {error}"))?;
    if !hermitian.is_finite() {
        return Err(format!("{name} produced a non-finite Hermitian error"));
    }
    Ok((hermitian, 1.0))
}

fn hermitian_amg_matrix() -> CsrMatrix<Complex64> {
    let mut entries = Vec::new();
    for row in 0..8 {
        entries.push((row, row, Complex64::new(4.0, 0.0)));
        if row + 1 < 8 {
            let upper = Complex64::new(-0.8, 0.2);
            entries.push((row, row + 1, upper));
            entries.push((row + 1, row, upper.conj()));
        }
    }
    CsrMatrix::from_triplets(8, 8, entries)
}

fn small_spd_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, Complex64::new(4.0, 0.0)),
            (0, 1, Complex64::new(1.0, 0.0)),
            (1, 0, Complex64::new(1.0, 0.0)),
            (1, 1, Complex64::new(3.0, 0.0)),
            (1, 2, Complex64::new(0.75, 0.0)),
            (2, 1, Complex64::new(0.75, 0.0)),
            (2, 2, Complex64::new(2.5, 0.0)),
        ],
    )
}

fn saddle_matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, Complex64::new(2.0, 1.0)),
            (1, 1, Complex64::new(3.0, -0.5)),
            (2, 2, Complex64::new(4.0, 0.25)),
            (3, 3, Complex64::new(5.0, -0.75)),
        ],
    )
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64), String> {
    let spd = small_spd_matrix();
    let chebyshev = ChebyshevPreconditioner::try_from_csr(&spd, ChebyshevConfig::default())
        .map_err(|error| format!("Chebyshev setup failed: {error}"))?;
    let (cheb_t, cheb_h, cheb_checks) = check_exact("Chebyshev", &chebyshev, 3, 0xC0EB)?;

    let amg_matrix = hermitian_amg_matrix();
    let amg = AmgPreconditioner::try_from_csr(
        &amg_matrix,
        AmgConfig {
            restriction: AmgRestriction::ConjugateTranspose,
            smoother: AmgSmoother::SymmetricGaussSeidel,
            ..AmgConfig::default()
        },
    )
    .map_err(|error| format!("Hermitian AMG setup failed: {error}"))?;
    if amg.transpose_capability() != AdjointCapability::Unsupported {
        return Err("Hermitian AMG must reject unconjugated transpose".into());
    }
    let (amg_h, amg_checks) = check_hermitian_exact("Hermitian AMG", &amg, 8, 0xA0A)?;
    let amg_t = 0.0;

    let saddle = SaddlePointPreconditioner::from_matrix(
        &saddle_matrix(),
        SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::Diagonal,
            schur_strategy: SchurStrategy::Diagonal,
            upper_triangular: false,
        },
    );
    let (saddle_t, saddle_h, saddle_checks) = check_exact("saddle point", &saddle, 4, 0x5ADD)?;

    let unsupported = AmgPreconditioner::try_from_csr(
        &amg_matrix,
        AmgConfig {
            restriction: AmgRestriction::Transpose,
            ..AmgConfig::default()
        },
    )
    .map_err(|error| format!("unsupported AMG setup failed: {error}"))?;
    if unsupported.transpose_capability() != AdjointCapability::Unsupported
        || unsupported.hermitian_capability() != AdjointCapability::Unsupported
    {
        return Err("default complex AMG must advertise unsupported adjoints".into());
    }
    let probe = vec![Complex64::new(0.25, -0.5); 8];
    let transpose_rejected = matches!(
        unsupported.try_apply_transpose(&probe),
        Err(SolverError::UnsupportedOperation { .. })
    );
    let hermitian_rejected = matches!(
        unsupported.try_apply_hermitian(&probe),
        Err(SolverError::UnsupportedOperation { .. })
    );
    if !transpose_rejected || !hermitian_rejected || unsupported.try_apply(&probe).is_err() {
        return Err("unsupported AMG adjoint boundary did not fail closed".into());
    }

    Ok((
        cheb_t.max(amg_t).max(saddle_t),
        cheb_h.max(amg_h).max(saddle_h),
        cheb_checks + amg_checks + saddle_checks,
        3.0,
        if transpose_rejected { 1.0 } else { 0.0 },
        if hermitian_rejected { 1.0 } else { 0.0 },
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            max_transpose,
            max_hermitian,
            checks,
            capability_cases,
            transpose_rejected,
            hermitian_rejected,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent exact-adjoint capability matrix for Chebyshev, AMG, and saddle-point preconditioners")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .formulation("exact transpose/Hermitian properties plus explicit unsupported-adjoint failure")
                .dimension("3x3 Chebyshev, 8x8 Hermitian AMG, 4x4 saddle-point, and complex AMG unsupported fixture")
                .tolerance("all exact capabilities advertise Exact; adjoint errors <= 1e-10; both unsupported actions reject; forward remains available")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .maximum("max_transpose_error", TOLERANCE)
                .maximum("max_hermitian_error", TOLERANCE)
                .metric("max_transpose_error", max_transpose)
                .metric("max_hermitian_error", max_hermitian)
                .metric("exact_property_checks", checks)
                .metric("capability_cases", capability_cases)
                .metric("unsupported_transpose_rejected", transpose_rejected)
                .metric("unsupported_hermitian_rejected", hermitian_rejected);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
                .reference("independent exact-adjoint capability matrix")
                .input_checksum(CHECKSUM)
                .reference_checksum(CHECKSUM)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
