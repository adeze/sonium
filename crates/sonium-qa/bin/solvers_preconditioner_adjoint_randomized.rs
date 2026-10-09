//! Independent randomized CPU evidence for preconditioner adjoint contracts.

use num_complex::{Complex32, Complex64};
use sonium_qa::QaResult;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{
    AdditiveSchwarzPreconditioner, BlockJacobiPreconditioner, DiagonalPreconditioner,
    HiluPreconditioner, IluPreconditioner, IlutConfig, IlutPreconditioner,
    MixedPrecisionPreconditioner, ParIlutPreconditioner,
};
use sonium_solvers_core::traits::Preconditioner;
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-adjoint-randomized.v1";
const TOLERANCE: f64 = 1.0e-6;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::try_from_raw_parts(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            Complex64::new(4.0, 0.5),
            Complex64::new(0.8, -0.2),
            Complex64::new(0.3, 0.1),
            Complex64::new(3.0, -0.75),
            Complex64::new(0.7, 0.2),
            Complex64::new(0.4, -0.15),
            Complex64::new(2.5, 0.3),
        ],
    )
    .expect("canonical complex property matrix")
}

fn vector(seed: u64) -> Vec<Complex64> {
    let mut state = seed;
    (0..3)
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
) -> Result<f64, String> {
    let mr = preconditioner.apply(r);
    let mh_s = preconditioner
        .try_apply_hermitian(s)
        .map_err(|error| format!("Hermitian action failed: {error}"))?;
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| a.conj() * b).sum();
    let rhs: Complex64 = r.iter().zip(&mh_s).map(|(a, b)| a.conj() * b).sum();
    Ok((lhs - rhs).norm())
}

fn transpose_error<P: Preconditioner<Complex64>>(
    preconditioner: &P,
    r: &[Complex64],
    s: &[Complex64],
) -> Result<f64, String> {
    let mr = preconditioner.apply(r);
    let mt_s = preconditioner
        .try_apply_transpose(s)
        .map_err(|error| format!("transpose action failed: {error}"))?;
    let lhs: Complex64 = mr.iter().zip(s).map(|(a, b)| *a * b).sum();
    let rhs: Complex64 = r.iter().zip(&mt_s).map(|(a, b)| *a * b).sum();
    Ok((lhs - rhs).norm())
}

fn update<P: Preconditioner<Complex64>>(
    name: &str,
    preconditioner: &P,
    seed: u64,
    max_hermitian: &mut f64,
    max_transpose: &mut f64,
) -> Result<(), String> {
    let r = vector(seed);
    let s = vector(seed.wrapping_add(0x9E37_79B9));
    *max_hermitian = (*max_hermitian).max(hermitian_error(preconditioner, &r, &s)?);
    *max_transpose = (*max_transpose).max(transpose_error(preconditioner, &r, &s)?);
    if !(*max_hermitian).is_finite() || !(*max_transpose).is_finite() {
        return Err(format!("{name} produced a non-finite property error"));
    }
    Ok(())
}

fn run() -> Result<(f64, f64, f64), String> {
    let matrix = matrix();
    let ilu = IluPreconditioner::try_from_csr(&matrix).map_err(|error| format!("ILU: {error}"))?;
    let ilut =
        IlutPreconditioner::try_from_csr(&matrix).map_err(|error| format!("ILUT: {error}"))?;
    let block_jacobi = BlockJacobiPreconditioner::try_from_csr(&matrix, 2, IlutConfig::default())
        .map_err(|error| format!("Block-Jacobi: {error}"))?;
    let diagonal = DiagonalPreconditioner::try_from_diagonal(&[
        Complex64::new(4.0, 0.5),
        Complex64::new(3.0, -0.75),
        Complex64::new(2.5, 0.3),
    ])
    .map_err(|error| format!("diagonal: {error}"))?;
    let schwarz = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, 1)
        .map_err(|error| format!("Schwarz: {error}"))?;
    let parilut = ParIlutPreconditioner::try_from_csr(&matrix)
        .map_err(|error| format!("ParILUT: {error}"))?;
    let hilu = HiluPreconditioner::try_from_csr_uncoupled(&matrix)
        .map_err(|error| format!("HILU: {error}"))?;
    let mixed_low = DiagonalPreconditioner::try_from_diagonal(&[
        Complex32::new(0.25, 0.1),
        Complex32::new(-0.2, 0.05),
        Complex32::new(0.4, -0.15),
    ])
    .map_err(|error| format!("mixed-precision diagonal: {error}"))?;
    let mixed = MixedPrecisionPreconditioner::<Complex64, Complex32>::new(Box::new(mixed_low));

    let mut max_hermitian = 0.0;
    let mut max_transpose = 0.0;
    for seed in 0..16_u64 {
        update(
            "ILU",
            &ilu,
            0xA5A5_0000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "ILUT",
            &ilut,
            0xA5A5_1000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "Block-Jacobi",
            &block_jacobi,
            0xA5A5_2000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "diagonal",
            &diagonal,
            0xA5A5_3000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "Schwarz",
            &schwarz,
            0xA5A5_4000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "ParILUT",
            &parilut,
            0xA5A5_5000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "HILU",
            &hilu,
            0xA5A5_6000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
        update(
            "MixedPrecision",
            &mixed,
            0xA5A5_7000 + seed,
            &mut max_hermitian,
            &mut max_transpose,
        )?;
    }
    Ok((max_hermitian, max_transpose, 16.0 * 8.0))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((max_hermitian, max_transpose, checks)) => {
            let result = QaResult::passed(CASE_ID)
            .reference(
                "independent Complex64 transpose/Hermitian identities including mixed precision",
            )
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-adjoint-randomized-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-adjoint-randomized-v1",
                )
            .dimension("3x3 CSR; 8 preconditioners; 16 deterministic vector pairs")
            .tolerance("maximum transpose and Hermitian identity error <= 1e-6")
                .seed("solvers-preconditioner-adjoint-randomized:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .maximum("max_hermitian_error", TOLERANCE)
                .maximum("max_transpose_error", TOLERANCE)
                .metric("max_hermitian_error", max_hermitian)
                .metric("max_transpose_error", max_transpose)
                .metric("property_checks", checks);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("FAILED: {error}");
            let result = QaResult::failed(CASE_ID, error)
            .reference(
                "independent Complex64 transpose/Hermitian identities including mixed precision",
            )
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-adjoint-randomized-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-adjoint-randomized-v1",
                )
            .dimension("3x3 CSR; 8 preconditioners; 16 deterministic vector pairs")
            .tolerance("maximum transpose and Hermitian identity error <= 1e-6")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            std::process::exit(1);
        }
    }
}
