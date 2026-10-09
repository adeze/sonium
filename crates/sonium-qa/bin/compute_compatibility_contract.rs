//! CPU contract for the infallible multi-dot compatibility wrappers.
//!
//! The checked backend APIs own typed failures.  The legacy `Vec`-returning
//! wrappers must still preserve valid results, reject malformed shapes without
//! panicking, and use the same Hermitian reduction semantics as their checked
//! counterparts.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, FromPrimitive};
use sonium_compute::{Backend, CpuBackend, DeviceScalar, ReductionMode};
use sonium_qa::QaResult;

const CASE_ID: &str = "compute.compatibility-contract.v1";

fn close<T: DeviceScalar>(actual: T, expected: T) -> bool {
    let tolerance = T::Real::from_f64(2.0e-5).unwrap_or_else(T::Real::epsilon);
    (actual - expected).norm_sqr() <= tolerance * tolerance
}

fn check_scalar<T: DeviceScalar>(backend: &CpuBackend) -> Result<(usize, usize, usize), String> {
    let real = |value: f64| {
        T::Real::from_f64(value)
            .map(T::from_real)
            .ok_or_else(|| format!("cannot represent fixture value {value}"))
    };
    let one = real(1.0)?;
    let two = real(2.0)?;
    let three = real(3.0)?;
    let four = real(4.0)?;
    let five = real(5.0)?;
    let six = real(6.0)?;

    let mut first = backend
        .raw_alloc::<T>(2)
        .map_err(|error| format!("first basis allocation failed: {error}"))?;
    let mut second = backend
        .raw_alloc::<T>(2)
        .map_err(|error| format!("second basis allocation failed: {error}"))?;
    let mut flat = backend
        .raw_alloc::<T>(4)
        .map_err(|error| format!("flat basis allocation failed: {error}"))?;
    let mut y = backend
        .raw_alloc::<T>(2)
        .map_err(|error| format!("input allocation failed: {error}"))?;
    backend.copy_h2d(&[one, two], &mut first);
    backend.copy_h2d(&[three, four], &mut second);
    backend.copy_h2d(&[one, two, three, four], &mut flat);
    backend.copy_h2d(&[five, six], &mut y);

    let expected = [real(17.0)?, real(39.0)?];
    let basis = [&first, &second];
    let fast = backend.multi_dot(&basis, &y);
    if fast.len() != 2 || !close(fast[0], expected[0]) || !close(fast[1], expected[1]) {
        return Err(format!("fast compatibility result mismatch: {fast:?}"));
    }
    let deterministic = backend.multi_dot_with_mode(ReductionMode::Deterministic, &basis, &y);
    if deterministic.len() != 2
        || !close(deterministic[0], expected[0])
        || !close(deterministic[1], expected[1])
    {
        return Err(format!(
            "deterministic compatibility result mismatch: {deterministic:?}"
        ));
    }
    let strided = backend.multi_dot_strided(&flat, 2, 2, &y);
    if strided.len() != 2 || !close(strided[0], expected[0]) || !close(strided[1], expected[1]) {
        return Err(format!(
            "strided compatibility result mismatch: {strided:?}"
        ));
    }

    let mut malformed = backend
        .raw_alloc::<T>(1)
        .map_err(|error| format!("malformed basis allocation failed: {error}"))?;
    backend.copy_h2d(&[one], &mut malformed);
    let malformed_basis = [&malformed];
    if !backend.multi_dot(&malformed_basis, &y).is_empty() {
        return Err("malformed multi-dot unexpectedly returned a result".into());
    }
    if !backend.multi_dot_strided(&flat, 3, 2, &y).is_empty() {
        return Err("malformed strided multi-dot unexpectedly returned a result".into());
    }

    Ok((3, 2, 2))
}

fn check_precision_cast_compatibility(backend: &CpuBackend) -> Result<(usize, usize), String> {
    let mut destination = backend
        .raw_alloc::<f32>(2)
        .map_err(|error| format!("precision-cast destination allocation failed: {error}"))?;
    backend.copy_h2d(&[3.0_f32, -4.0_f32], &mut destination);

    let mut observed = [0.0_f32; 2];
    backend.copy_h2d_cast(&[1.5_f64, -2.25_f64], &mut destination);
    backend.copy_d2h(&destination, &mut observed);
    if observed != [1.5, -2.25] {
        return Err(format!("valid precision cast mismatch: {observed:?}"));
    }

    backend.copy_h2d_cast(&[9.0_f64], &mut destination);
    backend.copy_d2h(&destination, &mut observed);
    if observed != [1.5, -2.25] {
        return Err(format!("legacy mismatch mutated destination: {observed:?}"));
    }

    if backend
        .try_copy_h2d_cast(&[9.0_f64], &mut destination)
        .is_ok()
    {
        return Err("checked precision cast accepted a length mismatch".into());
    }

    Ok((1, 2))
}

fn run() -> Result<(usize, usize, usize, usize, usize), String> {
    let backend = CpuBackend::new();
    let mut scalar_cases = 0;
    let mut valid_modes = 0;
    let mut rejection_cases = 0;
    for check in [
        check_scalar::<f32> as fn(&CpuBackend) -> Result<(usize, usize, usize), String>,
        check_scalar::<f64>,
        check_scalar::<Complex32>,
        check_scalar::<Complex64>,
    ] {
        let (scalars, valid, rejected) = check(&backend)?;
        scalar_cases += scalars;
        valid_modes += valid;
        rejection_cases += rejected;
    }
    let (cast_cases, cast_rejections) = check_precision_cast_compatibility(&backend)?;
    Ok((
        scalar_cases,
        valid_modes,
        rejection_cases,
        cast_cases,
        cast_rejections,
    ))
}

fn main() {
    match run() {
        Ok((scalar_cases, valid_modes, rejection_cases, cast_cases, cast_rejections)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("scalar_cases", scalar_cases as f64)
                .metric("valid_modes", valid_modes as f64)
                .metric("rejection_cases", rejection_cases as f64)
                .metric("cast_cases", cast_cases as f64)
                .metric("cast_rejections", cast_rejections as f64)
                .minimum("scalar_cases", 12.0)
                .minimum("valid_modes", 8.0)
                .minimum("rejection_cases", 8.0)
                .minimum("cast_cases", 1.0)
                .minimum("cast_rejections", 2.0)
                .reference("independent Hermitian multi-dot compatibility invariants")
                .reference_checksum(
                    "analytic/deterministic-fixture:compute-compatibility-contract-v1",
                )
                .dimension(
                    "f32/f64/Complex32/Complex64; fast/deterministic/strided; malformed shapes",
                )
                .seed("compute-compatibility-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compatibility QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu");
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("compatibility QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
