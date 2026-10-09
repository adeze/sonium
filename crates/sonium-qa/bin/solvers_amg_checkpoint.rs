//! Independent CPU evidence for host AMG checkpoint round trips.

use num_complex::Complex64;
use sonium_compute::{Backend, CpuBackend, DevicePreconditioner};
use sonium_qa::QaResult;
use sonium_solvers_core::traits::Preconditioner;
use sonium_solvers_preconditioners::{AmgConfig, AmgPreconditioner, BackendAmgPreconditioner};
use sonium_solvers_sparse::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "solvers.amg-checkpoint.v1";
const INPUT_CHECKSUM: &str = "analytic/deterministic-fixture:amg-checkpoint-v1";
const TOLERANCE: f64 = 1.0e-12;

fn matrix() -> CsrMatrix<Complex64> {
    CsrMatrix::from_raw_parts(
        8,
        8,
        (0..=8).collect(),
        (0..8).collect(),
        (0..8)
            .map(|index| Complex64::new(2.0 + index as f64, 0.0))
            .collect(),
    )
}

fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
}

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64), String> {
    let source = AmgPreconditioner::try_from_csr(&matrix(), AmgConfig::default())
        .map_err(|error| format!("AMG setup failed: {error:?}"))?;
    let bytes = source
        .to_checkpoint_bytes()
        .map_err(|error| format!("AMG checkpoint serialization failed: {error:?}"))?;
    let restored = AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes)
        .map_err(|error| format!("AMG checkpoint restore failed: {error:?}"))?;
    let rhs: Vec<_> = (0..8)
        .map(|index| Complex64::new(index as f64 + 0.25, -0.5 * index as f64))
        .collect();
    let forward_error = max_error(&source.apply(&rhs), &restored.apply(&rhs));
    let transpose_error = max_error(
        &source.apply_transpose(&rhs),
        &restored.apply_transpose(&rhs),
    );
    let hermitian_error = max_error(
        &source.apply_hermitian(&rhs),
        &restored.apply_hermitian(&rhs),
    );

    let mut malformed_rejections = 0.0;
    for length in [0, 1, bytes.len() / 2, bytes.len().saturating_sub(1)] {
        if AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bytes[..length]).is_err() {
            malformed_rejections += 1.0;
        }
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    if AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&trailing).is_err() {
        malformed_rejections += 1.0;
    }
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    if AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_magic).is_err() {
        malformed_rejections += 1.0;
    }
    let mut bad_flags = bytes.clone();
    bad_flags[13] = 1;
    if AmgPreconditioner::<Complex64>::from_checkpoint_bytes(&bad_flags).is_err() {
        malformed_rejections += 1.0;
    }
    let levels = restored.num_levels() as f64;
    if !forward_error.is_finite()
        || !transpose_error.is_finite()
        || !hermitian_error.is_finite()
        || malformed_rejections < 7.0
        || levels < 1.0
    {
        return Err(format!(
            "AMG checkpoint gate failed: errors=({forward_error}, {transpose_error}, {hermitian_error}), malformed={malformed_rejections}, levels={levels}"
        ));
    }
    let backend = CpuBackend::new();
    let backend_amg = BackendAmgPreconditioner::<CpuBackend, Complex64>::try_from_checkpoint_bytes(
        &backend, &bytes,
    )
    .map_err(|error| format!("backend AMG checkpoint restore failed: {error:?}"))?;
    let backend_bytes = backend_amg
        .to_checkpoint_bytes()
        .map_err(|error| format!("backend AMG checkpoint serialization failed: {error:?}"))?;
    let backend_round_trip =
        BackendAmgPreconditioner::<CpuBackend, Complex64>::try_from_checkpoint_bytes(
            &backend,
            &backend_bytes,
        )
        .map_err(|error| format!("backend AMG envelope restore failed: {error:?}"))?
        .to_checkpoint_bytes()
        .map_err(|error| format!("backend AMG envelope serialization failed: {error:?}"))?;
    if backend_round_trip != backend_bytes {
        return Err("backend AMG checkpoint bytes changed during round trip".into());
    }

    let mut rhs_device = backend
        .raw_alloc::<Complex64>(rhs.len())
        .map_err(|error| format!("backend RHS allocation failed: {error:?}"))?;
    let mut output_device = backend
        .raw_alloc::<Complex64>(rhs.len())
        .map_err(|error| format!("backend output allocation failed: {error:?}"))?;
    backend.copy_h2d(&rhs, &mut rhs_device);
    DevicePreconditioner::try_apply(&backend_amg, &backend, &rhs_device, &mut output_device)
        .map_err(|error| format!("backend AMG apply failed: {error:?}"))?;
    let mut backend_values = vec![Complex64::new(0.0, 0.0); rhs.len()];
    backend.copy_d2h(&output_device, &mut backend_values);
    let backend_action_error = max_error(&source.apply(&rhs), &backend_values);
    let backend_apply_count = backend_amg.diagnostics().apply_count as f64;

    let mut backend_bad_magic = bytes.clone();
    backend_bad_magic[0] ^= 0xff;
    let mut backend_trailing = bytes.clone();
    backend_trailing.push(0);
    let backend_malformed_rejections = [backend_bad_magic, backend_trailing]
        .into_iter()
        .filter(|packet| {
            BackendAmgPreconditioner::<CpuBackend, Complex64>::try_from_checkpoint_bytes(
                &backend, packet,
            )
            .is_err()
        })
        .count() as f64;
    if !backend_action_error.is_finite()
        || backend_action_error > 1.0e-9
        || backend_apply_count != 1.0
        || backend_malformed_rejections < 2.0
    {
        return Err(format!(
            "backend AMG checkpoint gate failed: action_error={backend_action_error}, apply_count={backend_apply_count}, malformed={backend_malformed_rejections}"
        ));
    }

    Ok((
        forward_error,
        transpose_error,
        hermitian_error,
        malformed_rejections,
        backend_action_error,
        backend_malformed_rejections,
        backend_apply_count,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            forward_error,
            transpose_error,
            hermitian_error,
            malformed_rejections,
            backend_action_error,
            backend_malformed_rejections,
            backend_apply_count,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum(INPUT_CHECKSUM)
                .reference_checksum(INPUT_CHECKSUM)
                .dimension("8x8 diagonal Complex64 CSR AMG; CPU backend restore/apply")
                .tolerance("host action errors <= 1e-12; backend action error <= 1e-9; seven host and two backend malformed packets rejected")
                .metric("forward_error", forward_error)
                .metric("transpose_error", transpose_error)
                .metric("hermitian_error", hermitian_error)
                .metric("malformed_rejections", malformed_rejections)
                .metric("backend_action_error", backend_action_error)
                .metric("backend_malformed_rejections", backend_malformed_rejections)
                .metric("backend_apply_count", backend_apply_count)
                .maximum("forward_error", TOLERANCE)
                .maximum("transpose_error", TOLERANCE)
                .maximum("hermitian_error", TOLERANCE)
                .minimum("malformed_rejections", 7.0)
                .maximum("backend_action_error", 1.0e-9)
                .minimum("backend_malformed_rejections", 2.0)
                .minimum("backend_apply_count", 1.0)
                .maximum("backend_apply_count", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            eprintln!("{error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
