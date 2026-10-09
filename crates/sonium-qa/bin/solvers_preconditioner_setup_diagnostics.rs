//! Aggregate setup diagnostics for composed preconditioners.

use sonium_compute::CpuBackend;
use sonium_qa::QaResult;
use sonium_solvers::CsrMatrix;
use sonium_solvers::preconditioners::{
    AdditiveSchwarzPreconditioner, AmgConfig, AmgPreconditioner, BackendAmgPreconditioner,
    BlockJacobiPreconditioner, ChebyshevConfig, ChebyshevError, ChebyshevPreconditioner,
    GeneoConfig, IlutConfig, IlutPreconditioner, SchwarzCoarseSetup,
};
use std::time::Instant;

const CASE_ID: &str = "solvers.preconditioner-setup-diagnostics.v1";
const SEED: &str = "solvers-preconditioner-setup-diagnostics:fixture-v1";

fn run() -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), String> {
    let identity = CsrMatrix::<f64>::identity(4);
    let clean_ilut = IlutPreconditioner::from_csr(&identity);
    let clean_ilut_diagnostics = clean_ilut.diagnostics();
    if clean_ilut_diagnostics.regularized_pivots != 0
        || clean_ilut_diagnostics.non_finite_entries != 0
    {
        return Err(format!(
            "clean ILUT diagnostics not empty: {clean_ilut_diagnostics:?}"
        ));
    }
    let block = BlockJacobiPreconditioner::from_csr(&identity, 2, IlutConfig::default());
    let block_diagnostics = block.diagnostics();
    if block_diagnostics.regularized_pivots != 0
        || block_diagnostics.blocks_with_regularized_pivots != 0
        || block_diagnostics.non_finite_entries != 0
    {
        return Err(format!(
            "clean Block-Jacobi diagnostics were not empty: {:?}",
            block_diagnostics
        ));
    }

    let schwarz = AdditiveSchwarzPreconditioner::from_csr(&identity, 2, 1);
    let schwarz_diagnostics = schwarz.diagnostics();
    let min_abs_pivot = schwarz_diagnostics
        .min_abs_pivot
        .ok_or_else(|| "clean Schwarz diagnostics omitted minimum pivot".to_string())?;
    if !min_abs_pivot.is_finite() || min_abs_pivot <= 0.0 {
        return Err(format!(
            "clean Schwarz minimum pivot was invalid: {min_abs_pivot:?}"
        ));
    }
    if schwarz_diagnostics.skipped_pivots != 0 || schwarz_diagnostics.non_finite_entries != 0 {
        return Err(format!(
            "clean Schwarz diagnostics were not empty: {:?}",
            schwarz_diagnostics
        ));
    }

    let overflow_matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 1.0_f64),
            (0, 1, f64::MAX),
            (0, 2, f64::MAX),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ],
    );
    let geneo_setup = std::panic::catch_unwind(|| {
        AdditiveSchwarzPreconditioner::from_csr_partitions(
            &overflow_matrix,
            &[vec![0], vec![1, 2]],
            1,
        )
        .with_geneo_coarse_space(&overflow_matrix, &GeneoConfig::default())
    })
    .map_err(|_| "GenEO setup panicked on generated non-finite local data".to_string())?;
    let geneo_fallback = geneo_setup.coarse_setup() == SchwarzCoarseSetup::FellBackToOneLevel;
    if !geneo_fallback || geneo_setup.has_coarse_space() {
        return Err("GenEO generated non-finite setup did not fall back cleanly".into());
    }
    let checked_geneo_error = AdditiveSchwarzPreconditioner::from_csr_partitions(
        &overflow_matrix,
        &[vec![0], vec![1, 2]],
        1,
    )
    .try_with_geneo_coarse_space(&overflow_matrix, &GeneoConfig::default())
    .expect_err("checked GenEO setup must reject generated non-finite values");
    let checked_geneo_generated_non_finite = matches!(
        checked_geneo_error,
        sonium_solvers::GeneoSetupError::GeneratedNonFinite { .. }
    );
    if !checked_geneo_generated_non_finite {
        return Err(format!(
            "checked GenEO setup returned unexpected error: {checked_geneo_error:?}"
        ));
    }

    let scale = f64::MAX;
    let off_diagonal = 0.49 * scale;
    let chebyshev_matrix = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, scale),
            (1, 1, scale),
            (2, 2, scale),
            (0, 1, off_diagonal),
            (1, 0, off_diagonal),
            (0, 2, off_diagonal),
            (2, 0, off_diagonal),
            (1, 2, off_diagonal),
            (2, 1, off_diagonal),
        ],
    );
    let chebyshev_rejected = matches!(
        ChebyshevPreconditioner::try_from_csr(&chebyshev_matrix, ChebyshevConfig::default()),
        Err(ChebyshevError::InvalidConfiguration {
            reason: "estimated spectral interval is not finite or usable",
        })
    );
    if !chebyshev_rejected {
        return Err("checked Chebyshev accepted an overflowing spectral estimate".into());
    }
    let amg_rejected =
        AmgPreconditioner::try_from_csr(&chebyshev_matrix, AmgConfig::default()).is_err();
    if !amg_rejected {
        return Err("checked AMG accepted an overflowing generated hierarchy".into());
    }
    let backend = CpuBackend::new();
    let backend_amg_rejected = std::panic::catch_unwind(|| {
        BackendAmgPreconditioner::<CpuBackend, f64>::try_from_csr(
            &backend,
            &chebyshev_matrix,
            AmgConfig::default(),
        )
    })
    .map_err(|_| "checked backend AMG setup panicked on generated non-finite hierarchy")?
    .is_err();
    if !backend_amg_rejected {
        return Err("checked backend AMG accepted an overflowing generated hierarchy".into());
    }

    let generated_non_finite = CsrMatrix::from_raw_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1e-100_f64, 1e154, 1e154, 1e100],
    );
    let config = IlutConfig {
        drop_tolerance: 0.0,
        pivot_tolerance: 1e-300,
        pivoting: false,
        ..IlutConfig::default()
    };
    let compatibility =
        BlockJacobiPreconditioner::from_csr(&generated_non_finite, 1, config.clone());
    let generated_non_finite_entries = compatibility.diagnostics().non_finite_entries;
    if generated_non_finite_entries == 0 {
        return Err("compatibility Block-Jacobi lost generated non-finite diagnostics".into());
    }
    let checked_rejected =
        BlockJacobiPreconditioner::try_from_csr(&generated_non_finite, 1, config.clone()).is_err();
    if !checked_rejected {
        return Err("checked Block-Jacobi accepted generated non-finite local factors".into());
    }

    let ilut_compatibility =
        IlutPreconditioner::from_csr_with_config(&generated_non_finite, &config);
    let ilut_diagnostics = ilut_compatibility.diagnostics();
    if ilut_diagnostics.non_finite_entries == 0 {
        return Err("compatibility ILUT lost generated non-finite diagnostics".into());
    }
    let ilut_checked_rejected =
        IlutPreconditioner::try_from_csr_with_config(&generated_non_finite, &config).is_err();
    if !ilut_checked_rejected {
        return Err("checked ILUT accepted generated non-finite factors".into());
    }

    let singular = CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 0.0]);
    let regularized = IlutPreconditioner::from_csr(&singular).diagnostics();
    if regularized.regularized_pivots == 0 {
        return Err("compatibility ILUT omitted regularized-pivot diagnostics".into());
    }

    Ok((
        min_abs_pivot,
        generated_non_finite_entries as f64,
        checked_rejected as u8 as f64,
        schwarz_diagnostics.skipped_pivots as f64,
        clean_ilut_diagnostics.regularized_pivots as f64,
        ilut_diagnostics.non_finite_entries as f64,
        regularized.regularized_pivots as f64,
        geneo_fallback as u8 as f64,
        checked_geneo_generated_non_finite as u8 as f64,
        chebyshev_rejected as u8 as f64,
        amg_rejected as u8 as f64,
        backend_amg_rejected as u8 as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            min_abs_pivot,
            generated_non_finite_entries,
            checked_rejected,
            skipped_pivots,
            clean_ilut_regularized,
            ilut_non_finite_entries,
            ilut_regularized,
            geneo_fallback,
            checked_geneo_generated_non_finite,
            chebyshev_rejected,
            amg_rejected,
            backend_amg_rejected,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed(SEED)
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-setup-diagnostics-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-setup-diagnostics-v1",
                )
                .reference("aggregate ILUT, Block-Jacobi, Schwarz, and AMG setup diagnostics")
                .metric("schwarz_min_abs_pivot", min_abs_pivot)
                .metric(
                    "block_generated_non_finite_entries",
                    generated_non_finite_entries,
                )
                .metric(
                    "checked_block_rejected_generated_non_finite",
                    checked_rejected,
                )
                .metric("schwarz_skipped_pivots", skipped_pivots)
                .metric("clean_ilut_regularized_pivots", clean_ilut_regularized)
                .metric("ilut_generated_non_finite_entries", ilut_non_finite_entries)
                .metric("ilut_regularized_pivots", ilut_regularized)
                .metric("geneo_generated_non_finite_fallback", geneo_fallback)
                .metric(
                    "checked_geneo_generated_non_finite_rejected",
                    checked_geneo_generated_non_finite,
                )
                .metric(
                    "chebyshev_overflowed_spectral_estimate_rejected",
                    chebyshev_rejected,
                )
                .metric("amg_overflowed_generated_hierarchy_rejected", amg_rejected)
                .metric(
                    "backend_amg_overflowed_generated_hierarchy_rejected",
                    backend_amg_rejected,
                )
                .minimum("block_generated_non_finite_entries", 1.0)
                .minimum("checked_block_rejected_generated_non_finite", 1.0)
                .minimum("ilut_generated_non_finite_entries", 1.0)
                .minimum("ilut_regularized_pivots", 1.0)
                .minimum("geneo_generated_non_finite_fallback", 1.0)
                .minimum("checked_geneo_generated_non_finite_rejected", 1.0)
                .minimum("chebyshev_overflowed_spectral_estimate_rejected", 1.0)
                .minimum("amg_overflowed_generated_hierarchy_rejected", 1.0)
                .minimum("backend_amg_overflowed_generated_hierarchy_rejected", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error.clone())
                .seed(SEED)
                .input_checksum(
                    "analytic/deterministic-fixture:preconditioner-setup-diagnostics-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:preconditioner-setup-diagnostics-v1",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().expect("QA JSON"));
            eprintln!("FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
