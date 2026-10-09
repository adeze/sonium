//! Bounded W2-01 FEM/BEM/XEM boundary-contract evidence.
//!
//! The fixture is intentionally small: it exercises the production
//! conservative trace transfer (including its weighted adjoint) and the
//! product power ledger's passive closure/error paths. It is an integration
//! contract, not a claim that every coupled FEM/BEM scheduler or
//! preconditioner is release-complete.

use num_complex::Complex64;
use sonium_bem::core::coupling::ConservativeTraceTransfer;
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::{PowerLedger, PowerLedgerTolerance};
use std::time::Instant;

const CASE_ID: &str = "coupling.conservative-power-contracts.v1";

#[derive(Debug)]
struct Metrics {
    transfer_constant_error: f64,
    weighted_adjoint_error: f64,
    power_relative_residual: f64,
    typed_rejections: usize,
    elapsed_ms: u64,
}

fn run() -> Result<Metrics, String> {
    let started = Instant::now();
    let transfer = ConservativeTraceTransfer::try_new(
        vec![vec![(0, 0.5), (1, 0.5)]],
        vec![1.0, 1.0],
        vec![2.0],
    )
    .map_err(|error| format!("valid conservative transfer rejected: {error:?}"))?;
    let source = [Complex64::new(2.0, -1.0), Complex64::new(2.0, -1.0)];
    let target = transfer
        .apply(&source)
        .map_err(|error| format!("trace transfer failed: {error:?}"))?;
    let transfer_constant_error = (target[0] - source[0]).norm();
    let weighted_adjoint_error = transfer
        .check_adjoint(
            &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 1.0)],
            &[Complex64::new(2.0, -1.0)],
        )
        .map_err(|error| format!("weighted adjoint check failed: {error:?}"))?;

    let ledger = PowerLedger::from_components(10.0, 2.0, 1.0, 0.5, 0.2, 0.3, 4.0, 2.0)
        .map_err(|error| format!("valid power ledger rejected: {error:?}"))?;
    let closure = ledger
        .close(PowerLedgerTolerance {
            absolute_w: 1.0e-12,
            relative: 1.0e-12,
        })
        .map_err(|error| format!("power ledger did not close: {error:?}"))?;

    let mut typed_rejections = 0;
    if ConservativeTraceTransfer::try_new(vec![vec![(0, 1.0), (0, 0.0)]], vec![1.0], vec![1.0])
        .is_err()
    {
        typed_rejections += 1;
    }
    if PowerLedger::from_components(f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).is_err() {
        typed_rejections += 1;
    }
    if typed_rejections != 2 {
        return Err("invalid coupling/power inputs were not rejected".into());
    }
    Ok(Metrics {
        transfer_constant_error,
        weighted_adjoint_error,
        power_relative_residual: closure.relative_residual,
        typed_rejections,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("transfer_constant_error", metrics.transfer_constant_error)
                .metric("weighted_adjoint_error", metrics.weighted_adjoint_error)
                .metric("power_relative_residual", metrics.power_relative_residual)
                .metric("typed_rejections", metrics.typed_rejections as f64)
                .maximum("transfer_constant_error", 1.0e-14)
                .maximum("weighted_adjoint_error", 1.0e-14)
                .maximum("power_relative_residual", 1.0e-12)
                .minimum("typed_rejections", 2.0)
                .input_checksum("analytic/deterministic-fixture:conservative-power-v1")
                .reference("constant preservation, weighted adjoint identity, analytic power accounting, typed invalid-input rejection")
                .reference_checksum("analytic/deterministic-fixture:conservative-power-v1")
                .dimension("two source trace cells to one target trace cell; scalar power ledger")
                .formulation("weighted conservative trace/adjoint and passive power closure contract")
                .tolerance("trace and power residuals <= 1e-12; two invalid inputs rejected")
                .seed("coupling-conservative-power-v1")
                .timing_ms("coupling_contracts", metrics.elapsed_ms)
                .elapsed_ms(metrics.elapsed_ms);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("coupling result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("coupling failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
