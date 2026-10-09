//! Release-backed active acoustic-power balance contract.

use sonium_qa::QaResult;
use sonium_sim_room::energy_validation::{
    AcousticPowerBalanceSample, check_acoustic_energy_balance,
};

fn run() -> Result<QaResult, String> {
    let report = check_acoustic_energy_balance(
        &[
            AcousticPowerBalanceSample {
                frequency_hz: 100.0,
                source_power_w: 1.0,
                absorbed_power_w: 0.25,
                radiated_power_w: 0.75,
            },
            AcousticPowerBalanceSample {
                frequency_hz: 200.0,
                source_power_w: 2.0,
                absorbed_power_w: 0.5,
                radiated_power_w: 1.5,
            },
            AcousticPowerBalanceSample {
                frequency_hz: 400.0,
                source_power_w: 4.0,
                absorbed_power_w: 1.0,
                radiated_power_w: 3.0,
            },
        ],
        1.0e-12,
    )?;
    report.require_within(1.0e-12, 1.0e-12, 1.0e-12)?;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("sample_count", report.sample_count as f64)
        .metric("rms_relative_error", report.rms_relative_error)
        .metric("max_relative_error", report.max_relative_error)
        .metric("max_absolute_error_w", report.max_absolute_error_w)
        .minimum("sample_count", 1.0)
        .maximum("rms_relative_error", 1.0e-12)
        .maximum("max_relative_error", 1.0e-12)
        .maximum("max_absolute_error_w", 1.0e-12)
        .input_checksum("analytic/deterministic-fixture:roomsim-energy-balance-v1")
        .reference("cycle-averaged active source power equals passive absorbed plus radiated power")
        .reference_checksum("analytic/deterministic-fixture:roomsim-energy-balance-v1")
        .dimension("one source, passive absorbed/radiated ledger, 100/200/400 Hz")
        .formulation(
            "checked finite non-negative active-power ledger with relative and absolute residuals",
        )
        .tolerance("RMS/max relative and absolute balance errors <= 1e-12")
        .seed("roomsim-energy-balance-v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim energy-balance result must serialize")
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
                    .expect("RoomSim energy-balance failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

const CASE_ID: &str = "roomsim.energy-balance-contract.v1";
