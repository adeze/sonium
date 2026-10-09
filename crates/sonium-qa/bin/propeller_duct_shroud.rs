//! Bounded quasi-one-dimensional duct/shroud flow reference QA.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{DuctShroudOptions, DuctShroudStation, evaluate_duct_shroud_reference};

const CASE_ID: &str = "propeller.duct-shroud-reference.v1";

fn run() -> Result<(f64, f64, f64), String> {
    let stations = [
        DuctShroudStation {
            axial_position_m: 0.0,
            flow_area_m2: 0.08,
            hydraulic_diameter_m: 0.30,
        },
        DuctShroudStation {
            axial_position_m: 0.5,
            flow_area_m2: 0.04,
            hydraulic_diameter_m: 0.22,
        },
        DuctShroudStation {
            axial_position_m: 1.0,
            flow_area_m2: 0.06,
            hydraulic_diameter_m: 0.26,
        },
    ];
    let result = evaluate_duct_shroud_reference(
        &stations,
        DuctShroudOptions {
            mass_flow_kg_s: 0.8,
            maximum_mach: 0.8,
            ..DuctShroudOptions::default()
        },
    )
    .map_err(|error| error.to_string())?;
    if !result.is_finite() {
        return Err("duct/shroud result is not finite".into());
    }
    if result.maximum_continuity_residual_kg_s > 1.0e-12 {
        return Err(format!(
            "continuity residual too large: {}",
            result.maximum_continuity_residual_kg_s
        ));
    }
    if result.stations[1].velocity_m_s <= result.stations[0].velocity_m_s
        || result.stations[1].static_pressure_pa >= result.stations[0].static_pressure_pa
    {
        return Err("contraction did not accelerate and reduce static pressure".into());
    }
    Ok((
        result.maximum_velocity_m_s,
        result.maximum_mach,
        result.maximum_continuity_residual_kg_s,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((maximum_velocity, maximum_mach, continuity_residual)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("maximum_velocity_m_s", maximum_velocity)
                .metric("maximum_mach", maximum_mach)
                .metric("maximum_continuity_residual_kg_s", continuity_residual)
                .maximum("maximum_continuity_residual_kg_s", 1.0e-12)
                .input_checksum("analytic/deterministic-fixture:propeller-duct-shroud-v1")
                .reference("constant-density continuity and Darcy/minor-loss station identities")
                .reference_checksum("analytic/invariant:propeller-duct-shroud-v1")
                .dimension("three-station geometry-derived duct ladder; f64 CPU")
                .formulation("quasi-one-dimensional incompressible duct/shroud mean-flow screen")
                .tolerance("continuity residual <= 1e-12 kg/s; finite subsonic station states")
                .seed("propeller-duct-shroud-v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result must serialize")
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
                    .expect("QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
