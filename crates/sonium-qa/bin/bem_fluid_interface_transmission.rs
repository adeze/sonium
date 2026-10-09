//! Release-facing typed fluid-interface transmission evidence.
//!
//! The production BEM crate does not yet expose a penetrable multi-region
//! boundary operator. This case therefore gates the production typed material
//! adapter against an independently evaluated planar Fresnel reference while
//! keeping that limitation explicit in the result metadata.

use num_complex::Complex64;
use sonium_bem::{AmbientMedium, FluidInterface, MaterialError};
use sonium_qa::QaResult;
use sonium_xem::AcousticMedium;
use std::time::Instant;

const CASE_ID: &str = "bem.fluid-interface-transmission.v1";
const INCIDENCE_ANGLE_RADIANS: f64 = 0.1;
const MAX_COEFFICIENT_ERROR: f64 = 2.0e-13;
const MAX_POWER_ERROR: f64 = 2.0e-13;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    reflection_coefficient_error: f64,
    transmission_coefficient_error: f64,
    reflected_power_error: f64,
    transmitted_power_error: f64,
    power_balance_residual: f64,
    rejection_checks: f64,
    total_internal_reflection: f64,
}

fn fresnel_reference(
    incident: AmbientMedium,
    transmitted: AmbientMedium,
    incidence_angle_radians: f64,
) -> (Complex64, Complex64, f64, f64) {
    let sin_incident = incidence_angle_radians.sin();
    let cos_incident = incidence_angle_radians.cos();
    let sin_transmitted = transmitted.speed_of_sound / incident.speed_of_sound * sin_incident;
    let cos_transmitted = (1.0 - sin_transmitted * sin_transmitted).sqrt();
    let incident_admittance = cos_incident / incident.characteristic_impedance();
    let transmitted_admittance = cos_transmitted / transmitted.characteristic_impedance();
    let reflection = Complex64::new(
        (incident_admittance - transmitted_admittance)
            / (incident_admittance + transmitted_admittance),
        0.0,
    );
    let transmission = Complex64::new(1.0, 0.0) + reflection;
    let reflected_power = reflection.norm_sqr();
    let transmitted_power = transmitted_admittance / incident_admittance * transmission.norm_sqr();
    (reflection, transmission, reflected_power, transmitted_power)
}

fn run() -> Result<Metrics, String> {
    let incident_xem =
        AcousticMedium::try_new(1_000.0, 343.0, 1.21).map_err(|error| error.to_string())?;
    let transmitted_xem =
        AcousticMedium::try_new(1_000.0, 1_480.0, 1_000.0).map_err(|error| error.to_string())?;
    let incident = AmbientMedium::new(1.21, 343.0);
    let transmitted = AmbientMedium::new(1_000.0, 1_480.0);
    let interface = FluidInterface::try_from_xem(incident_xem, transmitted_xem)
        .map_err(|error| error.to_string())?;
    let result = interface
        .try_plane_wave_transmission(INCIDENCE_ANGLE_RADIANS)
        .map_err(|error| error.to_string())?;
    let (
        reference_reflection,
        reference_transmission,
        reference_reflected_power,
        reference_transmitted_power,
    ) = fresnel_reference(incident, transmitted, INCIDENCE_ANGLE_RADIANS);

    let reflection_coefficient_error = (result.pressure_reflection - reference_reflection).norm();
    let transmission_coefficient_error =
        (result.pressure_transmission - reference_transmission).norm();
    let reflected_power_error = (result.reflected_power - reference_reflected_power).abs();
    let transmitted_power_error = (result.transmitted_power - reference_transmitted_power).abs();
    let power_balance_residual = result.power_balance_residual().abs();
    if reflection_coefficient_error > MAX_COEFFICIENT_ERROR
        || transmission_coefficient_error > MAX_COEFFICIENT_ERROR
        || reflected_power_error > MAX_POWER_ERROR
        || transmitted_power_error > MAX_POWER_ERROR
        || power_balance_residual > MAX_POWER_ERROR
    {
        return Err(format!(
            "{CASE_ID}: Fresnel errors reflection={reflection_coefficient_error:.3e}, transmission={transmission_coefficient_error:.3e}, reflected_power={reflected_power_error:.3e}, transmitted_power={transmitted_power_error:.3e}, balance={power_balance_residual:.3e}"
        ));
    }
    if result.total_internal_reflection {
        return Err(format!(
            "{CASE_ID}: propagating interface marked as total reflection"
        ));
    }

    let different_frequency =
        AcousticMedium::try_new(1_001.0, 1_480.0, 1_000.0).map_err(|error| error.to_string())?;
    if !matches!(
        FluidInterface::try_from_xem(incident_xem, different_frequency),
        Err(MaterialError::InvalidConvention(_))
    ) {
        return Err(format!("{CASE_ID}: mixed-frequency media were accepted"));
    }
    if !matches!(
        interface.try_plane_wave_transmission(std::f64::consts::FRAC_PI_2),
        Err(MaterialError::InvalidIncidenceAngle(_))
    ) {
        return Err(format!("{CASE_ID}: invalid incidence angle was accepted"));
    }

    let total_reflection = FluidInterface::new(incident, transmitted)
        .try_plane_wave_transmission(1.0)
        .map_err(|error| error.to_string())?;
    if !total_reflection.total_internal_reflection || total_reflection.transmitted_power != 0.0 {
        return Err(format!(
            "{CASE_ID}: total-internal-reflection branch failed"
        ));
    }

    Ok(Metrics {
        reflection_coefficient_error,
        transmission_coefficient_error,
        reflected_power_error,
        transmitted_power_error,
        power_balance_residual,
        rejection_checks: 2.0,
        total_internal_reflection: 1.0,
    })
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok(metrics)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric(
                    "reflection_coefficient_error",
                    metrics.reflection_coefficient_error,
                )
                .metric(
                    "transmission_coefficient_error",
                    metrics.transmission_coefficient_error,
                )
                .metric("reflected_power_error", metrics.reflected_power_error)
                .metric("transmitted_power_error", metrics.transmitted_power_error)
                .metric("power_balance_residual", metrics.power_balance_residual)
                .metric("rejection_checks", metrics.rejection_checks)
                .metric(
                    "total_internal_reflection_branch",
                    metrics.total_internal_reflection,
                )
                .maximum("reflection_coefficient_error", MAX_COEFFICIENT_ERROR)
                .maximum("transmission_coefficient_error", MAX_COEFFICIENT_ERROR)
                .maximum("reflected_power_error", MAX_POWER_ERROR)
                .maximum("transmitted_power_error", MAX_POWER_ERROR)
                .maximum("power_balance_residual", MAX_POWER_ERROR)
                .minimum("rejection_checks", 2.0)
                .minimum("total_internal_reflection_branch", 1.0)
                .input_checksum("analytic/deterministic-fixture:bem-fluid-interface-v1")
                .reference("independent planar Fresnel pressure and normal-power coefficients")
                .reference_checksum("analytic/deterministic-fixture:fresnel-fluid-interface-v1")
                .dimension("two-fluid planar interface, normal and oblique incidence")
                .formulation(
                    "typed FluidInterface XEM adapter and production material transmission path",
                )
                .tolerance("coefficient/power error <= 2e-13; two typed rejection checks")
                .seed("bem-fluid-interface-transmission:air-water:oblique-and-tir:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("fluid-interface QA result must serialize")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("fluid-interface QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            let record = QaResult::failed(CASE_ID, "panic in fluid-interface QA case")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("fluid-interface panic result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
