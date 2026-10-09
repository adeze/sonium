//! Independent physical contract for the typed production fluid interface.
//!
//! This is deliberately a planar-interface test: sonium-bem does not expose
//! a penetrable-sphere BEM operator yet.  The XEM adapter and the interface
//! evaluator are nevertheless production APIs, so compare their result with
//! independently evaluated Fresnel pressure and normal-power coefficients.

use num_complex::Complex64;
use sonium_bem::{AmbientMedium, FluidInterface, MaterialError};
use sonium_xem::AcousticMedium;

const CASE_ID: &str = "bem.fluid-interface-transmission.v1";
const MAX_COEFFICIENT_RELATIVE_ERROR: f64 = 2.0e-13;
const MAX_POWER_BALANCE_RESIDUAL: f64 = 2.0e-13;
const INCIDENCE_ANGLE_RADIANS: f64 = 0.1;

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

#[test]
fn typed_fluid_interface_matches_independent_fresnel_contract_v1() {
    let incident_xem = AcousticMedium::try_new(1_000.0, 343.0, 1.21).expect("incident medium");
    let transmitted_xem =
        AcousticMedium::try_new(1_000.0, 1_480.0, 1_000.0).expect("transmitted medium");
    let incident = AmbientMedium::new(1.21, 343.0);
    let transmitted = AmbientMedium::new(1_000.0, 1_480.0);

    let result = FluidInterface::try_from_xem(incident_xem, transmitted_xem)
        .expect("typed interface admission")
        .try_plane_wave_transmission(INCIDENCE_ANGLE_RADIANS)
        .expect("propagating oblique transmission");
    let (
        reference_reflection,
        reference_transmission,
        reference_reflected_power,
        reference_transmitted_power,
    ) = fresnel_reference(incident, transmitted, INCIDENCE_ANGLE_RADIANS);

    let reflection_error = (result.pressure_reflection - reference_reflection).norm();
    let transmission_error = (result.pressure_transmission - reference_transmission).norm();
    let reflected_power_error = (result.reflected_power - reference_reflected_power).abs();
    let transmitted_power_error = (result.transmitted_power - reference_transmitted_power).abs();
    let power_balance_residual = result.power_balance_residual().abs();

    assert!(!result.total_internal_reflection);
    assert!(
        reflection_error <= MAX_COEFFICIENT_RELATIVE_ERROR,
        "{CASE_ID}: reflection coefficient error {reflection_error:.3e} exceeds {MAX_COEFFICIENT_RELATIVE_ERROR:.3e}"
    );
    assert!(
        transmission_error <= MAX_COEFFICIENT_RELATIVE_ERROR,
        "{CASE_ID}: transmission coefficient error {transmission_error:.3e} exceeds {MAX_COEFFICIENT_RELATIVE_ERROR:.3e}"
    );
    assert!(
        reflected_power_error <= MAX_COEFFICIENT_RELATIVE_ERROR,
        "{CASE_ID}: reflected-power error {reflected_power_error:.3e} exceeds {MAX_COEFFICIENT_RELATIVE_ERROR:.3e}"
    );
    assert!(
        transmitted_power_error <= MAX_COEFFICIENT_RELATIVE_ERROR,
        "{CASE_ID}: transmitted-power error {transmitted_power_error:.3e} exceeds {MAX_COEFFICIENT_RELATIVE_ERROR:.3e}"
    );
    assert!(
        power_balance_residual <= MAX_POWER_BALANCE_RESIDUAL,
        "{CASE_ID}: power-balance residual {power_balance_residual:.3e} exceeds {MAX_POWER_BALANCE_RESIDUAL:.3e}"
    );

    let mismatched_frequency = AcousticMedium::try_new(1_001.0, 1_480.0, 1_000.0)
        .expect("mismatched medium remains individually valid");
    assert!(matches!(
        FluidInterface::try_from_xem(incident_xem, mismatched_frequency),
        Err(MaterialError::InvalidConvention(_))
    ));

    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"{CASE_ID}\",\"status\":\"passed\",\"path\":\"typed-xem-planar-fluid-interface\",\"incidence_angle_radians\":{INCIDENCE_ANGLE_RADIANS:.6},\"reflection_coefficient_error\":{reflection_error:.6e},\"transmission_coefficient_error\":{transmission_error:.6e},\"power_balance_residual\":{power_balance_residual:.6e}}}"
    );
}
