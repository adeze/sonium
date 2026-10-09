//! Analytic compact rotating-dipole directivity and power reference QA.

use std::f64::consts::PI;
use std::time::Instant;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_qa::QaResult;
use sonium_sim_propellers::RotatingDipoleReference;

const CASE_ID: &str = "propeller.rotating-dipole-reference.v1";

fn run() -> Result<(f64, f64, f64, f64, f64), String> {
    let source = PropellerSource::drone_5inch(18_000.0);
    let source_equatorial = source.harmonic_amplitudes_at_distance(1, 1.0)[0];
    let reference = RotatingDipoleReference::try_from_equatorial_amplitude(
        1,
        source.harmonic_frequency(1),
        1.0,
        source_equatorial,
        source.density,
        source.speed_of_sound,
    )
    .map_err(|error| error.to_string())?;
    let equatorial_error = (reference.equatorial_peak_pressure_pa - source_equatorial).abs()
        / source_equatorial.max(f64::MIN_POSITIVE);
    let axial_pressure = reference
        .peak_pressure_pa(0.0)
        .map_err(|error| error.to_string())?;
    let mirror_error = (reference
        .peak_pressure_pa(PI / 6.0)
        .map_err(|error| error.to_string())?
        - reference
            .peak_pressure_pa(5.0 * PI / 6.0)
            .map_err(|error| error.to_string())?)
    .abs();

    let n_theta = 256;
    let n_phi = 64;
    let d_theta = PI / n_theta as f64;
    let d_phi = 2.0 * PI / n_phi as f64;
    let mut sphere_integral = 0.0;
    for i in 0..n_theta {
        let theta = (i as f64 + 0.5) * d_theta;
        let pressure = reference
            .peak_pressure_pa(theta)
            .map_err(|error| error.to_string())?;
        sphere_integral += n_phi as f64 * pressure.powi(2) * theta.sin() * d_theta * d_phi;
    }
    let numerical_power_factor =
        sphere_integral / (4.0 * PI * reference.equatorial_peak_pressure_pa.powi(2));
    let analytic_power_factor = RotatingDipoleReference::DIRECTIVITY_POWER_FACTOR;
    let power_factor_error = (numerical_power_factor - analytic_power_factor).abs();
    let power_ratio_error = (reference.radiated_sound_power_w()
        / reference.isotropic_reference_power_w()
        - analytic_power_factor)
        .abs();
    let observer = reference
        .complex_pressure_pa(2.0, PI / 2.0, 0.0)
        .map_err(|error| error.to_string())?;
    let quarter_turn_observer = reference
        .complex_pressure_pa(2.0, PI / 2.0, PI / 2.0)
        .map_err(|error| error.to_string())?;
    let distance_scaling_error = (observer.magnitude_pa() / source_equatorial * 2.0 - 1.0).abs();
    let phase_quadrature_error = (observer.real_pa * quarter_turn_observer.real_pa
        + observer.imaginary_pa * quarter_turn_observer.imaginary_pa)
        .abs()
        / (observer.magnitude_pa().max(f64::MIN_POSITIVE)
            * quarter_turn_observer.magnitude_pa().max(f64::MIN_POSITIVE));

    if equatorial_error > 1.0e-12
        || axial_pressure.abs() > 1.0e-12
        || mirror_error > 1.0e-12
        || power_factor_error > 2.0e-5
        || power_ratio_error > 1.0e-12
        || distance_scaling_error > 1.0e-12
        || phase_quadrature_error > 1.0e-12
    {
        return Err(format!(
            "rotating-dipole reference failed: equatorial_error={equatorial_error:e}, \
             axial_pressure={axial_pressure:e}, mirror_error={mirror_error:e}, \
             power_factor_error={power_factor_error:e}, power_ratio_error={power_ratio_error:e}, \
             distance_scaling_error={distance_scaling_error:e}, \
             phase_quadrature_error={phase_quadrature_error:e}"
        ));
    }

    Ok((
        equatorial_error,
        power_factor_error,
        numerical_power_factor,
        distance_scaling_error,
        phase_quadrature_error,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((
            equatorial_error,
            power_factor_error,
            numerical_power_factor,
            distance_scaling_error,
            phase_quadrature_error,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("equatorial_source_relative_error", equatorial_error)
                .metric("sphere_power_factor", numerical_power_factor)
                .metric("sphere_power_factor_absolute_error", power_factor_error)
                .metric("analytic_dipole_power_factor", 2.0 / 3.0)
                .metric("observer_distance_scaling_error", distance_scaling_error)
                .metric("observer_phase_quadrature_error", phase_quadrature_error)
                .input_checksum("propeller-5inch:18000rpm:harmonic=1:rotating-dipole-v1")
                .reference(
                    "analytic compact rotating-dipole sin(theta) envelope and sphere power integral",
                )
                .reference_checksum("analytic/invariant:rotating-dipole-sin-theta-v1")
                .dimension(
                    "far-field compact observer phasor; 256 polar x 64 azimuth magnitude quadrature plus 2m phase pair",
                )
                .formulation(
                    "exp(i omega t), outgoing exp(-ikr), rotating phase exp(-iphi), p_peak(theta)=p_eq sin(theta); sphere-average sin^2(theta)=2/3",
                )
                .tolerance(
                    "source/directivity/phase checks <= 1e-12; quadrature power-factor error <= 2e-5",
                )
                .maximum("equatorial_source_relative_error", 1.0e-12)
                .maximum("sphere_power_factor_absolute_error", 2.0e-5)
                .maximum("observer_distance_scaling_error", 1.0e-12)
                .maximum("observer_phase_quadrature_error", 1.0e-12)
                .seed("propeller-rotating-dipole-v1:source=drone-5inch:rpm=18000:harmonic=1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PROPELLER ROTATING-DIPOLE QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
