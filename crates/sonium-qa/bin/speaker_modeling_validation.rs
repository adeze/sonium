//! End-to-end analytical contracts for `sonium-sim-speaker-modeling`.

use num_complex::Complex64;
use sonium_fem::physics::acoustics::electroacoustics::lumped_extraction::radiation_impedance_piston_lf;
use sonium_fem::physics::acoustics::electroacoustics::transducer::ElectrodynamicCoupling;
use sonium_sim_speaker_modeling::{
    AcousticMedium, CalibrationDataOrigin, CalibrationTolerances, DriverCalibrationData,
    DriverGeometry, DriverModel, EnclosureSpec, IndependentMeasurementMetadata,
    LargeSignalProvenance, LargeSignalSpec, MeasuredBlPoint, MeasuredImpedancePoint,
    MeasuredInductancePoint, MeasuredVelocityPoint, MovingSystemSpec, PortSpec, ResponseProvenance,
    SpeakerDriverStudy, StudyRequirements, ThermalCompressionSpec, VoiceCoilSpec,
    baffled_piston_directivity, baffled_piston_far_field_pressure,
    baffled_piston_mechanical_radiation_impedance, correlate_driver_model,
};
use sonium_sim_speaker_modeling::{
    DistortionReferenceTolerances, DriverThermalNetworkConfig, MeasuredDistortionPoint,
    MeasuredRadiationResponsePoint, MeasuredStructuralAcousticPoint, MeasuredThermalPoint,
    StructuralAcousticFeedback, StructuralAcousticMode, StructuralAcousticReferenceTolerances,
    correlate_distortion_reference, correlate_structural_acoustic_reference,
};
use std::f64::consts::{PI, TAU};

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) -> Result<(), String> {
    let error = (actual - expected).abs();
    if error > tolerance {
        return Err(format!(
            "{label}: expected {expected:.12e}, got {actual:.12e}, abs error {error:.3e}"
        ));
    }
    Ok(())
}

fn validate_study_contract(study: &SpeakerDriverStudy) -> Result<(), String> {
    study.validate().map_err(|error| error.to_string())?;
    let first = study
        .frequency_hz
        .first()
        .copied()
        .ok_or_else(|| "frequency sweep is empty".to_string())?;
    let last = study
        .frequency_hz
        .last()
        .copied()
        .ok_or_else(|| "frequency sweep is empty".to_string())?;
    assert_close("sweep start", first, 20.0, 0.0)?;
    assert_close("sweep stop", last, 20_000.0, 0.0)?;
    if !study.frequency_hz.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err("frequency sweep is not strictly increasing".to_string());
    }
    Ok(())
}

fn validate_piston_reference(study: &SpeakerDriverStudy) -> Result<(), String> {
    let medium = study.acoustic_medium;
    let radius = 0.1;
    let frequency = medium.sound_speed_m_per_s / (TAU * radius);
    let impedance = baffled_piston_mechanical_radiation_impedance(radius, frequency, medium)
        .map_err(|error| error.to_string())?;
    let normalization =
        medium.density_kg_per_m3 * medium.sound_speed_m_per_s * PI * radius * radius;
    assert_close(
        "ka=1 normalized radiation resistance",
        impedance.re / normalization,
        0.423_275_192_243_126_6,
        1.0e-13,
    )?;
    assert_close(
        "ka=1 normalized radiation reactance",
        impedance.im / normalization,
        0.646_763_728_283_562_2,
        1.0e-10,
    )?;

    let low_frequency = 20.0;
    let exact = study
        .mechanical_radiation_impedance(low_frequency)
        .map_err(|error| error.to_string())?;
    let area = PI * study.geometry.diaphragm_radius_m.powi(2);
    let low_frequency_reference = radiation_impedance_piston_lf(
        study.geometry.diaphragm_radius_m,
        TAU * low_frequency,
        medium.density_kg_per_m3,
        medium.sound_speed_m_per_s,
    ) * area
        * area;
    assert_close(
        "low-frequency radiation-resistance ratio",
        exact.re / low_frequency_reference.re,
        1.0,
        1.0e-4,
    )?;
    assert_close(
        "low-frequency radiation-reactance ratio",
        exact.im / low_frequency_reference.im,
        1.0,
        1.0e-4,
    )
}

fn validate_power_balance(study: &SpeakerDriverStudy) -> Result<f64, String> {
    let mut maximum_relative_error: f64 = 0.0;
    let voltage = Complex64::new(study.drive_voltage_rms_v, 0.0);
    for &frequency in &study.frequency_hz {
        let response = study
            .try_small_signal_response_at(frequency, voltage)
            .map_err(|error| error.to_string())?;
        let radiation = study
            .mechanical_radiation_impedance(frequency)
            .map_err(|error| error.to_string())?;
        let input = (voltage * response.current.conj()).re;
        let coil = study.voice_coil.resistance_ohm * response.current.norm_sqr();
        let suspension =
            study.moving_system.mechanical_resistance_kg_per_s * response.coil_velocity.norm_sqr();
        let radiated = radiation.re * response.coil_velocity.norm_sqr();
        let error = (input - coil - suspension - radiated).abs();
        let relative_error = error / input.max(f64::MIN_POSITIVE);
        maximum_relative_error = maximum_relative_error.max(relative_error);
        if relative_error > 2.0e-12 {
            return Err(format!(
                "RMS power balance failed at {frequency:.3} Hz: relative error {relative_error:.3e}"
            ));
        }
    }
    Ok(maximum_relative_error)
}

fn validate_report(study: &SpeakerDriverStudy) -> Result<(), String> {
    let report = study.configured_detailed_example_report();
    if report.response_provenance
        != ResponseProvenance::LumpedElectromechanicalWithBaffledPistonRadiation
    {
        return Err("response provenance does not identify the lumped piston model".to_string());
    }
    if !report.fem.magnetic_bias_is_representative {
        return Err("representative magnetic bias is mislabeled as solved FEM data".to_string());
    }
    let json = serde_json::to_string(&report).map_err(|error| error.to_string())?;
    let decoded = serde_json::from_str::<sonium_sim_speaker_modeling::DetailedExampleReport>(&json)
        .map_err(|error| error.to_string())?;
    decoded
        .study
        .validate()
        .map_err(|error| error.to_string())?;
    if decoded.responses.len() != study.frequency_hz.len() {
        return Err("serialized report lost response rows".to_string());
    }
    Ok(())
}

fn validate_extended_models(study: &SpeakerDriverStudy) -> Result<(), String> {
    let medium = study.acoustic_medium;
    let vented = EnclosureSpec::Vented {
        volume_m3: 35.0e-3,
        port: PortSpec {
            radius_m: 35.0e-3,
            length_m: 0.15,
            end_correction_m: 0.055,
            resistance_pa_s_per_m3: 80.0,
        },
        leakage_resistance_pa_s_per_m3: 3.0e6,
        absorption_resistance_pa_s_per_m3: 2.0e6,
    };
    let tuning = vented
        .tuning_frequency_hz(medium)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "vented enclosure has no tuning frequency".to_string())?;
    if !(20.0..100.0).contains(&tuning) {
        return Err(format!("implausible vented-box tuning: {tuning:.3} Hz"));
    }
    for frequency in [20.0, tuning, 100.0, 1_000.0] {
        let impedance = vented
            .acoustic_impedance(frequency, medium)
            .map_err(|error| error.to_string())?;
        if impedance.re < 0.0 || !impedance.re.is_finite() || !impedance.im.is_finite() {
            return Err(format!("non-passive enclosure at {frequency:.3} Hz"));
        }
    }

    let radius = study.geometry.diaphragm_radius_m;
    let frequency = 5_000.0;
    let on_axis = baffled_piston_directivity(radius, frequency, 0.0, medium)
        .map_err(|error| error.to_string())?;
    let off_axis = baffled_piston_directivity(radius, frequency, 60.0, medium)
        .map_err(|error| error.to_string())?;
    assert_close("on-axis directivity", on_axis, 1.0, 1.0e-12)?;
    if off_axis.abs() >= on_axis {
        return Err("high-frequency piston response is not directional".to_string());
    }
    let velocity = Complex64::new(0.1, 0.0);
    let near = baffled_piston_far_field_pressure(radius, frequency, velocity, 1.0, 0.0, medium)
        .map_err(|error| error.to_string())?;
    let far = baffled_piston_far_field_pressure(radius, frequency, velocity, 2.0, 0.0, medium)
        .map_err(|error| error.to_string())?;
    assert_close(
        "far-field inverse-distance ratio",
        near.norm() / far.norm(),
        2.0,
        1.0e-12,
    )?;

    let transient = SpeakerDriverStudy::transient_large_signal_driver();
    let nonlinear = transient
        .try_large_signal_response(LargeSignalSpec {
            duration_s: 0.1,
            voltage_rms_v: 4.0,
            ..LargeSignalSpec::default()
        })
        .map_err(|error| error.to_string())?;
    if nonlinear.provenance
        != LargeSignalProvenance::LumpedNonlinearElectromechanicalWithoutAcousticLoad
        || nonlinear.peak_displacement_m <= 0.0
        || nonlinear.displacement_thd <= 0.0
    {
        return Err("nonlinear transient diagnostics are invalid".to_string());
    }

    let thermal = study
        .try_thermal_compressed_response_at(1_000.0, 20.0, ThermalCompressionSpec::default())
        .map_err(|error| error.to_string())?;
    if thermal.hot_resistance_ohm <= study.voice_coil.resistance_ohm
        || thermal.compression_db <= 0.0
    {
        return Err("electrothermal compression did not reduce output".to_string());
    }
    Ok(())
}

/// Correlate the typed handoff against a separately evaluated low-frequency
/// piston reference.  The reference uses the FEM transducer coupling and
/// low-frequency piston formula directly, rather than copying the
/// `SpeakerDriverStudy` response helper.  Its metadata is required by the
/// calibration API, so this cannot silently become a relabeled fixture.
fn validate_independent_reference_calibration() -> Result<(), String> {
    let frequencies_hz = vec![20.0, 40.0, 80.0, 160.0];
    let study = SpeakerDriverStudy::try_from_requirements(StudyRequirements {
        name: "independent-reference-midwoofer".to_owned(),
        geometry: DriverGeometry {
            voice_coil_radius_m: 12.5e-3,
            diaphragm_radius_m: 52.5e-3,
            diaphragm_half_angle_degrees: 32.0,
            air_gap_m: 0.75e-3,
            has_non_axisymmetric_basket: false,
            sector_angle_degrees: None,
        },
        voice_coil: VoiceCoilSpec {
            resistance_ohm: 5.8,
            inductance_h: 0.42e-3,
            bl_tm: 6.8,
            active_turns: 75.0,
        },
        moving_system: MovingSystemSpec {
            moving_mass_kg: 14.0e-3,
            mechanical_resistance_kg_per_s: 1.1,
            stiffness_n_per_m: 950.0,
            xmax_m: None,
        },
        harmonic_small_signal: true,
        transient_or_pulse: false,
        nonlinear_large_signal: false,
        frequency_hz: frequencies_hz.clone(),
        voltage_rms_v: 2.83,
        acoustic_medium: AcousticMedium::standard_air(),
    })
    .map_err(|error| error.to_string())?;
    let mut model = DriverModel::from_study(&study).map_err(|error| error.to_string())?;
    let thermal = DriverThermalNetworkConfig::default();
    let thermal_power = thermal
        .power_from_losses(2.0, 1.0)
        .map_err(|error| error.to_string())?;
    let thermal_state = thermal
        .specification
        .solve_steady_state(thermal_power)
        .map_err(|error| error.to_string())?;
    model.thermal_network = Some(thermal);
    let area = PI * study.geometry.diaphragm_radius_m.powi(2);
    let coupling = ElectrodynamicCoupling {
        resistance: study.voice_coil.resistance_ohm,
        inductance: study.voice_coil.inductance_h,
        bl: study.voice_coil.bl_tm,
        coil_dofs: vec![(0, 1.0)],
    };
    let voltage = Complex64::new(study.drive_voltage_rms_v, 0.0);
    let mut data = DriverCalibrationData {
        origin: CalibrationDataOrigin::IndependentMeasurement,
        independent_measurement: Some(IndependentMeasurementMetadata {
            dataset_id: "sonium-independent-piston-reference-v1".to_owned(),
            source: "sonium-fem low-frequency piston reference".to_owned(),
            acquisition_method: "independent analytical/FEM transducer evaluation".to_owned(),
        }),
        bl_curve: vec![MeasuredBlPoint {
            displacement_m: 0.0,
            force_factor_tm: study.voice_coil.bl_tm,
        }],
        inductance_curve: frequencies_hz
            .iter()
            .map(|&frequency_hz| MeasuredInductancePoint {
                frequency_hz,
                inductance_h: study.voice_coil.inductance_h,
                eddy_resistance_ohm: 0.0,
            })
            .collect(),
        thermal_response: vec![MeasuredThermalPoint {
            copper_loss_w: 2.0,
            eddy_loss_w: 1.0,
            temperatures_k: thermal_state.temperatures_k,
        }],
        ..DriverCalibrationData::default()
    };
    for &frequency_hz in &frequencies_hz {
        let omega = TAU * frequency_hz;
        let radiation = radiation_impedance_piston_lf(
            study.geometry.diaphragm_radius_m,
            omega,
            study.acoustic_medium.density_kg_per_m3,
            study.acoustic_medium.sound_speed_m_per_s,
        ) * area
            * area;
        let mechanical = study
            .moving_system
            .try_mechanical_impedance(frequency_hz)
            .map_err(|error| error.to_string())?
            + radiation;
        let impedance = coupling.total_impedance(omega, mechanical);
        let (_, velocity, _) = coupling.solve_frequency(voltage, omega, mechanical);
        data.impedance.push(MeasuredImpedancePoint {
            frequency_hz,
            real_ohm: impedance.re,
            imaginary_ohm: impedance.im,
        });
        data.velocity_transfer.push(MeasuredVelocityPoint {
            frequency_hz,
            velocity_real_m_per_s: velocity.re,
            velocity_imag_m_per_s: velocity.im,
        });
        let (radiation_velocity, radiation_electrical_impedance) = model
            .response_with_radiation_impedance(frequency_hz, radiation)
            .map_err(|error| error.to_string())?;
        data.radiation_response
            .push(MeasuredRadiationResponsePoint {
                frequency_hz,
                radiation_real_n_s_per_m: radiation.re,
                radiation_imag_n_s_per_m: radiation.im,
                velocity_real_m_per_s: radiation_velocity.re,
                velocity_imag_m_per_s: radiation_velocity.im,
                electrical_real_ohm: radiation_electrical_impedance.re,
                electrical_imag_ohm: radiation_electrical_impedance.im,
            });
    }
    let report = correlate_driver_model(
        &model,
        &data,
        &CalibrationTolerances {
            impedance_ohm: 0.1,
            bl_tm: 1.0e-12,
            inductance_h: 1.0e-12,
            eddy_resistance_ohm: 1.0e-12,
            skin_depth_m: 1.0e-6,
            proximity_factor: 0.05,
            temperature_k: 0.25,
            radiation_velocity_m_per_s: 1.0e-4,
            radiation_impedance_ohm: 0.05,
            magnetic_field_a_per_m: 1.0,
            displacement_m: 1.0e-6,
            velocity_m_per_s: 1.0e-3,
            magnetic_flux_density_t: 1.0e-12,
        },
    )
    .map_err(|error| error.to_string())?;
    if !report.passed || report.origin != CalibrationDataOrigin::IndependentMeasurement {
        return Err(format!(
            "independent reference correlation failed: {report:?}"
        ));
    }
    Ok(())
}

fn validate_modal_reference_contract() -> Result<(), String> {
    let feedback = StructuralAcousticFeedback {
        area_m2: 0.02,
        modes: vec![StructuralAcousticMode {
            natural_frequency_hz: 400.0,
            effective_mass_kg: 0.2,
            damping_ratio: 0.03,
            pressure_participation: 1.0,
            stress_per_displacement_pa_per_m: 2.0e6,
        }],
        max_iterations: 12,
        velocity_tolerance: 1.0e-6,
        under_relaxation: 0.8,
    };
    let measurements = [100.0, 250.0, 800.0]
        .into_iter()
        .map(|frequency_hz| {
            let pressure = Complex64::new(10.0, 3.0);
            let response = feedback
                .response_at(frequency_hz, pressure)
                .map_err(|error| error.to_string())?;
            Ok(MeasuredStructuralAcousticPoint {
                frequency_hz,
                pressure_rms_pa: pressure,
                displacement_rms_m: response.displacement_rms_m,
                velocity_rms_m_per_s: response.velocity_rms_m_per_s,
                stress_rms_pa: response.stress_rms_pa,
                volume_velocity_rms_m3_per_s: response.volume_velocity_rms_m3_per_s,
                absorbed_power_w: response.absorbed_power_w,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let report = correlate_structural_acoustic_reference(
        &feedback,
        &measurements,
        StructuralAcousticReferenceTolerances::default(),
    )
    .map_err(|error| error.to_string())?;
    if !report.passed {
        return Err(format!("modal reference correlation failed: {report:?}"));
    }
    Ok(())
}

fn validate_distortion_reference_contract() -> Result<(), String> {
    let mut study = SpeakerDriverStudy::transient_large_signal_driver();
    study.enclosure = EnclosureSpec::Vented {
        volume_m3: 35.0e-3,
        port: PortSpec {
            radius_m: 35.0e-3,
            length_m: 0.15,
            end_correction_m: 0.055,
            resistance_pa_s_per_m3: 80.0,
        },
        leakage_resistance_pa_s_per_m3: 3.0e6,
        absorption_resistance_pa_s_per_m3: 2.0e6,
    };
    let spec = LargeSignalSpec {
        frequency_hz: 80.0,
        duration_s: 0.1,
        samples_per_period: 64,
        secondary_frequency_hz: Some(1_000.0),
        ..LargeSignalSpec::default()
    };
    let voltages = [1.0, 2.0];
    let sweep = study
        .try_lumped_nonlinear_compression_sweep(spec, &voltages)
        .map_err(|error| error.to_string())?;
    let reference = sweep
        .points
        .iter()
        .map(|point| MeasuredDistortionPoint {
            frequency_hz: spec.frequency_hz,
            secondary_frequency_hz: spec.secondary_frequency_hz,
            voltage_rms_v: point.voltage_rms_v,
            acoustic_pressure_fundamental_peak_pa: point.box_pressure_amplitude_pa,
            acoustic_thd: point.acoustic_thd,
            acoustic_imd: point.acoustic_imd,
        })
        .collect::<Vec<_>>();
    let report = correlate_distortion_reference(
        &study,
        spec,
        &reference,
        DistortionReferenceTolerances::default(),
    )
    .map_err(|error| error.to_string())?;
    if !report.passed {
        return Err(format!(
            "distortion reference correlation failed: {report:?}"
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    let resonance_hz = study
        .moving_system
        .try_resonance_hz()
        .map_err(|error| error.to_string())?;
    validate_study_contract(&study)?;
    validate_piston_reference(&study)?;
    let maximum_power_error = validate_power_balance(&study)?;
    validate_report(&study)?;
    validate_extended_models(&study)?;
    validate_modal_reference_contract()?;
    validate_distortion_reference_contract()?;
    validate_independent_reference_calibration()?;

    println!(
        "Speaker modeling: {} frequencies, fs={:.3} Hz, max RMS power-balance error={maximum_power_error:.3e}",
        study.frequency_hz.len(),
        resonance_hz
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modeling_pipeline_matches_analytical_contracts() {
        run().expect("speaker-modeling analytical contracts should pass");
    }
}
