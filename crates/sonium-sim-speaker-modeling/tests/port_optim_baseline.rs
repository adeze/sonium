//! Step-0 baseline checks: frozen archive gates and deterministic mesh.
//!
//! The default tests guard `fixtures/port_optim_baseline_v1.json` against
//! silent drift: the archive must match the frozen [`PortOptimReferenceCase`]
//! and pass [`check_baseline_gates`]. The ignored
//! `regenerate_port_optim_baseline_v1` test reproduces the archive from
//! scratch (Tier 1 linear reference plus Tier 2 coupled sweep at production
//! fidelity) and must run in release mode:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_baseline -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sonium_sim_speaker_modeling::port_flow::{
    PortOptimBaseline, PortOptimReferenceCase, build_slot_port_mesh, check_baseline_gates,
};

fn archive_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_baseline_v1.json")
}

fn load_archive() -> Result<PortOptimBaseline, Box<dyn std::error::Error + Send + Sync>> {
    let text = std::fs::read_to_string(archive_path())?;
    Ok(serde_json::from_str(&text)?)
}

#[test]
fn archived_baseline_matches_frozen_fixture_and_passes_gates()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let baseline = load_archive()?;
    assert_eq!(
        baseline.case,
        PortOptimReferenceCase::default(),
        "archive fixture drifted from the frozen Step-0 reference"
    );
    let report = check_baseline_gates(&baseline)?;
    println!(
        "port-optim baseline v1 gates: f_target={:.6} Hz, tier1_points={}, tier2_levels={}, \
         lumped_limit={:.6}%, linear_thd={:.6}%, energy_physical={:.6}%, \
         tier1_residual={:.3e}, tier1_flux={:.3e}, max_compression={:.4} dB",
        baseline.f_target_hz,
        baseline.tier1.len(),
        baseline.tier2.len(),
        100.0 * report.lumped_limit_relative_error,
        100.0 * report.linear_thd_fraction,
        100.0 * report.max_energy_physical_closure,
        report.max_tier1_relative_residual,
        report.max_tier1_flux_relative_difference,
        report.max_compression_db,
    );
    Ok(())
}

#[test]
fn slot_mesh_is_deterministic() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let case = PortOptimReferenceCase::default();
    let geometry = case.slot_geometry();
    let first = build_slot_port_mesh(&geometry)?;
    let second = build_slot_port_mesh(&geometry)?;
    assert_eq!(first.num_nodes(), second.num_nodes());
    assert_eq!(
        format!("{first:?}"),
        format!("{second:?}"),
        "slot mesh build is not bit-deterministic"
    );
    println!(
        "slot mesh: nodes={}, elements={}, boundaries={}",
        first.num_nodes(),
        first.elements.len(),
        first.boundaries.len()
    );
    Ok(())
}

/// Regenerate the frozen Step-0 archive at production fidelity.
///
/// Runs the Tier 1 straight-tube linear reference at `[0.75, 1.0, 1.25] x
/// f_target` and the Tier 2 coupled slot sweep at all three drive levels
/// (252 samples/cycle, 40 cycles each), then writes
/// `fixtures/port_optim_baseline_v1.json`. Release-only: debug builds are
/// rejected rather than run for hours.
#[test]
#[ignore = "production-fidelity baseline campaign; run explicitly in release mode"]
fn regenerate_port_optim_baseline_v1() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if cfg!(debug_assertions) {
        return Err("run the baseline campaign in release mode \
            (cargo test -p sonium-sim-speaker-modeling --release \
            --test port_optim_baseline -- --ignored --nocapture)"
            .into());
    }

    use rayon::prelude::*;
    use sonium_fem::basis::PolynomialDegree;
    use sonium_fem::spaces::{H1Space, VectorH1Space};
    use sonium_sim_speaker_modeling::free_turbulence_power_trend;
    use sonium_sim_speaker_modeling::port_flow::analysis::{dft_amplitude, thd_fraction};
    use sonium_sim_speaker_modeling::port_flow::baseline::{
        LumpedLimitBaseline, PORT_OPTIM_BASELINE_PROVENANCE, PORT_OPTIM_BASELINE_SCHEMA_VERSION,
        Tier1BaselinePoint, Tier2BaselineLevel,
    };
    use sonium_sim_speaker_modeling::port_flow::coupled::run_coupled_port;
    use sonium_sim_speaker_modeling::port_flow::{
        build_round_port_mesh, complex_pair, compression_db, duct_bulk_mach_number,
        solve_round_port_linear_reference,
    };

    /// Relative scale below which negative dissipation counts as roundoff.
    /// Matches the `qa-speaker-port-nonlinear` 2D leg.
    const DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

    let case = PortOptimReferenceCase::default();
    case.validate()?;
    let f_target_hz = case.f_target_hz()?;
    let drive_hz = case.tier2_drive_frequency_hz()?;
    println!(
        "baseline campaign: f_target={f_target_hz:.6} Hz, tier2_drive={drive_hz:.6} Hz, \
         dt={:.6e} s, steps={}",
        case.dt_s()?,
        case.num_steps()
    );

    // Tier 1: straight-tube linear reference at three tuning-relative points.
    let round_mesh = build_round_port_mesh(&case.round_geometry())?;
    println!(
        "tier 1 mesh: nodes={}, elements={}, boundaries={}",
        round_mesh.num_nodes(),
        round_mesh.elements.len(),
        round_mesh.boundaries.len()
    );
    let reference_config = case.linear_reference_config();
    let tier1_frequencies = case.tier1_frequencies_hz()?;
    let tier1: Vec<Tier1BaselinePoint> = tier1_frequencies
        .par_iter()
        .map(|frequency| {
            let transfer =
                solve_round_port_linear_reference(&round_mesh, &reference_config, *frequency)?;
            println!(
                "tier 1 f={frequency:.6} Hz: Z=({:+.6e},{:+.6e}) Pa s/m3, \
                 H=({:+.6e},{:+.6e}), Q_rms=({:+.6e},{:+.6e}) m3/s, \
                 residual={:.3e}, flux={:.3e}, dofs={}",
                transfer.port_impedance_pa_s_per_m3.re,
                transfer.port_impedance_pa_s_per_m3.im,
                transfer.box_to_port_flow_ratio.re,
                transfer.box_to_port_flow_ratio.im,
                transfer.port_volume_velocity_rms_m3_per_s.re,
                transfer.port_volume_velocity_rms_m3_per_s.im,
                transfer.relative_residual,
                transfer.strong_weak_exterior_flux_relative_difference,
                transfer.scalar_dofs,
            );
            Ok(Tier1BaselinePoint {
                frequency_hz: *frequency,
                port_impedance_pa_s_per_m3: complex_pair(transfer.port_impedance_pa_s_per_m3),
                box_to_port_flow_ratio: complex_pair(transfer.box_to_port_flow_ratio),
                port_volume_velocity_rms_m3_per_s: complex_pair(
                    transfer.port_volume_velocity_rms_m3_per_s,
                ),
                relative_residual: transfer.relative_residual,
                flux_relative_difference: transfer.strong_weak_exterior_flux_relative_difference,
                scalar_dofs: transfer.scalar_dofs,
            })
        })
        .collect::<Result<Vec<_>, sonium_sim_speaker_modeling::port_flow::PortFlowError>>()?;

    // Tier 2: coupled slot sweep, one production run per drive level.
    let slot_mesh = build_slot_port_mesh(&case.slot_geometry())?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let tail_length = case.tier2_tail_cycles * case.tier2_samples_per_cycle;
    let medium = case.medium();
    let duct_area = case.tier2_slot_area_m2();
    let expected_steps = case.num_steps();

    struct RawLevel {
        voltage_peak_v: f64,
        flow_amplitude_m3s: f64,
        flow_rms_m3s: f64,
        pressure_amplitude_pa: f64,
        peak_excursion_m: f64,
        thd: f64,
        mach: f64,
        trend: f64,
        physical_closure: f64,
        algebraic_closure: f64,
        interface_residual: f64,
        energy_scale_j: f64,
    }

    let raw_levels: Vec<RawLevel> = case
        .tier2_drives_v_peak
        .par_iter()
        .map(
            |voltage_peak_v| -> Result<RawLevel, Box<dyn std::error::Error + Send + Sync>> {
                let config = case.coupled_config(*voltage_peak_v)?;
                let dt = config.bdf.dt;
                let result =
                    run_coupled_port(&slot_mesh, &velocity_space, &pressure_space, &config)?;
                if result.steps_completed != expected_steps {
                    return Err(format!(
                        "Tier 2 at {voltage_peak_v} V peak completed {} of {expected_steps} steps",
                        result.steps_completed
                    )
                    .into());
                }
                let tail_start = result.port_flow_m3s.len() - tail_length;
                let flow_tail = &result.port_flow_m3s[tail_start..];
                let energy = result.energy_summary(tail_start..result.energy_steps.len());
                if !energy.is_finite() {
                    return Err(format!(
                        "Tier 2 energy summary at {voltage_peak_v} V peak is non-finite: {energy:?}"
                    )
                    .into());
                }
                let dissipation_tolerance =
                    DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE * energy.energy_scale_j;
                for (offset, step) in result.energy_steps[tail_start..].iter().enumerate() {
                    if !step.is_finite() {
                        return Err(format!(
                            "Tier 2 energy step {} at {voltage_peak_v} V peak is non-finite",
                            tail_start + offset
                        )
                        .into());
                    }
                    if !step.physical_dissipations_nonnegative(dissipation_tolerance) {
                        return Err(format!(
                        "Tier 2 energy step {} at {voltage_peak_v} V peak has negative dissipation",
                        tail_start + offset
                    )
                    .into());
                    }
                }
                let flow_amplitude = dft_amplitude(flow_tail, dt, drive_hz);
                let flow_rms =
                    (flow_tail.iter().map(|q| q * q).sum::<f64>() / flow_tail.len() as f64).sqrt();
                let pressure_amplitude =
                    dft_amplitude(&result.box_pressure_pa[tail_start..], dt, drive_hz);
                let peak_excursion = result.displacement_m[tail_start..]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f64, f64::max);
                let thd = thd_fraction(flow_tail, dt, drive_hz, 5);
                let duct_velocity = (flow_amplitude / duct_area).abs();
                let mach =
                    duct_bulk_mach_number(flow_amplitude, duct_area, medium.sound_speed_m_per_s);
                let trend = free_turbulence_power_trend(duct_velocity, case.tier2_duct_length_m)?;
                println!(
                    "tier 2 V={voltage_peak_v:.3} V peak: Q={flow_amplitude:.6e} m3/s, \
                 Q_rms={flow_rms:.6e} m3/s, p_box={pressure_amplitude:.6e} Pa, \
                 THD={:.6}%, Mach={mach:.6e}, trend={trend:.6e}",
                    100.0 * thd,
                );
                Ok(RawLevel {
                    voltage_peak_v: *voltage_peak_v,
                    flow_amplitude_m3s: flow_amplitude,
                    flow_rms_m3s: flow_rms,
                    pressure_amplitude_pa: pressure_amplitude,
                    peak_excursion_m: peak_excursion,
                    thd,
                    mach,
                    trend,
                    physical_closure: energy.physical_relative_closure,
                    algebraic_closure: energy.algebraic_relative_closure,
                    interface_residual: energy.interface_relative_residual,
                    energy_scale_j: energy.energy_scale_j,
                })
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let reference = raw_levels
        .first()
        .ok_or("Tier 2 produced no drive levels")?;
    let tier2: Vec<Tier2BaselineLevel> = raw_levels
        .iter()
        .map(|level| Tier2BaselineLevel {
            voltage_peak_v: level.voltage_peak_v,
            steps_completed: expected_steps,
            port_flow_fundamental_amplitude_m3s: level.flow_amplitude_m3s,
            port_flow_rms_m3s: level.flow_rms_m3s,
            box_pressure_amplitude_pa: level.pressure_amplitude_pa,
            peak_excursion_m: level.peak_excursion_m,
            thd_fraction: level.thd,
            compression_db: compression_db(
                level.voltage_peak_v,
                level.flow_amplitude_m3s,
                reference.voltage_peak_v,
                reference.flow_amplitude_m3s,
            ),
            duct_bulk_mach: level.mach,
            turbulence_power_trend: level.trend,
            energy_physical_relative_closure: level.physical_closure,
            energy_algebraic_relative_closure: level.algebraic_closure,
            energy_interface_relative_residual: level.interface_residual,
            energy_scale_j: level.energy_scale_j,
        })
        .collect();

    let lumped_reference = case.lumped_port_flow_amplitude(reference.voltage_peak_v)?;
    let lumped_limit = LumpedLimitBaseline {
        relative_error: (reference.flow_amplitude_m3s - lumped_reference).abs() / lumped_reference,
        coupled_flow_m3s: reference.flow_amplitude_m3s,
        reference_flow_m3s: lumped_reference,
    };
    let max_compression_db = tier2
        .iter()
        .map(|level| level.compression_db)
        .fold(f64::NEG_INFINITY, f64::max);

    let baseline = PortOptimBaseline {
        schema_version: PORT_OPTIM_BASELINE_SCHEMA_VERSION,
        provenance: PORT_OPTIM_BASELINE_PROVENANCE.to_string(),
        case: PortOptimReferenceCase::default(),
        f_target_hz,
        tier1,
        tier2,
        lumped_limit,
        max_compression_db,
    };
    let report = check_baseline_gates(&baseline)?;
    println!(
        "baseline gates pass: lumped_limit={:.6}%, linear_thd={:.6}%, \
         energy_physical={:.6}%, tier1_residual={:.3e}, tier1_flux={:.3e}, \
         max_compression={:.4} dB",
        100.0 * report.lumped_limit_relative_error,
        100.0 * report.linear_thd_fraction,
        100.0 * report.max_energy_physical_closure,
        report.max_tier1_relative_residual,
        report.max_tier1_flux_relative_difference,
        report.max_compression_db,
    );

    let path = archive_path();
    std::fs::write(&path, serde_json::to_string_pretty(&baseline)?)?;
    println!("wrote {}", path.display());
    Ok(())
}
