#![cfg(feature = "sim-wolfram")]

use num_complex::Complex64;
use serde::Deserialize;
use serde_json::{Value, json};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_qa::subprocess::WolframCommandExt;
use sonium_sim_pcfan::{
    PcFanBroadbandConfig, PcFanConfig, solve_pc_fan_acoustics, solve_pc_fan_mean_flow,
};
use sonium_sim_propellers::{EvaluationMode, PropellerDesign};
use sonium_sim_room::fast_room::{FastRoomConfig, run_fast_hybrid_with_config};
use sonium_sim_speaker::spline::CubicSpline;
use sonium_sim_speaker_modeling::{
    AcousticMedium, baffled_piston_directivity, baffled_piston_far_field_pressure,
    baffled_piston_mechanical_radiation_impedance,
};
use sonium_sim_structural_mechanics::{Beam2DStudy, run_beam2d_static};
use sonium_xem::{
    Directivity, DirectivityGrid, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source,
    SurfaceConfig,
};
use std::{collections::BTreeMap, env, path::PathBuf, process::Command};

const EXPECTED_CRATES: &[&str] = &[
    "sonium-sim-pcfan",
    "sonium-sim-propellers",
    "sonium-sim-room",
    "sonium-sim-speaker",
    "sonium-sim-speaker-modeling",
    "sonium-sim-structural-mechanics",
];

#[derive(Debug, Deserialize)]
struct AccuracyReport {
    schema_version: u32,
    passed: bool,
    crates: BTreeMap<String, CrateReport>,
}

#[derive(Debug, Deserialize)]
struct CrateReport {
    passed: bool,
    check_count: usize,
    checks: BTreeMap<String, AccuracyCheck>,
}

#[derive(Debug, Deserialize)]
struct AccuracyCheck {
    passed: bool,
    absolute_error: f64,
    relative_error: f64,
    tolerance: f64,
}

fn complex_json(value: Complex64) -> Value {
    json!([value.re, value.im])
}

fn room_case() -> Value {
    // Above the forced 20 Hz blend transition, with perfectly absorbing
    // boundaries, the hybrid solver reduces exactly to the direct free-field
    // Green function exp(+ikr)/(4 pi r) for its exp(-i omega t) convention.
    let source_position = Point3D::new(1.0, 1.2, 1.1);
    let listeners = vec![
        Point3D::new(2.1, 1.5, 1.15),
        Point3D::new(3.4, 2.7, 1.2),
        Point3D::new(4.2, 3.1, 2.2),
    ];
    let source = Source::new(
        source_position,
        Directivity::Grid(DirectivityGrid::omnidirectional()),
        1.0,
    )
    .with_name("wolfram-free-field".into());
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(5.0, 4.0, 2.8)),
        vec![source],
        listeners.clone(),
    );
    simulation.frequencies = vec![400.0, 1_000.0, 2_400.0];
    let absorbing = SurfaceConfig::Absorption { coefficient: 1.0 };
    simulation.boundaries.floor = absorbing.clone();
    simulation.boundaries.ceiling = absorbing.clone();
    simulation.boundaries.walls = absorbing;
    let config = FastRoomConfig {
        modal_damping_floor: 5.0e-4,
        modal_frequency_margin: 1.25,
        image_order: 1,
        max_path_time: 1.0,
        min_transition_hz: 20.0,
        max_transition_hz: 20.0,
    };
    let free_field = run_fast_hybrid_with_config(&simulation, &config)
        .expect("run direct-path room reference case");

    // Zero absorption adds exactly the six first-order rectangular-room
    // images to the direct path when image_order is one.
    let reflecting = SurfaceConfig::Absorption { coefficient: 0.0 };
    simulation.boundaries.floor = reflecting.clone();
    simulation.boundaries.ceiling = reflecting.clone();
    simulation.boundaries.walls = reflecting;
    let first_order = run_fast_hybrid_with_config(&simulation, &config)
        .expect("run first-order image room reference case");

    let pressure_grid = |pressures: &[Vec<Complex64>]| {
        pressures
            .iter()
            .map(|row| row.iter().copied().map(complex_json).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };

    json!({
        "room_dimensions_m": [5.0, 4.0, 2.8],
        "source_m": [source_position.x, source_position.y, source_position.z],
        "listeners_m": listeners.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
        "frequencies_hz": simulation.frequencies,
        "sound_speed_m_s": simulation.speed_of_sound,
        "actual_free_field_pressures": pressure_grid(&free_field.pressures[0]),
        "actual_first_order_pressures": pressure_grid(&first_order.pressures[0])
    })
}

fn pc_fan_case() -> Value {
    let mut config = PcFanConfig::bench_120mm();
    config.broadband = PcFanBroadbandConfig::off();
    config.mesh.harmonic_count = 3;
    let design = config.default_design_variables();
    let flow = solve_pc_fan_mean_flow(&config, &design);
    let acoustics = solve_pc_fan_acoustics(&config, &design);

    json!({
        "blade_count": config.blade_count,
        "rpm": config.operating_point.rpm,
        "sound_speed_m_s": config.operating_point.speed_of_sound_m_s,
        "air_density_kg_m3": config.operating_point.air_density_kg_m3,
        "reference_distance_m": config.enclosure.reference_distance_m,
        "hub_radius_m": config.dimensions.hub_radius_m,
        "tip_radius_m": config.dimensions.blade_tip_radius_m,
        "airflow_cfm": flow.airflow_cfm,
        "actual_axial_velocity_m_s": flow.axial_velocity_m_s,
        "actual_mach_number": flow.mach_number,
        "actual_blade_passing_frequency_hz": acoustics.blade_passing_frequency_hz,
        "actual_harmonic_indices": acoustics.harmonics.iter().map(|h| h.harmonic).collect::<Vec<_>>(),
        "actual_harmonic_frequencies_hz": acoustics.harmonics.iter().map(|h| h.frequency_hz).collect::<Vec<_>>(),
        "actual_harmonic_pressures_pa": acoustics.harmonics.iter().map(|h| h.pressure_pa).collect::<Vec<_>>(),
        "actual_harmonic_spl_db": acoustics.harmonics.iter().map(|h| h.sound_pressure_level_db).collect::<Vec<_>>(),
        "actual_harmonic_swl_db": acoustics.harmonics.iter().map(|h| h.sound_power_level_db).collect::<Vec<_>>(),
        "actual_overall_spl_db": acoustics.overall_spl_db,
        "actual_a_weighted_overall_spl_db": acoustics.a_weighted_overall_spl_db,
        "actual_overall_swl_db": acoustics.overall_swl_db
    })
}

fn propeller_case() -> Value {
    let source = PropellerSource::drone_5inch(18_000.0);
    let spectrum =
        PropellerDesign::conventional(source.clone()).spectrum(3, EvaluationMode::FastSurrogate);
    let load_scale = 1.7;
    let mut scaled_source = source.clone();
    scaled_source.thrust *= load_scale;
    scaled_source.torque *= load_scale;
    let scaled_spectrum =
        PropellerDesign::conventional(scaled_source).spectrum(3, EvaluationMode::FastSurrogate);
    let band_hz = [0.0, 100_000.0];

    json!({
        "blade_count": source.n_blades,
        "rpm": source.rpm,
        "thrust_n": source.thrust,
        "torque_nm": source.torque,
        "tip_radius_m": source.tip_radius,
        "sound_speed_m_s": source.speed_of_sound,
        "load_scale": load_scale,
        "a_weighting_band_hz": band_hz,
        "actual_harmonic_indices": spectrum.harmonics.iter().map(|h| h.harmonic).collect::<Vec<_>>(),
        "actual_frequencies_hz": spectrum.harmonics.iter().map(|h| h.frequency_hz).collect::<Vec<_>>(),
        "actual_base_pressures_pa": spectrum.harmonics.iter().map(|h| h.base_pressure_pa).collect::<Vec<_>>(),
        "actual_pressures_pa": spectrum.harmonics.iter().map(|h| h.pressure_pa).collect::<Vec<_>>(),
        "actual_amplitude_reductions": spectrum.harmonics.iter().map(|h| h.amplitude_reduction).collect::<Vec<_>>(),
        "actual_scaled_base_pressures_pa": scaled_spectrum.harmonics.iter().map(|h| h.base_pressure_pa).collect::<Vec<_>>(),
        "actual_a_weighted_pressure_power": spectrum.a_weighted_pressure_power_in_band(band_hz[0], band_hz[1])
    })
}

fn speaker_case() -> Value {
    let points = [[0.0, 0.05], [0.2, 0.04], [0.6, 0.015], [1.0, 0.0]];
    let queries = [-0.1, 0.0, 0.15, 0.2, 0.55, 0.6, 1.0, 1.2];
    let spline = CubicSpline::from_control_points(&points);
    let affine_points = [[-0.5, -1.25], [0.1, 0.55], [0.8, 2.65], [1.4, 4.45]];
    let affine_queries = [-0.8, -0.5, -0.13, 0.4, 1.4, 1.8];
    let affine_spline = CubicSpline::from_control_points(&affine_points);

    json!({
        "control_points": points,
        "queries": queries,
        "actual_values": queries.map(|x| spline.evaluate(x)),
        "actual_derivatives": queries.map(|x| spline.derivative(x)),
        "affine_control_points": affine_points,
        "affine_queries": affine_queries,
        "actual_affine_values": affine_queries.map(|x| affine_spline.evaluate(x)),
        "actual_affine_derivatives": affine_queries.map(|x| affine_spline.derivative(x))
    })
}

fn speaker_modeling_case() -> Value {
    let medium = AcousticMedium::standard_air();
    let radius_m = 0.082;
    let frequency_hz = 750.0;
    let angles_deg = [0.0, 30.0, 60.0, 90.0];
    let velocity_rms = Complex64::new(0.03, -0.01);
    let distances_m = [1.15, 2.3, 4.6];
    let far_field_angle_deg = 37.0;
    let directivity = angles_deg.map(|angle| {
        baffled_piston_directivity(radius_m, frequency_hz, angle, medium)
            .expect("evaluate piston directivity")
    });
    let pressures = distances_m.map(|distance_m| {
        baffled_piston_far_field_pressure(
            radius_m,
            frequency_hz,
            velocity_rms,
            distance_m,
            far_field_angle_deg,
            medium,
        )
        .expect("evaluate piston far field")
    });
    let impedance_frequencies_hz = [0.25, 125.0, 750.0, 2_500.0];
    let impedances = impedance_frequencies_hz.map(|frequency| {
        baffled_piston_mechanical_radiation_impedance(radius_m, frequency, medium)
            .expect("evaluate piston radiation impedance")
    });

    json!({
        "radius_m": radius_m,
        "frequency_hz": frequency_hz,
        "angles_deg": angles_deg,
        "density_kg_m3": medium.density_kg_per_m3,
        "sound_speed_m_s": medium.sound_speed_m_per_s,
        "velocity_rms": complex_json(velocity_rms),
        "distances_m": distances_m,
        "far_field_angle_deg": far_field_angle_deg,
        "actual_directivity": directivity,
        "actual_far_field_pressures": pressures.map(complex_json),
        "impedance_frequencies_hz": impedance_frequencies_hz,
        "actual_mechanical_radiation_impedances": impedances.map(complex_json)
    })
}

fn structural_case() -> Value {
    let study = Beam2DStudy::introductory();
    let result = run_beam2d_static(&study).expect("solve introductory cantilever");

    json!({
        "length_m": study.geometry.length_m,
        "width_m": study.geometry.width_m,
        "height_m": study.geometry.height_m,
        "youngs_modulus_pa": study.material.youngs_modulus_pa,
        "tip_force_n": study.vertical_tip_force_n,
        "actual_tip_deflection_m": result.tip_deflection_m,
        "actual_solver_residual": result.solver_residual,
        "actual_force_balance_residual_n": result.force_balance_residual_n,
        "actual_free_equilibrium_residual_n": result.free_equilibrium_residual_n,
        "actual_energy_balance_relative_error": result.energy_balance_relative_error,
        "actual_applied_force_n": result.applied_force_n,
        "actual_reaction_force_n": result.reaction_force_n,
        "actual_reaction_moment_nm": result.reaction_moment_nm,
        "actual_strain_energy_j": result.strain_energy_j,
        "actual_max_displacement_m": result.max_displacement_m,
        "elements": result.elements,
        "actual_nodal_displacement": result.displacement
    })
}

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/sim_crates_accuracy.wls")
}

fn wolfram_kernel_path() -> Option<PathBuf> {
    env::var_os("WOLFRAM_KERNEL")
        .or_else(|| env::var_os("WolframKernel"))
        .map(PathBuf::from)
        .or_else(|| {
            #[cfg(target_os = "macos")]
            {
                let standard = PathBuf::from(
                    "/Applications/Wolfram Engine.app/Contents/Resources/\
                     Wolfram Player.app/Contents/MacOS/WolframKernel",
                );
                standard.is_file().then_some(standard)
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        })
}

fn run_wolfram(input: &Value) -> AccuracyReport {
    let input_file = tempfile::NamedTempFile::new().expect("create Wolfram input file");
    std::fs::write(
        input_file.path(),
        serde_json::to_vec_pretty(input).expect("serialize Wolfram input"),
    )
    .expect("write Wolfram input");

    let kernel = wolfram_kernel_path();
    let executable = kernel.clone().unwrap_or_else(|| {
        PathBuf::from(env::var_os("WOLFRAMSCRIPT").unwrap_or_else(|| "wolframscript".into()))
    });
    let mut command = Command::new(&executable);
    command.arg(if kernel.is_some() { "-script" } else { "-file" });
    command
        .arg(script_path())
        .env("SONIUM_SIM_ACCURACY_INPUT", input_file.path());
    let output = command.wolfram_output().unwrap_or_else(|error| {
        panic!(
            "failed to launch {executable:?}; install Wolfram Engine or set \
             WOLFRAM_KERNEL/WolframKernel/WOLFRAMSCRIPT: {error}"
        )
    });
    assert!(
        output.status.success(),
        "Wolfram simulation accuracy script failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    serde_json::from_str(json).unwrap_or_else(|error| {
        panic!(
            "Wolfram simulation accuracy output was not JSON: {error}\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn every_simulation_crate_matches_an_independent_wolfram_reference() {
    let input = json!({
        "schema_version": 1,
        "cases": {
            "sonium-sim-pcfan": pc_fan_case(),
            "sonium-sim-propellers": propeller_case(),
            "sonium-sim-room": room_case(),
            "sonium-sim-speaker": speaker_case(),
            "sonium-sim-speaker-modeling": speaker_modeling_case(),
            "sonium-sim-structural-mechanics": structural_case()
        }
    });
    let report = run_wolfram(&input);

    assert_eq!(report.schema_version, 1);
    assert_eq!(report.crates.len(), EXPECTED_CRATES.len());
    for expected in EXPECTED_CRATES {
        let crate_report = report
            .crates
            .get(*expected)
            .unwrap_or_else(|| panic!("Wolfram report omitted {expected}"));
        assert_eq!(crate_report.check_count, crate_report.checks.len());
        assert!(
            crate_report.check_count > 0,
            "{expected} has no accuracy checks"
        );
        let failed = crate_report
            .checks
            .iter()
            .filter_map(|(name, check)| {
                (!check.passed).then_some(format!(
                    "{name}: abs={:.3e}, rel={:.3e}, tol={:.3e}",
                    check.absolute_error, check.relative_error, check.tolerance
                ))
            })
            .collect::<Vec<_>>();
        assert!(crate_report.passed, "{expected} failed: {failed:?}");
    }
    assert!(report.passed, "one or more sonium-sim-* crates failed");
}
