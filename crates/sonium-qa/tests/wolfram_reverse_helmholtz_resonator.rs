//! Sonium reverse example built from Wolfram's published Helmholtz-resonator
//! equation.
//!
//! This deliberately does not claim to reproduce Wolfram's bottle geometry:
//! it checks Sonium's cylindrical speaker cabinet plus external port against
//! the same compact-resonator formula and standard termination corrections.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::physics::acoustics::pressure::resolved::helmholtz_resonance_frequency_hz;
use sonium_qa::subprocess::WolframCommandExt;
use sonium_sim_speaker_modeling::fem_bem_coupling::{
    CouplingDrive, CylinderFixtureDiscretization, FemBemCoupling,
    cylindrical_box_with_external_port,
};
use sonium_solvers::{LinearSolveConfig, LinearSolverType};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.0;
const DENSITY_KG_M3: f64 = 1.204;
const BOX_RADIUS_M: f64 = 0.15;
const BOX_DEPTH_M: f64 = 0.25;
const PORT_RADIUS_M: f64 = 0.03;
const PORT_LENGTH_M: f64 = 0.10;

fn discretization() -> CylinderFixtureDiscretization {
    CylinderFixtureDiscretization {
        n_circumference: 16,
        n_box_depth: 5,
        n_cap_radial: 3,
        n_aperture_radial: 2,
        n_port_depth: 4,
    }
}

fn solver_config() -> LinearSolveConfig {
    LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 2_000,
        tolerance: 1.0e-9,
        restart: 100,
        exact_residual_check_interval: 0,
        verbosity: 0,
    }
}

fn parabolic_peak(curve: &[(f64, f64)]) -> Option<f64> {
    let (index, _) = curve
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.1.total_cmp(&right.1.1))?;
    if index == 0 || index + 1 >= curve.len() {
        return Some(curve[index].0);
    }
    let (x0, y0) = curve[index - 1];
    let (x1, y1) = curve[index];
    let (x2, y2) = curve[index + 1];
    let denominator = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if denominator.abs() < 1.0e-30 {
        return Some(x1);
    }
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denominator;
    if a.abs() < 1.0e-30 {
        return Some(x1);
    }
    Some(-b / (2.0 * a))
}

fn sonium_resonance_peak_hz() -> (f64, f64) {
    let fixture = cylindrical_box_with_external_port(
        BOX_RADIUS_M,
        BOX_DEPTH_M,
        PORT_RADIUS_M,
        PORT_LENGTH_M,
        discretization(),
    )
    .expect("Sonium should build the cabinet-plus-port fixture");
    let model = FemBemCoupling::from_fixture(&fixture, SOUND_SPEED_M_S, DENSITY_KG_M3)
        .expect("Sonium should build the coupled resonator model");
    let cavity_volume = fixture.cavity_volume_m3();
    let port_area = PI * PORT_RADIUS_M.powi(2);
    let effective_port_length = PORT_LENGTH_M + (0.85 + 0.61) * PORT_RADIUS_M;
    let reference = helmholtz_resonance_frequency_hz(
        SOUND_SPEED_M_S,
        cavity_volume,
        port_area,
        effective_port_length,
    );

    let areas = model.aperture_element_areas();
    let config = solver_config();
    let curve: Vec<(f64, f64)> = (0..=10)
        .map(|index| 40.0 + 4.0 * index as f64)
        .map(|frequency| {
            let solution = model
                .solve_frequency(
                    frequency,
                    CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0)),
                    &config,
                )
                .unwrap_or_else(|error| panic!("coupled solve failed at {frequency} Hz: {error}"));
            (frequency, solution.aperture_volume_velocity(&areas).norm())
        })
        .collect();
    (
        parabolic_peak(&curve).expect("frequency sweep is nonempty"),
        reference,
    )
}

#[derive(Deserialize)]
struct WolframHelmholtzReference {
    schema_version: u32,
    provenance: String,
    sound_speed_m_s: f64,
    density_kg_m3: f64,
    box_radius_m: f64,
    box_depth_m: f64,
    port_radius_m: f64,
    port_length_m: f64,
    cavity_volume_m3: f64,
    port_area_m2: f64,
    effective_port_length_m: f64,
    resonance_frequency_hz: f64,
}

fn wolfram_reference() -> Option<WolframHelmholtzReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/reverse_helmholtz_resonator.wls");
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using the local Helmholtz formula");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram Helmholtz reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram Helmholtz output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_relative_close(name: &str, actual: f64, reference: f64, tolerance: f64) {
    let relative_error = (actual - reference).abs() / reference.abs();
    assert!(
        relative_error < tolerance,
        "{name}: actual={actual:.6}, reference={reference:.6}, error={:.3}%",
        100.0 * relative_error,
    );
}

#[test]
fn sonium_reverse_helmholtz_resonator_matches_wolfram_formula() {
    let (sonium_peak, local_reference) = sonium_resonance_peak_hz();
    assert_relative_close(
        "coupled resonator peak versus compact formula",
        sonium_peak,
        local_reference,
        0.08,
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(reference.provenance.contains("reverse example"));
        assert_eq!(reference.sound_speed_m_s, SOUND_SPEED_M_S);
        assert_eq!(reference.density_kg_m3, DENSITY_KG_M3);
        assert_eq!(reference.box_radius_m, BOX_RADIUS_M);
        assert_eq!(reference.box_depth_m, BOX_DEPTH_M);
        assert_eq!(reference.port_radius_m, PORT_RADIUS_M);
        assert_eq!(reference.port_length_m, PORT_LENGTH_M);
        assert_relative_close(
            "cavity volume",
            reference.cavity_volume_m3,
            PI * BOX_RADIUS_M.powi(2) * BOX_DEPTH_M,
            1.0e-12,
        );
        assert_relative_close(
            "port area",
            reference.port_area_m2,
            PI * PORT_RADIUS_M.powi(2),
            1.0e-12,
        );
        assert_relative_close(
            "effective port length",
            reference.effective_port_length_m,
            PORT_LENGTH_M + (0.85 + 0.61) * PORT_RADIUS_M,
            1.0e-12,
        );
        assert_relative_close(
            "Wolfram/local Helmholtz formula",
            reference.resonance_frequency_hz,
            local_reference,
            1.0e-12,
        );
        assert_relative_close(
            "Sonium/Wolfram Helmholtz resonance",
            sonium_peak,
            reference.resonance_frequency_hz,
            0.08,
        );
    }
}
