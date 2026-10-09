//! Speaker-oriented acoustic BEM adaptation derived from Wolfram's prescribed
//! surface-velocity Motor Noise model.
//!
//! Wolfram does not publish a loudspeaker or baffled-piston BEM example. This
//! test therefore does not claim literal geometry parity. It drives a circular
//! patch on a finite sealed cabinet through Sonium's public low-level BEM API
//! and compares front-hemisphere pressure, directivity, spreading, and power
//! with the independent infinite-baffle piston limit.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_bem::{
    core::{
        BasisOrder, BoundaryCondition, IncidentField, PhysicsParams,
        assembly::tbem::build_tbem_system_with_beta,
        contracts::constant_panel_power,
        postprocess::pressure::compute_total_field,
        solver::{GmresConfig, gmres_solve_tbem_with_ilu},
    },
    room_acoustics::{RectangularRoom, room_mesh_to_core_elements},
};
use sonium_mesher::RoomSurfaceMeshing;
use sonium_qa::subprocess::WolframCommandExt;
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.0;
const DENSITY_KG_M3: f64 = 1.21;
const CABINET_DEPTH_M: f64 = 0.15;
const CABINET_WIDTH_M: f64 = 0.30;
const CABINET_HEIGHT_M: f64 = 0.30;
const PISTON_RADIUS_M: f64 = 0.065;
const PISTON_VELOCITY_PEAK_M_S: f64 = 0.01;
const NOMINAL_KA: f64 = 2.0;
const ELEMENTS_PER_METER: usize = 40;
const ANGLES_DEG: [f64; 3] = [0.0, 30.0, 60.0];
const DIRECTIVITY_RADIUS_M: f64 = 2.0;
const AXIAL_RADII_M: [f64; 2] = [2.0, 4.0];

fn frequency_hz() -> f64 {
    NOMINAL_KA * SOUND_SPEED_M_S / (2.0 * PI * PISTON_RADIUS_M)
}

fn bessel_j1(x: f64) -> f64 {
    let mut term = 0.5 * x;
    let mut sum = term;
    for order in 1..80 {
        term *= -0.25 * x * x / (order as f64 * (order as f64 + 1.0));
        sum += term;
        if term.abs() < 1.0e-16 * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

fn piston_directivity(ka: f64, angle_deg: f64) -> f64 {
    let argument = ka * angle_deg.to_radians().sin();
    if argument.abs() < 1.0e-12 {
        1.0
    } else {
        2.0 * bessel_j1(argument) / argument
    }
}

fn piston_on_axis_pressure_peak(radius_m: f64, distance_m: f64) -> f64 {
    let wave_number = 2.0 * PI * frequency_hz() / SOUND_SPEED_M_S;
    DENSITY_KG_M3 * SOUND_SPEED_M_S * wave_number * PISTON_VELOCITY_PEAK_M_S * radius_m.powi(2)
        / (2.0 * distance_m)
}

fn piston_power(radius_m: f64) -> f64 {
    let wave_number = 2.0 * PI * frequency_hz() / SOUND_SPEED_M_S;
    let ka = wave_number * radius_m;
    let normalized_resistance = 1.0 - bessel_j1(2.0 * ka) / ka;
    0.5 * DENSITY_KG_M3
        * SOUND_SPEED_M_S
        * PI
        * radius_m.powi(2)
        * PISTON_VELOCITY_PEAK_M_S.powi(2)
        * normalized_resistance
}

fn receiver(angle_deg: f64, distance_m: f64) -> [f64; 3] {
    let angle = angle_deg.to_radians();
    [
        CABINET_DEPTH_M + distance_m * angle.cos(),
        0.5 * CABINET_WIDTH_M + distance_m * angle.sin(),
        0.5 * CABINET_HEIGHT_M,
    ]
}

#[derive(Debug)]
struct BemResult {
    pressure_pa: Vec<Complex64>,
    radiated_power_w: f64,
    active_area_m2: f64,
    panel_count: usize,
    relative_residual: f64,
}

fn solve_bem() -> BemResult {
    let room = RectangularRoom::new(CABINET_DEPTH_M, CABINET_WIDTH_M, CABINET_HEIGHT_M);
    let mesh = room.generate_mesh(ELEMENTS_PER_METER);
    let wave_number = 2.0 * PI * frequency_hz() / SOUND_SPEED_M_S;
    let (mut elements, nodes) = room_mesh_to_core_elements(&mesh, wave_number);
    let piston_center = [
        CABINET_DEPTH_M,
        0.5 * CABINET_WIDTH_M,
        0.5 * CABINET_HEIGHT_M,
    ];
    let prescribed_velocity = Complex64::new(PISTON_VELOCITY_PEAK_M_S, 0.0);
    let mut active = Vec::new();
    let mut surface_velocity = vec![Complex64::new(0.0, 0.0); elements.len()];
    for (index, element) in elements.iter_mut().enumerate() {
        let on_front = (element.center[0] - piston_center[0]).abs() < 1.0e-12;
        let radial_distance =
            (element.center[1] - piston_center[1]).hypot(element.center[2] - piston_center[2]);
        if on_front && radial_distance <= PISTON_RADIUS_M {
            element.boundary_condition = BoundaryCondition::Velocity(vec![prescribed_velocity]);
            surface_velocity[index] = prescribed_velocity;
            active.push(index);
        } else {
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        }
        element.dof_addresses = vec![index];
    }
    assert!(!active.is_empty(), "piston patch must contain panels");

    let physics = PhysicsParams::new(frequency_hz(), SOUND_SPEED_M_S, DENSITY_KG_M3, false);
    let beta = physics.burton_miller_beta_scaled(4.0);
    let system = build_tbem_system_with_beta(&elements, &nodes, &physics, beta);
    let solved = gmres_solve_tbem_with_ilu(
        &system.matrix,
        &system.rhs,
        &GmresConfig {
            max_iterations: 1_000,
            restart: 100,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 1,
        },
    );
    assert!(
        solved.converged,
        "speaker-piston BEM solve failed: residual={}",
        solved.residual
    );

    let mut receivers: Vec<_> = ANGLES_DEG
        .iter()
        .map(|&angle| receiver(angle, DIRECTIVITY_RADIUS_M))
        .collect();
    receivers.push(receiver(0.0, AXIAL_RADII_M[1]));
    let no_incident = IncidentField::PlaneWave {
        direction: [1.0, 0.0, 0.0],
        amplitude: Complex64::new(0.0, 0.0),
    };
    let pressure_pa = compute_total_field(
        &receivers,
        &elements,
        &nodes,
        &solved.x,
        Some(&surface_velocity),
        &no_incident,
        &physics,
        BasisOrder::Constant,
    )
    .into_iter()
    .map(|field| field.p_total)
    .collect();

    let active_areas: Vec<_> = active.iter().map(|&index| elements[index].area).collect();
    let active_pressures: Vec<_> = active.iter().map(|&index| solved.x[index]).collect();
    let active_velocities = vec![prescribed_velocity; active.len()];
    let radiated_power_w =
        constant_panel_power(&active_areas, &active_pressures, &active_velocities)
            .expect("speaker-piston power");

    BemResult {
        pressure_pa,
        radiated_power_w,
        active_area_m2: active_areas.iter().sum(),
        panel_count: elements.len(),
        relative_residual: solved.residual,
    }
}

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    provenance: String,
    ka: f64,
    angles_deg: [f64; ANGLES_DEG.len()],
    normalized_directivity: [f64; ANGLES_DEG.len()],
    axial_radii_m: [f64; AXIAL_RADII_M.len()],
    axial_pressure_peak_pa: [f64; AXIAL_RADII_M.len()],
}

fn optional_wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/bem_speaker_piston_reference.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram piston reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram piston output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn speaker_bem_recovers_baffled_piston_field_and_power_trends() {
    let result = solve_bem();
    let effective_radius = (result.active_area_m2 / PI).sqrt();
    let effective_ka = 2.0 * PI * frequency_hz() / SOUND_SPEED_M_S * effective_radius;
    let pressure_magnitude: Vec<_> = result
        .pressure_pa
        .iter()
        .map(|value| value.norm())
        .collect();
    let numerical_directivity = ANGLES_DEG.map(|angle| {
        let index = ANGLES_DEG
            .iter()
            .position(|&candidate| candidate == angle)
            .unwrap();
        pressure_magnitude[index] / pressure_magnitude[0]
    });
    let exact_directivity = ANGLES_DEG.map(|angle| piston_directivity(effective_ka, angle));
    let exact_axis = piston_on_axis_pressure_peak(effective_radius, DIRECTIVITY_RADIUS_M);
    let exact_power = piston_power(effective_radius);
    let on_axis_error = (pressure_magnitude[0] - exact_axis).abs() / exact_axis;
    let spreading_error =
        (pressure_magnitude[ANGLES_DEG.len()] / pressure_magnitude[0] - 0.5).abs() / 0.5;
    let power_error = (result.radiated_power_w - exact_power).abs() / exact_power;

    eprintln!(
        "speaker piston BEM: panels={}, area={:.8e} m2, a_eff={:.6e} m, ka_eff={:.5}, residual={:.3e}, p2m={:.6e}/{:.6e} Pa, power={:.6e}/{:.6e} W, errors(axis/spreading/power)={:.3e}/{:.3e}/{:.3e}, directivity={:?}/{:?}",
        result.panel_count,
        result.active_area_m2,
        effective_radius,
        effective_ka,
        result.relative_residual,
        pressure_magnitude[0],
        exact_axis,
        result.radiated_power_w,
        exact_power,
        on_axis_error,
        spreading_error,
        power_error,
        numerical_directivity,
        exact_directivity,
    );

    assert!(result.relative_residual < 1.0e-8);
    assert!(result.radiated_power_w > 0.0, "radiation must be passive");
    assert!((effective_radius / PISTON_RADIUS_M - 1.0).abs() < 0.12);
    assert!(on_axis_error < 0.35);
    assert!(spreading_error < 0.08);
    assert!(power_error < 0.35);
    for sample in 1..ANGLES_DEG.len() {
        assert!(
            (numerical_directivity[sample] - exact_directivity[sample]).abs() < 0.20,
            "{} degree directivity: BEM={}, baffled piston={}",
            ANGLES_DEG[sample],
            numerical_directivity[sample],
            exact_directivity[sample]
        );
    }

    if let Some(reference) = optional_wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(reference.provenance.contains("derived adaptation"));
        assert_eq!(reference.ka, NOMINAL_KA);
        assert_eq!(reference.angles_deg, ANGLES_DEG);
        assert_eq!(reference.axial_radii_m, AXIAL_RADII_M);
        for (sample, &angle) in ANGLES_DEG.iter().enumerate() {
            assert!(
                (reference.normalized_directivity[sample] - piston_directivity(NOMINAL_KA, angle))
                    .abs()
                    < 1.0e-12
            );
        }
        for (sample, &distance) in AXIAL_RADII_M.iter().enumerate() {
            assert!(
                (reference.axial_pressure_peak_pa[sample]
                    - piston_on_axis_pressure_peak(PISTON_RADIUS_M, distance))
                .abs()
                    / piston_on_axis_pressure_peak(PISTON_RADIUS_M, distance)
                    < 1.0e-12
            );
        }
    }
}
