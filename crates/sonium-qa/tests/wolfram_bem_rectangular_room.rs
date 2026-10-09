//! Driven rigid-room BEM check derived from Wolfram's Room Eigenfrequencies.
//!
//! The literal Wolfram example supplies the `7.5 x 5 x 3 m` geometry and
//! rigid boundaries. It does not publish a driven BEM solution, so this test
//! compares Sonium's interior room BEM with an independently evaluated
//! normalized Neumann modal Green-function sum.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_bem::room_acoustics::{
    Point3D, RectangularRoom, Source, calculate_field_pressure, solve_bem_system_direct,
};
use sonium_mesher::RoomSurfaceMeshing;
use sonium_qa::subprocess::WolframCommandExt;
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const DIMENSIONS_M: [f64; 3] = [7.5, 5.0, 3.0];
const SOUND_SPEED_M_S: f64 = 343.2;
const FREQUENCY_HZ: f64 = 80.0;
const SOURCE_M: [f64; 3] = [1.1, 1.3, 1.2];
const RECEIVER_M: [f64; 3] = [5.8, 3.7, 1.6];
const MAX_MODE_ORDER: usize = 30;

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    provenance: String,
    dimensions_m: [f64; 3],
    sound_speed_m_s: f64,
    frequency_hz: f64,
    source_m: [f64; 3],
    receiver_m: [f64; 3],
    max_mode_order: usize,
    receiver_pressure: [f64; 2],
    receiver_pressure_order_20: [f64; 2],
}

fn epsilon(order: usize) -> f64 {
    if order == 0 { 1.0 } else { 2.0 }
}

fn mode_product(nx: usize, ny: usize, nz: usize, point: [f64; 3]) -> f64 {
    (nx as f64 * PI * point[0] / DIMENSIONS_M[0]).cos()
        * (ny as f64 * PI * point[1] / DIMENSIONS_M[1]).cos()
        * (nz as f64 * PI * point[2] / DIMENSIONS_M[2]).cos()
}

fn modal_pressure(max_order: usize) -> Complex64 {
    let k = 2.0 * PI * FREQUENCY_HZ / SOUND_SPEED_M_S;
    let volume = DIMENSIONS_M.iter().product::<f64>();
    let mut pressure = 0.0;
    for nx in 0..=max_order {
        for ny in 0..=max_order {
            for nz in 0..=max_order {
                let mode_k_squared = (nx as f64 * PI / DIMENSIONS_M[0]).powi(2)
                    + (ny as f64 * PI / DIMENSIONS_M[1]).powi(2)
                    + (nz as f64 * PI / DIMENSIONS_M[2]).powi(2);
                pressure += epsilon(nx) * epsilon(ny) * epsilon(nz) / volume
                    * mode_product(nx, ny, nz, SOURCE_M)
                    * mode_product(nx, ny, nz, RECEIVER_M)
                    / (mode_k_squared - k * k);
            }
        }
    }
    Complex64::new(pressure, 0.0)
}

fn sonium_pressure(elements_per_meter: usize) -> Complex64 {
    let room = RectangularRoom::new(DIMENSIONS_M[0], DIMENSIONS_M[1], DIMENSIONS_M[2]);
    let mesh = room.generate_mesh(elements_per_meter);
    let source_position = Point3D::new(SOURCE_M[0], SOURCE_M[1], SOURCE_M[2]);
    let receiver = Point3D::new(RECEIVER_M[0], RECEIVER_M[1], RECEIVER_M[2]);
    let sources = [Source::omnidirectional(source_position, 1.0)];
    let k = 2.0 * PI * FREQUENCY_HZ / SOUND_SPEED_M_S;
    let surface_pressure = solve_bem_system_direct(&mesh, &sources, k, FREQUENCY_HZ)
        .expect("solve driven rigid rectangular room");
    calculate_field_pressure(
        &mesh,
        &surface_pressure,
        &sources,
        &[receiver],
        k,
        FREQUENCY_HZ,
    )[0]
}

fn relative_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-30)
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/bem_rectangular_room_driven.wls");
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
        "Wolfram driven-room reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram driven-room output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

#[test]
fn sonium_room_bem_matches_rectangular_neumann_green_function() {
    let modal_20 = modal_pressure(20);
    let modal = modal_pressure(MAX_MODE_ORDER);
    let modal_tail = relative_error(modal_20, modal);
    let coarse = sonium_pressure(1);
    let refined = sonium_pressure(2);
    let coarse_error = relative_error(coarse, modal);
    let refined_error = relative_error(refined, modal);
    let mesh_change = relative_error(refined, coarse);
    eprintln!(
        "driven room: modal20/modal30={modal_tail:.3e}, coarse/modal={coarse_error:.3e}, refined/modal={refined_error:.3e}, mesh_change={mesh_change:.3e}, refined={refined:.6e}, modal={modal:.6e}"
    );
    assert!(modal_tail < 0.02, "modal truncation error {modal_tail:.3e}");
    assert!(
        refined_error < 0.06,
        "refined room-BEM error {refined_error:.3e}"
    );
    assert!(
        refined_error < coarse_error,
        "room-BEM refinement did not reduce error: coarse={coarse_error:.3e}, refined={refined_error:.3e}"
    );
    assert!(
        mesh_change < 0.1,
        "room-BEM mesh change is too large: {mesh_change:.3e}"
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(reference.provenance.contains("not a literal BEM example"));
        assert_eq!(reference.dimensions_m, DIMENSIONS_M);
        assert_eq!(reference.sound_speed_m_s, SOUND_SPEED_M_S);
        assert_eq!(reference.frequency_hz, FREQUENCY_HZ);
        assert_eq!(reference.source_m, SOURCE_M);
        assert_eq!(reference.receiver_m, RECEIVER_M);
        assert_eq!(reference.max_mode_order, MAX_MODE_ORDER);
        let wolfram = Complex64::new(
            reference.receiver_pressure[0],
            reference.receiver_pressure[1],
        );
        let wolfram_20 = Complex64::new(
            reference.receiver_pressure_order_20[0],
            reference.receiver_pressure_order_20[1],
        );
        assert!(relative_error(wolfram, modal) < 1.0e-12);
        assert!(relative_error(wolfram_20, modal_20) < 1.0e-12);
        assert!(relative_error(refined, wolfram) < 0.06);
    }
}
