//! Literal reproduction of Wolfram's 2D Electric Motor Noise Analysis.
//!
//! A 0.1 m radius motor casing vibrates in circumferential mode 2 inside a
//! 3 m circular truncation boundary. Wolfram prescribes
//! `v_hat = 2 i omega d exp(2 i phi)`, with `d=2 mm`, and uses the cylindrical
//! absorbing coefficient `1/(2 R)-i k`. The published acceptance values are
//! SPL at `(2 m, 0)` for 15000, 17500, and 20000 rpm. The continuation removes
//! Wolfram's exact hard annular sector and checks directional insertion loss.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_neumann_complex, apply_robin_complex},
    mesh::{
        BoundaryType, ElementType, Mesh, Point, annular_mesh_triangles,
        interpolation::interpolate_scalar_at_point,
    },
    physics::acoustics::pressure::HelmholtzForm,
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::SparseLuFactorization;
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.2;
const AIR_DENSITY_KG_M3: f64 = 1.204;
const MOTOR_RADIUS_M: f64 = 0.1;
const DOMAIN_RADIUS_M: f64 = 3.0;
const DEFORMATION_AMPLITUDE_M: f64 = 2.0e-3;
const INNER_MARKER: i32 = 1;
const OUTER_MARKER: i32 = 2;
const BARRIER_MARKER: i32 = 3;
const RPM: [f64; 3] = [15_000.0, 17_500.0, 20_000.0];
const WOLFRAM_SPL_DB: [f64; 3] = [122.286, 126.941, 131.007];
const REFERENCE_PRESSURE_PA: f64 = 20.0e-6;
const RADIAL_CELLS: usize = 36;
const ANGULAR_CELLS: usize = 224;

fn interpolate_complex(mesh: &Mesh, values: &[Complex64], point: [f64; 3]) -> Complex64 {
    let real = values.iter().map(|value| value.re).collect::<Vec<_>>();
    let imag = values.iter().map(|value| value.im).collect::<Vec<_>>();
    Complex64::new(
        interpolate_scalar_at_point(mesh, &real, point).expect("receiver must lie in annulus"),
        interpolate_scalar_at_point(mesh, &imag, point).expect("receiver must lie in annulus"),
    )
}

fn solve_motor(mesh: &Mesh, rpm: f64) -> Vec<Complex64> {
    let frequency_hz = rpm / 60.0;
    let omega = 2.0 * PI * frequency_hz;
    let k = omega / SOUND_SPEED_M_S;
    let space = H1Space::new(PolynomialDegree::P1);
    let matrix = assemble_bilinear(&HelmholtzForm::from_real(k), &space, &space, mesh, 2);
    let mut system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); mesh.nodes.len()]);

    // For exp(+i omega t), Euler's equation gives dp/dn = i rho omega v_hat
    // on the fluid domain's inner normal. A global sign would not affect the
    // published SPL, but retaining it makes the mode-phase checks meaningful.
    apply_neumann_complex(&mut system, mesh, INNER_MARKER, &|x, y, _| {
        let phi = y.atan2(x);
        let mode = Complex64::from_polar(1.0, 2.0 * phi);
        let velocity = Complex64::new(0.0, 2.0 * omega * DEFORMATION_AMPLITUDE_M) * mode;
        Complex64::new(0.0, AIR_DENSITY_KG_M3 * omega) * velocity
    });
    let absorbing = Complex64::new(1.0 / (2.0 * DOMAIN_RADIUS_M), -k);
    apply_robin_complex(
        &mut system,
        mesh,
        OUTER_MARKER,
        &|_, _, _| absorbing,
        &|_, _, _| Complex64::new(0.0, 0.0),
    );

    let matrix = system.matrix.to_csr();
    SparseLuFactorization::factorize(&matrix)
        .unwrap_or_else(|error| panic!("motor-noise factorization failed at {rpm} rpm: {error}"))
        .solve(&system.rhs)
        .unwrap_or_else(|error| panic!("motor-noise solve failed at {rpm} rpm: {error}"))
}

fn sound_pressure_level_db(pressure: Complex64) -> f64 {
    20.0 * (pressure.norm() / REFERENCE_PRESSURE_PA).log10()
}

/// Polar mesh of Wolfram's literal hard barrier:
/// `Annulus[{0,0},{1,6/5},{Pi/4,3 Pi/4}]` removed from the acoustic domain.
fn barrier_mesh() -> Mesh {
    let mut radii = (0..=12)
        .map(|index| MOTOR_RADIUS_M + 0.9 * index as f64 / 12.0)
        .collect::<Vec<_>>();
    radii.extend((1..=3).map(|index| 1.0 + 0.2 * index as f64 / 3.0));
    radii.extend((1..=24).map(|index| 1.2 + 1.8 * index as f64 / 24.0));

    let node_id = |radial: usize, angular: usize| radial * ANGULAR_CELLS + angular;
    let mut cells = Vec::new();
    let mut used = vec![false; radii.len() * ANGULAR_CELLS];
    for radial in 0..radii.len() - 1 {
        let radius_midpoint = 0.5 * (radii[radial] + radii[radial + 1]);
        for angular in 0..ANGULAR_CELLS {
            let next_angular = (angular + 1) % ANGULAR_CELLS;
            let angle_midpoint = 2.0 * PI * (angular as f64 + 0.5) / ANGULAR_CELLS as f64;
            let inside_barrier = (1.0..1.2).contains(&radius_midpoint)
                && (PI / 4.0..3.0 * PI / 4.0).contains(&angle_midpoint);
            if inside_barrier {
                continue;
            }
            let nodes = [
                node_id(radial, angular),
                node_id(radial + 1, angular),
                node_id(radial, next_angular),
                node_id(radial + 1, next_angular),
            ];
            for node in nodes {
                used[node] = true;
            }
            cells.push(nodes);
        }
    }

    let mut mesh = Mesh::new(2);
    let mut compact = vec![usize::MAX; used.len()];
    for (background_node, is_used) in used.into_iter().enumerate() {
        if !is_used {
            continue;
        }
        let radial = background_node / ANGULAR_CELLS;
        let angular = background_node % ANGULAR_CELLS;
        let angle = 2.0 * PI * angular as f64 / ANGULAR_CELLS as f64;
        compact[background_node] = mesh.add_node(Point::new_2d(
            radii[radial] * angle.cos(),
            radii[radial] * angle.sin(),
        ));
    }
    for [inner, outer, inner_next, outer_next] in cells {
        let [inner, outer, inner_next, outer_next] = [
            compact[inner],
            compact[outer],
            compact[inner_next],
            compact[outer_next],
        ];
        mesh.add_element(ElementType::Triangle, vec![inner, outer, outer_next]);
        mesh.add_element(ElementType::Triangle, vec![inner, outer_next, inner_next]);
    }

    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        let mean_radius = boundary
            .nodes
            .iter()
            .map(|&node| mesh.nodes[node].x.hypot(mesh.nodes[node].y))
            .sum::<f64>()
            / boundary.nodes.len() as f64;
        if (mean_radius - MOTOR_RADIUS_M).abs() < 1.0e-8 {
            boundary.marker = INNER_MARKER;
            boundary.boundary_type = BoundaryType::Neumann;
        } else if (mean_radius - DOMAIN_RADIUS_M).abs() < 1.0e-8 {
            boundary.marker = OUTER_MARKER;
            boundary.boundary_type = BoundaryType::Robin;
        } else {
            boundary.marker = BARRIER_MARKER;
            boundary.boundary_type = BoundaryType::Neumann;
        }
    }
    mesh
}

fn upper_receiver_spl_db(mesh: &Mesh, rpm: f64) -> f64 {
    let pressure = solve_motor(mesh, rpm);
    sound_pressure_level_db(interpolate_complex(mesh, &pressure, [0.0, 2.0, 0.0]))
}

fn sonium_spl_db() -> [f64; 3] {
    let mesh = annular_mesh_triangles(
        0.0,
        0.0,
        MOTOR_RADIUS_M,
        DOMAIN_RADIUS_M,
        RADIAL_CELLS,
        ANGULAR_CELLS,
    );
    let mut result = [0.0; 3];
    for (index, rpm) in RPM.into_iter().enumerate() {
        let pressure = solve_motor(&mesh, rpm);
        let receiver = interpolate_complex(&mesh, &pressure, [2.0, 0.0, 0.0]);
        result[index] = sound_pressure_level_db(receiver);

        // Circumferential mode 2 has equal magnitude on the four axes, with
        // the y-axis pressure opposite in phase to the x-axis pressure.
        let py = interpolate_complex(&mesh, &pressure, [0.0, 2.0, 0.0]);
        let magnitude_error = (py.norm() - receiver.norm()).abs() / receiver.norm();
        assert!(
            magnitude_error < 0.015,
            "mode-2 magnitude symmetry failed at {rpm} rpm"
        );
        let phase_residual = (py + receiver).norm() / receiver.norm();
        assert!(
            phase_residual < 0.025,
            "mode-2 phase symmetry failed at {rpm} rpm"
        );
    }
    result
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    provenance: String,
    rpm: [f64; 3],
    spl_at_2m_db: [f64; 3],
    barrier_spl_at_upper_2m_db: [f64; 3],
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/motor_noise_analysis.wls");
    let output = match Command::new(&executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using published Motor Noise SPL values");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram Motor Noise reference failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram Motor Noise output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_spl_close(actual: &[f64; 3], reference: &[f64; 3], tolerance_db: f64) {
    for (index, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        let error = (actual - reference).abs();
        assert!(
            error < tolerance_db,
            "{} rpm SPL: actual={actual:.3} dB, reference={reference:.3} dB, error={error:.3} dB",
            RPM[index]
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_motor_noise_hard_barrier_directionality() {
    let free_mesh = annular_mesh_triangles(
        0.0,
        0.0,
        MOTOR_RADIUS_M,
        DOMAIN_RADIUS_M,
        RADIAL_CELLS,
        ANGULAR_CELLS,
    );
    let barrier_mesh = barrier_mesh();
    let free_spl = upper_receiver_spl_db(&free_mesh, RPM[2]);
    let barrier_spl = upper_receiver_spl_db(&barrier_mesh, RPM[2]);
    eprintln!(
        "20,000 rpm upper receiver: free={free_spl:.3} dB, barrier={barrier_spl:.3} dB, insertion loss={:.3} dB",
        free_spl - barrier_spl
    );
    assert!(free_spl.is_finite() && barrier_spl.is_finite());
    assert!(
        barrier_spl < free_spl,
        "hard barrier should reduce upper-receiver SPL: free={free_spl:.3} dB, barrier={barrier_spl:.3} dB"
    );

    if let Some(reference) = wolfram_reference() {
        assert!(
            (barrier_spl - reference.barrier_spl_at_upper_2m_db[2]).abs() < 2.0,
            "20,000 rpm upper-receiver barrier SPL: Sonium={barrier_spl:.3} dB, Wolfram={:.3} dB",
            reference.barrier_spl_at_upper_2m_db[2]
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_motor_noise_spl_and_mode_shape() {
    let sonium = sonium_spl_db();
    eprintln!("Sonium Motor Noise SPL at (2 m, 0): {sonium:?} dB");
    assert_spl_close(&sonium, &WOLFRAM_SPL_DB, 1.0);
    assert!(sonium.windows(2).all(|pair| pair[1] > pair[0]));

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(
            reference
                .provenance
                .contains("Wolfram Motor Noise Analysis")
        );
        assert_eq!(reference.rpm, RPM);
        assert_spl_close(&reference.spl_at_2m_db, &WOLFRAM_SPL_DB, 0.01);
        assert_spl_close(&sonium, &reference.spl_at_2m_db, 1.0);
    }
}
