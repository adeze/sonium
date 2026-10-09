//! Literal 2D reproduction of Wolfram's double-slit acoustic diffraction case.
//!
//! Wolfram uses a 0.6 m semicircular radiation region above a 0.25 m
//! rectangular source chamber. Two 30 mm slits, separated by 0.2 m, cross a
//! 10 mm screen. A 5 Pa plane wave with wavelength 0.1 m enters from below.
//! The published deterministic observables are the classical double-slit
//! extrema angles, not numerical pressure samples.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_robin_complex},
    mesh::{BoundaryType, ElementType, Mesh, Point, interpolation::interpolate_scalar_at_point},
    physics::acoustics::pressure::HelmholtzForm,
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::SparseLuFactorization;
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.2;
const WAVELENGTH_M: f64 = 0.1;
const INCIDENT_PRESSURE_PA: f64 = 5.0;
const DOMAIN_RADIUS_M: f64 = 0.6;
const CHAMBER_HEIGHT_M: f64 = 0.25;
const SCREEN_THICKNESS_M: f64 = 0.01;
const SLIT_WIDTH_M: f64 = 0.03;
const SLIT_SEPARATION_M: f64 = 0.2;
const INLET_MARKER: i32 = 1;
const ABSORBING_MARKER: i32 = 2;
const HARD_WALL_MARKER: i32 = 3;

fn double_slit_mesh() -> Mesh {
    // dx = dy = lambda/12, matching Wolfram's requested maximum edge scale.
    let nx = 144;
    let ny = 102;
    let x_min = -DOMAIN_RADIUS_M;
    let x_max = DOMAIN_RADIUS_M;
    let y_min = 0.0;
    let y_max = CHAMBER_HEIGHT_M + DOMAIN_RADIUS_M;
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    let mut mask = vec![vec![false; nx]; ny];
    for (j, row) in mask.iter_mut().enumerate() {
        let y = y_min + (j as f64 + 0.5) * dy;
        for (i, active) in row.iter_mut().enumerate() {
            let x = x_min + (i as f64 + 0.5) * dx;
            let lower_chamber = y <= CHAMBER_HEIGHT_M - SCREEN_THICKNESS_M;
            let upper_semicircle = y >= CHAMBER_HEIGHT_M
                && x * x + (y - CHAMBER_HEIGHT_M).powi(2) <= DOMAIN_RADIUS_M.powi(2);
            let in_screen_layer =
                (CHAMBER_HEIGHT_M - SCREEN_THICKNESS_M..=CHAMBER_HEIGHT_M).contains(&y);
            let in_left_slit = (x + SLIT_SEPARATION_M / 2.0).abs() <= SLIT_WIDTH_M / 2.0;
            let in_right_slit = (x - SLIT_SEPARATION_M / 2.0).abs() <= SLIT_WIDTH_M / 2.0;
            *active = lower_chamber
                || upper_semicircle
                || (in_screen_layer && (in_left_slit || in_right_slit));
        }
    }

    let mut mesh = symmetric_binary_mask_mesh(&mask, x_min, x_max, y_min, y_max);
    let tolerance = 0.1 * dx.min(dy);
    for boundary in &mut mesh.boundaries {
        let points = boundary
            .nodes
            .iter()
            .map(|&node| &mesh.nodes[node])
            .collect::<Vec<_>>();
        if points.iter().all(|point| point.y.abs() < tolerance) {
            boundary.marker = INLET_MARKER;
            boundary.boundary_type = BoundaryType::Robin;
        } else if points
            .iter()
            .all(|point| point.y >= CHAMBER_HEIGHT_M - SCREEN_THICKNESS_M - tolerance)
        {
            // This deliberately follows Wolfram's predicate `y >= L - w`,
            // including the exposed screen faces as well as the outer arc.
            boundary.marker = ABSORBING_MARKER;
            boundary.boundary_type = BoundaryType::Robin;
        } else {
            boundary.marker = HARD_WALL_MARKER;
            boundary.boundary_type = BoundaryType::Neumann;
        }
    }
    mesh
}

fn symmetric_binary_mask_mesh(
    mask: &[Vec<bool>],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Mesh {
    let ny = mask.len();
    let nx = mask[0].len();
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let background_node = |i: usize, j: usize| j * (nx + 1) + i;
    let mut used = vec![false; (nx + 1) * (ny + 1)];
    let mut cells = Vec::new();
    for (j, row) in mask.iter().enumerate() {
        for (i, &active) in row.iter().enumerate() {
            if !active {
                continue;
            }
            let nodes = [
                background_node(i, j),
                background_node(i + 1, j),
                background_node(i, j + 1),
                background_node(i + 1, j + 1),
            ];
            for node in nodes {
                used[node] = true;
            }
            cells.push((i, nodes));
        }
    }

    let mut mesh = Mesh::new(2);
    let mut compact = vec![usize::MAX; used.len()];
    for (background, is_used) in used.into_iter().enumerate() {
        if is_used {
            let i = background % (nx + 1);
            let j = background / (nx + 1);
            compact[background] =
                mesh.add_node(Point::new_2d(x_min + i as f64 * dx, y_min + j as f64 * dy));
        }
    }

    for (i, [n00, n10, n01, n11]) in cells {
        let [n00, n10, n01, n11] = [compact[n00], compact[n10], compact[n01], compact[n11]];
        if i < nx / 2 {
            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        } else {
            mesh.add_element(ElementType::Triangle, vec![n00, n10, n01]);
            mesh.add_element(ElementType::Triangle, vec![n10, n11, n01]);
        }
    }
    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        boundary.marker = HARD_WALL_MARKER;
        boundary.boundary_type = BoundaryType::Neumann;
    }
    mesh
}

fn solve_double_slit(mesh: &Mesh) -> Vec<Complex64> {
    let k = 2.0 * PI / WAVELENGTH_M;
    let space = H1Space::new(PolynomialDegree::P1);
    let matrix = assemble_bilinear(&HelmholtzForm::from_real(k), &space, &space, mesh, 2);
    let mut system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); mesh.nodes.len()]);

    apply_robin_complex(
        &mut system,
        mesh,
        INLET_MARKER,
        &|_, _, _| Complex64::new(0.0, -k),
        &|_, _, _| Complex64::new(0.0, -2.0 * k * INCIDENT_PRESSURE_PA),
    );
    apply_robin_complex(
        &mut system,
        mesh,
        ABSORBING_MARKER,
        // Wolfram's 2D outgoing boundary is cylindrical. The 1/(2R)
        // curvature correction is small but keeps the same phasor convention.
        &|_, _, _| Complex64::new(1.0 / (2.0 * DOMAIN_RADIUS_M), -k),
        &|_, _, _| Complex64::new(0.0, 0.0),
    );

    let matrix = system.matrix.to_csr();
    SparseLuFactorization::factorize(&matrix)
        .expect("double-slit factorization")
        .solve(&system.rhs)
        .expect("double-slit solve")
}

fn interpolate_complex(mesh: &Mesh, values: &[Complex64], point: [f64; 3]) -> Complex64 {
    let real = values.iter().map(|value| value.re).collect::<Vec<_>>();
    let imaginary = values.iter().map(|value| value.im).collect::<Vec<_>>();
    Complex64::new(
        interpolate_scalar_at_point(mesh, &real, point).expect("real pressure interpolation"),
        interpolate_scalar_at_point(mesh, &imaginary, point)
            .expect("imaginary pressure interpolation"),
    )
}

fn pressure_on_arc(mesh: &Mesh, pressure: &[Complex64], angle_from_vertical: f64) -> f64 {
    let receiver_radius = 0.5;
    let point = [
        receiver_radius * angle_from_vertical.sin(),
        CHAMBER_HEIGHT_M + receiver_radius * angle_from_vertical.cos(),
        0.0,
    ];
    interpolate_complex(mesh, pressure, point).norm()
}

fn extremum_angle(
    mesh: &Mesh,
    pressure: &[Complex64],
    expected: f64,
    half_window: f64,
    maximum: bool,
) -> f64 {
    let samples = 41;
    (0..samples)
        .map(|index| {
            let fraction = index as f64 / (samples - 1) as f64;
            let angle = expected - half_window + 2.0 * half_window * fraction;
            (angle, pressure_on_arc(mesh, pressure, angle))
        })
        .reduce(|best, candidate| {
            if (maximum && candidate.1 > best.1) || (!maximum && candidate.1 < best.1) {
                candidate
            } else {
                best
            }
        })
        .expect("nonempty angular sample window")
        .0
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WolframReference {
    schema_version: u32,
    analytic_maxima_degrees: [f64; 3],
    analytic_minima_degrees: [f64; 4],
    numerical_maxima_degrees: [f64; 3],
    numerical_minima_degrees: [f64; 4],
}

fn optional_wolfram_reference() -> Option<WolframReference> {
    // Opt in explicitly: the locally installed kernel is not reliable enough
    // for deterministic CI and this solve can be expensive.
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/acoustic_wave_diffraction.wls");
    let output = Command::new(executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .expect("run explicit WOLFRAMSCRIPT diffraction reference");
    assert!(
        output.status.success(),
        "Wolfram diffraction reference failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("Wolfram UTF-8 output");
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .expect("Wolfram diffraction JSON output");
    Some(serde_json::from_str(json).expect("parse Wolfram diffraction JSON"))
}

#[test]
fn sonium_reproduces_wolfram_double_slit_interference_angles() {
    let mesh = double_slit_mesh();
    let pressure = solve_double_slit(&mesh);

    let expected_maxima = [
        -(WAVELENGTH_M / SLIT_SEPARATION_M).asin(),
        0.0,
        (WAVELENGTH_M / SLIT_SEPARATION_M).asin(),
    ];
    let expected_minima = [
        -(1.5 * WAVELENGTH_M / SLIT_SEPARATION_M).asin(),
        -(0.5 * WAVELENGTH_M / SLIT_SEPARATION_M).asin(),
        (0.5 * WAVELENGTH_M / SLIT_SEPARATION_M).asin(),
        (1.5 * WAVELENGTH_M / SLIT_SEPARATION_M).asin(),
    ];

    let search_window = 8.0_f64.to_radians();
    let angle_tolerance = 5.0_f64.to_radians();
    let actual_maxima = expected_maxima
        .map(|expected| extremum_angle(&mesh, &pressure, expected, search_window, true));
    let actual_minima = expected_minima
        .map(|expected| extremum_angle(&mesh, &pressure, expected, search_window, false));
    for (expected, actual) in expected_maxima.into_iter().zip(actual_maxima) {
        assert!(
            (actual - expected).abs() <= angle_tolerance,
            "maximum: expected {:.3} deg, got {:.3} deg",
            expected.to_degrees(),
            actual.to_degrees()
        );
    }
    for (expected, actual) in expected_minima.into_iter().zip(actual_minima) {
        assert!(
            (actual - expected).abs() <= angle_tolerance,
            "minimum: expected {:.3} deg, got {:.3} deg",
            expected.to_degrees(),
            actual.to_degrees()
        );
    }

    // Symmetric geometry and excitation must produce a symmetric magnitude
    // pattern, independently of the analytical far-field approximation.
    for angle_degrees in [10.0_f64, 20.0, 30.0, 40.0, 50.0] {
        let angle = angle_degrees.to_radians();
        let positive = pressure_on_arc(&mesh, &pressure, angle);
        let negative = pressure_on_arc(&mesh, &pressure, -angle);
        let relative = (positive - negative).abs() / positive.max(negative);
        assert!(
            relative < 0.03,
            "magnitude symmetry failed at {angle_degrees} deg: {relative:.3}"
        );
    }

    // The Wolfram model uses lambda/12. Record the relevant scale check: the
    // 30 mm slit is 0.3 lambda and the two-slit separation is 2 lambda.
    assert!((SLIT_WIDTH_M / WAVELENGTH_M - 0.3).abs() < 1.0e-12);
    assert!((SLIT_SEPARATION_M / WAVELENGTH_M - 2.0).abs() < 1.0e-12);
    assert!((SOUND_SPEED_M_S / WAVELENGTH_M - 3432.0).abs() < 1.0e-12);

    if let Some(reference) = optional_wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        for (actual, expected) in reference
            .analytic_maxima_degrees
            .into_iter()
            .zip(expected_maxima.map(f64::to_degrees))
        {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        for (actual, expected) in reference
            .analytic_minima_degrees
            .into_iter()
            .zip(expected_minima.map(f64::to_degrees))
        {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        for (sonium, wolfram) in actual_maxima
            .map(f64::to_degrees)
            .into_iter()
            .zip(reference.numerical_maxima_degrees)
        {
            assert!((sonium - wolfram).abs() <= 6.0);
        }
        for (sonium, wolfram) in actual_minima
            .map(f64::to_degrees)
            .into_iter()
            .zip(reference.numerical_minima_degrees)
        {
            assert!((sonium - wolfram).abs() <= 6.0);
        }
    }
}
