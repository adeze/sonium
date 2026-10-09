//! Literal axisymmetric diagnostic for Wolfram's `Helmholtz Resonator` model.
//!
//! Geometry, air properties, mouth pressure, sweep definition, compact
//! formula, and the published `180.158 Hz` FEM peak are taken from the model
//! page and mirrored by `wolfram/helmholtz_resonator_axisymmetric.wls`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::{
    basis::PolynomialDegree,
    mesh::{Mesh, Point, generators::implicit_region_mesh_triangles},
    physics::acoustics::pressure::{
        AcousticsAssemblerExt, axisymmetric::build_axi_acoustics_assembler,
    },
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{CsrMatrix, SparseLuFactorization, traits::Preconditioner};
use std::{collections::HashMap, env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.2;
const DENSITY_KG_M3: f64 = 1.204;
const NECK_LENGTH_M: f64 = 0.05;
const NECK_RADIUS_M: f64 = 0.01;
const CAVITY_RADIUS_M: f64 = 0.05;
const END_CORRECTION_RADII: f64 = 0.82;
const WOLFRAM_PEAK_HZ: f64 = 180.158;

#[derive(Debug, Deserialize)]
struct WolframReference {
    schema_version: u32,
    provenance: String,
    neck_length_m: f64,
    neck_radius_m: f64,
    cavity_radius_m: f64,
    resonance_formula_hz: f64,
    numerical_peak_hz: f64,
    published_numerical_peak_hz: f64,
    published_formula_error_percent: f64,
}

fn optional_wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/helmholtz_resonator_axisymmetric.wls");
    let output = Command::new(executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .expect("run explicitly configured Wolfram reference extractor");
    assert!(
        output.status.success(),
        "Wolfram Helmholtz extractor failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .expect("Wolfram Helmholtz extractor JSON");
    Some(serde_json::from_str(json).expect("parse Wolfram Helmholtz extractor JSON"))
}

fn compact_frequency_hz() -> f64 {
    let effective_length = NECK_LENGTH_M + END_CORRECTION_RADII * NECK_RADIUS_M;
    SOUND_SPEED_M_S / (2.0 * PI)
        * (NECK_RADIUS_M.powi(2) / (effective_length * (4.0 / 3.0) * CAVITY_RADIUS_M.powi(3)))
            .sqrt()
}

/// Shape-regular background-grid version of Wolfram's half-disk/rectangle
/// union in `(r,z)`.
fn bottle_mesh(_arc_segments: usize, refinement_levels: usize) -> Mesh {
    let center_z = -(NECK_LENGTH_M + CAVITY_RADIUS_M - NECK_RADIUS_M / 2.0);
    let nx = 10usize << refinement_levels.saturating_sub(1);
    let ny = 29usize << refinement_levels.saturating_sub(1);
    implicit_region_mesh_triangles(
        0.0,
        CAVITY_RADIUS_M,
        center_z - CAVITY_RADIUS_M,
        0.0,
        nx,
        ny,
        1,
        |r, z| {
            let in_neck = r <= NECK_RADIUS_M && z >= -NECK_LENGTH_M;
            let in_cavity = r * r + (z - center_z).powi(2) <= CAVITY_RADIUS_M.powi(2);
            in_neck || in_cavity
        },
    )
}

fn impose_dirichlet_values(
    matrix: &mut CsrMatrix<Complex64>,
    rhs: &mut [Complex64],
    values: &[Option<Complex64>],
) {
    for row in 0..matrix.num_rows {
        if values[row].is_some() {
            continue;
        }
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if let Some(value) = values[matrix.col_indices[entry]] {
                rhs[row] -= matrix.values[entry] * value;
                matrix.values[entry] = Complex64::new(0.0, 0.0);
            }
        }
    }
    for (dof, value) in values.iter().enumerate() {
        let Some(value) = value else { continue };
        rhs[dof] = *value;
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }
}

fn triangle_area(points: [Point; 3]) -> f64 {
    0.5 * ((points[1].x - points[0].x) * (points[2].y - points[0].y)
        - (points[2].x - points[0].x) * (points[1].y - points[0].y))
        .abs()
}

fn axisymmetric_volume(mesh: &Mesh) -> f64 {
    mesh.elements
        .iter()
        .map(|element| {
            let nodes = element.vertices();
            let points = [
                mesh.nodes[nodes[0]],
                mesh.nodes[nodes[1]],
                mesh.nodes[nodes[2]],
            ];
            let centroid_r = points.iter().map(|point| point.x).sum::<f64>() / 3.0;
            2.0 * PI * triangle_area(points) * centroid_r
        })
        .sum()
}

fn cavity_average_pressure(mesh: &Mesh, pressure: &[Complex64]) -> Complex64 {
    let center_z = -(NECK_LENGTH_M + CAVITY_RADIUS_M - NECK_RADIUS_M / 2.0);
    let mut weighted_pressure = Complex64::new(0.0, 0.0);
    let mut weight = 0.0;
    for element in &mesh.elements {
        let nodes = element.vertices();
        let points = [
            mesh.nodes[nodes[0]],
            mesh.nodes[nodes[1]],
            mesh.nodes[nodes[2]],
        ];
        let centroid_r = points.iter().map(|point| point.x).sum::<f64>() / 3.0;
        let centroid_z = points.iter().map(|point| point.y).sum::<f64>() / 3.0;
        if centroid_r.powi(2) + (centroid_z - center_z).powi(2) > CAVITY_RADIUS_M.powi(2) {
            continue;
        }
        let area = triangle_area(points);
        let r_sum = points.iter().map(|point| point.x).sum::<f64>();
        let p_sum = nodes.iter().map(|&node| pressure[node]).sum::<Complex64>();
        let rp_sum = nodes
            .iter()
            .zip(points)
            .map(|(&node, point)| point.x * pressure[node])
            .sum::<Complex64>();
        weighted_pressure += area / 12.0 * (r_sum * p_sum + rp_sum);
        weight += area * r_sum / 3.0;
    }
    weighted_pressure / weight
}

fn solve_response(mesh: &Mesh, frequencies_hz: &[f64]) -> Vec<f64> {
    let assembler = build_axi_acoustics_assembler(mesh, PolynomialDegree::P1);
    let mouth_values = mesh
        .nodes
        .iter()
        .map(|point| (point.y.abs() < 1.0e-12).then_some(Complex64::new(1.0, 0.0)))
        .collect::<Vec<_>>();
    assert!(mouth_values.iter().flatten().count() >= 2);

    frequencies_hz
        .iter()
        .map(|&frequency_hz| {
            let k = Complex64::new(2.0 * PI * frequency_hz / SOUND_SPEED_M_S, 0.0);
            let mut matrix = assembler.assemble_helmholtz(k, &HashMap::new());
            let mut rhs = vec![Complex64::new(0.0, 0.0); mesh.num_nodes()];
            impose_dirichlet_values(&mut matrix, &mut rhs, &mouth_values);
            let factor = SparseLuFactorization::factorize(&matrix)
                .expect("factor axisymmetric Helmholtz matrix");
            let pressure = factor.apply(&rhs);
            cavity_average_pressure(mesh, &pressure).norm()
        })
        .collect()
}

fn discrete_peak(frequencies_hz: &[f64], response: &[f64]) -> (f64, f64) {
    frequencies_hz
        .iter()
        .copied()
        .zip(response.iter().copied())
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty response sweep")
}

#[test]
fn sonium_axisymmetric_helmholtz_resonator_matches_wolfram_peak() {
    let formula_hz = compact_frequency_hz();
    let wolfram_sweep = (0..=24)
        .map(|index| formula_hz * (0.76 + 0.02 * index as f64))
        .collect::<Vec<_>>();
    let coarse = bottle_mesh(72, 2);
    let refined = bottle_mesh(72, 3);
    let coarse_response = solve_response(&coarse, &wolfram_sweep);
    let refined_response = solve_response(&refined, &wolfram_sweep);
    let coarse_peak = discrete_peak(&wolfram_sweep, &coarse_response);
    let refined_peak = discrete_peak(&wolfram_sweep, &refined_response);
    let coarse_volume = axisymmetric_volume(&coarse);
    let refined_volume = axisymmetric_volume(&refined);
    eprintln!(
        "Wolfram Helmholtz resonator: formula={formula_hz:.6} Hz, published={WOLFRAM_PEAK_HZ:.6} Hz, official-sweep coarse={coarse_peak:?}, refined={refined_peak:?}, volume={coarse_volume:.9e}/{refined_volume:.9e} m^3, nodes={}/{} rho={DENSITY_KG_M3}",
        coarse.num_nodes(),
        refined.num_nodes()
    );

    let sphere_volume = 4.0 / 3.0 * PI * CAVITY_RADIUS_M.powi(3);
    let neck_volume = PI * NECK_RADIUS_M.powi(2) * NECK_LENGTH_M;
    assert!((formula_hz / WOLFRAM_PEAK_HZ - 1.0).abs() < 0.03);
    assert!((coarse_volume / refined_volume - 1.0).abs() < 0.01);
    assert!((sphere_volume..sphere_volume + neck_volume).contains(&refined_volume));
    assert!(refined_peak.1.is_finite());
    assert!(refined_peak.1 > 1.0);
    // This is an undamped pole sampled on a discrete grid, so peak amplitude
    // is not a convergent observable; frequency and volume are.
    assert!((refined_peak.0 / coarse_peak.0 - 1.0).abs() < 0.03);
    assert!((coarse_peak.0 / WOLFRAM_PEAK_HZ - 1.0).abs() < 0.02);
    assert!((refined_peak.0 / WOLFRAM_PEAK_HZ - 1.0).abs() < 0.02);

    if let Some(reference) = optional_wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(reference.provenance.contains("literal Wolfram Helmholtz"));
        assert_eq!(reference.neck_length_m, NECK_LENGTH_M);
        assert_eq!(reference.neck_radius_m, NECK_RADIUS_M);
        assert_eq!(reference.cavity_radius_m, CAVITY_RADIUS_M);
        assert!((reference.resonance_formula_hz - formula_hz).abs() < 1.0e-9);
        assert_eq!(reference.published_numerical_peak_hz, WOLFRAM_PEAK_HZ);
        assert_eq!(reference.published_formula_error_percent, 2.6811);
        assert!((reference.numerical_peak_hz / WOLFRAM_PEAK_HZ - 1.0).abs() < 0.05);
    }
}
