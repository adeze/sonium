//! Sonium reproduction of Wolfram's straight and curved acoustic horns.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/ModelCollection/AcousticHorn.html?view=all>

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_robin_complex},
    mesh::{
        BoundaryType, Mesh,
        generators::{
            FittedImplicitRegionConfig, compact_mesh, fitted_implicit_region_mesh_triangles,
        },
    },
    physics::acoustics::pressure::HelmholtzForm,
    spaces::H1Space,
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::SparseLuFactorization;
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.344;
const DOMAIN_RADIUS_M: f64 = 1.0;
const INLET_HEIGHT_M: f64 = 1.0 / 15.0;
const WALL_THICKNESS_M: f64 = INLET_HEIGHT_M / 2.0;
const MOUTH_HEIGHT_M: f64 = 0.3;
const STRAIGHT_LENGTH_M: f64 = 0.5;
const CURVATURE_RADIUS_M: f64 = 2.0;
const INCIDENT_PRESSURE_PA: f64 = 1.0;

const INLET_MARKER: i32 = 1;
const OUTLET_MARKER: i32 = 2;
const HARD_WALL_MARKER: i32 = 3;

#[derive(Clone, Copy)]
enum HornWall {
    Straight,
    Curved,
}

fn interval_level_set(value: f64, minimum: f64, maximum: f64) -> f64 {
    (minimum - value).max(value - maximum)
}

fn straight_wall_level_set(x: f64, y: f64) -> f64 {
    let horizontal =
        interval_level_set(x, -DOMAIN_RADIUS_M, -DOMAIN_RADIUS_M + STRAIGHT_LENGTH_M).max(
            interval_level_set(y, INLET_HEIGHT_M, INLET_HEIGHT_M + WALL_THICKNESS_M),
        );

    let start = [-DOMAIN_RADIUS_M + STRAIGHT_LENGTH_M, INLET_HEIGHT_M];
    let end = [0.0, MOUTH_HEIGHT_M];
    let tangent = [end[0] - start[0], end[1] - start[1]];
    let length = tangent[0].hypot(tangent[1]);
    let unit_tangent = [tangent[0] / length, tangent[1] / length];
    let upper_normal = [-unit_tangent[1], unit_tangent[0]];
    let relative = [x - start[0], y - start[1]];
    let along = relative[0] * unit_tangent[0] + relative[1] * unit_tangent[1];
    let across = relative[0] * upper_normal[0] + relative[1] * upper_normal[1];
    let sloped = interval_level_set(along, 0.0, length).max(interval_level_set(
        across,
        0.0,
        WALL_THICKNESS_M,
    ));
    horizontal.min(sloped)
}

fn curved_wall_center() -> [f64; 2] {
    let first = [-DOMAIN_RADIUS_M, INLET_HEIGHT_M];
    let second = [0.0, MOUTH_HEIGHT_M];
    let chord = [second[0] - first[0], second[1] - first[1]];
    let chord_length = chord[0].hypot(chord[1]);
    let midpoint = [0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])];
    let offset = (CURVATURE_RADIUS_M.powi(2) - 0.25 * chord_length.powi(2)).sqrt();
    let upper_normal = [-chord[1] / chord_length, chord[0] / chord_length];
    [
        midpoint[0] + offset * upper_normal[0],
        midpoint[1] + offset * upper_normal[1],
    ]
}

fn curved_wall_level_set(x: f64, y: f64) -> f64 {
    let center = curved_wall_center();
    let start = [-DOMAIN_RADIUS_M, INLET_HEIGHT_M];
    let mouth = [0.0, MOUTH_HEIGHT_M];
    let start_angle = (start[1] - center[1]).atan2(start[0] - center[0]);
    let mouth_angle = (mouth[1] - center[1]).atan2(mouth[0] - center[0]);
    let radius = (x - center[0]).hypot(y - center[1]);
    let angle = (y - center[1]).atan2(x - center[0]);
    let radial = interval_level_set(
        radius,
        CURVATURE_RADIUS_M - WALL_THICKNESS_M,
        CURVATURE_RADIUS_M,
    );
    let angular = interval_level_set(
        angle,
        start_angle.min(mouth_angle),
        start_angle.max(mouth_angle),
    );
    radial.max(angular)
}

fn closest_point_on_segment(point: [f64; 2], first: [f64; 2], second: [f64; 2]) -> [f64; 2] {
    let tangent = [second[0] - first[0], second[1] - first[1]];
    let length_squared = tangent[0].powi(2) + tangent[1].powi(2);
    let parameter = (((point[0] - first[0]) * tangent[0] + (point[1] - first[1]) * tangent[1])
        / length_squared)
        .clamp(0.0, 1.0);
    [
        first[0] + parameter * tangent[0],
        first[1] + parameter * tangent[1],
    ]
}

fn closest_candidate(point: [f64; 2], candidates: impl IntoIterator<Item = [f64; 2]>) -> [f64; 2] {
    candidates
        .into_iter()
        .min_by(|left, right| {
            let left_distance = (left[0] - point[0]).powi(2) + (left[1] - point[1]).powi(2);
            let right_distance = (right[0] - point[0]).powi(2) + (right[1] - point[1]).powi(2);
            left_distance.total_cmp(&right_distance)
        })
        .expect("horn projection candidates")
}

fn unit_circle_segment_intersections(first: [f64; 2], second: [f64; 2]) -> Vec<[f64; 2]> {
    let direction = [second[0] - first[0], second[1] - first[1]];
    let a = direction[0].powi(2) + direction[1].powi(2);
    let b = 2.0 * (first[0] * direction[0] + first[1] * direction[1]);
    let c = first[0].powi(2) + first[1].powi(2) - DOMAIN_RADIUS_M.powi(2);
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }
    [-1.0, 1.0]
        .into_iter()
        .filter_map(|sign| {
            let parameter = (-b + sign * discriminant.sqrt()) / (2.0 * a);
            (-1.0e-12..=1.0 + 1.0e-12).contains(&parameter).then_some([
                first[0] + parameter * direction[0],
                first[1] + parameter * direction[1],
            ])
        })
        .collect()
}

fn straight_wall_projection_candidates(point: [f64; 2]) -> Vec<[f64; 2]> {
    let lower_start = [-DOMAIN_RADIUS_M, INLET_HEIGHT_M];
    let lower_joint = [-DOMAIN_RADIUS_M + STRAIGHT_LENGTH_M, INLET_HEIGHT_M];
    let lower_mouth = [0.0, MOUTH_HEIGHT_M];
    let tangent = [
        lower_mouth[0] - lower_joint[0],
        lower_mouth[1] - lower_joint[1],
    ];
    let length = tangent[0].hypot(tangent[1]);
    let normal = [-tangent[1] / length, tangent[0] / length];
    let upper_joint = [
        lower_joint[0] + WALL_THICKNESS_M * normal[0],
        lower_joint[1] + WALL_THICKNESS_M * normal[1],
    ];
    let upper_mouth = [
        lower_mouth[0] + WALL_THICKNESS_M * normal[0],
        lower_mouth[1] + WALL_THICKNESS_M * normal[1],
    ];
    let upper_horizontal_joint = [lower_joint[0], INLET_HEIGHT_M + WALL_THICKNESS_M];
    let upper_start = [-DOMAIN_RADIUS_M, INLET_HEIGHT_M + WALL_THICKNESS_M];
    let segments = [
        [lower_start, lower_joint],
        [lower_joint, lower_mouth],
        [lower_mouth, upper_mouth],
        [upper_mouth, upper_joint],
        [upper_joint, upper_horizontal_joint],
        [upper_horizontal_joint, upper_start],
    ];
    let mut candidates = segments
        .map(|segment| closest_point_on_segment(point, segment[0], segment[1]))
        .to_vec();
    candidates.extend(
        segments
            .into_iter()
            .flat_map(|segment| unit_circle_segment_intersections(segment[0], segment[1])),
    );
    candidates
}

fn curved_arc_projection(point: [f64; 2], radius: f64) -> [f64; 2] {
    let center = curved_wall_center();
    let start = [-DOMAIN_RADIUS_M, INLET_HEIGHT_M];
    let mouth = [0.0, MOUTH_HEIGHT_M];
    let start_angle = (start[1] - center[1]).atan2(start[0] - center[0]);
    let mouth_angle = (mouth[1] - center[1]).atan2(mouth[0] - center[0]);
    let angle = (point[1] - center[1])
        .atan2(point[0] - center[0])
        .clamp(start_angle.min(mouth_angle), start_angle.max(mouth_angle));
    [
        center[0] + radius * angle.cos(),
        center[1] + radius * angle.sin(),
    ]
}

fn curved_wall_projection_candidates(point: [f64; 2]) -> Vec<[f64; 2]> {
    let center = curved_wall_center();
    let outer_start = [-DOMAIN_RADIUS_M, INLET_HEIGHT_M];
    let outer_mouth = [0.0, MOUTH_HEIGHT_M];
    let scale = (CURVATURE_RADIUS_M - WALL_THICKNESS_M) / CURVATURE_RADIUS_M;
    let inner_start = [
        center[0] + scale * (outer_start[0] - center[0]),
        center[1] + scale * (outer_start[1] - center[1]),
    ];
    let inner_mouth = [
        center[0] + scale * (outer_mouth[0] - center[0]),
        center[1] + scale * (outer_mouth[1] - center[1]),
    ];
    let end_segments = [[outer_start, inner_start], [outer_mouth, inner_mouth]];
    let mut candidates = vec![
        curved_arc_projection(point, CURVATURE_RADIUS_M),
        curved_arc_projection(point, CURVATURE_RADIUS_M - WALL_THICKNESS_M),
        closest_point_on_segment(point, outer_start, inner_start),
        closest_point_on_segment(point, outer_mouth, inner_mouth),
    ];
    candidates.extend(
        end_segments
            .into_iter()
            .flat_map(|segment| unit_circle_segment_intersections(segment[0], segment[1])),
    );
    for wall_radius in [CURVATURE_RADIUS_M, CURVATURE_RADIUS_M - WALL_THICKNESS_M] {
        let distance = center[0].hypot(center[1]);
        let along =
            (DOMAIN_RADIUS_M.powi(2) - wall_radius.powi(2) + distance.powi(2)) / (2.0 * distance);
        let height_squared = DOMAIN_RADIUS_M.powi(2) - along.powi(2);
        if height_squared >= 0.0 {
            let base = [center[0] * along / distance, center[1] * along / distance];
            let normal = [-center[1] / distance, center[0] / distance];
            for sign in [-1.0, 1.0] {
                let candidate = [
                    base[0] + sign * height_squared.sqrt() * normal[0],
                    base[1] + sign * height_squared.sqrt() * normal[1],
                ];
                let projected = curved_arc_projection(candidate, wall_radius);
                if (projected[0] - candidate[0]).hypot(projected[1] - candidate[1]) < 1.0e-10 {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates
}

fn project_horn_boundary_exact(mesh: &mut Mesh, wall: HornWall) {
    let boundary_nodes: std::collections::BTreeSet<_> = mesh
        .boundaries
        .iter()
        .flat_map(|boundary| boundary.nodes.iter().copied())
        .collect();
    for node in boundary_nodes {
        let point = [mesh.nodes[node].x, mesh.nodes[node].y];
        let radius = point[0].hypot(point[1]);
        let mut candidates = vec![[point[0].clamp(-DOMAIN_RADIUS_M, DOMAIN_RADIUS_M), 0.0]];
        if radius > 0.0 {
            candidates.push([
                point[0] * DOMAIN_RADIUS_M / radius,
                point[1] * DOMAIN_RADIUS_M / radius,
            ]);
        }
        candidates.extend(match wall {
            HornWall::Straight => straight_wall_projection_candidates(point),
            HornWall::Curved => curved_wall_projection_candidates(point),
        });
        let valid_candidates: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|candidate| {
                horn_domain_level_set(wall, candidate[0], candidate[1]).abs() < 1.0e-8
            })
            .collect();
        let projected = closest_candidate(
            point,
            if valid_candidates.is_empty() {
                candidates
            } else {
                valid_candidates
            },
        );
        mesh.nodes[node].x = projected[0];
        mesh.nodes[node].y = projected[1];
    }
}

fn horn_domain_level_set(wall: HornWall, x: f64, y: f64) -> f64 {
    let half_disk = (x * x + y * y - DOMAIN_RADIUS_M.powi(2)).max(-y);
    let wall = match wall {
        HornWall::Straight => straight_wall_level_set(x, y),
        HornWall::Curved => curved_wall_level_set(x, y),
    };
    half_disk.max(-wall)
}

fn horn_mesh(wall: HornWall, nx: usize, ny: usize) -> Mesh {
    // Wolfram requests h = lambda(1000 Hz)/12 = 0.0286 m. Start from a
    // finer 0.0167 m background grid and project every exposed node onto the
    // exact disk, floor, and horn-wall level sets.
    let dx = 2.0 * DOMAIN_RADIUS_M / nx as f64;
    let dy = DOMAIN_RADIUS_M / ny as f64;
    let mut mesh = fitted_implicit_region_mesh_triangles(
        -DOMAIN_RADIUS_M,
        DOMAIN_RADIUS_M,
        0.0,
        DOMAIN_RADIUS_M,
        HARD_WALL_MARKER,
        FittedImplicitRegionConfig {
            nx,
            ny,
            boundary_refinement_levels: 0,
            projection_iterations: 16,
        },
        |x, y| horn_domain_level_set(wall, x, y),
    );
    for _ in 0..3 {
        project_horn_boundary_exact(&mut mesh, wall);
        // A cut cell can place all three of one triangle's vertices on the
        // same exact wall segment. Projection then turns that zero-measure
        // exterior sliver into a collinear triangle; discard it, rebuild the
        // exposed boundary, and project any newly exposed nodes.
        let nodes = &mesh.nodes;
        mesh.elements.retain(|element| {
            let [a, b, c] = [
                &nodes[element.nodes[0]],
                &nodes[element.nodes[1]],
                &nodes[element.nodes[2]],
            ];
            0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() > 1.0e-10
        });
        mesh.detect_boundaries();
    }
    mesh = compact_mesh(&mesh);
    let (minimum_area, minimum_area_nodes) = mesh
        .elements
        .iter()
        .map(|element| {
            let [a, b, c] = [
                &mesh.nodes[element.nodes[0]],
                &mesh.nodes[element.nodes[1]],
                &mesh.nodes[element.nodes[2]],
            ];
            (
                0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs(),
                [[a.x, a.y], [b.x, b.y], [c.x, c.y]],
            )
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .expect("horn elements");
    assert!(
        minimum_area > 1.0e-10,
        "boundary fitting collapsed a horn element: area={minimum_area:.3e}, nodes={minimum_area_nodes:?}"
    );
    let (maximum_boundary_residual, worst_boundary_point) = mesh
        .boundaries
        .iter()
        .flat_map(|boundary| boundary.nodes.iter())
        .map(|&node| {
            let point = &mesh.nodes[node];
            (
                horn_domain_level_set(wall, point.x, point.y).abs(),
                [point.x, point.y],
            )
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .expect("horn boundary nodes");
    assert!(
        maximum_boundary_residual < 1.0e-8,
        "horn boundary is not fitted to its exact geometry: residual={maximum_boundary_residual:.3e} at {worst_boundary_point:?}"
    );
    let outer_threshold = DOMAIN_RADIUS_M - 2.0 * dx.hypot(dy);
    for boundary in &mut mesh.boundaries {
        let points = boundary
            .nodes
            .iter()
            .map(|&node| &mesh.nodes[node])
            .collect::<Vec<_>>();
        let x = points.iter().map(|point| point.x).sum::<f64>() / points.len() as f64;
        let y = points.iter().map(|point| point.y).sum::<f64>() / points.len() as f64;
        let edge_dx = (points[1].x - points[0].x).abs();
        let edge_dy = (points[1].y - points[0].y).abs();
        if x < -0.95
            && y <= INLET_HEIGHT_M + dy
            && edge_dy > edge_dx
            && x.hypot(y) >= outer_threshold
        {
            boundary.marker = INLET_MARKER;
            boundary.boundary_type = BoundaryType::Robin;
        } else if y > 0.5 * dy
            && !(x < -0.95 && y <= INLET_HEIGHT_M + WALL_THICKNESS_M + dy)
            && x.hypot(y) >= outer_threshold
        {
            boundary.marker = OUTLET_MARKER;
            boundary.boundary_type = BoundaryType::Robin;
        } else {
            boundary.marker = HARD_WALL_MARKER;
            boundary.boundary_type = BoundaryType::Neumann;
        }
    }
    mesh
}

fn reflection_at_frequency(mesh: &Mesh, frequency_hz: f64) -> f64 {
    let k = 2.0 * PI * frequency_hz / SOUND_SPEED_M_S;
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
        OUTLET_MARKER,
        &|_, _, _| Complex64::new(1.0 / (2.0 * DOMAIN_RADIUS_M), -k),
        &|_, _, _| Complex64::new(0.0, 0.0),
    );
    let pressure = SparseLuFactorization::factorize(&system.matrix.to_csr())
        .expect("horn factorization")
        .solve(&system.rhs)
        .expect("horn solve");

    let mut integral = Complex64::new(0.0, 0.0);
    let mut length = 0.0;
    for boundary in mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == INLET_MARKER)
    {
        let [first, second] = [boundary.nodes[0], boundary.nodes[1]];
        let a = &mesh.nodes[first];
        let b = &mesh.nodes[second];
        let edge_length = (a.x - b.x).hypot(a.y - b.y);
        integral += 0.5 * edge_length * (pressure[first] + pressure[second]);
        length += edge_length;
    }
    assert!(length > 0.0, "horn inlet marker must have nonzero length");
    (integral / length - Complex64::new(INCIDENT_PRESSURE_PA, 0.0)).norm()
}

#[derive(Deserialize)]
struct WolframHornReference {
    schema_version: u32,
    frequencies_hz: Vec<f64>,
    straight_reflection: Vec<f64>,
    curved_reflection: Vec<f64>,
}

fn optional_wolfram_reference() -> Option<WolframHornReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/acoustic_horn.wls");
    let output = Command::new(executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .expect("run explicit WOLFRAMSCRIPT horn reference");
    assert!(
        output.status.success(),
        "Wolfram horn reference failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("Wolfram UTF-8 output");
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .expect("Wolfram horn JSON output");
    Some(serde_json::from_str(json).expect("parse Wolfram horn JSON"))
}

#[test]
fn sonium_reproduces_wolfram_acoustic_horn_reflection_spectra() {
    // Values digitized from Wolfram's published log-scale reflection plot.
    // The acceptance tolerance includes plot-pixel uncertainty.
    let straight_frequencies = [150.0, 300.0, 650.0, 800.0, 950.0];
    let published_straight = [0.1097, 0.3717, 0.1653, 0.0827, 0.1119];
    let curved_frequencies = [150.0, 300.0, 650.0];
    let published_curved = [0.2272, 0.0940, 0.0348];
    let curved_high_frequencies = [800.0, 950.0];
    // Pixel-coordinate calibration against the published logarithmic axes.
    // The 800 Hz dip is first-order mesh sensitive, so compare its two-grid
    // extrapolation rather than selecting a mesh whose resonance happens to
    // cross the raster curve. At 950 Hz the refined value is already stable.
    let published_curved_high = [0.0410, 0.0218];
    let straight_mesh = horn_mesh(HornWall::Straight, 120, 60);
    let curved_mesh = horn_mesh(HornWall::Curved, 120, 60);
    let straight =
        straight_frequencies.map(|frequency| reflection_at_frequency(&straight_mesh, frequency));
    let curved =
        curved_frequencies.map(|frequency| reflection_at_frequency(&curved_mesh, frequency));
    let curved_high =
        curved_high_frequencies.map(|frequency| reflection_at_frequency(&curved_mesh, frequency));
    let refined_curved_mesh = horn_mesh(HornWall::Curved, 180, 90);
    let curved_high_refined = curved_high_frequencies
        .map(|frequency| reflection_at_frequency(&refined_curved_mesh, frequency));
    let refinement_ratio = 1.5;
    let curved_800_extrapolated =
        (refinement_ratio * curved_high_refined[0] - curved_high[0]) / (refinement_ratio - 1.0);
    eprintln!("boundary-fitted straight horn: {straight:?}");
    eprintln!("boundary-fitted curved horn: {curved:?}");
    eprintln!("boundary-fitted curved horn high-frequency diagnostic: {curved_high:?}");
    eprintln!(
        "boundary-fitted curved horn refined high-frequency: {curved_high_refined:?}; 800 Hz extrapolated={curved_800_extrapolated:.10}"
    );

    for (index, (&got, &want)) in straight.iter().zip(&published_straight).enumerate() {
        let log_error = (got / want).log10().abs();
        assert!(
            log_error < 0.25,
            "straight horn at {} Hz: Sonium {got:.6}, Wolfram plot {want:.6}, log10 error {log_error:.4}",
            straight_frequencies[index]
        );
    }
    for (index, (&got, &want)) in curved.iter().zip(&published_curved).enumerate() {
        let log_error = (got / want).log10().abs();
        assert!(
            log_error < 0.30,
            "curved horn at {} Hz: Sonium {got:.6}, Wolfram plot {want:.6}, log10 error {log_error:.4}",
            curved_frequencies[index]
        );
    }
    assert!(
        curved_high[0] < curved_high_refined[0] && curved_high_refined[0] < curved_800_extrapolated,
        "800 Hz refinement must approach the published dip monotonically: base={:.6}, refined={:.6}, extrapolated={curved_800_extrapolated:.6}",
        curved_high[0],
        curved_high_refined[0]
    );
    assert!(
        (curved_800_extrapolated / published_curved_high[0])
            .log10()
            .abs()
            < 0.12,
        "curved horn 800 Hz extrapolation: Sonium {curved_800_extrapolated:.6}, Wolfram raster {:.6}",
        published_curved_high[0]
    );
    assert!(
        (curved_high_refined[1] / published_curved_high[1])
            .log10()
            .abs()
            < 0.13,
        "curved horn 950 Hz refined: Sonium {:.6}, Wolfram raster {:.6}",
        curved_high_refined[1],
        published_curved_high[1]
    );
    if let Some(reference) = optional_wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        for (index, &frequency) in straight_frequencies.iter().enumerate() {
            let Some(position) = reference
                .frequencies_hz
                .iter()
                .position(|&candidate| candidate == frequency)
            else {
                continue;
            };
            let Some(&want) = reference.straight_reflection.get(position) else {
                continue;
            };
            let got = straight[index];
            assert!(
                (got / want).log10().abs() < 0.2,
                "Wolfram/Sonium straight horn at {frequency} Hz: {got:.6} versus {want:.6}"
            );
        }
        for (index, &frequency) in curved_frequencies.iter().enumerate() {
            let Some(position) = reference
                .frequencies_hz
                .iter()
                .position(|&candidate| candidate == frequency)
            else {
                continue;
            };
            let Some(&want) = reference.curved_reflection.get(position) else {
                continue;
            };
            let got = curved[index];
            assert!(
                (got / want).log10().abs() < 0.25,
                "Wolfram/Sonium curved horn at {frequency} Hz: {got:.6} versus {want:.6}"
            );
        }
        // These high-frequency values are intentionally gated on a real
        // Wolfram run: the published raster does not support trustworthy
        // digitization of the deep, narrow curved-horn reflection null.
        for (index, &frequency) in curved_high_frequencies.iter().enumerate() {
            let Some(position) = reference
                .frequencies_hz
                .iter()
                .position(|&candidate| candidate == frequency)
            else {
                continue;
            };
            let Some(&want) = reference.curved_reflection.get(position) else {
                continue;
            };
            let got = if index == 0 {
                curved_800_extrapolated
            } else {
                curved_high_refined[index]
            };
            assert!(
                (got / want).log10().abs() < 0.25,
                "Wolfram/Sonium high-frequency curved horn at {frequency} Hz: {got:.6} versus {want:.6}"
            );
        }
    }
}
