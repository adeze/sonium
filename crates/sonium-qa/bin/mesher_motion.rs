//! Release-facing checked mesh-motion contract.
//!
//! This CPU gate validates deterministic nodal motion, quality admission, and
//! transactional rollback. It is intentionally not a claim about an
//! elasticity/RBF/harmonic motion solve, ALE time integration, remeshing, or
//! a shape-adjoint optimization loop.

use sonium_mesher::mesh::generators::unit_square_triangles;
use sonium_mesher::mesh::motion::{HarmonicMotionOptions, MeshMotion, MeshMotionError};
use sonium_mesher::mesh::quality::QualityThresholds;
use sonium_mesher::mesh::{BoundaryType, Mesh};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "mesher.motion.v1";
const AREA_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct Metrics {
    area_relative_error: f64,
    moved_nodes: usize,
    max_displacement: f64,
    min_mean_ratio: f64,
    min_aspect_quality: f64,
    min_element_measure_ratio: f64,
    max_element_measure_ratio: f64,
    harmonic_iterations: usize,
    harmonic_residual: f64,
    markers_preserved: bool,
    deterministic_repeat: bool,
    quality_rejection_transactional: bool,
}

fn mesh_area(mesh: &Mesh) -> f64 {
    (0..mesh.num_elements())
        .map(|index| mesh.element_measure(index))
        .sum()
}

fn mesh_signature(mesh: &Mesh) -> Result<Vec<u8>, String> {
    serde_json::to_vec(mesh).map_err(|error| format!("mesh serialization failed: {error}"))
}

fn motion_fixture() -> Mesh {
    let mut mesh = unit_square_triangles(2);
    mesh.detect_boundaries();
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 13, |_| true);
    mesh
}

fn harmonic_scaled_motion(mesh: &Mesh, scale_x: f64) -> Result<(MeshMotion, usize, f64), String> {
    let prescribed = mesh
        .nodes
        .iter()
        .map(|point| {
            if point.x == 0.0 || point.x == 1.0 || point.y == 0.0 || point.y == 1.0 {
                Some([(scale_x - 1.0) * point.x, 0.0, 0.0])
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let (motion, report) = MeshMotion::harmonic(
        mesh,
        &prescribed,
        HarmonicMotionOptions {
            max_iterations: 1_000,
            tolerance: 1.0e-13,
        },
    )
    .map_err(|error| format!("construct harmonic affine motion: {error}"))?;
    Ok((motion, report.iterations, report.residual))
}

fn run() -> Result<Metrics, String> {
    let scale_x = 1.25;
    let mut mesh = motion_fixture();
    let base_area = mesh_area(&mesh);
    let marker_signature: Vec<_> = mesh
        .boundaries
        .iter()
        .map(|boundary| {
            (
                boundary.nodes.clone(),
                boundary.marker,
                boundary.boundary_type,
            )
        })
        .collect();
    let (motion, harmonic_iterations, harmonic_residual) = harmonic_scaled_motion(&mesh, scale_x)?;
    let report = motion
        .apply_checked(&mut mesh)
        .map_err(|error| format!("apply affine motion: {error}"))?;
    let area_relative_error = (mesh_area(&mesh) - scale_x * base_area).abs() / base_area;
    let markers_preserved = marker_signature
        == mesh
            .boundaries
            .iter()
            .map(|boundary| {
                (
                    boundary.nodes.clone(),
                    boundary.marker,
                    boundary.boundary_type,
                )
            })
            .collect::<Vec<_>>();
    if area_relative_error > AREA_TOLERANCE || !markers_preserved {
        return Err(format!(
            "affine motion mismatch: area_error={area_relative_error:.3e}, markers_preserved={markers_preserved}"
        ));
    }
    if (report.min_element_measure_ratio - scale_x).abs() > AREA_TOLERANCE
        || (report.max_element_measure_ratio - scale_x).abs() > AREA_TOLERANCE
    {
        return Err(format!(
            "affine Jacobian ratio mismatch: min={}, max={}, expected={scale_x}",
            report.min_element_measure_ratio, report.max_element_measure_ratio
        ));
    }

    let mut repeated = motion_fixture();
    harmonic_scaled_motion(&repeated, scale_x)?
        .0
        .apply_checked(&mut repeated)
        .map_err(|error| format!("repeat affine motion: {error}"))?;
    let deterministic_repeat = mesh_signature(&mesh)? == mesh_signature(&repeated)?;
    if !deterministic_repeat {
        return Err("motion changed serialized mesh ordering on replay".into());
    }

    let mut rejected = unit_square_triangles(1);
    let before = mesh_signature(&rejected)?;
    let quality_motion = MeshMotion::new(vec![
        [0.0, 0.0, 0.0],
        [-0.9, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    .map_err(|error| format!("construct quality-rejection motion: {error}"))?;
    let thresholds = QualityThresholds {
        min_mean_ratio: 0.5,
        ..QualityThresholds::default()
    };
    let quality_rejection_transactional = matches!(
        quality_motion.apply_checked_with_quality(&mut rejected, thresholds),
        Err(MeshMotionError::Quality(_))
    ) && mesh_signature(&rejected)? == before;
    if !quality_rejection_transactional {
        return Err("quality-barrier rejection was not transactional".into());
    }

    Ok(Metrics {
        area_relative_error,
        moved_nodes: report.moved_nodes,
        max_displacement: report.max_displacement,
        min_mean_ratio: report.min_mean_ratio,
        min_aspect_quality: report.min_aspect_quality,
        min_element_measure_ratio: report.min_element_measure_ratio,
        max_element_measure_ratio: report.max_element_measure_ratio,
        harmonic_iterations,
        harmonic_residual,
        markers_preserved,
        deterministic_repeat,
        quality_rejection_transactional,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-motion-affine-scale-x-1.25")
                .metric("area_relative_error", metrics.area_relative_error)
                .metric("moved_nodes", metrics.moved_nodes as f64)
                .metric("max_displacement", metrics.max_displacement)
                .metric("min_mean_ratio", metrics.min_mean_ratio)
                .metric("min_aspect_quality", metrics.min_aspect_quality)
                .metric(
                    "min_element_measure_ratio",
                    metrics.min_element_measure_ratio,
                )
                .metric(
                    "max_element_measure_ratio",
                    metrics.max_element_measure_ratio,
                )
                .metric("harmonic_iterations", metrics.harmonic_iterations as f64)
                .metric("harmonic_residual", metrics.harmonic_residual)
                .metric(
                    "markers_preserved",
                    if metrics.markers_preserved { 1.0 } else { 0.0 },
                )
                .metric(
                    "deterministic_repeat",
                    if metrics.deterministic_repeat {
                        1.0
                    } else {
                        0.0
                    },
                )
                .metric(
                    "quality_rejection_transactional",
                    if metrics.quality_rejection_transactional {
                        1.0
                    } else {
                        0.0
                    },
                )
                .maximum("area_relative_error", AREA_TOLERANCE)
                .minimum("min_element_measure_ratio", 1.25 - AREA_TOLERANCE)
                .maximum("max_element_measure_ratio", 1.25 + AREA_TOLERANCE)
                .minimum("harmonic_iterations", 1.0)
                .maximum("harmonic_residual", 1.0e-13)
                .minimum("moved_nodes", 1.0)
                .minimum("markers_preserved", 1.0)
                .minimum("deterministic_repeat", 1.0)
                .minimum("quality_rejection_transactional", 1.0)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            let json = result
                .to_json_line()
                .expect("motion QA result must serialize");
            println!("QA_RESULT: {json}");
            println!("SUCCESS");
        }
        Err(error) => {
            let json = QaResult::failed(CASE_ID, &error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("mesher-motion-affine-scale-x-1.25")
                .elapsed_ms(started.elapsed().as_millis() as u64)
                .to_json_line()
                .expect("motion QA failure result must serialize");
            println!("QA_RESULT: {json}");
            eprintln!("MESHER MOTION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
