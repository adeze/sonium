//! Production-backed 3-D solid harmonic mesh-convergence evidence.
//!
//! The report runs the direct assembled P1 solid harmonic path on a coarse
//! mesh and a uniformly refined mesh. The observation DOF is resolved by its
//! physical coordinate on the refined mesh, so refinement cannot silently
//! compare different node locations.

use crate::{
    Solid3DGeometry, Solid3DHarmonicResponseResult, Solid3DHarmonicResponseStudy,
    StructuralMechanicsError, run_solid3d_harmonic_response,
};
use serde::{Deserialize, Serialize};
use sonium_fem::mesh::Mesh;

/// Schema version for the 3-D solid harmonic convergence report.
pub const SOLID3D_HARMONIC_CONVERGENCE_SCHEMA_VERSION: u32 = 1;

/// Coarse/refined direct-solid harmonic convergence evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DHarmonicMeshConvergenceReport {
    pub schema_version: u32,
    pub coarse: Solid3DHarmonicResponseResult,
    pub refined: Solid3DHarmonicResponseResult,
    pub coarse_resolution: [usize; 3],
    pub refined_resolution: [usize; 3],
    pub coarse_observation_node: usize,
    pub refined_observation_node: usize,
    pub observation_coordinate_m: [f64; 3],
    pub displacement_relative_changes: Vec<f64>,
    pub phase_changes_rad: Vec<f64>,
    pub max_displacement_relative_change: f64,
    pub max_phase_change_rad: f64,
    pub peak_displacement_relative_change: f64,
    pub peak_frequency_relative_change: f64,
    pub topology_refined: bool,
    pub response_tolerance: f64,
    pub phase_tolerance_rad: f64,
    pub converged: bool,
}

/// Run direct assembled 3-D harmonic response on uniformly refined meshes.
pub fn run_solid3d_harmonic_mesh_convergence(
    study: &Solid3DHarmonicResponseStudy,
    response_tolerance: f64,
    phase_tolerance_rad: f64,
) -> Result<Solid3DHarmonicMeshConvergenceReport, StructuralMechanicsError> {
    if !response_tolerance.is_finite() || response_tolerance <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d.harmonic_convergence.response_tolerance",
            value: response_tolerance.to_string(),
        });
    }
    if !phase_tolerance_rad.is_finite()
        || phase_tolerance_rad <= 0.0
        || phase_tolerance_rad > std::f64::consts::PI
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d.harmonic_convergence.phase_tolerance_rad",
            value: phase_tolerance_rad.to_string(),
        });
    }

    let coarse_resolution = [
        study.base.geometry.nx,
        study.base.geometry.ny,
        study.base.geometry.nz,
    ];
    let refined_resolution = [
        refine_resolution(study.base.geometry.nx, "nx")?,
        refine_resolution(study.base.geometry.ny, "ny")?,
        refine_resolution(study.base.geometry.nz, "nz")?,
    ];
    let coarse_mesh = study.base.mesh()?;
    let observation_coordinate =
        coarse_mesh
            .nodes
            .get(study.observation_node)
            .ok_or_else(|| StructuralMechanicsError::InvalidConfiguration {
                field: "solid3d_harmonic_response.observation_node",
                value: study.observation_node.to_string(),
            })?;
    let observation_coordinate_m = [
        observation_coordinate.x,
        observation_coordinate.y,
        observation_coordinate.z,
    ];

    let coarse = run_solid3d_harmonic_response(study)?;
    let mut refined_study = study.clone();
    refined_study.base.geometry = Solid3DGeometry {
        nx: refined_resolution[0],
        ny: refined_resolution[1],
        nz: refined_resolution[2],
        ..study.base.geometry
    };
    let refined_mesh = refined_study.base.mesh()?;
    refined_study.observation_node = resolve_observation_node(
        &refined_mesh,
        observation_coordinate_m,
        refined_study.base.geometry,
    )?;
    let refined_observation_node = refined_study.observation_node;
    let refined = run_solid3d_harmonic_response(&refined_study)?;

    if coarse.points.len() != refined.points.len() {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "solid-3d harmonic convergence alignment",
            message: "coarse and refined frequency result counts differ".to_string(),
        });
    }

    let mut displacement_relative_changes = Vec::with_capacity(coarse.points.len());
    let mut phase_changes_rad = Vec::with_capacity(coarse.points.len());
    for (coarse_point, refined_point) in coarse.points.iter().zip(&refined.points) {
        if coarse_point.frequency_hz != refined_point.frequency_hz {
            return Err(StructuralMechanicsError::SolverFailure {
                stage: "solid-3d harmonic convergence alignment",
                message: "coarse and refined frequency grids differ".to_string(),
            });
        }
        let displacement_change = relative_change(
            refined_point.displacement_abs_m,
            coarse_point.displacement_abs_m,
        );
        let phase_change =
            principal_phase_distance(refined_point.phase_lag_rad, coarse_point.phase_lag_rad);
        if !displacement_change.is_finite() || !phase_change.is_finite() {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "solid3d_harmonic_convergence.finite_response_changes",
                value: displacement_change.max(phase_change),
            });
        }
        displacement_relative_changes.push(displacement_change);
        phase_changes_rad.push(phase_change);
    }

    let max_displacement_relative_change = displacement_relative_changes
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let max_phase_change_rad = phase_changes_rad.iter().copied().fold(0.0, f64::max);
    let peak_displacement_relative_change = relative_change(
        refined.peak_displacement_abs_m,
        coarse.peak_displacement_abs_m,
    );
    let peak_frequency_relative_change =
        relative_change(refined.peak_frequency_hz, coarse.peak_frequency_hz);
    let topology_refined = refined.nodes > coarse.nodes && refined.elements > coarse.elements;
    let converged = coarse.converged
        && refined.converged
        && topology_refined
        && max_displacement_relative_change <= response_tolerance
        && peak_displacement_relative_change <= response_tolerance
        && peak_frequency_relative_change <= response_tolerance
        && max_phase_change_rad <= phase_tolerance_rad;

    Ok(Solid3DHarmonicMeshConvergenceReport {
        schema_version: SOLID3D_HARMONIC_CONVERGENCE_SCHEMA_VERSION,
        coarse,
        refined,
        coarse_resolution,
        refined_resolution,
        coarse_observation_node: study.observation_node,
        refined_observation_node,
        observation_coordinate_m,
        displacement_relative_changes,
        phase_changes_rad,
        max_displacement_relative_change,
        max_phase_change_rad,
        peak_displacement_relative_change,
        peak_frequency_relative_change,
        topology_refined,
        response_tolerance,
        phase_tolerance_rad,
        converged,
    })
}

fn refine_resolution(value: usize, field: &'static str) -> Result<usize, StructuralMechanicsError> {
    value
        .checked_mul(2)
        .ok_or_else(|| StructuralMechanicsError::InvalidConfiguration {
            field,
            value: "uniform refinement overflows usize".to_string(),
        })
}

fn resolve_observation_node(
    mesh: &Mesh,
    coordinate: [f64; 3],
    geometry: Solid3DGeometry,
) -> Result<usize, StructuralMechanicsError> {
    let tolerance = 1.0e-10
        * geometry
            .length_m
            .max(geometry.width_m)
            .max(geometry.height_m)
            .max(1.0);
    let mut best: Option<(usize, f64)> = None;
    for (index, point) in mesh.nodes.iter().enumerate() {
        let distance_squared = (point.x - coordinate[0]).powi(2)
            + (point.y - coordinate[1]).powi(2)
            + (point.z - coordinate[2]).powi(2);
        if !distance_squared.is_finite() {
            continue;
        }
        if best
            .map(|(_, best_distance)| distance_squared < best_distance)
            .unwrap_or(true)
        {
            best = Some((index, distance_squared));
        }
    }
    let Some((index, distance_squared)) = best else {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "refined harmonic mesh has no observation nodes".to_string(),
        });
    };
    if distance_squared.sqrt() > tolerance {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "refined harmonic mesh does not retain the coarse observation coordinate"
                .to_string(),
        });
    }
    Ok(index)
}

fn relative_change(refined: f64, coarse: f64) -> f64 {
    (refined - coarse).abs() / refined.abs().max(coarse.abs()).max(1.0e-30)
}

fn principal_phase_distance(refined: f64, coarse: f64) -> f64 {
    let delta = refined - coarse;
    delta.sin().atan2(delta.cos()).abs()
}
