//! Production-backed 3-D solid static mesh-convergence evidence.
//!
//! The report runs the existing `Solid3DStudy` static path at a caller's
//! coarse resolution and a uniformly refined resolution. Displacement and
//! strain energy are the hard convergence quantities. Element-centroid stress
//! is retained as a diagnostic under the crate's stress-singularity policy and
//! is never silently promoted to a point/edge peak-stress gate.

use crate::{
    Solid3DGeometry, Solid3DStaticResult, Solid3DStudy, StructuralMechanicsError,
    run_solid3d_static,
};
use serde::{Deserialize, Serialize};

/// Schema version for the 3-D solid static convergence report.
pub const SOLID3D_STATIC_CONVERGENCE_SCHEMA_VERSION: u32 = 1;

/// Coarse/refined production-solid convergence evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DStaticMeshConvergenceReport {
    pub schema_version: u32,
    pub coarse: Solid3DStaticResult,
    pub refined: Solid3DStaticResult,
    pub coarse_resolution: [usize; 3],
    pub refined_resolution: [usize; 3],
    pub displacement_relative_change: f64,
    pub energy_relative_change: f64,
    pub centroid_stress_relative_change: f64,
    pub centroid_stress_converged: bool,
    pub topology_refined: bool,
    pub tolerance: f64,
    pub converged: bool,
}

/// Run the production 3-D solid static solve on uniformly refined meshes.
pub fn run_solid3d_static_mesh_convergence(
    study: &Solid3DStudy,
    tolerance: f64,
) -> Result<Solid3DStaticMeshConvergenceReport, StructuralMechanicsError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "solid3d.mesh_convergence_tolerance",
            value: tolerance.to_string(),
        });
    }

    let coarse_resolution = [study.geometry.nx, study.geometry.ny, study.geometry.nz];
    let refined_resolution = [
        refine_resolution(study.geometry.nx, "nx")?,
        refine_resolution(study.geometry.ny, "ny")?,
        refine_resolution(study.geometry.nz, "nz")?,
    ];
    let coarse = run_solid3d_static(study)?;
    let mut refined_study = study.clone();
    refined_study.geometry = Solid3DGeometry {
        nx: refined_resolution[0],
        ny: refined_resolution[1],
        nz: refined_resolution[2],
        ..study.geometry
    };
    let refined = run_solid3d_static(&refined_study)?;

    let displacement_relative_change =
        relative_change(refined.max_displacement_m, coarse.max_displacement_m);
    let energy_relative_change = relative_change(refined.strain_energy_j, coarse.strain_energy_j);
    let centroid_stress_relative_change =
        relative_change(refined.max_von_mises_pa, coarse.max_von_mises_pa);
    let topology_refined = refined.nodes > coarse.nodes && refined.elements > coarse.elements;
    let centroid_stress_converged = centroid_stress_relative_change <= tolerance;
    let converged = coarse.converged
        && refined.converged
        && topology_refined
        && displacement_relative_change <= tolerance
        && energy_relative_change <= tolerance;

    Ok(Solid3DStaticMeshConvergenceReport {
        schema_version: SOLID3D_STATIC_CONVERGENCE_SCHEMA_VERSION,
        coarse,
        refined,
        coarse_resolution,
        refined_resolution,
        displacement_relative_change,
        energy_relative_change,
        centroid_stress_relative_change,
        centroid_stress_converged,
        topology_refined,
        tolerance,
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

fn relative_change(refined: f64, coarse: f64) -> f64 {
    (refined - coarse).abs() / refined.abs().max(coarse.abs()).max(1.0e-30)
}
