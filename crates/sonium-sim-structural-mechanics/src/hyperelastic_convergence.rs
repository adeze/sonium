//! Production-backed 3-D hyperelastic mesh-convergence evidence.
//!
//! This wrapper runs the existing displacement-controlled FEM/Newton
//! hyperelastic path at a coarse resolution and a uniformly refined
//! resolution. It is a nonlinear spatial-convergence contract, not a claim of
//! transient, post-buckling, fracture, or product-scale material validation.

use crate::{
    Hyperelastic3DResult, Hyperelastic3DStudy, StructuralMechanicsError, run_hyperelastic_3d,
};
use serde::{Deserialize, Serialize};

/// Schema version for the 3-D hyperelastic convergence report.
pub const HYPERELASTIC_3D_CONVERGENCE_SCHEMA_VERSION: u32 = 1;

/// Coarse/refined convergence evidence for the actual 3-D hyperelastic FEM path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyperelastic3DMeshConvergenceReport {
    pub schema_version: u32,
    pub coarse: Hyperelastic3DResult,
    pub refined: Hyperelastic3DResult,
    pub coarse_resolution: [usize; 3],
    pub refined_resolution: [usize; 3],
    pub relative_strain_energy_change: f64,
    pub relative_displacement_change: f64,
    pub refined_affine_displacement_error_m: f64,
    pub refined_deformation_gradient_error: f64,
    pub refined_constitutive_gradient_error: f64,
    pub topology_refined: bool,
    pub response_tolerance: f64,
    pub field_tolerance_m: f64,
    pub gradient_tolerance: f64,
    pub converged: bool,
}

/// Run the actual 3-D hyperelastic FEM/Newton path on coarse/refined meshes.
pub fn run_hyperelastic_3d_mesh_convergence(
    study: &Hyperelastic3DStudy,
    response_tolerance: f64,
    field_tolerance_m: f64,
    gradient_tolerance: f64,
) -> Result<Hyperelastic3DMeshConvergenceReport, StructuralMechanicsError> {
    validate_tolerance(
        response_tolerance,
        "hyperelastic3d.mesh_convergence.response_tolerance",
    )?;
    validate_tolerance(
        field_tolerance_m,
        "hyperelastic3d.mesh_convergence.field_tolerance_m",
    )?;
    validate_tolerance(
        gradient_tolerance,
        "hyperelastic3d.mesh_convergence.gradient_tolerance",
    )?;

    let coarse_resolution = [study.nx, study.ny, study.nz];
    let refined_resolution = [
        refine_resolution(study.nx, "nx")?,
        refine_resolution(study.ny, "ny")?,
        refine_resolution(study.nz, "nz")?,
    ];
    let coarse = run_hyperelastic_3d(study)?;
    let mut refined_study = *study;
    refined_study.nx = refined_resolution[0];
    refined_study.ny = refined_resolution[1];
    refined_study.nz = refined_resolution[2];
    let refined = run_hyperelastic_3d(&refined_study)?;

    let relative_strain_energy_change =
        relative_change(refined.strain_energy_j, coarse.strain_energy_j);
    let relative_displacement_change =
        relative_change(refined.max_displacement_m, coarse.max_displacement_m);
    let refined_affine_displacement_error_m = refined.max_affine_displacement_error_m;
    let refined_deformation_gradient_error = refined.deformation_gradient_error;
    let refined_constitutive_gradient_error = refined.constitutive_gradient_error;
    let topology_refined = refined.mesh_nodes > coarse.mesh_nodes
        && refined.elements > coarse.elements
        && refined.dofs > coarse.dofs;
    let converged = coarse.converged
        && refined.converged
        && topology_refined
        && relative_strain_energy_change <= response_tolerance
        && relative_displacement_change <= response_tolerance
        && refined_affine_displacement_error_m <= field_tolerance_m
        && refined_deformation_gradient_error <= gradient_tolerance
        && refined_constitutive_gradient_error <= gradient_tolerance;

    Ok(Hyperelastic3DMeshConvergenceReport {
        schema_version: HYPERELASTIC_3D_CONVERGENCE_SCHEMA_VERSION,
        coarse,
        refined,
        coarse_resolution,
        refined_resolution,
        relative_strain_energy_change,
        relative_displacement_change,
        refined_affine_displacement_error_m,
        refined_deformation_gradient_error,
        refined_constitutive_gradient_error,
        topology_refined,
        response_tolerance,
        field_tolerance_m,
        gradient_tolerance,
        converged,
    })
}

fn validate_tolerance(value: f64, field: &'static str) -> Result<(), StructuralMechanicsError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
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
