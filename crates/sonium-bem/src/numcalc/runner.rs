//! Run a Mesh2HRTF project across all frequencies and capture the surface
//! pressure plus optional evaluation-grid pressure for each one.

use std::fs;
use std::path::Path;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::core::bem_solver::{AssemblyMethod, BemError};
use crate::core::incident::IncidentField;
use crate::core::postprocess::pressure::compute_total_field;
use crate::core::solver::frequency_sweep::{FrequencySweepConfig, frequency_sweep};
use crate::core::types::{BasisOrder, ElementProperty};

use super::project::NumCalcProject;

/// One row per frequency × eval-point complex pressure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepOutput {
    /// Solve frequencies (Hz).
    pub frequencies: Vec<f64>,
    /// Speed of sound (m/s) used for the sweep.
    pub speed_of_sound: f64,
    /// Air density (kg/m³) used for the sweep.
    pub density: f64,
    /// Number of surface DOFs.
    pub num_surface_dofs: usize,
    /// Per-frequency surface pressure (`frequencies.len()` × `num_surface_dofs`),
    /// flattened row-major. Each entry stored as `[re, im]`.
    pub surface_pressure: Vec<[f64; 2]>,
    /// Evaluation-grid coordinates (`n_eval` × 3), flattened row-major.
    pub eval_points: Vec<f64>,
    /// Per-frequency pressure on the evaluation grid (`frequencies.len()` ×
    /// `n_eval`), flattened row-major; empty if there is no evaluation grid.
    pub eval_pressure: Vec<[f64; 2]>,
}

impl SweepOutput {
    /// Write the output as JSON next to the project.
    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, s)
    }
}

/// Run the BEM frequency sweep described by `project`, optionally restricted
/// to `[freq_start_idx, freq_end_idx)` of the frequency curve. Both indices
/// are inclusive on the lower bound and exclusive on the upper bound; pass
/// `None` to run the full sweep.
pub fn run_project(
    project: &NumCalcProject,
    freq_start_idx: Option<usize>,
    freq_end_idx: Option<usize>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<SweepOutput, BemError> {
    let total = project.frequencies.len();
    let start = freq_start_idx.unwrap_or(0).min(total);
    let end = freq_end_idx.unwrap_or(total).min(total).max(start);
    let frequencies: Vec<f64> = project.frequencies[start..end].to_vec();

    let surface_elements: Vec<_> = project
        .elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .cloned()
        .collect();

    let incident = pick_incident(project);

    let config = FrequencySweepConfig {
        frequencies: frequencies.clone(),
        tolerance,
        max_iterations,
        recycle_solution: true,
        gmres_restart: 30,
        beta_scale: 4.0,
        recycle_subspace_size: 0,
        max_recycle_relative_frequency_delta: 0.1,
    };

    let sweep = frequency_sweep(
        &surface_elements,
        &project.nodes,
        &incident,
        &config,
        AssemblyMethod::Tbem,
        project.speed_of_sound,
        project.density,
    )?;

    let n_dof = sweep.solutions.first().map(|s| s.len()).unwrap_or(0);
    let mut surface_flat: Vec<[f64; 2]> = Vec::with_capacity(sweep.solutions.len() * n_dof.max(1));
    for sol in &sweep.solutions {
        for &c in sol {
            surface_flat.push([c.re, c.im]);
        }
    }

    let eval_points = project.evaluation_centers();
    let eval_flat = flatten_eval_points(&eval_points);
    let eval_pressure = if project.evaluation_indices.is_empty() {
        Vec::new()
    } else {
        eval_pressure_per_frequency(
            project,
            &frequencies,
            &incident,
            &surface_elements,
            &sweep.solutions,
            &eval_points,
        )
    };

    Ok(SweepOutput {
        frequencies,
        speed_of_sound: project.speed_of_sound,
        density: project.density,
        num_surface_dofs: n_dof,
        surface_pressure: surface_flat,
        eval_points: eval_flat,
        eval_pressure,
    })
}

fn flatten_eval_points(pts: &[[f64; 3]]) -> Vec<f64> {
    pts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn eval_pressure_per_frequency(
    project: &NumCalcProject,
    frequencies: &[f64],
    incident: &IncidentField,
    surface_elements: &[crate::core::types::Element],
    solutions: &[Vec<Complex64>],
    eval_points: &[[f64; 3]],
) -> Vec<[f64; 2]> {
    debug_assert_eq!(solutions.len(), frequencies.len());
    let mut out = Vec::with_capacity(solutions.len() * eval_points.len());
    for (i, surface_pressure) in solutions.iter().enumerate() {
        let physics = project.physics_at(frequencies[i]);
        let field = compute_total_field(
            eval_points,
            surface_elements,
            &project.nodes,
            surface_pressure,
            None,
            incident,
            &physics,
            BasisOrder::Constant,
        );
        for fp in field {
            out.push([fp.p_total.re, fp.p_total.im]);
        }
    }
    out
}

fn pick_incident(project: &NumCalcProject) -> IncidentField {
    if let Some(pw) = project.plane_waves.first() {
        IncidentField::PlaneWave {
            direction: pw.direction,
            amplitude: Complex64::new(pw.amplitude_re, pw.amplitude_im),
        }
    } else if let Some(ps) = project.point_sources.first() {
        IncidentField::PointSource {
            position: ps.position,
            strength: Complex64::new(ps.amplitude_re, ps.amplitude_im),
        }
    } else {
        // No external incident field — only velocity BCs drive the system.
        IncidentField::PlaneWave {
            direction: [0.0, 0.0, 1.0],
            amplitude: Complex64::new(0.0, 0.0),
        }
    }
}

// Convenience re-export so downstream callers don't need to know about the
// internal element type when filtering by property.
#[allow(dead_code)]
pub(crate) fn is_surface(p: ElementProperty) -> bool {
    !p.is_evaluation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numcalc::load_project;
    use std::path::PathBuf;

    fn sample_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample_project/NC.inp")
    }

    #[test]
    fn run_project_partial_range_uses_correct_frequencies() {
        let project = load_project(sample_path()).unwrap();
        // Sample has 3 frequencies. Solve only [1, 3) → indices 1 and 2.
        let out = run_project(&project, Some(1), Some(3), 50, 1e-4).unwrap();
        assert_eq!(out.frequencies.len(), 2);
        assert_eq!(out.frequencies, project.frequencies[1..3]);
    }

    #[test]
    fn run_project_clamps_out_of_range_indices() {
        let project = load_project(sample_path()).unwrap();
        // start > total → empty sweep, no panic.
        let out = run_project(&project, Some(99), Some(200), 50, 1e-4).unwrap();
        assert!(out.frequencies.is_empty());
        assert!(out.surface_pressure.is_empty());
    }

    #[test]
    fn run_project_inverted_range_returns_empty() {
        let project = load_project(sample_path()).unwrap();
        // end < start should be clamped, not panic.
        let out = run_project(&project, Some(2), Some(0), 50, 1e-4).unwrap();
        assert!(out.frequencies.is_empty());
    }
}
