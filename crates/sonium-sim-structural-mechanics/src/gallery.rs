//! Reusable, checked gallery contracts for structural design results.

use crate::{FemTopology2DResult, Solid3DTopologyResult, StructuralMechanicsError};
use serde::{Deserialize, Serialize};
use sonium_xem::io::GalleryRenderer;
use std::path::Path;

pub const FEM_TOPOLOGY_GALLERY_SCHEMA_VERSION: u32 = 1;
pub const SOLID3D_TOPOLOGY_GALLERY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FemTopologyBeforeAfterSummary {
    pub schema_version: u32,
    pub element_count: usize,
    pub grid_columns: usize,
    pub objective: String,
    pub initial_mean_density: f64,
    pub final_mean_density: f64,
    pub mean_absolute_density_change: f64,
    pub maximum_absolute_density_change: f64,
}

/// Append deterministic before/after topology figures and their checked source
/// artifact to an existing gallery renderer.
///
/// The function deliberately receives a completed public topology result: the
/// renderer cannot silently invent an initial field or claim a design change
/// when the solver did not retain one. The resulting manifest entries are
/// suitable for website/gallery consumers, while the summary remains a compact
/// machine-readable provenance boundary.
pub fn append_fem_topology_before_after_gallery(
    gallery: &mut GalleryRenderer,
    output_dir: impl AsRef<Path>,
    result: &FemTopology2DResult,
) -> Result<FemTopologyBeforeAfterSummary, StructuralMechanicsError> {
    let initial = &result.initial_physical_density;
    let final_density = &result.final_physical_density;
    if initial.is_empty()
        || initial.len() != final_density.len()
        || initial
            .iter()
            .chain(final_density)
            .any(|value| !value.is_finite())
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "fem_topology_gallery.density_fields",
            value: initial.len().max(final_density.len()) as f64,
        });
    }

    let delta: Vec<f64> = final_density
        .iter()
        .zip(initial)
        .map(|(final_value, initial_value)| final_value - initial_value)
        .collect();
    let initial_mean_density = mean(initial);
    let final_mean_density = mean(final_density);
    let mean_absolute_density_change =
        delta.iter().map(|value| value.abs()).sum::<f64>() / delta.len() as f64;
    let maximum_absolute_density_change = delta.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let grid_columns = result.study.geometry.nx.saturating_mul(2);
    let grid_columns = if grid_columns > 0 && initial.len().is_multiple_of(grid_columns) {
        grid_columns
    } else {
        0
    };
    let summary = FemTopologyBeforeAfterSummary {
        schema_version: FEM_TOPOLOGY_GALLERY_SCHEMA_VERSION,
        element_count: initial.len(),
        grid_columns,
        objective: format!("{:?}", result.objective),
        initial_mean_density,
        final_mean_density,
        mean_absolute_density_change,
        maximum_absolute_density_change,
    };
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir).map_err(|error| gallery_io_error("create", error))?;
    let source_path = output_dir.join("topology-before-after.json");
    let source_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(|error| gallery_io_error("serialize", error))?;
    std::fs::write(&source_path, source_bytes).map_err(|error| gallery_io_error("write", error))?;
    let source = "topology-before-after.json";
    let caption_prefix = "Fixed top-down element-index view; physical density";
    if grid_columns > 0 {
        gallery
            .grid_heatmap_png(
                "topology-density-initial.png",
                initial,
                grid_columns,
                source,
                format!("{caption_prefix} before optimization"),
            )
            .map_err(|error| gallery_io_error("render initial", error))?;
        gallery
            .grid_heatmap_png(
                "topology-density-final.png",
                final_density,
                grid_columns,
                source,
                format!("{caption_prefix} after optimization"),
            )
            .map_err(|error| gallery_io_error("render final", error))?;
    } else {
        gallery
            .heatmap_png(
                "topology-density-initial.png",
                initial,
                source,
                format!("{caption_prefix} before optimization"),
            )
            .map_err(|error| gallery_io_error("render initial", error))?;
        gallery
            .heatmap_png(
                "topology-density-final.png",
                final_density,
                source,
                format!("{caption_prefix} after optimization"),
            )
            .map_err(|error| gallery_io_error("render final", error))?;
    }
    let x: Vec<f64> = (0..delta.len()).map(|index| index as f64).collect();
    gallery
        .line_svg(
            "topology-density-delta.svg",
            &x,
            &delta,
            "Topology density change",
            source,
            "Final minus initial physical density by element index",
        )
        .map_err(|error| gallery_io_error("render delta", error))?;
    gallery
        .line_svg(
            "topology-density-volume.svg",
            &[0.0, 1.0],
            &[initial_mean_density, final_mean_density],
            "Topology mean density before and after",
            source,
            "Mean physical density at the initial and final design states",
        )
        .map_err(|error| gallery_io_error("render volume", error))?;
    Ok(summary)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Solid3DTopologyBeforeAfterSummary {
    pub schema_version: u32,
    pub element_count: usize,
    pub objective: String,
    pub initial_mean_density: f64,
    pub final_mean_density: f64,
    pub mean_absolute_density_change: f64,
    pub maximum_absolute_density_change: f64,
}

/// Append deterministic before/after figures for the retained 3-D physical
/// density fields. These are fixed element-index views, rather than a claim
/// of spatial 3-D rendering; the source JSON keeps the mapping auditable.
pub fn append_solid3d_topology_before_after_gallery(
    gallery: &mut GalleryRenderer,
    output_dir: impl AsRef<Path>,
    result: &Solid3DTopologyResult,
) -> Result<Solid3DTopologyBeforeAfterSummary, StructuralMechanicsError> {
    let initial = &result.initial_density;
    let final_density = &result.final_density;
    if initial.is_empty()
        || initial.len() != final_density.len()
        || initial
            .iter()
            .chain(final_density)
            .any(|value| !value.is_finite())
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "solid3d_topology_gallery.density_fields",
            value: initial.len().max(final_density.len()) as f64,
        });
    }

    let delta: Vec<f64> = final_density
        .iter()
        .zip(initial)
        .map(|(final_value, initial_value)| final_value - initial_value)
        .collect();
    let initial_mean_density = mean(initial);
    let final_mean_density = mean(final_density);
    let summary = Solid3DTopologyBeforeAfterSummary {
        schema_version: SOLID3D_TOPOLOGY_GALLERY_SCHEMA_VERSION,
        element_count: initial.len(),
        objective: format!("{:?}", result.objective),
        initial_mean_density,
        final_mean_density,
        mean_absolute_density_change: delta.iter().map(|value| value.abs()).sum::<f64>()
            / delta.len() as f64,
        maximum_absolute_density_change: delta.iter().map(|value| value.abs()).fold(0.0, f64::max),
    };

    let output_dir = output_dir.as_ref();
    let source_path = output_dir.join("solid3d-topology-before-after.json");
    let source_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(|error| gallery_io_error("serialize 3-D summary", error))?;
    std::fs::write(&source_path, source_bytes)
        .map_err(|error| gallery_io_error("write 3-D summary", error))?;
    let source = "solid3d-topology-before-after.json";
    let caption_prefix = "Fixed element-index view; physical density";

    gallery
        .heatmap_png(
            "solid3d-density-initial.png",
            initial,
            source,
            format!("{caption_prefix} before optimization"),
        )
        .map_err(|error| gallery_io_error("render 3-D initial", error))?;
    gallery
        .heatmap_png(
            "solid3d-density-final.png",
            final_density,
            source,
            format!("{caption_prefix} after optimization"),
        )
        .map_err(|error| gallery_io_error("render 3-D final", error))?;

    let x: Vec<f64> = (0..delta.len()).map(|index| index as f64).collect();
    gallery
        .line_svg(
            "solid3d-density-delta.svg",
            &x,
            &delta,
            "3-D topology density change",
            source,
            "Final minus initial physical density by element index",
        )
        .map_err(|error| gallery_io_error("render 3-D delta", error))?;
    gallery
        .line_svg(
            "solid3d-density-volume.svg",
            &[0.0, 1.0],
            &[initial_mean_density, final_mean_density],
            "3-D topology mean density before and after",
            source,
            "Mean physical density at initial and final design states",
        )
        .map_err(|error| gallery_io_error("render 3-D volume", error))?;

    Ok(summary)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn gallery_io_error(
    stage: &'static str,
    error: impl std::fmt::Display,
) -> StructuralMechanicsError {
    StructuralMechanicsError::SolverFailure {
        stage,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FemTopology2DStudy, Solid3DTopologyStudy, run_fem_topology_2d, run_solid3d_topology,
    };
    use tempfile::tempdir;

    #[test]
    fn topology_gallery_records_checked_before_after_artifacts() {
        let mut study = FemTopology2DStudy::introductory();
        study.geometry.nx = 4;
        study.geometry.ny = 1;
        study.volume_fraction = 0.95;
        study.continuation_steps = 1;
        study.iterations_per_stage = 2;
        study.filter_radius_m = 0.02;
        let result = run_fem_topology_2d(&study).expect("small topology result");
        let directory = tempdir().expect("temporary gallery directory");
        let mut renderer = GalleryRenderer::new(directory.path()).expect("gallery renderer");
        let summary =
            append_fem_topology_before_after_gallery(&mut renderer, directory.path(), &result)
                .expect("before/after gallery");
        let manifest = renderer
            .finish("Topology before/after", vec!["fixed top-down view".into()])
            .expect("gallery manifest");
        assert_eq!(summary.element_count, result.final_physical_density.len());
        assert!(summary.maximum_absolute_density_change.is_finite());
        assert!(manifest.is_file());
        for filename in [
            "topology-before-after.json",
            "topology-density-initial.png",
            "topology-density-final.png",
            "topology-density-delta.svg",
            "topology-density-volume.svg",
            "gallery.json",
        ] {
            assert!(
                directory.path().join(filename).is_file(),
                "missing {filename}"
            );
        }
    }

    #[test]
    fn topology_gallery_rejects_missing_initial_density() {
        let mut study = FemTopology2DStudy::introductory();
        study.geometry.nx = 4;
        study.geometry.ny = 1;
        study.volume_fraction = 0.95;
        study.continuation_steps = 1;
        study.iterations_per_stage = 2;
        study.filter_radius_m = 0.02;
        let result = FemTopology2DResult {
            initial_physical_density: Vec::new(),
            ..run_fem_topology_2d(&study).expect("topology result")
        };
        let directory = tempdir().expect("temporary gallery directory");
        let mut renderer = GalleryRenderer::new(directory.path()).expect("gallery renderer");
        assert!(matches!(
            append_fem_topology_before_after_gallery(&mut renderer, directory.path(), &result),
            Err(StructuralMechanicsError::ValidationFailure {
                check: "fem_topology_gallery.density_fields",
                ..
            })
        ));
    }

    #[test]
    fn solid3d_topology_gallery_records_checked_before_after_artifacts() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.volume_fraction = 0.90;
        study.iterations = 2;
        let result = run_solid3d_topology(&study).expect("small 3-D topology result");
        let directory = tempdir().expect("temporary gallery directory");
        let mut renderer = GalleryRenderer::new(directory.path()).expect("gallery renderer");
        let summary =
            append_solid3d_topology_before_after_gallery(&mut renderer, directory.path(), &result)
                .expect("3-D before/after gallery");
        let manifest_path = renderer
            .finish(
                "Structural 3-D topology before/after gallery",
                vec!["fixed element-index density view".into()],
            )
            .expect("gallery manifest");
        let manifest = sonium_xem::io::validate_gallery_manifest(&manifest_path)
            .expect("valid gallery manifest");

        assert_eq!(manifest.figures.len(), 4);
        assert_eq!(summary.element_count, result.final_density.len());
        assert!(summary.initial_mean_density.is_finite());
        assert!(summary.final_mean_density.is_finite());
        for filename in [
            "solid3d-topology-before-after.json",
            "solid3d-density-initial.png",
            "solid3d-density-final.png",
            "solid3d-density-delta.svg",
            "solid3d-density-volume.svg",
            "gallery.json",
        ] {
            assert!(
                directory.path().join(filename).is_file(),
                "missing {filename}"
            );
        }
    }

    #[test]
    fn solid3d_topology_gallery_rejects_missing_initial_density() {
        let mut study = Solid3DTopologyStudy::introductory();
        study.base.geometry.nx = 2;
        study.base.geometry.ny = 1;
        study.base.geometry.nz = 1;
        study.volume_fraction = 0.90;
        study.iterations = 1;
        let result = Solid3DTopologyResult {
            initial_density: Vec::new(),
            ..run_solid3d_topology(&study).expect("small 3-D topology result")
        };
        let directory = tempdir().expect("temporary gallery directory");
        let mut renderer = GalleryRenderer::new(directory.path()).expect("gallery renderer");
        assert!(matches!(
            append_solid3d_topology_before_after_gallery(&mut renderer, directory.path(), &result,),
            Err(StructuralMechanicsError::ValidationFailure {
                check: "solid3d_topology_gallery.density_fields",
                ..
            })
        ));
    }
}
