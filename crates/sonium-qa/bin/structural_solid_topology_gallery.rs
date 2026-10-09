//! Compile-only QA contract for the 3-D solid topology before/after gallery.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    Solid3DTopologyStudy, append_solid3d_topology_before_after_gallery, run_solid3d_topology,
};
use sonium_xem::io::{GalleryRenderer, validate_gallery_manifest};

const CASE_ID: &str = "structural.solid-topology-gallery-before-after.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    figures: usize,
    elements: usize,
    mean_absolute_density_change: f64,
    maximum_absolute_density_change: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = Solid3DTopologyStudy::introductory();
    study.base.geometry.nx = 2;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.volume_fraction = 0.90;
    study.iterations = 2;
    study.allowable_stress_pa = 1.0e14;

    let result = run_solid3d_topology(&study).map_err(|error| error.to_string())?;
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let mut renderer = GalleryRenderer::new(directory.path()).map_err(|error| error.to_string())?;
    let summary =
        append_solid3d_topology_before_after_gallery(&mut renderer, directory.path(), &result)
            .map_err(|error| error.to_string())?;
    let manifest_path = renderer
        .finish(
            "Structural 3-D topology before/after gallery",
            vec![
                "fixed element-index density view".into(),
                "source JSON preserves physical density values".into(),
            ],
        )
        .map_err(|error| error.to_string())?;
    let manifest = validate_gallery_manifest(&manifest_path).map_err(|error| error.to_string())?;

    if manifest.figures.len() != 4
        || summary.element_count != result.final_density.len()
        || !summary.mean_absolute_density_change.is_finite()
        || !summary.maximum_absolute_density_change.is_finite()
    {
        return Err("3-D topology before/after gallery contract failed".into());
    }
    for filename in [
        "solid3d-topology-before-after.json",
        "solid3d-density-initial.png",
        "solid3d-density-final.png",
        "solid3d-density-delta.svg",
        "solid3d-density-volume.svg",
        "gallery.json",
    ] {
        if !directory.path().join(filename).is_file() {
            return Err(format!("missing 3-D topology gallery artifact {filename}"));
        }
    }

    Ok(Metrics {
        figures: manifest.figures.len(),
        elements: summary.element_count,
        mean_absolute_density_change: summary.mean_absolute_density_change,
        maximum_absolute_density_change: summary.maximum_absolute_density_change,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("figures", metrics.figures as f64)
                .metric("elements", metrics.elements as f64)
                .metric("mean_absolute_density_change", metrics.mean_absolute_density_change)
                .metric(
                    "maximum_absolute_density_change",
                    metrics.maximum_absolute_density_change,
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-solid-topology-gallery-before-after-v1",
                )
                .dimension("3-D fixed element-index initial/final density")
                .formulation("SIMP before/after source JSON plus PNG/SVG artifacts")
                .seed("structural-solid-topology-gallery-before-after:2x1x1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("structural-solid-topology-gallery-before-after:2x1x1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL SOLID TOPOLOGY GALLERY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
