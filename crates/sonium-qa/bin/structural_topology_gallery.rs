//! Compile-only QA contract for the structural topology before/after gallery.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    FemTopology2DStudy, append_fem_topology_before_after_gallery, run_fem_topology_2d,
};
use sonium_xem::io::{GalleryRenderer, validate_gallery_manifest};

const CASE_ID: &str = "structural.topology-gallery-before-after.v1";

#[derive(Debug, Clone, Copy)]
struct Metrics {
    figures: usize,
    elements: usize,
    mean_absolute_density_change: f64,
    maximum_absolute_density_change: f64,
}

fn run() -> Result<Metrics, String> {
    let mut study = FemTopology2DStudy::introductory();
    study.geometry.nx = 4;
    study.geometry.ny = 1;
    study.volume_fraction = 0.95;
    study.continuation_steps = 1;
    study.iterations_per_stage = 2;
    study.filter_radius_m = 0.02;
    let result = run_fem_topology_2d(&study).map_err(|error| error.to_string())?;
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    let mut renderer = GalleryRenderer::new(directory.path()).map_err(|error| error.to_string())?;
    let summary =
        append_fem_topology_before_after_gallery(&mut renderer, directory.path(), &result)
            .map_err(|error| error.to_string())?;
    let manifest_path = renderer
        .finish(
            "Structural topology before/after gallery",
            vec![
                "fixed top-down element-index view".into(),
                "physical density remains unscaled in source JSON".into(),
            ],
        )
        .map_err(|error| error.to_string())?;
    let manifest = validate_gallery_manifest(&manifest_path).map_err(|error| error.to_string())?;
    if manifest.figures.len() != 4
        || summary.element_count != result.final_physical_density.len()
        || !summary.mean_absolute_density_change.is_finite()
        || !summary.maximum_absolute_density_change.is_finite()
    {
        return Err("topology before/after gallery contract failed".into());
    }
    for filename in [
        "topology-before-after.json",
        "topology-density-initial.png",
        "topology-density-final.png",
        "topology-density-delta.svg",
        "topology-density-volume.svg",
        "gallery.json",
    ] {
        if !directory.path().join(filename).is_file() {
            return Err(format!("missing topology gallery artifact {filename}"));
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
                .metric(
                    "mean_absolute_density_change",
                    metrics.mean_absolute_density_change,
                )
                .metric(
                    "maximum_absolute_density_change",
                    metrics.maximum_absolute_density_change,
                )
                .minimum("figures", 4.0)
                .minimum("elements", 1.0)
                .reference("assembled P1 2-D SIMP topology before/after gallery contract")
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-topology-gallery-before-after-v1",
                )
                .dimension("2-D P1 triangles; initial/final density fields plus delta and mean-density views")
                .formulation("fixed-camera deterministic gallery renderer with source JSON provenance")
                .tolerance("all source and rendered scalar values finite")
                .seed("structural-topology-gallery-before-after:4x1:cpu")
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
                .seed("structural-topology-gallery-before-after:4x1:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL TOPOLOGY GALLERY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
