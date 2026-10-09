//! Release-backed native RoomSim gallery artifact gate.

use sonium_qa::QaResult;
use sonium_sim_room::{gallery, hf_extension, scenarios};
use std::path::Path;

const CASE_ID: &str = "roomsim.gallery-artifacts.v1";

fn count_kinds(manifest: &serde_json::Value, kind: &str) -> usize {
    manifest["figures"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|figure| figure["kind"].as_str() == Some(kind))
        .count()
}

fn run() -> Result<QaResult, String> {
    let scenario = scenarios::scenario_by_name("small_stereo_2_0")
        .ok_or_else(|| "small_stereo_2_0 scenario is unavailable".to_string())?;
    let output = sonium_sim_room::fast_room::run_fast_hybrid(&scenario.simulation)
        .map_err(|error| error.to_string())?;
    let output = hf_extension::try_extend_to_full_range_with_room(&output, &scenario.simulation)?;
    output.validate_layout()?;

    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    gallery::write_gallery(
        &output,
        directory.path(),
        &scenario.name,
        "fast-hybrid",
        &scenario.simulation,
        &gallery::GalleryOptions::default(),
    )?;

    let manifest_path = directory.path().join("gallery.json");
    let manifest = sonium_xem::io::validate_gallery_manifest(&manifest_path)
        .map_err(|error| error.to_string())?;
    if manifest.width_px != 1920 || manifest.height_px != 1080 {
        return Err(format!(
            "gallery viewport is {}x{}, expected 1920x1080",
            manifest.width_px, manifest.height_px
        ));
    }
    let manifest_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let raster_count = count_kinds(&manifest_json, "png");
    let vector_count = count_kinds(&manifest_json, "svg");
    if raster_count < 4 || vector_count < 6 {
        return Err(format!(
            "gallery coverage too small: {raster_count} PNG and {vector_count} SVG figures"
        ));
    }
    for artifact in [
        "frequency_response.csv",
        "source_listener_response.csv",
        "spatial_metrics.csv",
        "room_metrics.json",
        "listener_frequency_waterfall.png",
        "index.html",
    ] {
        require_file(directory.path(), artifact)?;
    }
    let shell = std::fs::read_to_string(directory.path().join("index.html"))
        .map_err(|error| error.to_string())?;
    for link in [
        "figure.source",
        "provenance.audio",
        "frequency_response.csv",
    ] {
        if !shell.contains(link) {
            return Err(format!("gallery shell omits {link}"));
        }
    }
    let metrics: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("room_metrics.json"))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if metrics["schema"] != "sonium-sim-room-metrics/v1"
        || metrics["listener_count"].as_u64().unwrap_or(0) == 0
        || metrics["provenance"].as_array().is_none()
    {
        return Err("room metrics artifact violates its schema".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("raster_figure_count", raster_count as f64)
        .metric("vector_figure_count", vector_count as f64)
        .metric("frequency_sample_count", output.frequencies.len() as f64)
        .minimum("raster_figure_count", 4.0)
        .minimum("vector_figure_count", 6.0)
        .input_checksum("analytic/deterministic-fixture:roomsim-gallery-artifacts-v1")
        .reference("validated native gallery manifest, source checksums, and fixed viewport")
        .reference_checksum("analytic/deterministic-fixture:roomsim-gallery-artifacts-v1")
        .dimension("small_stereo_2_0 fast-hybrid response plus 20 kHz synthetic extension")
        .formulation("production gallery renderer; CSV/WAV and provenance remain authoritative")
        .tolerance(
            ">=4 fixed-size PNG figures, >=6 fixed-size SVG figures, linked source/audio artifacts",
        )
        .seed("roomsim-gallery-artifacts:cpu:v1"))
}

fn require_file(directory: &Path, name: &str) -> Result<(), String> {
    if directory.join(name).is_file() {
        Ok(())
    } else {
        Err(format!("gallery is missing {name}"))
    }
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim gallery result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("RoomSim gallery failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
