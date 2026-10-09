//! Compile-only contract for the PC-fan native gallery image gate.
//!
//! This is a deterministic renderer fixture: it does not run PC-fan CFD or a
//! full simulation. The runtime case, when explicitly invoked, checks that a
//! native PNG is decodable, fixed-size, visibly nonblank, and admitted by the
//! checksummed gallery manifest.

use sonium_qa::QaResult;
use sonium_xem::io::{GalleryRenderer, validate_gallery_manifest, validate_png_nonblank};

const CASE_ID: &str = "pcfan.gallery-image-contract.v1";

fn run() -> Result<[f64; 3], String> {
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    std::fs::write(
        directory.path().join("field.vtu"),
        b"deterministic gallery source",
    )
    .map_err(|error| error.to_string())?;

    let mut gallery = GalleryRenderer::new(directory.path()).map_err(|error| error.to_string())?;
    gallery
        .heatmap_png(
            "native_field.png",
            &[0.0, 1.0, 0.5],
            "field.vtu",
            "Native deterministic field image",
        )
        .map_err(|error| error.to_string())?;
    gallery
        .line_svg(
            "source_curve.svg",
            &[0.0, 1.0],
            &[0.0, 1.0],
            "Source curve",
            "field.vtu",
            "Native source curve",
        )
        .map_err(|error| error.to_string())?;
    let manifest_path = gallery
        .finish(
            "PC-fan native image contract",
            vec!["fixed 1920x1080 viewport".into()],
        )
        .map_err(|error| error.to_string())?;

    let manifest = validate_gallery_manifest(&manifest_path).map_err(|error| error.to_string())?;
    let dimensions = validate_png_nonblank(directory.path().join("native_field.png"))
        .map_err(|error| error.to_string())?;
    if dimensions != (1920, 1080) {
        return Err(format!(
            "native gallery PNG dimensions {:?}, expected (1920, 1080)",
            dimensions
        ));
    }

    let image_bytes = std::fs::metadata(directory.path().join("native_field.png"))
        .map_err(|error| error.to_string())?
        .len();
    if image_bytes == 0 {
        return Err("native gallery PNG is empty".into());
    }

    Ok([
        manifest.figures.len() as f64,
        dimensions.0 as f64,
        image_bytes as f64,
    ])
}

fn main() {
    match run() {
        Ok(metrics) => {
            let result = QaResult::passed(CASE_ID)
                .metric("manifest_figure_count", metrics[0])
                .metric("png_width_px", metrics[1])
                .metric("png_file_size_bytes", metrics[2])
                .reference_checksum("analytic/deterministic-fixture:pcfan-gallery-image-v1")
                .input_checksum("sonium-xem-gallery-renderer:native-png:1920x1080")
                .seed("pcfan-gallery-image-contract:cpu:v1");
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("QA result serialization")
            );
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .to_json_line()
                    .expect("QA failure serialization")
            );
            std::process::exit(1);
        }
    }
}
