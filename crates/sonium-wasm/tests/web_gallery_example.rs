use std::fs;
use std::path::PathBuf;
use std::process::Command;

use sonium_roomsim::BrowserCaptureContract;

#[test]
fn web_gallery_example_writes_validated_bundle() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("sonium-roomsim must live below the workspace root")
        .to_owned();
    let output_dir =
        std::env::temp_dir().join(format!("sonium-web-gallery-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&output_dir);

    let output = Command::new("cargo")
        .current_dir(&workspace)
        .args([
            "run",
            "--quiet",
            "-p",
            "sonium-roomsim",
            "--example",
            "web_gallery",
            "--offline",
            "--",
        ])
        .arg(&output_dir)
        .output()
        .expect("web_gallery example should start");
    assert!(
        output.status.success(),
        "web_gallery failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    for artifact in [
        "index.html",
        "gallery.json",
        "web-gallery-provenance.json",
        "scenario.json",
        "results.json",
        "pressure_slice_250hz.json",
        "progressive_state.json",
        "web-gallery-capture.json",
    ] {
        assert!(
            output_dir.join(artifact).is_file(),
            "missing web-gallery artifact {artifact}"
        );
    }
    sonium_xem::io::validate_gallery_manifest(output_dir.join("gallery.json"))
        .expect("web-gallery manifest should validate");
    BrowserCaptureContract::from_json(
        &fs::read_to_string(output_dir.join("web-gallery-capture.json"))
            .expect("capture contract should be readable"),
    )
    .expect("web-gallery capture contract should validate");

    let index = fs::read_to_string(output_dir.join("index.html"))
        .expect("web-gallery index should be readable");
    assert!(index.contains("gallery.json"));
    assert!(index.contains("fetch("));

    fs::remove_dir_all(&output_dir).expect("remove temporary web-gallery output");
}

#[test]
fn web_gallery_shell_exposes_provenance_and_authoritative_source_links() {
    let html = include_str!("../assets/web_gallery.html");

    for required in [
        "web-gallery-provenance.json",
        "scenario.json",
        "pressure_slice_250hz.json",
        "progressive_state.json",
        "id=\"provenance\"",
        "id=\"integrity\"",
        "data-status=\"pending\"",
        "verifyArtifactChecksums",
        "fnv1a64:",
        "Artifact integrity failed",
        "Gallery withheld until artifact integrity passes",
        "source_checksum_fnv1a64",
        "fetch(check.path, {cache: \"no-store\"})",
        "status.dataset.status = failed.length === 0 ? \"ok\" : \"failed\"",
        "Promise.all([fetch(\"gallery.json\"), fetch(\"web-gallery-provenance.json\")])",
        "figure.source",
        "source data:",
        "Linked JSON remains authoritative",
    ] {
        assert!(
            html.contains(required),
            "gallery shell is missing {required:?}"
        );
    }
    assert!(
        html.contains("escapeHtml"),
        "gallery captions and provenance must use the HTML escaping helper"
    );
}
