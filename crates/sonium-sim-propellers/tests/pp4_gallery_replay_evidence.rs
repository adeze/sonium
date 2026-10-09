//! Native PP4 deterministic replay evidence for the production gallery CLI.
//!
//! Two clean child-process invocations of the real `propeller-3d-opt --gallery`
//! path are compared byte-for-byte.  The packet covers reproducibility of the
//! complete gallery/provenance bundle; it does not promote the surrogate
//! renderer to resolved flow or close the sonium-qa release fan-in.

use serde::Serialize;
use sonium_sim_propellers::sha256_hex;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SCHEMA: &str = "sonium.propellers.pp4-gallery-replay-evidence.v1";
const EXPECTED_FILES: [&str; 13] = [
    "gallery.json",
    "execution_evidence.json",
    "optimized_blade.vtu",
    "baseline_blade.vtu",
    "source_spectrum.csv",
    "blade_load_distribution.csv",
    "optimized-blade-loading.png",
    "axial-wake-proxy.png",
    "spectrum.svg",
    "thrust-versus-operating-point.svg",
    "torque-versus-operating-point.svg",
    "blade-load-distribution.svg",
    "conventional-toroidal-comparison.svg",
    // rotation.gif is included in the figure list but would make this array
    // 14 entries; retain the complete file set explicitly below.
];
const ROTATION_GIF: &str = "rotation.gif";
const FIGURES: [&str; 10] = [
    "source_spectrum.csv",
    "blade_load_distribution.csv",
    "optimized-blade-loading.png",
    "axial-wake-proxy.png",
    "spectrum.svg",
    "thrust-versus-operating-point.svg",
    "torque-versus-operating-point.svg",
    "blade-load-distribution.svg",
    "conventional-toroidal-comparison.svg",
    ROTATION_GIF,
];

#[derive(Debug, Serialize)]
struct ReplayFile {
    path: &'static str,
    first_sha256: String,
    second_sha256: String,
    first_bytes: usize,
    second_bytes: usize,
    byte_identical: bool,
}

#[derive(Debug, Serialize)]
struct ReplayArtifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    claim_class: &'static str,
    reference: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    run_count: usize,
    deterministic: bool,
    files: Vec<ReplayFile>,
    excluded_files: Vec<&'static str>,
    canonical_output: String,
    output_sha256: String,
}

fn all_expected_files() -> Vec<&'static str> {
    let mut files = EXPECTED_FILES.to_vec();
    files.push(ROTATION_GIF);
    files
}

fn remove_owned(path: &Path) {
    if path.is_dir() {
        fs::remove_dir_all(path).expect("remove owned replay directory");
    } else if path.exists() {
        fs::remove_file(path).expect("remove owned replay file");
    }
}

fn run_gallery(output_dir: &Path) {
    let command = Command::new(env!("CARGO_BIN_EXE_propeller-3d-opt"))
        .args(["--iterations", "1", "--rpm", "18000", "--gallery"])
        .arg(output_dir)
        .env("RAYON_NUM_THREADS", "1")
        .env_remove("SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION")
        .env_remove("SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE")
        .output()
        .expect("run production gallery child");
    assert!(
        command.status.success(),
        "production gallery failed: {}",
        String::from_utf8_lossy(&command.stderr)
    );
    assert!(String::from_utf8_lossy(&command.stdout).contains("Gallery manifest written to"));
}

fn validate_gallery(output_dir: &Path) -> BTreeSet<String> {
    let manifest = sonium_xem::io::validate_gallery_manifest(output_dir.join("gallery.json"))
        .expect("replay must publish a valid gallery manifest");
    assert_eq!(manifest.schema_version, 1);
    assert_eq!((manifest.width_px, manifest.height_px), (1920, 1080));
    assert_eq!(
        manifest
            .figures
            .iter()
            .map(|figure| figure.filename.as_str())
            .collect::<BTreeSet<_>>(),
        FIGURES.into_iter().collect::<BTreeSet<_>>()
    );
    let expected = all_expected_files();
    for name in &expected {
        assert!(
            output_dir.join(name).is_file(),
            "missing replay file {name}"
        );
    }
    let actual = fs::read_dir(output_dir)
        .expect("read replay output")
        .map(|entry| {
            let path = entry.expect("read replay entry").path();
            assert!(
                path.is_file(),
                "replay output contains a directory {path:?}"
            );
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("replay filename is UTF-8")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected.into_iter().map(str::to_owned).collect());
    actual
}

fn digest(path: &Path) -> (String, usize) {
    let bytes = fs::read(path).expect("read replay file");
    (sha256_hex(&bytes), bytes.len())
}

fn evidence(output_dir: Option<&Path>) -> ReplayArtifact {
    assert_eq!(std::env::consts::OS, "linux", "PP4 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP4 requires native x86_64"
    );
    let root = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".target-pp4-next-fresh/evidence"));
    fs::create_dir_all(&root).expect("create PP4 replay evidence root");
    let first = root.join("replay-one");
    let second = root.join("replay-two");
    remove_owned(&first);
    remove_owned(&second);
    fs::create_dir_all(&first).expect("create first replay directory");
    fs::create_dir_all(&second).expect("create second replay directory");
    run_gallery(&first);
    run_gallery(&second);
    let first_files = validate_gallery(&first);
    let second_files = validate_gallery(&second);
    assert_eq!(first_files, second_files);

    let mut files = Vec::with_capacity(all_expected_files().len());
    for name in all_expected_files() {
        let (first_sha256, first_bytes) = digest(&first.join(name));
        let (second_sha256, second_bytes) = digest(&second.join(name));
        assert_eq!(
            first_sha256, second_sha256,
            "replay digest differs for {name}"
        );
        assert_eq!(first_bytes, second_bytes, "replay size differs for {name}");
        files.push(ReplayFile {
            path: name,
            first_sha256,
            second_sha256,
            first_bytes,
            second_bytes,
            byte_identical: true,
        });
    }
    let canonical_input = "command=propeller-3d-opt --iterations 1 --rpm 18000 --gallery;RAYON_NUM_THREADS=1;fault_hooks=removed;clean_output_dirs=two".to_owned();
    let canonical_output = serde_json::to_string(&files).expect("serialize replay files");
    let artifact = ReplayArtifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 production CLI CPU/f64; Windows/Arm64/AVX512 waived",
        claim_class: "native production gallery byte-identical clean replay",
        reference: "two clean invocations of the admitted production gallery CLI produce identical bytes for every gallery and provenance artifact",
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        canonical_input,
        run_count: 2,
        deterministic: true,
        files,
        excluded_files: Vec::new(),
        output_sha256: sha256_hex(canonical_output.as_bytes()),
        canonical_output,
    };
    if let Some(output_dir) = output_dir {
        fs::write(
            output_dir.join("native-x86-pp4-gallery-replay-v1.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize PP4 replay artifact"),
        )
        .expect("write PP4 replay artifact");
    }
    artifact
}

#[test]
fn native_pp4_gallery_replay_evidence_v1() {
    let artifact = evidence(
        std::env::var_os("SONIUM_OUTPUT_DIR")
            .map(|path| Path::new(&path).to_path_buf())
            .as_deref(),
    );
    assert_eq!(artifact.status, "pass");
    assert_eq!(artifact.run_count, 2);
    assert!(artifact.deterministic);
    assert_eq!(artifact.files.len(), 14);
    assert!(artifact.files.iter().all(|file| file.byte_identical));
}
