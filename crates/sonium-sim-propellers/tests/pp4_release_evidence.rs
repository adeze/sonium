//! Native PP4 gallery/release evidence.
//!
//! This producer validates the real headless bundle and its transactional
//! failure boundaries. It deliberately reports the release join as gated:
//! QA-manifest ownership remains in `sonium-qa`, and PP1--PP3 are not silently
//! converted into a complete release claim.

use serde::Serialize;
use sonium_sim_propellers::{
    Blade3dConfig, Propeller3dOptConfig, PropellerArtifactEvidence, PropellerExecutionProvenance,
    PropellerHostMetadata, optimize_blade_shape, validate_gallery_bundle, write_gallery_bundle,
};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SCHEMA: &str = "sonium.propellers.native-x86-pp4.v1";
const GALLERY_SCHEMA: &str = "sonium-xem-gallery/v1";
const EXPECTED_FIGURES: [&str; 10] = [
    "source_spectrum.csv",
    "blade_load_distribution.csv",
    "optimized-blade-loading.png",
    "axial-wake-proxy.png",
    "spectrum.svg",
    "thrust-versus-operating-point.svg",
    "torque-versus-operating-point.svg",
    "blade-load-distribution.svg",
    "conventional-toroidal-comparison.svg",
    "rotation.gif",
];

#[derive(Debug, Serialize)]
struct GalleryEvidence {
    path: String,
    manifest_schema: String,
    figure_count: usize,
    png_count: usize,
    svg_count: usize,
    gif_dimensions: [u16; 2],
    gif_frames: usize,
    spectrum_rows: usize,
    panel_load_rows: usize,
    vtu_files: usize,
    manifest_sha256: String,
    artifact_checksums_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct FailureProbe {
    id: String,
    status: String,
    typed_error: String,
    partial_publish: bool,
    exit_code: Option<i32>,
    sentinel_preserved: bool,
    retry_succeeded: bool,
    published_artifact_count: usize,
}

#[derive(Debug, Serialize)]
struct Pp4Capability {
    renderer: &'static str,
    renderer_status: &'static str,
    locale: String,
    locale_policy: &'static str,
    pre_publication_hook: &'static str,
    renderer_fault_hook: &'static str,
    fault_directory: &'static str,
}

#[derive(Debug, Serialize)]
struct QaJoin {
    status: String,
    registered_propeller_cases: Vec<String>,
    reason: String,
}

#[derive(Debug, Serialize)]
struct Pp4Artifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    host: PropellerHostMetadata,
    execution: PropellerExecutionProvenance,
    deterministic_mode: bool,
    thread_policy: String,
    capability: Pp4Capability,
    gallery: GalleryEvidence,
    failure_probes: Vec<FailureProbe>,
    qa_join: QaJoin,
    final_release_gate: &'static str,
}

fn config() -> Propeller3dOptConfig {
    Propeller3dOptConfig {
        mesh: Blade3dConfig {
            radial_segments: 8,
            chord_segments: 4,
            loop_segments: 4,
            ..Blade3dConfig::default()
        },
        n_harmonics: 4,
        max_iterations: 0,
        ..Propeller3dOptConfig::default()
    }
}

fn remove_owned(path: &Path) {
    if path.is_dir() {
        fs::remove_dir_all(path).expect("remove owned PP4 directory");
    } else if path.exists() {
        fs::remove_file(path).expect("remove owned PP4 file");
    }
}

fn sha256_file(path: &Path) -> String {
    sonium_sim_propellers::sha256_hex(&fs::read(path).expect("read PP4 artifact"))
}

fn collect_files(root: &Path, current: &Path, files: &mut BTreeMap<String, String>) {
    for entry in fs::read_dir(current).expect("read PP4 artifact directory") {
        let path = entry.expect("read PP4 directory entry").path();
        if path.is_dir() {
            collect_files(root, &path, files);
        } else {
            let relative = path
                .strip_prefix(root)
                .expect("PP4 artifact remains below root")
                .to_string_lossy()
                .replace('\\', "/");
            files.insert(relative, sha256_file(&path));
        }
    }
}

fn capability_metadata() -> Pp4Capability {
    Pp4Capability {
        renderer: "sonium_xem::io::GalleryRenderer",
        renderer_status: "available",
        locale: std::env::var("LC_ALL")
            .or_else(|_| std::env::var("LANG"))
            .unwrap_or_else(|_| "C".into()),
        locale_policy: "locale-independent Rust f64 formatting; renderer does not require a GUI locale",
        pre_publication_hook: "SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION",
        renderer_fault_hook: "SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE",
        fault_directory: ".target-unix-pp4-faults",
    }
}

fn assert_no_staging_entries(parent: &Path) {
    for entry in fs::read_dir(parent).expect("read PP4 fault parent") {
        let path = entry.expect("read PP4 fault entry").path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        assert!(
            !name.starts_with(".propeller-gallery-stage-")
                && !name.starts_with(".propeller-gallery-backup-"),
            "fault probe left staging/backup entry {path:?}"
        );
    }
}

fn run_cli_gallery_fault(output_dir: &Path, fault: Option<&str>) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_propeller-3d-opt"));
    command
        .args(["--iterations", "1", "--rpm", "18000", "--gallery"])
        .arg(output_dir)
        .env("RAYON_NUM_THREADS", "1")
        .env_remove("SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION")
        .env_remove("SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE");
    if let Some(fault) = fault {
        command.env(fault, "1");
    }
    command.output().expect("run PP4 child fault probe")
}

fn write_cli_gallery(root: &Path) {
    let gallery = root.join("cli-gallery");
    remove_owned(&gallery);
    let command = Command::new(env!("CARGO_BIN_EXE_propeller-3d-opt"))
        .args(["--iterations", "1", "--rpm", "18000", "--gallery"])
        .arg(&gallery)
        .env("RAYON_NUM_THREADS", "1")
        .env_remove("SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION")
        .env_remove("SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE")
        .output()
        .expect("run PP4 CLI gallery");
    let success = command.status.success();
    fs::write(
        root.join("cli.log"),
        [&command.stdout[..], &command.stderr[..]].concat(),
    )
    .expect("write PP4 CLI log");
    assert!(
        success,
        "PP4 CLI gallery failed: {}",
        fs::read_to_string(root.join("cli.log")).unwrap_or_default()
    );
    assert_cli_gallery(&gallery);
}

fn assert_cli_gallery(output_dir: &Path) {
    let manifest = sonium_xem::io::validate_gallery_manifest(output_dir.join("gallery.json"))
        .expect("retry must publish a valid gallery manifest");
    assert_eq!(manifest.figures.len(), EXPECTED_FIGURES.len());
    for figure in manifest.figures {
        assert!(
            output_dir.join(figure.filename).is_file(),
            "retry missing gallery figure"
        );
    }
    for artifact in [
        "optimized_blade.vtu",
        "baseline_blade.vtu",
        "execution_evidence.json",
    ] {
        assert!(
            output_dir.join(artifact).is_file(),
            "retry missing {artifact}"
        );
    }
}

fn assert_no_published_gallery(output_dir: &Path) {
    for artifact in [
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
        "rotation.gif",
        "gallery.json",
        "execution_evidence.json",
    ] {
        assert!(
            !output_dir.join(artifact).exists(),
            "fault probe published partial artifact {artifact}"
        );
    }
}

fn probe_child_process_faults() -> Vec<FailureProbe> {
    let fault_root = std::env::var_os("SONIUM_PP4_FAULTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".target-unix-pp4-faults/pp4-faults"));
    remove_owned(&fault_root);
    fs::create_dir_all(&fault_root).expect("create PP4 fault directory");

    let interrupted_dir = fault_root.join("interrupted-before-publication");
    fs::create_dir_all(&interrupted_dir).expect("create interruption probe directory");
    let interrupted_sentinel = interrupted_dir.join("sentinel.txt");
    fs::write(&interrupted_sentinel, b"published-gallery-sentinel")
        .expect("write interruption sentinel");
    let interrupted = run_cli_gallery_fault(
        &interrupted_dir,
        Some("SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION"),
    );
    let interrupted_stderr = String::from_utf8_lossy(&interrupted.stderr);
    assert!(!interrupted.status.success());
    assert_eq!(interrupted.status.code(), Some(1));
    assert!(interrupted_stderr.contains("ErrorKind::Interrupted"));
    assert_eq!(
        fs::read(&interrupted_sentinel).expect("read preserved interruption sentinel"),
        b"published-gallery-sentinel"
    );
    assert_no_published_gallery(&interrupted_dir);
    assert_no_staging_entries(&interrupted_dir);

    let interrupted_retry = run_cli_gallery_fault(&interrupted_dir, None);
    assert!(
        interrupted_retry.status.success(),
        "interruption retry failed: {}",
        String::from_utf8_lossy(&interrupted_retry.stderr)
    );
    assert_cli_gallery(&interrupted_dir);
    assert!(!interrupted_sentinel.exists());
    assert_no_staging_entries(&interrupted_dir);

    let renderer_dir = fault_root.join("renderer-unavailable");
    fs::create_dir_all(&renderer_dir).expect("create renderer probe directory");
    let renderer_sentinel = renderer_dir.join("sentinel.txt");
    fs::write(&renderer_sentinel, b"renderer-gallery-sentinel").expect("write renderer sentinel");
    let renderer =
        run_cli_gallery_fault(&renderer_dir, Some("SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE"));
    let renderer_stderr = String::from_utf8_lossy(&renderer.stderr);
    assert!(!renderer.status.success());
    assert_eq!(renderer.status.code(), Some(1));
    assert!(renderer_stderr.contains("ErrorKind::Unsupported"));
    assert_eq!(
        fs::read(&renderer_sentinel).expect("read preserved renderer sentinel"),
        b"renderer-gallery-sentinel"
    );
    assert_no_published_gallery(&renderer_dir);
    assert_no_staging_entries(&renderer_dir);

    let renderer_retry = run_cli_gallery_fault(&renderer_dir, None);
    assert!(
        renderer_retry.status.success(),
        "renderer retry failed: {}",
        String::from_utf8_lossy(&renderer_retry.stderr)
    );
    assert_cli_gallery(&renderer_dir);
    assert!(!renderer_sentinel.exists());
    assert_no_staging_entries(&renderer_dir);

    vec![
        FailureProbe {
            id: "child-process.pre-publication-interruption".into(),
            status: "typed_rejected".into(),
            typed_error: "std::io::ErrorKind::Interrupted: pre-publication child process stopped before commit".into(),
            partial_publish: false,
            exit_code: interrupted.status.code(),
            sentinel_preserved: true,
            retry_succeeded: true,
            published_artifact_count: 0,
        },
        FailureProbe {
            id: "child-process.renderer-unavailable".into(),
            status: "typed_rejected".into(),
            typed_error: "std::io::ErrorKind::Unsupported: headless renderer unavailable".into(),
            partial_publish: false,
            exit_code: renderer.status.code(),
            sentinel_preserved: true,
            retry_succeeded: true,
            published_artifact_count: 0,
        },
    ]
}

fn validate_gallery(
    gallery: &Path,
    config: &Propeller3dOptConfig,
    result: &sonium_sim_propellers::OptimizedBlade,
) -> GalleryEvidence {
    let validation =
        validate_gallery_bundle(gallery, config, result).expect("PP4 gallery validation");
    assert_eq!(validation.figure_count, EXPECTED_FIGURES.len());
    assert_eq!(validation.spectrum_rows, result.spectrum.harmonics.len());
    assert_eq!(validation.panel_load_rows, result.mesh.panels.len());
    sonium_xem::io::validate_vtu_file(gallery.join("optimized_blade.vtu"))
        .expect("optimized VTU validation");
    sonium_xem::io::validate_vtu_file(gallery.join("baseline_blade.vtu"))
        .expect("baseline VTU validation");

    let manifest_path = gallery.join("gallery.json");
    let manifest = sonium_xem::io::validate_gallery_manifest(&manifest_path)
        .expect("generic gallery manifest validation");
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.width_px, 1920);
    assert_eq!(manifest.height_px, 1080);
    assert_eq!(
        manifest
            .figures
            .iter()
            .map(|figure| figure.filename.as_str())
            .collect::<Vec<_>>(),
        EXPECTED_FIGURES
    );
    assert!(
        manifest
            .conventions
            .iter()
            .any(|line| line.contains("fidelity=PanelGeometrySurrogate"))
    );

    let mut png_count = 0;
    let mut svg_count = 0;
    for figure in &manifest.figures {
        assert!(!figure.caption.trim().is_empty());
        assert!(figure.caption.contains("fidelity=PanelGeometrySurrogate"));
        let path = gallery.join(&figure.filename);
        match figure.kind.as_str() {
            "png" => {
                assert_eq!(
                    sonium_xem::io::validate_png_nonblank(&path).expect("PNG validation"),
                    (1920, 1080)
                );
                png_count += 1;
            }
            "svg" => {
                let svg = fs::read_to_string(&path).expect("read SVG");
                assert!(svg.contains("<svg "));
                assert!(svg.contains("width=\"1920\""));
                assert!(svg.contains("height=\"1080\""));
                assert!(svg.contains("<polyline"));
                svg_count += 1;
            }
            "gif" => {
                assert_eq!(
                    sonium_xem::io::validate_gif(&path).expect("GIF validation"),
                    (320, 180, 12)
                );
            }
            "csv" => {
                let csv = fs::read_to_string(&path).expect("read CSV");
                assert!(!csv.ends_with("\n\n"));
                assert!(csv.lines().skip(1).all(|line| {
                    line.split(',').all(|value| {
                        value
                            .parse::<f64>()
                            .map(|number| number.is_finite())
                            .unwrap_or(false)
                    })
                }));
            }
            other => panic!("unexpected PP4 figure kind {other}"),
        }
    }
    assert_eq!(png_count, 2);
    assert_eq!(svg_count, 5);

    let evidence_path = gallery.join("execution_evidence.json");
    let execution: PropellerArtifactEvidence =
        serde_json::from_slice(&fs::read(&evidence_path).expect("read execution evidence"))
            .expect("parse execution evidence");
    assert_eq!(execution.provenance, result.provenance.execution);
    assert_eq!(execution.artifact_checksums_sha256.len(), 13);
    assert!(
        execution
            .artifact_checksums_sha256
            .values()
            .all(|digest| digest.len() == 64)
    );

    let mut artifact_checksums_sha256 = BTreeMap::new();
    collect_files(gallery, gallery, &mut artifact_checksums_sha256);
    assert!(artifact_checksums_sha256.contains_key("gallery.json"));
    assert!(artifact_checksums_sha256.contains_key("execution_evidence.json"));
    GalleryEvidence {
        path: "library-gallery".into(),
        manifest_schema: GALLERY_SCHEMA.into(),
        figure_count: manifest.figures.len(),
        png_count,
        svg_count,
        gif_dimensions: [320, 180],
        gif_frames: 12,
        spectrum_rows: validation.spectrum_rows,
        panel_load_rows: validation.panel_load_rows,
        vtu_files: 2,
        manifest_sha256: sha256_file(&manifest_path),
        artifact_checksums_sha256,
    }
}

fn probe_failures(
    root: &Path,
    gallery: &Path,
    config: &Propeller3dOptConfig,
    result: &sonium_sim_propellers::OptimizedBlade,
) -> Vec<FailureProbe> {
    let mut probes = Vec::new();
    let before = fs::read(gallery.join("gallery.json")).expect("read valid gallery");

    let mut invalid = result.clone();
    invalid.loads.shaft_power_w = f64::NAN;
    assert!(write_gallery_bundle(gallery, config, &invalid).is_err());
    assert_eq!(fs::read(gallery.join("gallery.json")).unwrap(), before);
    probes.push(FailureProbe {
        id: "existing-output.invalid-load-ledger".into(),
        status: "typed_rejected".into(),
        typed_error: "non-admissible production result; existing gallery preserved".into(),
        partial_publish: false,
        exit_code: None,
        sentinel_preserved: true,
        retry_succeeded: false,
        published_artifact_count: 0,
    });

    let regular_file = root.join("output-path-is-file");
    fs::write(&regular_file, b"sentinel").expect("write filesystem probe");
    assert!(write_gallery_bundle(&regular_file, config, result).is_err());
    assert_eq!(fs::read(&regular_file).unwrap(), b"sentinel");
    probes.push(FailureProbe {
        id: "filesystem.output-path-is-file".into(),
        status: "typed_rejected".into(),
        typed_error: "InvalidData: gallery output path is not a directory".into(),
        partial_publish: false,
        exit_code: None,
        sentinel_preserved: true,
        retry_succeeded: false,
        published_artifact_count: 0,
    });

    let unwritable = Path::new("/proc/sonium-pp4-unwritable");
    assert!(write_gallery_bundle(unwritable, config, result).is_err());
    probes.push(FailureProbe {
        id: "filesystem.unwritable-output".into(),
        status: "typed_rejected".into(),
        typed_error: "io::Error: unwritable Unix output path".into(),
        partial_publish: false,
        exit_code: None,
        sentinel_preserved: false,
        retry_succeeded: false,
        published_artifact_count: 0,
    });

    let rejected = root.join("rejected-invalid-result");
    remove_owned(&rejected);
    fs::create_dir_all(&rejected).expect("create rejection directory");
    assert!(write_gallery_bundle(&rejected, config, &invalid).is_err());
    assert!(fs::read_dir(&rejected).unwrap().next().is_none());
    probes.push(FailureProbe {
        id: "clean-output.invalid-result".into(),
        status: "typed_rejected".into(),
        typed_error: "non-admissible production result; zero partial artifacts".into(),
        partial_publish: false,
        exit_code: None,
        sentinel_preserved: false,
        retry_succeeded: false,
        published_artifact_count: 0,
    });

    let corruption = root.join("corruption-probe");
    remove_owned(&corruption);
    write_gallery_bundle(&corruption, config, result).expect("write corruption probe gallery");
    let spectrum = corruption.join("source_spectrum.csv");
    let original = fs::read(&spectrum).unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&spectrum)
        .unwrap()
        .write_all(b"corruption\n")
        .unwrap();
    assert!(validate_gallery_bundle(&corruption, config, result).is_err());
    fs::write(&spectrum, original).unwrap();
    probes.push(FailureProbe {
        id: "corruption.csv-row".into(),
        status: "typed_rejected".into(),
        typed_error: "InvalidData: CSV row does not match admitted ledger/checksum".into(),
        partial_publish: false,
        exit_code: None,
        sentinel_preserved: true,
        retry_succeeded: false,
        published_artifact_count: 0,
    });

    let cli_probe = root.join("invalid-cli-output");
    remove_owned(&cli_probe);
    let command = Command::new(env!("CARGO_BIN_EXE_propeller-3d-opt"))
        .args(["--iterations", "0", "--rpm", "1000", "--gallery"])
        .arg(&cli_probe)
        .output()
        .expect("run invalid propeller CLI");
    assert!(!command.status.success());
    assert!(!cli_probe.exists());
    probes.push(FailureProbe {
        id: "cli.invalid-rpm".into(),
        status: "typed_rejected".into(),
        typed_error: "clap value-parser rejected RPM below 3000; no output".into(),
        partial_publish: false,
        exit_code: command.status.code(),
        sentinel_preserved: false,
        retry_succeeded: false,
        published_artifact_count: 0,
    });

    probes.extend(probe_child_process_faults());
    probes
}

#[test]
fn pp4_native_gallery_release_evidence_is_strict() {
    assert_eq!(std::env::consts::OS, "linux");
    assert_eq!(std::env::consts::ARCH, "x86_64");
    let started = Instant::now();
    let root = std::env::var_os("SONIUM_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/sonium-unix-pp4"));
    fs::create_dir_all(&root).expect("create PP4 root");
    let config = config();
    let result = optimize_blade_shape(&config);
    assert!(result.is_exportable());
    write_cli_gallery(&root);
    let gallery = root.join("library-gallery");
    remove_owned(&gallery);
    write_gallery_bundle(&gallery, &config, &result).expect("write clean PP4 gallery");
    let gallery_evidence = validate_gallery(&gallery, &config, &result);

    let manifest_before = fs::read(gallery.join("gallery.json")).unwrap();
    fs::write(gallery.join("stale-file.txt"), b"must be removed").unwrap();
    write_gallery_bundle(&gallery, &config, &result).expect("replace existing PP4 gallery");
    assert!(!gallery.join("stale-file.txt").exists());
    assert_eq!(
        fs::read(gallery.join("gallery.json")).unwrap(),
        manifest_before
    );
    assert_eq!(
        validate_gallery(&gallery, &config, &result).manifest_sha256,
        gallery_evidence.manifest_sha256
    );

    let failure_probes = probe_failures(&root, &gallery, &config, &result);
    assert!(
        failure_probes
            .iter()
            .all(|probe| { probe.status == "typed_rejected" && !probe.partial_publish })
    );

    let artifact = Pp4Artifact {
        schema: SCHEMA,
        status: "gated",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64 waived",
        host: PropellerHostMetadata::collect(),
        execution: result.provenance.execution.clone(),
        deterministic_mode: true,
        thread_policy: format!("RAYON_NUM_THREADS={}", rayon::current_num_threads()),
        capability: capability_metadata(),
        gallery: gallery_evidence,
        failure_probes,
        qa_join: QaJoin {
            status: "blocked".into(),
            registered_propeller_cases: vec![
                "propeller.capability-matrix.v1".into(),
                "propeller.load-power.v1".into(),
                "propeller.operating-envelope.v1".into(),
                "propeller.application-reference-correlation.v1".into(),
                "propeller.gallery.v1".into(),
                "propeller.rotating-dipole-reference.v1".into(),
                "propeller.multipoint-operating-envelope.v1".into(),
                "propeller.dynamic-stall-reference.v1".into(),
                "propeller.duct-shroud-reference.v1".into(),
                "propeller.body-fitted-meridional-flow-reference.v1".into(),
                "propeller.body-fitted-turbulence-reference.v1".into(),
                "propeller.body-fitted-rotating-wall-reference.v1".into(),
                "propeller.body-fitted-swirl-transport-reference.v1".into(),
                "propeller.body-fitted-coupled-swirl-transport-reference.v1".into(),
                "propeller.axisymmetric-meridional-flow-reference.v1".into(),
                "propeller.axisymmetric-weighted-rotating-wall-reference.v1".into(),
                "propeller.fwh-reference.v1".into(),
                "propeller.viscous-core-reference.v1".into(),
                "propeller.vortex-lattice-reference.v1".into(),
                "propeller.measured-calibration-reference.v1".into(),
                "propeller.fidelity-promotion-reference.v1".into(),
                "propeller.measured-dataset-provenance.v1".into(),
                "propeller.measured-dataset-attestation.v1".into(),
                "propeller.moving-fwh-reference.v1".into(),
                "propeller.mean-flow-fwh-reference.v1".into(),
                "propeller.measured-uncertainty-reference.v1".into(),
                "propeller.mesh-state-transfer-reference.v1".into(),
                "propeller.panel-response-refinement-reference.v1".into(),
                "propeller.farassat-fwh-reference.v1".into(),
            ],
            reason: "sonium-qa owns manifest/release fan-in; PP1--PP3 remain partial, so no QA or final-release pass is claimed".into(),
        },
        final_release_gate: "blocked until PP1, PP2, PP3 and sonium-qa fan-in are complete",
    };
    let artifact_path = root.join("native-x86-pp4-release-v1.json");
    fs::write(
        &artifact_path,
        serde_json::to_vec_pretty(&artifact).expect("serialize PP4 artifact"),
    )
    .expect("write PP4 artifact");
    let mut checksums = BTreeMap::new();
    collect_files(&root, &root, &mut checksums);
    assert!(checksums.contains_key("library-gallery/gallery.json"));
    fs::write(
        root.join("checksums.sha256"),
        checksums
            .iter()
            .map(|(path, digest)| format!("{digest}  {path}\n"))
            .collect::<String>(),
    )
    .expect("write PP4 checksum index");
    assert!(started.elapsed().as_secs() < 120);
}
