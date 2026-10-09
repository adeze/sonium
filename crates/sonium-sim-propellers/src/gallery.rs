//! Deterministic, headless propeller gallery export.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{
    EXECUTION_EVIDENCE_SCHEMA_V1, OptimizedBlade, Propeller3dOptConfig, PropellerArtifactEvidence,
    baseline_blade_shape, sha256_hex,
};

// These opt-in hooks are intentionally environment-driven so PP4 can exercise
// the real binary in a child process. They are inert for normal gallery users
// and are checked only at the publication boundaries they describe.
const PP4_INTERRUPT_BEFORE_PUBLICATION: &str = "SONIUM_PP4_FAULT_INTERRUPT_BEFORE_PUBLICATION";
const PP4_RENDERER_UNAVAILABLE: &str = "SONIUM_PP4_FAULT_RENDERER_UNAVAILABLE";

/// Validated source-row counts for a published propeller gallery bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GalleryBundleValidation {
    /// Number of manifest figures after checksum and source validation.
    pub figure_count: usize,
    /// Number of rows in `source_spectrum.csv`.
    pub spectrum_rows: usize,
    /// Number of rows in `blade_load_distribution.csv`.
    pub panel_load_rows: usize,
}

/// Write the complete production gallery bundle.
///
/// The admissibility check deliberately precedes every artifact write. A
/// failed production result therefore cannot leave a partial gallery that
/// looks publishable to a downstream website or QA consumer.
pub fn write_gallery_bundle(
    output_dir: &Path,
    config: &Propeller3dOptConfig,
    result: &OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    if !result.is_exportable() {
        return Err("propeller gallery refuses a non-admissible production result".into());
    }
    if output_dir == Path::new(".") || output_dir == Path::new("") {
        return Err(validation_error(
            "propeller gallery output must be a dedicated directory",
        ));
    }
    if output_dir.exists() && !output_dir.is_dir() {
        return Err(validation_error(
            "propeller gallery output path is not a directory",
        ));
    }

    // Render and validate in a sibling staging directory. The caller's
    // existing gallery remains intact if any source, renderer, or validation
    // step fails, and a successful rerun replaces the directory as one commit.
    let parent = gallery_parent(output_dir);
    std::fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".propeller-gallery-stage-")
        .tempdir_in(parent)?;
    write_gallery_bundle_in_place(staging.path(), config, result)?;
    if std::env::var(PP4_INTERRUPT_BEFORE_PUBLICATION).as_deref() == Ok("1") {
        return Err(fault_error(
            std::io::ErrorKind::Interrupted,
            "PP4 fault injection interrupted the process before gallery publication",
        ));
    }
    let staging_path = staging.keep();
    if let Err(error) = commit_staged_gallery(&staging_path, output_dir) {
        let _ = std::fs::remove_dir_all(&staging_path);
        return Err(error);
    }
    Ok(())
}

fn write_gallery_bundle_in_place(
    output_dir: &Path,
    config: &Propeller3dOptConfig,
    result: &OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    result
        .mesh
        .write_vtu(output_dir.join("optimized_blade.vtu"))?;
    baseline_blade_shape(config)
        .mesh
        .write_vtu(output_dir.join("baseline_blade.vtu"))?;
    write_gallery(output_dir, config, result)?;
    validate_gallery_bundle(output_dir, config, result)?;
    Ok(())
}

fn commit_staged_gallery(
    staging: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let parent = gallery_parent(output_dir);
    if !output_dir.exists() {
        std::fs::rename(staging, output_dir)?;
        return Ok(());
    }

    let backup = tempfile::Builder::new()
        .prefix(".propeller-gallery-backup-")
        .tempdir_in(parent)?;
    let backup_path = backup.path().to_path_buf();
    std::fs::remove_dir(&backup_path)?;
    std::fs::rename(output_dir, &backup_path)?;
    if let Err(error) = std::fs::rename(staging, output_dir) {
        let _ = std::fs::rename(&backup_path, output_dir);
        return Err(error.into());
    }
    std::fs::remove_dir_all(backup_path)?;
    Ok(())
}

fn gallery_parent(output_dir: &Path) -> &Path {
    output_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Validate a gallery against the exact admitted production result that wrote it.
///
/// The generic XEM gallery validator proves that files, dimensions, paths, and
/// checksums are coherent. This propeller-specific layer additionally proves
/// that the two numerical CSV sources are byte-round-trippable representations
/// of the result spectrum and panel-load ledger, so a stale or hand-edited
/// gallery cannot be promoted merely because it remains finite.
pub fn validate_gallery_bundle(
    output_dir: &Path,
    config: &Propeller3dOptConfig,
    result: &OptimizedBlade,
) -> Result<GalleryBundleValidation, Box<dyn std::error::Error>> {
    if !result.is_exportable() {
        return Err(validation_error(
            "propeller gallery refuses non-admissible production result",
        ));
    }

    let manifest = sonium_xem::io::validate_gallery_manifest(output_dir.join("gallery.json"))?;
    if !manifest
        .conventions
        .iter()
        .any(|convention| convention.contains("fidelity=PanelGeometrySurrogate"))
    {
        return Err(validation_error(
            "propeller gallery conventions must declare PanelGeometrySurrogate fidelity",
        ));
    }
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
    if manifest.figures.len() != EXPECTED_FIGURES.len()
        || EXPECTED_FIGURES.iter().any(|name| {
            !manifest
                .figures
                .iter()
                .any(|figure| figure.filename == *name)
        })
    {
        return Err(validation_error(format!(
            "propeller gallery manifest must contain exactly the expected ten figures (got {})",
            manifest.figures.len()
        )));
    }
    if manifest
        .figures
        .iter()
        .any(|figure| !figure.caption.contains("fidelity=PanelGeometrySurrogate"))
    {
        return Err(validation_error(
            "propeller gallery figure captions must declare PanelGeometrySurrogate fidelity",
        ));
    }

    for artifact in [
        "optimized_blade.vtu",
        "baseline_blade.vtu",
        "gallery.json",
        "execution_evidence.json",
    ] {
        if !output_dir.join(artifact).is_file() {
            return Err(validation_error(format!(
                "propeller gallery missing required artifact {artifact}"
            )));
        }
    }

    let baseline = baseline_blade_shape(config);
    let spectrum_rows = read_and_validate_spectrum(
        &output_dir.join("source_spectrum.csv"),
        &result.spectrum,
        &baseline.spectrum,
    )?;
    let panel_load_rows =
        read_and_validate_panel_loads(&output_dir.join("blade_load_distribution.csv"), result)?;

    let rotation_gif = std::fs::read(output_dir.join("rotation.gif"))?;
    if rotation_gif.len() < 6 || &rotation_gif[..6] != b"GIF89a" {
        return Err(validation_error(
            "propeller gallery rotation.gif has no GIF89a header",
        ));
    }

    validate_execution_evidence(output_dir, result)?;

    Ok(GalleryBundleValidation {
        figure_count: manifest.figures.len(),
        spectrum_rows,
        panel_load_rows,
    })
}

fn validate_execution_evidence(
    output_dir: &Path,
    result: &OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = output_dir.join("execution_evidence.json");
    let evidence: PropellerArtifactEvidence = serde_json::from_slice(&std::fs::read(&path)?)
        .map_err(|error| validation_error(format!("invalid execution evidence: {error}")))?;
    if evidence.schema != EXECUTION_EVIDENCE_SCHEMA_V1
        || evidence.provenance != result.provenance.execution
        || evidence.artifact_checksums_sha256.is_empty()
    {
        return Err(validation_error(
            "propeller execution evidence schema or provenance mismatch",
        ));
    }
    if evidence
        .artifact_checksums_sha256
        .contains_key("execution_evidence.json")
    {
        return Err(validation_error(
            "execution evidence may not checksum itself",
        ));
    }
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
    ] {
        if !evidence.artifact_checksums_sha256.contains_key(artifact) {
            return Err(validation_error(format!(
                "execution evidence is missing checksum for {artifact}"
            )));
        }
    }
    for (filename, expected_checksum) in evidence.artifact_checksums_sha256 {
        let artifact = output_dir.join(&filename);
        let relative = artifact
            .strip_prefix(output_dir)
            .map_err(|_| validation_error("execution evidence artifact escapes gallery root"))?;
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            || expected_checksum.len() != 64
            || !expected_checksum
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(validation_error(
                "execution evidence contains an invalid artifact entry",
            ));
        }
        let actual = sha256_hex(&std::fs::read(&artifact)?);
        if actual != expected_checksum {
            return Err(validation_error(format!(
                "execution evidence checksum mismatch for {filename}"
            )));
        }
    }
    Ok(())
}

fn write_execution_evidence(
    output_dir: &Path,
    result: &OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut artifact_checksums_sha256 = BTreeMap::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let filename = path
                .strip_prefix(output_dir)
                .map_err(|_| validation_error("gallery artifact escapes output directory"))?
                .to_string_lossy()
                .replace('\\', "/");
            artifact_checksums_sha256.insert(filename, sha256_hex(&std::fs::read(path)?));
        }
    }
    let evidence = PropellerArtifactEvidence {
        schema: EXECUTION_EVIDENCE_SCHEMA_V1.into(),
        provenance: result.provenance.execution.clone(),
        artifact_checksums_sha256,
    };
    std::fs::write(
        output_dir.join("execution_evidence.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    Ok(())
}

fn read_and_validate_spectrum(
    path: &Path,
    candidate: &crate::PropellerSpectrum,
    baseline: &crate::PropellerSpectrum,
) -> Result<usize, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let mut lines = contents.lines();
    if lines.next() != Some("frequency_hz,pressure_pa,baseline_pressure_pa") {
        return Err(validation_error("propeller spectrum CSV header mismatch"));
    }
    if candidate.harmonics.len() != baseline.harmonics.len() {
        return Err(validation_error(
            "propeller candidate and baseline spectrum lengths differ",
        ));
    }
    let mut rows = 0;
    for (index, line) in lines.enumerate() {
        let values = parse_csv_floats(line, 3)?;
        let candidate_harmonic = candidate
            .harmonics
            .get(index)
            .ok_or_else(|| validation_error("propeller spectrum CSV contains too many rows"))?;
        let baseline_harmonic = &baseline.harmonics[index];
        let expected = [
            candidate_harmonic.frequency_hz,
            candidate_harmonic.pressure_pa,
            baseline_harmonic.pressure_pa,
        ];
        if values
            .iter()
            .zip(expected)
            .any(|(actual, expected)| !close_enough(*actual, expected))
        {
            return Err(validation_error(format!(
                "propeller spectrum CSV row {index} does not match the admitted result"
            )));
        }
        rows += 1;
    }
    if rows != candidate.harmonics.len() {
        return Err(validation_error(format!(
            "propeller spectrum CSV has {rows} rows, expected {}",
            candidate.harmonics.len()
        )));
    }
    Ok(rows)
}

fn read_and_validate_panel_loads(
    path: &Path,
    result: &OptimizedBlade,
) -> Result<usize, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let mut lines = contents.lines();
    if lines.next() != Some("radius_fraction,loading_proxy_m2") {
        return Err(validation_error("propeller panel-load CSV header mismatch"));
    }
    let mut expected = result
        .mesh
        .panels
        .iter()
        .map(|panel| {
            (
                panel.radius_fraction,
                panel.area * panel.radius_fraction.powi(2),
            )
        })
        .collect::<Vec<_>>();
    expected.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut rows = 0;
    for (index, line) in lines.enumerate() {
        let values = parse_csv_floats(line, 2)?;
        let expected_row = expected
            .get(index)
            .ok_or_else(|| validation_error("propeller panel-load CSV contains too many rows"))?;
        if !close_enough(values[0], expected_row.0) || !close_enough(values[1], expected_row.1) {
            return Err(validation_error(format!(
                "propeller panel-load CSV row {index} does not match the admitted result"
            )));
        }
        rows += 1;
    }
    if rows != expected.len() {
        return Err(validation_error(format!(
            "propeller panel-load CSV has {rows} rows, expected {}",
            expected.len()
        )));
    }
    Ok(rows)
}

fn parse_csv_floats(
    line: &str,
    expected_count: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let values = line
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != expected_count || values.iter().any(|value| !value.is_finite()) {
        return Err(validation_error(
            "propeller gallery CSV row is malformed or non-finite",
        ));
    }
    Ok(values)
}

fn close_enough(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 2.0e-14 * actual.abs().max(expected.abs()).max(1.0)
}

fn validation_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

fn fault_error(kind: std::io::ErrorKind, message: &'static str) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::new(
        kind,
        format!("ErrorKind::{kind:?}: {message}"),
    ))
}

fn write_gallery(
    output_dir: &Path,
    config: &Propeller3dOptConfig,
    result: &OptimizedBlade,
) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_xem::io::GalleryRenderer;

    if !result.is_exportable() {
        return Err("propeller gallery refuses a non-admissible production result".into());
    }
    let panel_area = result
        .mesh
        .panels
        .iter()
        .map(|panel| panel.area)
        .collect::<Vec<_>>();
    let frequency = result
        .spectrum
        .harmonics
        .iter()
        .map(|harmonic| harmonic.frequency_hz)
        .collect::<Vec<_>>();
    let pressure = result
        .spectrum
        .harmonics
        .iter()
        .map(|harmonic| harmonic.pressure_pa)
        .collect::<Vec<_>>();
    let baseline = baseline_blade_shape(config);
    let baseline_pressure = baseline
        .spectrum
        .harmonics
        .iter()
        .map(|harmonic| harmonic.pressure_pa)
        .collect::<Vec<_>>();
    let mut blade_load = result
        .mesh
        .panels
        .iter()
        .map(|panel| {
            (
                panel.radius_fraction,
                panel.area * panel.radius_fraction.powi(2),
            )
        })
        .collect::<Vec<_>>();
    blade_load.sort_by(|left, right| left.0.total_cmp(&right.0));
    let blade_load_radius = blade_load
        .iter()
        .map(|(radius, _)| *radius)
        .collect::<Vec<_>>();
    let blade_load_value = blade_load.iter().map(|(_, load)| *load).collect::<Vec<_>>();
    if panel_area.is_empty()
        || frequency.len() < 2
        || frequency.len() != pressure.len()
        || frequency.len() != baseline_pressure.len()
        || panel_area
            .iter()
            .chain(frequency.iter())
            .chain(pressure.iter())
            .chain(baseline_pressure.iter())
            .any(|value| !value.is_finite())
        || blade_load
            .iter()
            .any(|(radius, load)| !radius.is_finite() || !load.is_finite())
    {
        return Err("propeller production result contains invalid gallery arrays".into());
    }

    if std::env::var(PP4_RENDERER_UNAVAILABLE).as_deref() == Ok("1") {
        return Err(fault_error(
            std::io::ErrorKind::Unsupported,
            "PP4 fault injection reports the headless gallery renderer unavailable",
        ));
    }
    let mut gallery = GalleryRenderer::new(output_dir)?;
    let spectrum_csv = output_dir.join("source_spectrum.csv");
    let mut spectrum_csv_contents = String::from("frequency_hz,pressure_pa,baseline_pressure_pa\n");
    for ((frequency_hz, pressure_pa), baseline_pressure_pa) in
        frequency.iter().zip(&pressure).zip(&baseline_pressure)
    {
        spectrum_csv_contents.push_str(&format!(
            "{frequency_hz:.17e},{pressure_pa:.17e},{baseline_pressure_pa:.17e}\n"
        ));
    }
    std::fs::write(&spectrum_csv, spectrum_csv_contents)?;

    let blade_load_csv = output_dir.join("blade_load_distribution.csv");
    let mut blade_load_csv_contents = String::from("radius_fraction,loading_proxy_m2\n");
    for (radius, load) in &blade_load {
        blade_load_csv_contents.push_str(&format!("{radius:.17e},{load:.17e}\n"));
    }
    std::fs::write(&blade_load_csv, blade_load_csv_contents)?;

    gallery.existing_figure(
        "source_spectrum.csv",
        "csv",
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; scalar=harmonic peak pressure; columns are frequency (Hz), pressure (Pa), and baseline pressure (Pa)",
    )?;
    gallery.existing_figure(
        "blade_load_distribution.csv",
        "csv",
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; scalar=spanwise loading proxy; columns are radius fraction and area (m²)",
    )?;
    gallery.heatmap_png(
        "optimized-blade-loading.png",
        &panel_area,
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed camera and deterministic colour map; scalar=panel area (m²)",
    )?;
    gallery.heatmap_png(
        "axial-wake-proxy.png",
        &result
            .mesh
            .panels
            .iter()
            .map(|panel| panel.radius_fraction * panel.chord_fraction)
            .collect::<Vec<_>>(),
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed camera and deterministic colour map; scalar=axial wake/loading proxy (dimensionless), not resolved transient flow",
    )?;
    gallery.line_svg(
        "spectrum.svg",
        &frequency,
        &pressure,
        "Propeller harmonic pressure spectrum",
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed 1920x1080 axes; scalar=harmonic peak pressure (Pa) versus frequency (Hz)",
    )?;
    gallery.line_svg(
        "thrust-versus-operating-point.svg",
        &[baseline.design.source.rpm, result.design.source.rpm],
        &[baseline.design.source.thrust, result.design.source.thrust],
        "Baseline and optimized thrust",
        "baseline_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed 1920x1080 axes; scalar=thrust (N) versus RPM",
    )?;
    gallery.line_svg(
        "torque-versus-operating-point.svg",
        &[baseline.design.source.rpm, result.design.source.rpm],
        &[baseline.design.source.torque, result.design.source.torque],
        "Baseline and optimized torque",
        "baseline_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed 1920x1080 axes; scalar=torque (N·m) versus RPM",
    )?;
    gallery.line_svg(
        "blade-load-distribution.svg",
        &blade_load_radius,
        &blade_load_value,
        "Spanwise blade loading proxy",
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed 1920x1080 axes; scalar=panel area weighted by radial loading factor (m²), ordered by pinned panel generation",
    )?;
    gallery.line_svg(
        "conventional-toroidal-comparison.svg",
        &frequency,
        &baseline_pressure
            .iter()
            .zip(&pressure)
            .map(|(reference, candidate)| {
                20.0 * (reference.max(1.0e-30) / candidate.max(1.0e-30)).log10()
            })
            .collect::<Vec<_>>(),
        "Conventional-to-toroidal harmonic reduction",
        "baseline_blade.vtu",
        "fidelity=PanelGeometrySurrogate; fixed 1920x1080 axes; scalar=peak-pressure reduction (dB) by harmonic; baseline and candidate share the pinned solver configuration",
    )?;
    let rotation_frames = (0..12)
        .map(|phase| {
            panel_area
                .iter()
                .enumerate()
                .map(|(index, value)| value * (1.0 + 0.2 * ((index + phase) as f64 * 0.31).sin()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    gallery.animation_gif(
        "rotation.gif",
        &rotation_frames,
        "optimized_blade.vtu",
        "fidelity=PanelGeometrySurrogate; 320x180 twelve-frame deterministic loading animation derived from production blade panels; not resolved transient flow",
    )?;
    let manifest_path = gallery.finish(
        "Optimized propeller gallery",
        vec![
            "Pinned 18,000 RPM two-blade 3 N operating point; actual_backend=CPU; precision=f64; fidelity=PanelGeometrySurrogate".into(),
            "1920x1080 fixed raster canvas, deterministic colour mapping and SVG axes; all rendered fields are scalar presentation data".into(),
            "CSV units: frequency Hz, pressure Pa, baseline pressure Pa, area m², radius fraction; SVG units are declared in captions".into(),
            "rotation.gif is a deterministic panel-loading proxy, not resolved transient flow".into(),
            format!(
                "thrust loss {:.6}; calibrated 1–5 kHz reduction {:.6} dB",
                result.thrust_loss, result.panel_summary.sensitive_band_reduction_db
            ),
            format!(
                "execution evidence: requested_backend={}; actual_backend={}; precision=f64; dispatch={}",
                result.provenance.execution.requested_backend,
                result
                    .provenance
                    .execution
                    .actual_backend
                    .map_or_else(|| "unavailable".into(), |backend| backend.to_string()),
                result.provenance.execution.scalar_or_simd,
            ),
        ],
    )?;
    write_execution_evidence(output_dir, result)?;
    sonium_xem::io::validate_gallery_manifest(&manifest_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Blade3dConfig, optimize_blade_shape};

    fn test_config() -> Propeller3dOptConfig {
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

    #[test]
    fn gallery_bundle_registers_finite_csv_sources() {
        let config = test_config();
        let result = optimize_blade_shape(&config);
        let directory = tempfile::tempdir().unwrap();

        write_gallery_bundle(directory.path(), &config, &result).unwrap();
        let validation = validate_gallery_bundle(directory.path(), &config, &result).unwrap();
        assert_eq!(validation.figure_count, 10);
        assert_eq!(validation.spectrum_rows, result.spectrum.harmonics.len());
        assert_eq!(validation.panel_load_rows, result.mesh.panels.len());
        let manifest =
            sonium_xem::io::validate_gallery_manifest(directory.path().join("gallery.json"))
                .unwrap();
        assert_eq!(manifest.figures.len(), 10);

        let spectrum =
            std::fs::read_to_string(directory.path().join("source_spectrum.csv")).unwrap();
        assert_eq!(
            spectrum.lines().next(),
            Some("frequency_hz,pressure_pa,baseline_pressure_pa")
        );
        assert_eq!(
            spectrum.lines().count(),
            result.spectrum.harmonics.len() + 1
        );
        assert!(spectrum.lines().skip(1).all(|row| {
            row.split(',').count() == 3
                && row
                    .split(',')
                    .all(|value| value.parse::<f64>().unwrap().is_finite())
        }));

        let loads =
            std::fs::read_to_string(directory.path().join("blade_load_distribution.csv")).unwrap();
        assert_eq!(
            loads.lines().next(),
            Some("radius_fraction,loading_proxy_m2")
        );
        assert_eq!(loads.lines().count(), result.mesh.panels.len() + 1);
        assert!(loads.lines().skip(1).all(|row| {
            row.split(',').count() == 2
                && row
                    .split(',')
                    .all(|value| value.parse::<f64>().unwrap().is_finite())
        }));
    }

    #[test]
    fn gallery_bundle_rejects_result_provenance_drift() {
        let config = test_config();
        let result = optimize_blade_shape(&config);
        let directory = tempfile::tempdir().unwrap();
        write_gallery_bundle(directory.path(), &config, &result).unwrap();

        let mut drifted = result.clone();
        drifted.spectrum.harmonics[0].pressure_pa += 0.25;
        let error = validate_gallery_bundle(directory.path(), &config, &drifted).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("non-admissible production result")
        );

        let mut panel_drifted = result.clone();
        panel_drifted.mesh.panels[0].area += 0.25;
        let error = validate_gallery_bundle(directory.path(), &config, &panel_drifted).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("non-admissible production result")
        );
    }

    #[test]
    fn gallery_bundle_rejects_before_writing_any_artifact() {
        let config = test_config();
        let mut result = optimize_blade_shape(&config);
        result.loads.shaft_power_w = f64::NAN;
        let directory = tempfile::tempdir().unwrap();

        let error = write_gallery_bundle(directory.path(), &config, &result).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("non-admissible production result")
        );
        for artifact in [
            "optimized_blade.vtu",
            "baseline_blade.vtu",
            "source_spectrum.csv",
            "blade_load_distribution.csv",
            "gallery.json",
        ] {
            assert!(!directory.path().join(artifact).exists(), "left {artifact}");
        }
    }

    #[test]
    fn gallery_bundle_replaces_existing_output_without_stale_artifacts() {
        let config = test_config();
        let result = optimize_blade_shape(&config);
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("gallery");

        write_gallery_bundle(&output, &config, &result).unwrap();
        std::fs::write(output.join("stale.txt"), b"must not survive").unwrap();
        let before = std::fs::read(output.join("gallery.json")).unwrap();

        write_gallery_bundle(&output, &config, &result).unwrap();

        assert!(!output.join("stale.txt").exists());
        assert!(output.join("execution_evidence.json").is_file());
        assert_eq!(std::fs::read(output.join("gallery.json")).unwrap(), before);
        validate_gallery_bundle(&output, &config, &result).unwrap();
    }

    #[test]
    fn failed_replacement_keeps_the_previous_valid_gallery() {
        let config = test_config();
        let result = optimize_blade_shape(&config);
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("gallery");
        write_gallery_bundle(&output, &config, &result).unwrap();
        let before = std::fs::read(output.join("gallery.json")).unwrap();

        let mut invalid = result.clone();
        invalid.loads.shaft_power_w = f64::NAN;
        assert!(write_gallery_bundle(&output, &config, &invalid).is_err());

        assert_eq!(std::fs::read(output.join("gallery.json")).unwrap(), before);
        validate_gallery_bundle(&output, &config, &result).unwrap();
    }

    #[test]
    fn gallery_bundle_rejects_artifact_checksum_corruption() {
        let config = test_config();
        let result = optimize_blade_shape(&config);
        let directory = tempfile::tempdir().unwrap();
        write_gallery_bundle(directory.path(), &config, &result).unwrap();

        let evidence_path = directory.path().join("execution_evidence.json");
        let mut evidence = serde_json::from_slice::<PropellerArtifactEvidence>(
            &std::fs::read(&evidence_path).unwrap(),
        )
        .unwrap();
        let checksum = evidence
            .artifact_checksums_sha256
            .get_mut("spectrum.svg")
            .unwrap();
        checksum.replace_range(..1, if checksum.starts_with('0') { "1" } else { "0" });
        std::fs::write(evidence_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();

        let error = validate_gallery_bundle(directory.path(), &config, &result).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("checksum mismatch for spectrum.svg")
        );
    }
}
