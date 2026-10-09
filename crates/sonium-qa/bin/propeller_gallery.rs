//! Production propeller gallery artifact and fail-closed export QA.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_propellers::{
    Blade3dConfig, Propeller3dOptConfig, PropellerExecutionProvenance,
    gallery::{validate_gallery_bundle, write_gallery_bundle},
    optimize_blade_shape,
};

const CASE_ID: &str = "propeller.gallery.v1";

fn csv_rows(path: &std::path::Path, expected_header: &str) -> Result<usize, String> {
    let contents = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut rows = contents.lines();
    if rows.next() != Some(expected_header) {
        return Err(format!("{path:?} has an unexpected CSV header"));
    }
    let mut count = 0;
    for row in rows {
        let values = row.split(',').collect::<Vec<_>>();
        if values.iter().any(|value| {
            value
                .parse::<f64>()
                .map(|number| !number.is_finite())
                .unwrap_or(true)
        }) {
            return Err(format!("{path:?} contains a non-finite CSV value"));
        }
        count += 1;
    }
    Ok(count)
}

fn run() -> Result<(usize, usize, usize, PropellerExecutionProvenance), String> {
    let config = Propeller3dOptConfig {
        mesh: Blade3dConfig {
            radial_segments: 8,
            chord_segments: 4,
            loop_segments: 4,
            ..Blade3dConfig::default()
        },
        n_harmonics: 4,
        max_iterations: 0,
        ..Propeller3dOptConfig::default()
    };
    let result = optimize_blade_shape(&config);
    let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    write_gallery_bundle(directory.path(), &config, &result).map_err(|error| error.to_string())?;
    let validation = validate_gallery_bundle(directory.path(), &config, &result)
        .map_err(|error| error.to_string())?;

    let manifest = sonium_xem::io::validate_gallery_manifest(directory.path().join("gallery.json"))
        .map_err(|error| error.to_string())?;
    let figure_names = manifest
        .figures
        .iter()
        .map(|figure| figure.filename.as_str())
        .collect::<std::collections::HashSet<_>>();
    for required in [
        "optimized_blade.vtu",
        "baseline_blade.vtu",
        "source_spectrum.csv",
        "blade_load_distribution.csv",
    ] {
        if !directory.path().join(required).is_file() {
            return Err(format!("gallery is missing {required}"));
        }
    }
    if !figure_names.contains("source_spectrum.csv")
        || !figure_names.contains("blade_load_distribution.csv")
    {
        return Err("CSV artifacts are not registered in gallery.json".into());
    }
    let spectrum_rows = csv_rows(
        &directory.path().join("source_spectrum.csv"),
        "frequency_hz,pressure_pa,baseline_pressure_pa",
    )?;
    let load_rows = csv_rows(
        &directory.path().join("blade_load_distribution.csv"),
        "radius_fraction,loading_proxy_m2",
    )?;
    if spectrum_rows != result.spectrum.harmonics.len() || load_rows != result.mesh.panels.len() {
        return Err("gallery CSV row counts do not match production data".into());
    }

    let mut invalid = result.clone();
    invalid.loads.shaft_power_w = f64::NAN;
    let rejected_directory = tempfile::tempdir().map_err(|error| error.to_string())?;
    if write_gallery_bundle(rejected_directory.path(), &config, &invalid).is_ok() {
        return Err("gallery accepted a non-admissible production result".into());
    }
    for artifact in [
        "optimized_blade.vtu",
        "baseline_blade.vtu",
        "source_spectrum.csv",
        "gallery.json",
    ] {
        if rejected_directory.path().join(artifact).exists() {
            return Err(format!("rejected gallery left behind {artifact}"));
        }
    }
    if spectrum_rows != validation.spectrum_rows || load_rows != validation.panel_load_rows {
        return Err("gallery wrapper and library validation row counts differ".into());
    }
    Ok((
        validation.figure_count,
        validation.spectrum_rows,
        validation.panel_load_rows,
        result.provenance.execution,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((figure_count, spectrum_rows, load_rows, execution)) => {
            let mut result = QaResult::passed(CASE_ID)
                .requested_backend(execution.requested_backend.to_string())
                .actual_backend(
                    execution
                        .actual_backend
                        .map_or_else(|| "unavailable".into(), |backend| backend.to_string()),
                )
                .metric("spectrum_csv_rows", spectrum_rows as f64)
            .metric("blade_load_csv_rows", load_rows as f64)
                .metric("gallery_figure_count", figure_count as f64)
                .metric("rejected_gallery_artifact_count", 0.0)
                .reference("production gallery manifest, finite CSV sources, and fail-closed export gate")
                .tolerance("all CSV values finite; manifest checksums valid; rejected export writes zero artifacts")
                .minimum("spectrum_csv_rows", 2.0)
                .minimum("blade_load_csv_rows", 1.0)
                .maximum("rejected_gallery_artifact_count", 0.0)
                .input_checksum(execution.input_checksum_sha256.clone())
                .reference_checksum(execution.output_checksum_sha256.clone())
                .dimension(format!(
                    "CPU/f64/{:?}; fidelity={:?}; dispatch={}",
                    execution.precision, execution.fidelity, execution.scalar_or_simd
                ))
                .formulation(format!(
                    "converged={}; cancelled={}; device_lost={}; allocation_failures={}; transfers={}; synchronizations={}; launches={}",
                    execution.converged,
                    execution.cancelled,
                    execution.device_lost,
                    execution.allocation_failures,
                    execution.transfer_count,
                    execution.synchronization_count,
                    execution.launch_count,
                ))
                .seed(execution.seed.clone())
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            result
                .environment
                .insert("compiler".into(), execution.host.compiler);
            result.environment.insert("os".into(), execution.host.os);
            result
                .environment
                .insert("kernel".into(), execution.host.kernel);
            result
                .environment
                .insert("cpu_model".into(), execution.host.cpu_model);
            result
                .environment
                .insert("microcode".into(), execution.host.microcode);
            result
                .environment
                .insert("target".into(), execution.host.target);
            result
                .environment
                .insert("target_features".into(), execution.host.target_features);
            result.environment.insert(
                "rayon_threads".into(),
                execution.host.rayon_threads.to_string(),
            );
            result
                .environment
                .insert("affinity_policy".into(), execution.host.affinity_policy);
            if let Ok(json) = result.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("unavailable")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PROPELLER GALLERY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
