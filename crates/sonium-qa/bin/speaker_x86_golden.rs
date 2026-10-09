//! Native-x86 deterministic speaker golden artifact.
//!
//! This runs the CPU/f64 reference on a native x86_64 runner, or in the
//! explicit `SONIUM_AARCH64_REFERENCE_ONLY=1` producer mode used to create an
//! accepted native-aarch64 comparison artifact. It records dispatch
//! information and a schema/checksum-bearing artifact. Rosetta is not treated
//! as native x86 evidence.

use serde::{Deserialize, Serialize};
use serde_json::json;
use sonium_qa::{
    QaContextReuse, QaExecutionCounts, QaFailureStage, QaMemoryEvidence, QaResult,
    QaRuntimeCounters, QaRuntimeEvidence, host_hardware_provenance, write_qa_artifact,
};
use sonium_sim_speaker::{cea2034_output, mlg_loader, simulation};
use sonium_sim_speaker_modeling::{
    DriverModel, DriverModelReferenceArtifact, DriverModelReferenceTolerance, SpeakerDriverStudy,
    conventions::ArtifactProvenance,
};
use std::path::PathBuf;
use std::time::Instant;

const CASE_ID: &str = "speaker.native-x86-golden.v1";
const ARTIFACT_SCHEMA_VERSION: u32 = 3;
const COMPARISON_TOLERANCE: f64 = 1.0e-8;

fn failed_x86(error: String) -> QaResult {
    let requested = if std::env::var("SONIUM_AARCH64_REFERENCE_ONLY")
        .ok()
        .as_deref()
        == Some("1")
    {
        "cpu-native-aarch64-reference"
    } else {
        "cpu-native-x86_64"
    };
    QaResult::failed(CASE_ID, error.clone())
        .requested_backend(requested)
        .actual_backend("unavailable")
        .backend_failed(requested, error)
        .failure_stage(QaFailureStage::Environment)
        .hardware_provenance(host_hardware_provenance())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComplexSample {
    re: f64,
    im: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NativeX86GoldenArtifact {
    schema_version: u32,
    target_arch: String,
    #[serde(default)]
    target_os: String,
    #[serde(default)]
    cpu_model: String,
    #[serde(default)]
    microcode: String,
    #[serde(default)]
    compiler: String,
    #[serde(default)]
    cpu_vendor: String,
    #[serde(default)]
    runtime_features: serde_json::Value,
    dispatch: Vec<String>,
    model: sonium_sim_speaker::model::LoudspeakerModel,
    solver_config: simulation::SolverConfig,
    frequencies: Vec<f64>,
    directivity: Vec<Vec<ComplexSample>>,
    cea_on_axis: Vec<f64>,
    cea_preference_score: f64,
    reduced_order_preview: simulation::SpeakerPreviewArtifact,
    residual_gate: simulation::ResidualGateReport,
    driver_model_reference: DriverModelReferenceArtifact,
    projected_driver_model_reference: DriverModelReferenceArtifact,
}

#[derive(Debug, Clone, Copy)]
struct ComparisonMetrics {
    field: f64,
    impedance: f64,
    power: f64,
    residual: f64,
    gradient: f64,
}

fn dispatch_metadata() -> Vec<String> {
    let mut dispatch = vec![
        format!("target_arch={}", std::env::consts::ARCH),
        "scalar=f64".into(),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        dispatch.push(format!("avx2={}", std::is_x86_feature_detected!("avx2")));
        dispatch.push(format!("fma={}", std::is_x86_feature_detected!("fma")));
        dispatch.push(format!(
            "sse4.2={}",
            std::is_x86_feature_detected!("sse4.2")
        ));
    }
    #[cfg(target_arch = "aarch64")]
    dispatch.push("simd=neon".into());
    dispatch.push(format!("vendor={}", host_cpu_identity()));
    dispatch
}

fn runtime_feature_metadata() -> serde_json::Value {
    #[cfg(target_arch = "x86_64")]
    {
        return json!({
            "avx2": std::is_x86_feature_detected!("avx2"),
            "fma": std::is_x86_feature_detected!("fma"),
            "sse4.2": std::is_x86_feature_detected!("sse4.2"),
            "avx512f": std::is_x86_feature_detected!("avx512f"),
        });
    }
    #[cfg(target_arch = "aarch64")]
    {
        return json!({"neon": true});
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    json!({})
}

fn validate_aarch64_dispatch(dispatch: &[String]) -> Result<(), String> {
    if !dispatch.iter().any(|value| value == "simd=neon") {
        return Err("accepted aarch64 golden is missing simd=neon dispatch metadata".into());
    }
    let has_cpu_identity = dispatch.iter().any(|value| {
        value.strip_prefix("vendor=").is_some_and(|identity| {
            !identity.trim().is_empty() && !identity.eq_ignore_ascii_case("unknown")
        })
    });
    if !has_cpu_identity {
        return Err(
            "accepted aarch64 golden is missing a non-unknown vendor dispatch identity".into(),
        );
    }
    Ok(())
}

/// Return a stable, host-provided CPU identity on the native reference
/// platforms. Linux exposes the vendor in `/proc/cpuinfo`; macOS does not
/// mount that pseudo-file, so query the read-only `sysctl` interface instead.
/// The identity is metadata only and never controls numerical dispatch.
fn host_cpu_identity() -> String {
    if let Some(identity) = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|cpuinfo| {
            cpuinfo.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == "vendor_id")
                    .then(|| value.trim())
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            })
        })
    {
        return identity;
    }

    #[cfg(target_os = "macos")]
    {
        for key in ["machdep.cpu.brand_string", "hw.model"] {
            let output = std::process::Command::new("sysctl")
                .args(["-n", key])
                .output();
            if let Ok(output) = output {
                if output.status.success() {
                    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                    if !value.is_empty() {
                        return format!("{key}={value}");
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    if let Ok(identifier) = std::env::var("PROCESSOR_IDENTIFIER") {
        if !identifier.trim().is_empty() {
            return identifier;
        }
    }

    "unknown".into()
}

fn compare_aarch64(
    artifact: &NativeX86GoldenArtifact,
    path: &std::path::Path,
) -> Result<ComparisonMetrics, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read accepted aarch64 golden {}: {error}", path.display()))?;
    let reference: NativeX86GoldenArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode accepted aarch64 golden: {error}"))?;
    validate_aarch64_dispatch(&reference.dispatch)?;
    if reference.schema_version != ARTIFACT_SCHEMA_VERSION
        || reference.target_arch != "aarch64"
        || reference.frequencies != artifact.frequencies
        || reference.directivity.len() != artifact.directivity.len()
    {
        return Err("aarch64 golden schema, frequency, or shape mismatch".into());
    }
    let actual_model = serde_json::to_vec(&artifact.model)
        .map_err(|error| format!("serialize native x86 model for comparison: {error}"))?;
    let accepted_model = serde_json::to_vec(&reference.model)
        .map_err(|error| format!("serialize aarch64 model for comparison: {error}"))?;
    let actual_solver = serde_json::to_vec(&artifact.solver_config)
        .map_err(|error| format!("serialize native x86 solver config for comparison: {error}"))?;
    let accepted_solver = serde_json::to_vec(&reference.solver_config)
        .map_err(|error| format!("serialize aarch64 solver config for comparison: {error}"))?;
    if actual_model != accepted_model || actual_solver != accepted_solver {
        return Err("native x86/aarch64 model or solver configuration differs".into());
    }
    let mut maximum = 0.0_f64;
    for (actual_row, reference_row) in artifact.directivity.iter().zip(&reference.directivity) {
        if actual_row.len() != reference_row.len() {
            return Err("aarch64 golden directivity row shape mismatch".into());
        }
        for (actual, expected) in actual_row.iter().zip(reference_row) {
            let error = relative_complex_error([actual.re, actual.im], [expected.re, expected.im]);
            maximum = maximum.max(error);
        }
    }
    if maximum > COMPARISON_TOLERANCE {
        return Err(format!(
            "native x86/aarch64 directivity error {maximum:.3e} exceeds {COMPARISON_TOLERANCE:.3e}"
        ));
    }

    if artifact.cea_on_axis.len() != reference.cea_on_axis.len()
        || artifact.reduced_order_preview.frequencies != reference.reduced_order_preview.frequencies
        || artifact.reduced_order_preview.spl.len() != reference.reduced_order_preview.spl.len()
        || artifact.reduced_order_preview.phase.len() != reference.reduced_order_preview.phase.len()
    {
        return Err("aarch64 golden observable or preview shape mismatch".into());
    }
    let mut observable = 0.0_f64;
    for (actual, expected) in artifact.cea_on_axis.iter().zip(&reference.cea_on_axis) {
        observable = observable.max(finite_absolute_difference(*actual, *expected));
    }
    observable = observable.max(finite_absolute_difference(
        artifact.cea_preference_score,
        reference.cea_preference_score,
    ));
    for (actual_row, expected_row) in artifact
        .reduced_order_preview
        .spl
        .iter()
        .zip(&reference.reduced_order_preview.spl)
        .chain(
            artifact
                .reduced_order_preview
                .phase
                .iter()
                .zip(&reference.reduced_order_preview.phase),
        )
    {
        if actual_row.len() != expected_row.len() {
            return Err("aarch64 golden preview row shape mismatch".into());
        }
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            observable = observable.max(finite_absolute_difference(*actual, *expected));
        }
    }
    if observable > COMPARISON_TOLERANCE {
        return Err(format!(
            "native x86/aarch64 observable error {observable:.3e} exceeds {COMPARISON_TOLERANCE:.3e}"
        ));
    }
    let gate_residual = optional_metric_difference(
        artifact.residual_gate.max_relative_residual,
        reference.residual_gate.max_relative_residual,
        "residual-gate residual",
    )?;
    let gate_physical = optional_metric_difference(
        artifact.residual_gate.max_physical_error,
        reference.residual_gate.max_physical_error,
        "residual-gate physical",
    )?;
    if artifact.residual_gate.status != reference.residual_gate.status
        || finite_absolute_difference(
            artifact.residual_gate.tolerance,
            reference.residual_gate.tolerance,
        ) > COMPARISON_TOLERANCE
        || gate_residual > COMPARISON_TOLERANCE
        || gate_physical > COMPARISON_TOLERANCE
    {
        return Err("native x86/aarch64 residual-gate evidence differs".into());
    }

    let native_driver = &artifact.driver_model_reference;
    let accepted_driver = &reference.driver_model_reference;
    if accepted_driver.schema_version != native_driver.schema_version
        || accepted_driver.points.len() != native_driver.points.len()
        || accepted_driver.execution.frequency_grid_identity
            != native_driver.execution.frequency_grid_identity
    {
        return Err("aarch64 typed DriverModel artifact schema/grid mismatch".into());
    }
    let native_canonical = native_driver
        .canonical_json()
        .map_err(|error| format!("serialize native canonical DriverModel artifact: {error}"))?;
    let accepted_canonical = accepted_driver
        .canonical_json()
        .map_err(|error| format!("serialize aarch64 canonical DriverModel artifact: {error}"))?;
    if native_canonical != accepted_canonical {
        return Err("native x86/aarch64 canonical DriverModel artifact bytes differ".into());
    }
    let native_projected = &artifact.projected_driver_model_reference;
    let accepted_projected = &reference.projected_driver_model_reference;
    if native_projected.provenance != accepted_projected.provenance
        || native_projected.schema_version != accepted_projected.schema_version
    {
        return Err("native x86/aarch64 projected DriverModel provenance differs".into());
    }
    native_projected
        .validate_against_golden(native_driver, DriverModelReferenceTolerance::default())
        .map_err(|error| format!("native projected DriverModel replay failed: {error}"))?;
    accepted_projected
        .validate_against_golden(accepted_driver, DriverModelReferenceTolerance::default())
        .map_err(|error| format!("aarch64 projected DriverModel replay failed: {error}"))?;
    native_projected
        .validate_rom_promotion()
        .map_err(|error| format!("native projected DriverModel promotion failed: {error}"))?;
    accepted_projected
        .validate_rom_promotion()
        .map_err(|error| format!("aarch64 projected DriverModel promotion failed: {error}"))?;
    let native_projected_canonical = native_projected
        .canonical_json()
        .map_err(|error| format!("serialize native projected DriverModel artifact: {error}"))?;
    let accepted_projected_canonical = accepted_projected
        .canonical_json()
        .map_err(|error| format!("serialize aarch64 projected DriverModel artifact: {error}"))?;
    if native_projected_canonical != accepted_projected_canonical {
        return Err("native x86/aarch64 projected DriverModel artifact bytes differ".into());
    }
    let mut impedance = 0.0_f64;
    for (native, accepted) in native_driver.points.iter().zip(&accepted_driver.points) {
        if native.frequency_hz != accepted.frequency_hz {
            return Err("aarch64 typed DriverModel frequency mismatch".into());
        }
        let velocity_error = relative_complex_error(
            [native.velocity_real_m_per_s, native.velocity_imag_m_per_s],
            [
                accepted.velocity_real_m_per_s,
                accepted.velocity_imag_m_per_s,
            ],
        );
        impedance = impedance.max(relative_complex_error(
            [
                native.electrical_impedance_real_ohm,
                native.electrical_impedance_imag_ohm,
            ],
            [
                accepted.electrical_impedance_real_ohm,
                accepted.electrical_impedance_imag_ohm,
            ],
        ));
        if velocity_error > COMPARISON_TOLERANCE {
            return Err(format!(
                "native x86/aarch64 velocity field error {velocity_error:.3e} exceeds {COMPARISON_TOLERANCE:.3e}"
            ));
        }
    }
    let power = finite_absolute_difference(
        native_driver
            .execution
            .convergence
            .power_closure_relative_error,
        accepted_driver
            .execution
            .convergence
            .power_closure_relative_error,
    );
    let residual = finite_absolute_difference(
        native_driver.execution.convergence.max_relative_residual,
        accepted_driver.execution.convergence.max_relative_residual,
    );
    let gradient = finite_absolute_difference(
        native_driver.execution.promotion.gradient_relative_error,
        accepted_driver.execution.promotion.gradient_relative_error,
    );
    for (name, value) in [
        ("impedance", impedance),
        ("power", power),
        ("residual", residual),
        ("gradient", gradient),
    ] {
        if value > COMPARISON_TOLERANCE {
            return Err(format!(
                "native x86/aarch64 {name} error {value:.3e} exceeds {COMPARISON_TOLERANCE:.3e}"
            ));
        }
    }
    Ok(ComparisonMetrics {
        field: maximum,
        impedance,
        power,
        residual,
        gradient,
    })
}

fn relative_complex_error(actual: [f64; 2], expected: [f64; 2]) -> f64 {
    if actual
        .iter()
        .chain(expected.iter())
        .any(|value| !value.is_finite())
    {
        return f64::INFINITY;
    }
    let numerator = (actual[0] - expected[0]).hypot(actual[1] - expected[1]);
    let denominator = expected[0].hypot(expected[1]).max(1.0e-30);
    let error = numerator / denominator;
    if error.is_finite() {
        error
    } else {
        f64::INFINITY
    }
}

fn finite_absolute_difference(actual: f64, expected: f64) -> f64 {
    if actual.is_finite() && expected.is_finite() {
        let error = (actual - expected).abs();
        if error.is_finite() {
            error
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn relative_complex_error_rejects_nonfinite_values() {
        assert!(
            super::relative_complex_error([f64::NAN, 0.0], [0.0, 0.0]).is_infinite(),
            "non-finite cross-architecture samples must fail closed"
        );
        assert!(
            super::finite_absolute_difference(f64::NAN, 0.0).is_infinite(),
            "non-finite scalar comparisons must fail closed"
        );
    }

    #[test]
    fn aarch64_dispatch_validation_requires_neon_and_cpu_identity() {
        let valid = vec![
            "target_arch=aarch64".to_owned(),
            "scalar=f64".to_owned(),
            "simd=neon".to_owned(),
            "vendor=Apple M4".to_owned(),
        ];
        assert!(super::validate_aarch64_dispatch(&valid).is_ok());

        for invalid in [
            vec![
                "target_arch=aarch64".to_owned(),
                "vendor=Apple M4".to_owned(),
            ],
            vec![
                "target_arch=aarch64".to_owned(),
                "simd=neon".to_owned(),
                "vendor=unknown".to_owned(),
            ],
            vec![
                "target_arch=aarch64".to_owned(),
                "simd=neon".to_owned(),
                "vendor=".to_owned(),
            ],
        ] {
            assert!(
                super::validate_aarch64_dispatch(&invalid).is_err(),
                "invalid aarch64 dispatch metadata must fail closed: {invalid:?}"
            );
        }
    }
}

fn optional_metric_difference(
    actual: Option<f64>,
    expected: Option<f64>,
    name: &str,
) -> Result<f64, String> {
    match (actual, expected) {
        (Some(actual), Some(expected)) if actual.is_finite() && expected.is_finite() => {
            Ok(finite_absolute_difference(actual, expected))
        }
        (None, None) => Ok(0.0),
        _ => Err(format!(
            "native x86/aarch64 {name} metric presence or finiteness differs"
        )),
    }
}

fn run() -> Result<QaResult, String> {
    let aarch64_reference_only = std::env::var("SONIUM_AARCH64_REFERENCE_ONLY")
        .ok()
        .as_deref()
        == Some("1");
    if aarch64_reference_only && std::env::consts::ARCH != "aarch64" {
        return Err(format!(
            "SONIUM_AARCH64_REFERENCE_ONLY=1 requires target_arch=aarch64, got {}",
            std::env::consts::ARCH
        ));
    }
    if !aarch64_reference_only && std::env::consts::ARCH != "x86_64" {
        return Err(format!(
            "native x86 golden requires target_arch=x86_64, got {}",
            std::env::consts::ARCH
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if !aarch64_reference_only
        && (!std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma"))
    {
        return Err(
            "native x86 golden requires runtime AVX2 and FMA; refusing portable fallback".into(),
        );
    }
    let source = include_str!("../../sonium-sim-speaker/examples/two_way_speaker.mlg");
    let mut model = mlg_loader::load_model(source, "two_way_speaker.mlg")
        .map_err(|error| format!("load x86 golden speaker fixture: {error}"))?;
    model.simulation.freq_start = 500.0;
    model.simulation.freq_end = 1000.0;
    model.simulation.freq_points = 2;
    model.simulation.max_mesh_frequency = Some(1000.0);
    let eval_points = cea2034_output::generate_eval_points([0.0, 0.0, 0.0]);
    let config = simulation::SolverConfig::default();
    let started = Instant::now();
    let result = simulation::run_simulation_with_config(&model, &eval_points, &config)
        .map_err(|error| format!("native x86 CPU golden solve failed: {error}"))?;
    let cea = cea2034_output::compute_spinorama_metrics(&result)
        .map_err(|error| format!("native x86 CEA post-processing failed: {error}"))?;
    let preview = result
        .to_wasm_preview(2, eval_points.len())
        .map_err(|error| format!("native x86 reduced-order artifact failed: {error}"))?;
    let driver = DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver())
        .map_err(|error| format!("build native x86 DriverModel reference: {error}"))?;
    let driver_model_reference = DriverModelReferenceArtifact::from_model(
        &driver,
        &[100.0, 1_000.0],
        &[
            num_complex::Complex64::new(0.2, 0.1),
            num_complex::Complex64::new(0.4, 0.3),
        ],
    )
    .map_err(|error| format!("build native x86 typed reference artifact: {error}"))?;
    driver_model_reference
        .validate_against_model(&driver, DriverModelReferenceTolerance::default())
        .map_err(|error| format!("validate native x86 typed reference artifact: {error}"))?;
    let projected_driver_model_reference = driver_model_reference
        .clone()
        .with_provenance(ArtifactProvenance::wasm_preview());
    projected_driver_model_reference
        .validate_against_golden(
            &driver_model_reference,
            DriverModelReferenceTolerance::default(),
        )
        .map_err(|error| format!("validate native x86 projected reference replay: {error}"))?;
    projected_driver_model_reference
        .validate_rom_promotion()
        .map_err(|error| format!("validate native x86 projected ROM promotion: {error}"))?;
    let host = host_hardware_provenance();
    let target_os = std::env::consts::OS.to_owned();
    let artifact = NativeX86GoldenArtifact {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        target_arch: std::env::consts::ARCH.into(),
        target_os,
        cpu_model: host
            .cpu_model
            .clone()
            .unwrap_or_else(|| "unavailable".into()),
        microcode: host
            .microcode
            .clone()
            .unwrap_or_else(|| "unavailable".into()),
        compiler: host
            .compiler
            .clone()
            .unwrap_or_else(|| "unavailable".into()),
        cpu_vendor: host_cpu_identity(),
        runtime_features: runtime_feature_metadata(),
        dispatch: dispatch_metadata(),
        model: model.clone(),
        solver_config: config,
        frequencies: result.frequencies.clone(),
        directivity: (0..result.frequencies.len())
            .map(|frequency| {
                (0..eval_points.len())
                    .map(|point| {
                        let value = result.complex_pressure[[frequency, point]];
                        ComplexSample {
                            re: value.re,
                            im: value.im,
                        }
                    })
                    .collect()
            })
            .collect(),
        cea_on_axis: cea.on_axis,
        cea_preference_score: cea.preference_score,
        reduced_order_preview: preview,
        residual_gate: result.provenance.residual_gate,
        driver_model_reference,
        projected_driver_model_reference,
    };
    if aarch64_reference_only {
        validate_aarch64_dispatch(&artifact.dispatch)?;
    }
    let bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| format!("serialize native x86 golden: {error}"))?;
    let comparison_error = if let Some(path) = std::env::var_os("SONIUM_AARCH64_GOLDEN") {
        Some(compare_aarch64(&artifact, &PathBuf::from(path))?)
    } else {
        None
    };
    if !aarch64_reference_only
        && std::env::var("SONIUM_REQUIRE_AARCH64_GOLDEN")
            .ok()
            .as_deref()
            == Some("1")
        && comparison_error.is_none()
    {
        return Err(
            "SONIUM_REQUIRE_AARCH64_GOLDEN=1 requires an accepted aarch64 golden artifact".into(),
        );
    }
    let comparison_requested = comparison_error.is_some();
    let mut qa = QaResult::passed(CASE_ID)
        .requested_backend(if aarch64_reference_only {
            "cpu-native-aarch64-reference"
        } else {
            "cpu-native-x86_64"
        })
        .actual_backend(if aarch64_reference_only {
            "cpu-native-aarch64-reference"
        } else {
            "cpu-native-x86_64"
        })
        .backend_passed(if aarch64_reference_only {
            "cpu-native-aarch64-reference"
        } else {
            "cpu-native-x86_64"
        })
        .reference(
            "deterministic f64 CPU speaker BEM golden with CEA/directivity and reduced preview",
        )
        .reference_checksum("analytic/deterministic-fixture:speaker-native-x86-golden-v1")
        .input_checksum("sha256:embedded-two-way-speaker-mlg-v1")
        .dimension(if aarch64_reference_only {
            "native aarch64 3D speaker BEM reference, 2 frequencies, 72 directivity points"
        } else {
            "native x86_64 3D speaker BEM, 2 frequencies, 72 directivity points"
        })
        .formulation("CPU f64 full-order solve; CEA host post-process; reduced preview sidecar")
        .tolerance("native x86 artifact schema/checksum; accepted aarch64 field/impedance/power/residual/gradient comparison <= 1e-8")
        .metric(
            "directivity_samples",
            (result.frequencies.len() * eval_points.len()) as f64,
        )
        .metric("cea_preference_score", artifact.cea_preference_score)
        .metric(
            "native_avx2",
            artifact.dispatch.iter().any(|v| v == "avx2=true") as u8 as f64,
        )
        .metric(
            "native_fma",
            artifact.dispatch.iter().any(|v| v == "fma=true") as u8 as f64,
        )
        .maximum("artifact_schema_version", ARTIFACT_SCHEMA_VERSION as f64)
        .timing_ms("cpu_full_order", started.elapsed().as_millis() as u64)
        .elapsed_ms(started.elapsed().as_millis() as u64);
    qa.metrics.insert(
        "artifact_schema_version".into(),
        artifact.schema_version as f64,
    );
    if let Some(comparison) = comparison_error {
        qa.metrics.insert(
            "x86_aarch64_field_max_relative_error".into(),
            comparison.field,
        );
        qa.metrics.insert(
            "x86_aarch64_impedance_max_relative_error".into(),
            comparison.impedance,
        );
        qa.metrics
            .insert("x86_aarch64_power_error".into(), comparison.power);
        qa.metrics
            .insert("x86_aarch64_residual_error".into(), comparison.residual);
        qa.metrics
            .insert("x86_aarch64_gradient_error".into(), comparison.gradient);
    }
    let (relative, digest) = write_qa_artifact("speaker_native_x86_golden.json", &bytes)?;
    let checksum_contents = format!(
        "{}  {relative}\n",
        digest
            .strip_prefix("sha256:")
            .ok_or_else(|| format!("unexpected golden artifact checksum {digest}"))?
    );
    let (checksum_relative, checksum_digest) = write_qa_artifact(
        "speaker_native_x86_golden.json.sha256",
        checksum_contents.as_bytes(),
    )?;
    let golden_generation = if aarch64_reference_only {
        "deterministic native aarch64 speaker reference"
    } else {
        "deterministic native x86 speaker golden"
    };
    let mut hardware = host_hardware_provenance();
    hardware.target_features = Some(if aarch64_reference_only {
        "neon,scalar-f64".into()
    } else {
        "avx2,fma,sse4.2,scalar-f64".into()
    });
    hardware.runtime_dispatch = Some(if aarch64_reference_only {
        "native-aarch64-reference".into()
    } else {
        "native-x86-runtime-detected".into()
    });
    qa = qa
        .artifact_with_provenance(
            relative,
            "crates/sonium-qa/bin/speaker_x86_golden.rs",
            golden_generation,
            "v2",
            digest,
        )
        .artifact_with_provenance(
            checksum_relative,
            "crates/sonium-qa/bin/speaker_x86_golden.rs",
            "checksum for deterministic native speaker artifact",
            "v1",
            checksum_digest,
        )
        .runtime_evidence(QaRuntimeEvidence {
            context_reuse: QaContextReuse::NotApplicable,
            counts: QaExecutionCounts {
                executed: 1,
                passed: 1,
                ..QaExecutionCounts::default()
            },
            counters: QaRuntimeCounters::default(),
            memory: QaMemoryEvidence::default(),
            hardware,
            ..QaRuntimeEvidence::default()
        })
        .physical_residual(
            artifact
                .residual_gate
                .max_relative_residual
                .unwrap_or_default(),
        );
    qa.environment
        .insert("dispatch".into(), artifact.dispatch.join(","));
    qa.environment.insert(
        "aarch64_comparison".into(),
        if comparison_requested {
            "passed"
        } else {
            "not_requested"
        }
        .into(),
    );
    Ok(qa)
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok(result) => result,
        Err(error) => failed_x86(error),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("native x86 golden QA result must serialize")
    );
    if result.status != sonium_qa::QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
