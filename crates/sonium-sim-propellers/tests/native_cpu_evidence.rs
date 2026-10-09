//! Native CPU artifact and cross-architecture parity evidence for propellers.

use serde::{Deserialize, Serialize};
use sonium_sim_propellers::{
    Blade3dConfig, EvaluationMode, Propeller3dOptConfig, PropellerActuatorDiskReference,
    PropellerBackend, PropellerDesign, PropellerExecutionProvenance, PropellerModelFidelity,
    PropellerPrecision, baseline_blade_shape, optimize_blade_shape,
};

const EVIDENCE_SCHEMA: u32 = 1;
const CROSS_ARCH_ABSOLUTE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Deserialize, Serialize)]
struct NativePropellerEvidence {
    schema: u32,
    status: String,
    target_arch: String,
    execution: PropellerExecutionProvenance,
    dimensions: Vec<String>,
    tolerances: Vec<f64>,
    observables: Vec<f64>,
    #[serde(default)]
    aarch64_comparison_max_abs_error: Option<f64>,
}

fn test_config() -> Propeller3dOptConfig {
    Propeller3dOptConfig {
        mesh: Blade3dConfig {
            radial_segments: 8,
            chord_segments: 4,
            loop_segments: 4,
            ..Blade3dConfig::default()
        },
        n_harmonics: 6,
        max_iterations: 0,
        ..Propeller3dOptConfig::default()
    }
}

fn append_spectrum(observables: &mut Vec<f64>, design: &PropellerDesign, mode: EvaluationMode) {
    observables.extend(
        design
            .spectrum(6, mode)
            .harmonics
            .into_iter()
            .flat_map(|harmonic| [harmonic.frequency_hz, harmonic.pressure_pa]),
    );
}

fn evidence() -> NativePropellerEvidence {
    let config = test_config();
    let result = optimize_blade_shape(&config);
    assert!(
        result.is_exportable(),
        "native CPU result must be admissible"
    );
    let baseline = baseline_blade_shape(&config);
    let actuator_disk = PropellerActuatorDiskReference::try_from_source(&result.design.source)
        .expect("admitted result has an actuator-disk reference");

    let source = config.source.clone();
    let conventional = PropellerDesign::conventional(source.clone());
    let toroidal_serrated = PropellerDesign::toroidal_serrated(source);
    let mut observables = vec![
        baseline.loads.thrust_n,
        result.loads.thrust_n,
        baseline.loads.torque_nm,
        result.loads.torque_nm,
        baseline.loads.shaft_power_w,
        result.loads.shaft_power_w,
        actuator_disk.ideal_hover_power_w,
        result.loads.acoustic_power_w,
        result.panel_summary.sensitive_band_reduction_db,
        result.panel_summary.vortex_source_index,
        result.panel_summary.edge_coherence_index,
        result.panel_summary.structural_rigidity_index,
        result.objective,
    ];
    append_spectrum(
        &mut observables,
        &conventional,
        EvaluationMode::FastSurrogate,
    );
    append_spectrum(
        &mut observables,
        &toroidal_serrated,
        EvaluationMode::FastSurrogate,
    );
    append_spectrum(
        &mut observables,
        &conventional,
        EvaluationMode::ResearchControl,
    );
    append_spectrum(
        &mut observables,
        &toroidal_serrated,
        EvaluationMode::ResearchControl,
    );
    assert!(observables.iter().all(|value| value.is_finite()));

    let dimensions = vec![
        "hover/load-power-ledger/f64".into(),
        "panel-geometry-surrogate/research-control/f64".into(),
        "conventional-vs-toroidal-serrated/fast-surrogate+research-control/f64".into(),
        "actuator-disk-independent-reference/f64".into(),
    ];
    let tolerances = vec![CROSS_ARCH_ABSOLUTE_TOLERANCE; observables.len()];
    assert_eq!(
        result.provenance.execution.requested_backend,
        PropellerBackend::Cpu
    );
    assert_eq!(
        result.provenance.execution.actual_backend,
        Some(PropellerBackend::Cpu)
    );
    assert_eq!(
        result.provenance.execution.precision,
        PropellerPrecision::F64
    );
    assert_eq!(
        result.provenance.execution.fidelity,
        PropellerModelFidelity::PanelGeometrySurrogate
    );
    assert_eq!(result.provenance.execution.scalar_or_simd, "scalar");
    assert!(!result.provenance.execution.cancelled);
    assert!(!result.provenance.execution.device_lost);
    assert_eq!(result.provenance.execution.allocation_failures, 0);

    NativePropellerEvidence {
        schema: EVIDENCE_SCHEMA,
        status: "pass".into(),
        target_arch: std::env::consts::ARCH.into(),
        execution: result.provenance.execution,
        dimensions,
        tolerances,
        observables,
        aarch64_comparison_max_abs_error: None,
    }
}

fn compare_aarch64(
    artifact: &mut NativePropellerEvidence,
    path: &std::path::Path,
) -> Result<(), String> {
    let reference: NativePropellerEvidence = serde_json::from_slice(
        &std::fs::read(path).map_err(|error| format!("read aarch64 evidence: {error}"))?,
    )
    .map_err(|error| format!("decode aarch64 evidence: {error}"))?;
    if reference.schema != EVIDENCE_SCHEMA
        || reference.status != "pass"
        || reference.target_arch != "aarch64"
        || reference.dimensions != artifact.dimensions
        || reference.observables.len() != artifact.observables.len()
        || reference.tolerances.len() != artifact.tolerances.len()
    {
        return Err("aarch64 propeller evidence schema or observable shape mismatch".into());
    }
    let mut maximum = 0.0_f64;
    for ((actual, expected), tolerance) in artifact
        .observables
        .iter()
        .zip(&reference.observables)
        .zip(&artifact.tolerances)
    {
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        if error > *tolerance {
            return Err(format!(
                "native x86/aarch64 propeller error {error:.6e} exceeds {tolerance:.6e}"
            ));
        }
    }
    artifact.aarch64_comparison_max_abs_error = Some(maximum);
    Ok(())
}

#[test]
fn native_cpu_propeller_evidence_v1() {
    let mut artifact = evidence();
    if let Ok(reference) = std::env::var("SONIUM_AARCH64_REFERENCE") {
        compare_aarch64(&mut artifact, std::path::Path::new(&reference)).unwrap();
    }
    if let Ok(output_dir) = std::env::var("SONIUM_OUTPUT_DIR") {
        let output_dir = std::path::Path::new(&output_dir);
        std::fs::create_dir_all(output_dir).unwrap();
        let bytes = serde_json::to_vec_pretty(&artifact).unwrap();
        std::fs::write(
            output_dir.join("sonium_propeller_native_cpu_evidence.json"),
            &bytes,
        )
        .unwrap();
    }
}
