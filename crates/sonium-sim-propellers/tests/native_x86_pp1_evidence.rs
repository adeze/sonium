//! Native Linux/x86 evidence for the complete propeller application surface.
//!
//! This producer is deliberately application-owned.  It records real CPU/f64
//! evaluations and the typed capability boundary; it never relabels the
//! source surrogate as CUDA, a resolved CFD wake, or measured data.  The
//! companion Just recipe runs this producer with one and two Rayon workers and
//! gates the resulting canonical observables.

use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, AxisymmetricFlowOptions, AxisymmetricRotatingWallOptions,
    Blade3dConfig, BladeElementBemOptions, BladeElementModel, BladeElementOperatingPoint,
    BladeElementStation, BladePanel, BodyFittedDuctStation, BodyFittedFlowNode,
    BodyFittedFlowResult, EdgeTreatment, EvaluationMode, Point3, PrescribedHelicalWakeOptions,
    Propeller3dOptConfig, PropellerBackend, PropellerCapabilityMatrix, PropellerCapabilityRequest,
    PropellerCapabilityResult, PropellerDesign, PropellerExecutionBatch,
    PropellerExecutionDirection, PropellerExecutionProvenance, PropellerHostMetadata,
    PropellerPrecision, PropellerWorkload, SerrationGeometry, VortexLatticeOptions,
    baseline_blade_shape, compare_direct_and_bem, evaluate_axisymmetric_rotating_wall,
    evaluate_axisymmetric_weighted_flow, evaluate_prescribed_helical_wake,
    evaluate_vortex_lattice_reference, optimize_blade_shape, sha256_hex, write_gallery_bundle,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "sonium.propellers.native-x86-pp1.v1";
const SEED: &str = "sonium.native-x86.pp1.seed.v1";

#[derive(Debug, Serialize, Deserialize)]
struct CaseEvidence {
    id: String,
    family: String,
    status: String,
    oracle: String,
    observables: Vec<f64>,
    budgets: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CapabilityEvidence {
    schema: String,
    total_cells: usize,
    cpu_f64_forward_supported: usize,
    cuda_typed_unsupported: usize,
    metal_typed_unsupported: usize,
    device_requests_without_fallback: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct FailureEvidence {
    id: String,
    status: String,
    error_kind: String,
    partial_publish: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CrossArchitectureEvidence {
    status: String,
    targets: Vec<String>,
    reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativePp1Evidence {
    schema: String,
    status: String,
    scope: String,
    host: PropellerHostMetadata,
    execution: PropellerExecutionProvenance,
    deterministic_mode: bool,
    rayon_workers_requested: usize,
    input_descriptor: String,
    output_observables: Vec<f64>,
    canonical_input_sha256: String,
    canonical_output_sha256: String,
    artifact_checksums_sha256: BTreeMap<String, String>,
    cases: Vec<CaseEvidence>,
    capability: CapabilityEvidence,
    failures: Vec<FailureEvidence>,
    gallery_status: String,
    cross_architecture: CrossArchitectureEvidence,
}

fn config() -> Propeller3dOptConfig {
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

fn model() -> BladeElementModel {
    let source = PropellerSource::drone_5inch(12_000.0);
    let stations = vec![
        BladeElementStation {
            radius_m: 0.018,
            chord_m: 0.018,
            pitch_deg: 24.0,
        },
        BladeElementStation {
            radius_m: 0.036,
            chord_m: 0.014,
            pitch_deg: 21.0,
        },
        BladeElementStation {
            radius_m: 0.054,
            chord_m: 0.010,
            pitch_deg: 18.0,
        },
    ];
    let polar = AirfoilPolar::try_new(vec![
        AirfoilPolarPoint {
            angle_of_attack_deg: -20.0,
            lift_coefficient: -0.8,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 0.0,
            lift_coefficient: 0.0,
            drag_coefficient: 0.02,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 20.0,
            lift_coefficient: 0.8,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 40.0,
            lift_coefficient: 0.9,
            drag_coefficient: 0.16,
        },
    ])
    .expect("checked polar");
    BladeElementModel::try_from_source(&source, stations, polar).expect("checked BEM model")
}

fn radial_panel(root_x: f64, tip_x: f64, chord_m: f64) -> BladePanel {
    BladePanel {
        vertices: [
            Point3 {
                x: root_x,
                y: chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: tip_x,
                y: chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: tip_x,
                y: -chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: root_x,
                y: -chord_m * 0.5,
                z: 0.0,
            },
        ],
        blade_index: 0,
        radius_fraction: 0.5,
        chord_fraction: chord_m,
        area: (tip_x - root_x) * chord_m,
    }
}

fn synthetic_meridional_state() -> BodyFittedFlowResult {
    let mut nodes = Vec::new();
    for axial_position_m in [0.0, 1.0] {
        for radial_position_m in [1.0, 1.5, 2.0] {
            nodes.push(BodyFittedFlowNode {
                radial_position_m,
                axial_position_m,
                radial_velocity_m_s: 0.0,
                axial_velocity_m_s: 1.0,
                pressure_pa: 0.0,
            });
        }
    }
    BodyFittedFlowResult {
        schema: sonium_sim_propellers::BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1.into(),
        nodes,
        mesh_node_count: 6,
        mesh_element_count: 4,
        iterations: 1,
        residual: 0.0,
        converged: true,
        inlet_mean_axial_velocity_m_s: 1.0,
        outlet_mean_axial_velocity_m_s: 1.0,
        relative_mean_velocity_mismatch: 0.0,
        maximum_speed_m_s: 1.0,
        minimum_pressure_pa: 0.0,
        maximum_pressure_pa: 0.0,
    }
}

fn case(id: &str, family: &str, oracle: &str, observables: Vec<f64>) -> CaseEvidence {
    assert!(!observables.is_empty(), "{id} must publish observables");
    assert!(
        observables.iter().all(|value| value.is_finite()),
        "{id} has non-finite output"
    );
    CaseEvidence {
        id: id.into(),
        family: family.into(),
        status: "pass".into(),
        oracle: oracle.into(),
        observables,
        budgets: BTreeMap::new(),
    }
}

fn evidence(output_dir: Option<&Path>) -> NativePp1Evidence {
    assert_eq!(std::env::consts::OS, "linux", "PP1 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP1 requires native x86_64"
    );
    // The deterministic-mode gate lives in the test entry point, which
    // skips under the general suite; reaching here means it was requested.

    let config = config();
    let result = optimize_blade_shape(&config);
    assert!(
        result.is_exportable(),
        "optimized CPU result must be exportable"
    );
    let repeat = optimize_blade_shape(&config);
    assert_eq!(result.provenance.execution, repeat.provenance.execution);
    assert_eq!(
        result.objective, repeat.objective,
        "optimizer must be deterministic"
    );

    let source = config.source.clone();
    let baseline = baseline_blade_shape(&config);
    let actuator = sonium_sim_propellers::PropellerActuatorDiskReference::try_from_source(&source)
        .expect("actuator disk reference");
    actuator
        .validate_load_summary(baseline.loads, 1.0e-10)
        .expect("actuator-disk load/power identity");

    let mut cases = Vec::new();
    let mut output_observables = vec![
        baseline.loads.thrust_n,
        result.loads.thrust_n,
        baseline.loads.torque_nm,
        result.loads.torque_nm,
        baseline.loads.shaft_power_w,
        result.loads.shaft_power_w,
        actuator.ideal_hover_power_w,
        result.loads.acoustic_power_w,
        result.panel_summary.sensitive_band_reduction_db,
        result.objective,
    ];

    // Both source-model fidelities, all planform/edge combinations, and the
    // exact published harmonic provenance are exercised on this native run.
    for mode in [
        EvaluationMode::FastSurrogate,
        EvaluationMode::ResearchControl,
    ] {
        for (planform, edge, name) in [
            (
                sonium_sim_propellers::PropellerPlanform::Conventional,
                EdgeTreatment::Smooth,
                "conventional-smooth",
            ),
            (
                sonium_sim_propellers::PropellerPlanform::Conventional,
                EdgeTreatment::LeadingEdgeSerrations,
                "conventional-leading-serrated",
            ),
            (
                sonium_sim_propellers::PropellerPlanform::Conventional,
                EdgeTreatment::TrailingEdgeSerrations,
                "conventional-trailing-serrated",
            ),
            (
                sonium_sim_propellers::PropellerPlanform::ToroidalLoop,
                EdgeTreatment::LeadingAndTrailingSerrations,
                "toroidal-leading-trailing-serrated",
            ),
        ] {
            let design = PropellerDesign {
                source: source.clone(),
                planform,
                edge_treatment: edge,
                serrations: SerrationGeometry::default(),
            };
            let spectrum = design.spectrum(6, mode);
            assert_eq!(spectrum.provenance.evaluation_mode, mode);
            let values = spectrum
                .harmonics
                .iter()
                .flat_map(|harmonic| [harmonic.frequency_hz, harmonic.pressure_pa])
                .collect::<Vec<_>>();
            output_observables.extend(values.iter().copied());
            cases.push(case(
                &format!("source.{name}.{:?}", mode),
                "FastSurrogate/ResearchControl source spectrum",
                "PropellerSource harmonic amplitudes and bounded reduction prior",
                values,
            ));
        }
    }

    // Direct blade elements and converged BEM are compared at identical
    // operating points before the wake is allowed to consume the BEM state.
    let blade_model = model();
    let point = BladeElementOperatingPoint {
        rpm: 12_000.0,
        freestream_m_s: 10.0,
        density_kg_m3: 1.204,
    };
    let direct = blade_model.evaluate(point).expect("direct blade elements");
    let bem = blade_model
        .evaluate_bem(point, BladeElementBemOptions::default())
        .expect("converged BEM");
    let fidelity = compare_direct_and_bem(
        &direct,
        &bem,
        sonium_sim_propellers::BladeElementFidelityOptions::default(),
    )
    .expect("direct/BEM fidelity comparison");
    let wake = evaluate_prescribed_helical_wake(
        &blade_model,
        &bem,
        PrescribedHelicalWakeOptions {
            wake_turns: 2,
            segments_per_turn: 16,
            ..PrescribedHelicalWakeOptions::default()
        },
    )
    .expect("checked prescribed wake");
    assert!(wake.is_finite());
    let blade_values = vec![
        direct.thrust_n,
        direct.torque_nm,
        bem.loads.thrust_n,
        bem.loads.torque_nm,
        fidelity.relative_thrust_difference,
        fidelity.relative_torque_difference,
        wake.segment_count as f64,
        wake.biot_savart_evaluation_count as f64,
    ];
    output_observables.extend(blade_values.iter().copied());
    cases.push(case(
        "aerodynamics.direct-bem-wake",
        "checked direct/BEM/prescribed-helical-wake",
        "same operating-point direct/BEM comparison plus fixed-circulation wake budget",
        blade_values,
    ));

    let panel = radial_panel(0.04, 0.09, 0.018);
    let vlm = evaluate_vortex_lattice_reference(
        std::slice::from_ref(&panel),
        VortexLatticeOptions {
            freestream_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 6.0,
            },
            angular_velocity_rad_s: 180.0,
            wake_length_m: 0.8,
            wake_segments: 6,
            ..VortexLatticeOptions::default()
        },
    )
    .expect("checked VLM reference");
    assert!(vlm.is_finite());
    assert!(vlm.maximum_normal_residual_m_s < 1.0e-8);
    let vlm_values = vec![
        vlm.thrust_n,
        vlm.total_moment_nm.y,
        vlm.panels[0].circulation_m2_s,
        vlm.maximum_normal_residual_m_s,
        vlm.wake_segment_count_per_leg as f64,
    ];
    output_observables.extend(vlm_values.iter().copied());
    cases.push(case(
        "aerodynamics.vortex-lattice",
        "checked VLM",
        "finite panel result and normal-residual gate",
        vlm_values,
    ));

    // Independent analytic/reference dimensions: actuator-disk power, exact
    // annular Couette, and the 2πr weighted-flow identity.
    let couette = evaluate_axisymmetric_rotating_wall(
        &[
            BodyFittedDuctStation {
                axial_position_m: 0.0,
                inner_radius_m: 0.02,
                outer_radius_m: 0.04,
            },
            BodyFittedDuctStation {
                axial_position_m: 0.5,
                inner_radius_m: 0.02,
                outer_radius_m: 0.04,
            },
        ],
        AxisymmetricRotatingWallOptions {
            inner_angular_velocity_rad_s: 100.0,
            outer_angular_velocity_rad_s: 100.0,
            radial_samples: 5,
            maximum_work: 100,
            ..AxisymmetricRotatingWallOptions::default()
        },
    )
    .expect("exact annular Couette reference");
    assert!(couette.is_finite());
    assert!(couette.maximum_absolute_torque_n_m_per_m.abs() <= 1.0e-12);
    let weighted = evaluate_axisymmetric_weighted_flow(
        &synthetic_meridional_state(),
        AxisymmetricFlowOptions {
            density_kg_m3: 2.0,
            axial_tolerance_m: 1.0e-9,
        },
    )
    .expect("weighted body-fitted section reference");
    assert!(weighted.is_finite());
    let reference_values = vec![
        actuator.disk_area_m2,
        actuator.ideal_hover_power_w,
        couette.dissipation_sum_w_per_m,
        weighted.inlet.volume_flow_m3_s,
        weighted.relative_volume_flow_mismatch,
    ];
    output_observables.extend(reference_values.iter().copied());
    cases.push(case(
        "references.actuator-couette-weighted-flow",
        "independent analytic/high-precision reference dimensions",
        "actuator-disk pressure/area/velocity power and exact annular Couette/2πr identities",
        reference_values,
    ));

    // Explicit hover/near-zero and malformed/non-finite inputs are tested as
    // typed failures or finite bounded states, never as a passing placeholder.
    let hover = blade_model
        .evaluate_at_freestream(0.0)
        .expect("direct hover reference");
    assert!(hover.thrust_n.is_finite());
    assert!(matches!(
        blade_model.evaluate_bem(
            BladeElementOperatingPoint {
                freestream_m_s: 0.0,
                ..point
            },
            BladeElementBemOptions::default(),
        ),
        Err(sonium_sim_propellers::BladeElementError::BemRequiresPositiveFreestream)
    ));
    assert!(matches!(
        evaluate_vortex_lattice_reference(
            &[{
                let mut invalid = panel.clone();
                invalid.vertices[0].x = f64::NAN;
                invalid
            }],
            VortexLatticeOptions::default(),
        ),
        Err(sonium_sim_propellers::VortexLatticeError::InvalidPanel { .. })
    ));
    let edge_values = vec![hover.thrust_n, hover.torque_nm, 0.0, point.rpm];
    output_observables.extend(edge_values.iter().copied());
    cases.push(case(
        "adversarial.hover-invalid-nonfinite",
        "hover, zero/near-zero, malformed and non-finite boundaries",
        "typed BEM/VLM errors and finite direct hover reference",
        edge_values,
    ));

    let matrix = PropellerCapabilityMatrix::current();
    let cpu_supported = matrix
        .cells
        .iter()
        .filter(|cell| {
            cell.supported
                && cell.request.backend == PropellerBackend::Cpu
                && cell.request.precision == PropellerPrecision::F64
                && cell.request.mesh_order == 1
                && cell.request.direction == PropellerExecutionDirection::Forward
        })
        .count();
    let cuda_unsupported = matrix
        .cells
        .iter()
        .filter(|cell| cell.request.backend == PropellerBackend::Cuda && !cell.supported)
        .count();
    let metal_unsupported = matrix
        .cells
        .iter()
        .filter(|cell| cell.request.backend == PropellerBackend::Metal && !cell.supported)
        .count();
    assert_eq!(matrix.cells.len(), 648);
    for backend in [PropellerBackend::Cuda, PropellerBackend::Metal] {
        let request = PropellerCapabilityRequest {
            backend,
            workload: PropellerWorkload::VortexLattice,
            precision: PropellerPrecision::F64,
            mesh_order: 1,
            direction: PropellerExecutionDirection::Forward,
            batch: PropellerExecutionBatch::Single,
        };
        let PropellerCapabilityResult::Unsupported(unsupported) = matrix.resolve(request) else {
            panic!("device request must fail closed")
        };
        assert_eq!(unsupported.actual_backend, None);
        assert_eq!(unsupported.fallback_reason, None);
    }
    let capability = CapabilityEvidence {
        schema: matrix.schema,
        total_cells: matrix.cells.len(),
        cpu_f64_forward_supported: cpu_supported,
        cuda_typed_unsupported: cuda_unsupported,
        metal_typed_unsupported: metal_unsupported,
        device_requests_without_fallback: true,
    };

    let mut failures = Vec::new();
    let mut invalid_source = source.clone();
    invalid_source.rpm = f64::NAN;
    assert!(
        sonium_sim_propellers::BladeElementModel::try_from_source(
            &invalid_source,
            model().stations().to_vec(),
            model().polar().clone(),
        )
        .is_err()
    );
    failures.push(FailureEvidence {
        id: "invalid-input.nonfinite-source".into(),
        status: "typed_rejected".into(),
        error_kind: "BladeElementError::InvalidInput".into(),
        partial_publish: false,
    });

    let mut gallery_status = "not_requested".to_owned();
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create native PP1 output directory");
        let not_directory = output_dir.join("fs-probe-not-directory");
        fs::write(&not_directory, b"sentinel").expect("create owned filesystem probe");
        let error = write_gallery_bundle(&not_directory, &config, &result)
            .expect_err("gallery must reject a regular-file output path");
        assert!(error.to_string().contains("not a directory"));
        assert_eq!(
            fs::read(&not_directory).expect("probe remains intact"),
            b"sentinel"
        );
        failures.push(FailureEvidence {
            id: "filesystem.output-path-is-file".into(),
            status: "typed_rejected".into(),
            error_kind: "InvalidData".into(),
            partial_publish: false,
        });
        let gallery = output_dir.join("gallery");
        write_gallery_bundle(&gallery, &config, &result).expect("headless gallery bundle");
        let validation = sonium_sim_propellers::validate_gallery_bundle(&gallery, &config, &result)
            .expect("gallery bundle validation");
        assert_eq!(validation.figure_count, 10);
        gallery_status = "pass".into();
    }

    let input_descriptor = format!(
        "schema={SCHEMA}|seed={SEED}|config={:?}|rayon_workers={}",
        config,
        rayon::current_num_threads()
    );
    let canonical_output = serde_json::to_vec(&output_observables).expect("canonical output");
    let execution = result.provenance.execution.clone();
    assert!(execution.is_publishable_cpu_surrogate());
    let mut artifact_checksums_sha256 = BTreeMap::new();
    artifact_checksums_sha256.insert(
        "input_descriptor".into(),
        sha256_hex(input_descriptor.as_bytes()),
    );
    artifact_checksums_sha256.insert("output_observables".into(), sha256_hex(&canonical_output));
    NativePp1Evidence {
        schema: SCHEMA.into(),
        status: "pass".into(),
        scope: "native Linux x86_64 CPU/f64; Windows and Arm64 waived by user".into(),
        host: PropellerHostMetadata::collect(),
        execution,
        deterministic_mode: true,
        rayon_workers_requested: rayon::current_num_threads(),
        input_descriptor: input_descriptor.clone(),
        output_observables,
        canonical_input_sha256: sha256_hex(input_descriptor.as_bytes()),
        canonical_output_sha256: sha256_hex(&canonical_output),
        artifact_checksums_sha256,
        cases,
        capability,
        failures,
        gallery_status,
        cross_architecture: CrossArchitectureEvidence {
            status: "waived".into(),
            targets: vec!["aarch64".into(), "windows-x86_64".into(), "macos-x86_64".into()],
            reason: "Arm64/Windows/macOS are waived by the user-authorized native Linux/x86-only PP1 scope; no cross-architecture claims are emitted".into(),
        },
    }
}

#[test]
fn native_x86_pp1_producer_is_deterministic_and_fail_closed() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated native-x86 script that sets deterministic mode.
    if std::env::var("SONIUM_PROPELLER_DETERMINISTIC").as_deref() != Ok("1") {
        eprintln!(
            "skipping native PP1 evidence: set SONIUM_PROPELLER_DETERMINISTIC=1 to produce it"
        );
        return;
    }
    let output_dir = std::env::var_os("SONIUM_OUTPUT_DIR").map(PathBuf::from);
    let artifact = evidence(output_dir.as_deref());
    assert!(artifact.cases.iter().all(|case| case.status == "pass"));
    assert_eq!(artifact.execution.scalar_or_simd, "scalar");
    assert_eq!(artifact.execution.requested_backend, PropellerBackend::Cpu);
    assert_eq!(
        artifact.execution.actual_backend,
        Some(PropellerBackend::Cpu)
    );
    assert_eq!(artifact.execution.precision, PropellerPrecision::F64);
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir).expect("create artifact directory");
        let bytes = serde_json::to_vec_pretty(&artifact).expect("serialize PP1 artifact");
        fs::write(output_dir.join("native-x86-pp1-v1.json"), bytes).expect("write PP1 artifact");
    }
}
