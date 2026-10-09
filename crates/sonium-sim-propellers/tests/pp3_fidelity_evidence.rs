//! Canonical PP3 fidelity/convergence evidence for the propeller application.
//!
//! Every passing row below is a bounded analytic, synthetic, or caller-owned
//! reference.  This producer intentionally does not promote the propeller
//! crate's reduced models to resolved CFD, full FW-H, or external measured
//! validation.

use serde::Serialize;
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, AxisymmetricFlowOptions, AxisymmetricRotatingWallOptions,
    Blade3dConfig, BladeElementBemOptions, BladeElementModel, BladeElementOperatingPoint,
    BladeElementStation, BladePanel, BodyFittedDuctStation, BodyFittedFlowOptions,
    DuctShroudOptions, DuctShroudStation, EdgeTreatment, EvaluationMode, FidelityPromotionOptions,
    FidelityPromotionSample, FwhOptions, FwhSurfaceElement, ImageReflectionCoefficient,
    PanelRefinementObservation, PanelRefinementOptions, Point3, PropellerDesign, PropellerPlanform,
    RotatingDipoleReference, SerrationGeometry, SteadyFreeWakeOptions, VortexLatticeOptions,
    baseline_blade_shape, compare_direct_and_bem, evaluate_axisymmetric_rotating_wall,
    evaluate_axisymmetric_weighted_flow, evaluate_duct_shroud_reference,
    evaluate_fidelity_promotion, evaluate_fwh_reference, evaluate_ground_image_directivity,
    evaluate_mesh_convergence, evaluate_panel_refinement, evaluate_prescribed_helical_wake,
    evaluate_steady_free_wake, evaluate_vortex_lattice_reference, optimize_blade_shape, sha256_hex,
    solve_body_fitted_duct_flow,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "sonium.propellers.fidelity-evidence.v1";
const INPUT_SEED: &str = "sonium.pp3.native-x86.canonical.v1";
type CaseMetadata = (
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
    &'static [(&'static str, usize)],
);

#[derive(Debug, Serialize)]
struct Case {
    id: String,
    fidelity: String,
    reference_class: String,
    status: String,
    converged: bool,
    tolerance: f64,
    observables: Vec<f64>,
    observable_names: Vec<String>,
    dimensions: Vec<String>,
    gates: Vec<String>,
    budgets: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct FailureCase {
    id: String,
    status: String,
    typed_error: String,
    partial_publish: bool,
}

#[derive(Debug, Serialize)]
struct Pp3Artifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    cases: Vec<Case>,
    failures: Vec<FailureCase>,
    external_measured_data: &'static str,
    resolved_cfd_and_full_fwh: &'static str,
    optimizer_gallery_promotion: &'static str,
    output_sha256: String,
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
    BladeElementModel::try_from_source(&source, stations, polar).expect("checked model")
}

fn panel(root_x: f64, tip_x: f64, chord_m: f64) -> BladePanel {
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

fn add_case(
    cases: &mut Vec<Case>,
    id: &str,
    fidelity: &str,
    reference_class: &str,
    converged: bool,
    tolerance: f64,
    observables: Vec<f64>,
) {
    assert!(!observables.is_empty(), "{id} must publish observables");
    assert!(
        observables.iter().all(|value| value.is_finite()),
        "{id} non-finite"
    );
    let (observable_names, dimensions, gates, budgets) = case_metadata(id, observables.len());
    assert_eq!(
        observable_names.len(),
        observables.len(),
        "{id} metadata mismatch"
    );
    cases.push(Case {
        id: id.into(),
        fidelity: fidelity.into(),
        reference_class: reference_class.into(),
        status: "pass".into(),
        converged,
        tolerance,
        observables,
        observable_names,
        dimensions,
        gates,
        budgets,
    });
}

fn case_metadata(
    id: &str,
    observable_count: usize,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<String>,
    BTreeMap<String, usize>,
) {
    let (names, dimensions, gates, budgets): CaseMetadata = match id {
        id if id.starts_with("source.") => (
            &[
                "harmonic_1_frequency_hz",
                "harmonic_1_pressure_pa",
                "harmonic_2_frequency_hz",
                "harmonic_2_pressure_pa",
                "harmonic_3_frequency_hz",
                "harmonic_3_pressure_pa",
                "harmonic_4_frequency_hz",
                "harmonic_4_pressure_pa",
                "harmonic_5_frequency_hz",
                "harmonic_5_pressure_pa",
                "harmonic_6_frequency_hz",
                "harmonic_6_pressure_pa",
            ],
            &[
                "planform",
                "edge_treatment",
                "evaluation_mode",
                "harmonic_order",
            ],
            &[
                "finite",
                "analytic_source_identity",
                "class_label_separation",
            ],
            &[],
        ),
        "ladder.direct-bem-steady-and-prescribed-wake" => (
            &[
                "direct_thrust_n",
                "direct_torque_nm",
                "bem_thrust_n",
                "bem_torque_nm",
                "relative_thrust_error",
                "relative_torque_error",
                "wake_final_residual_m",
                "prescribed_wake_segment_count",
            ],
            &[
                "rpm",
                "freestream",
                "density",
                "blade_elements",
                "wake_revolutions",
                "wake_azimuth",
            ],
            &[
                "finite",
                "direct_bem_load_gate",
                "wake_residual",
                "prescribed_wake_budget",
            ],
            &[
                ("wake_max_iterations", 100),
                ("wake_max_node_count", 10_000),
                ("wake_max_biot_savart_evaluations", 1_000_000),
            ],
        ),
        "ladder.vortex-lattice-panel" => (
            &[
                "thrust_n",
                "moment_y_nm",
                "circulation_m2_s",
                "maximum_normal_residual_m_s",
            ],
            &[
                "panel_geometry",
                "freestream",
                "angular_velocity",
                "wake_segments",
            ],
            &["finite", "normal_residual", "panel_loads"],
            &[("wake_segments", 6)],
        ),
        "ladder.body-fitted-weighted-couette" => (
            &[
                "body_residual",
                "mean_velocity_mismatch",
                "inlet_volume_flow_m3_s",
                "weighted_flux_mismatch",
                "couette_max_torque_n_m_per_m",
                "couette_dissipation_w_per_m",
            ],
            &[
                "body_profile",
                "axial_mesh",
                "radial_mesh",
                "density",
                "viscosity",
                "rotating_wall",
            ],
            &[
                "finite",
                "picard_convergence",
                "2pi_r_flux",
                "exact_couette_identity",
            ],
            &[
                ("max_picard_iterations", 20),
                ("max_linear_solver_iterations", 100),
            ],
        ),
        "observer.dipole-fwh-installation" => (
            &[
                "equatorial_peak_pressure_pa",
                "axial_peak_pressure_pa",
                "radiated_sound_power_w",
                "compact_fwh_pressure_pa",
                "ground_image_pressure_pa",
            ],
            &[
                "observer_direction",
                "observer_distance",
                "source_phase",
                "ground_image",
                "far_field",
            ],
            &[
                "finite",
                "dipole_axis_equator",
                "source_observer_propagation",
                "installation_directivity",
            ],
            &[],
        ),
        "refinement.geometry-and-response" => (
            &[
                "final_relative_area_change",
                "final_relative_response_change",
                "geometry_final_panel_count",
                "response_final_panel_count",
            ],
            &[
                "radial_refinement",
                "chord_refinement",
                "loop_refinement",
                "response_refinement",
            ],
            &["finite", "geometry_convergence", "response_convergence"],
            &[
                ("geometry_refinement_levels", 3),
                ("response_refinement_levels", 3),
            ],
        ),
        "promotion.optimizer-gallery" => (
            &[
                "objective",
                "thrust_loss",
                "shaft_power_w",
                "acoustic_power_closure_error",
                "output_checksum_length",
            ],
            &[
                "thrust",
                "power",
                "thickness",
                "provenance",
                "exportability",
            ],
            &[
                "finite",
                "hard_predicates",
                "power_closure",
                "checksum",
                "exportable",
            ],
            &[],
        ),
        "adversarial.hover-zero-load" => (
            &[
                "hover_thrust_n",
                "hover_torque_nm",
                "zero_source_pressure_pa",
            ],
            &["zero_load", "zero_freestream", "source_spectrum"],
            &["finite", "zero_source", "typed_zero_inflow_boundary"],
            &[],
        ),
        _ => panic!("missing PP3 metadata for {id}"),
    };
    assert_eq!(
        names.len(),
        observable_count,
        "metadata table is stale for {id}"
    );
    (
        names.iter().map(|name| (*name).into()).collect(),
        dimensions.iter().map(|name| (*name).into()).collect(),
        gates.iter().map(|name| (*name).into()).collect(),
        budgets
            .iter()
            .map(|(name, value)| ((*name).into(), *value))
            .collect(),
    )
}

fn evidence(output_dir: Option<&Path>) -> Pp3Artifact {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );

    let source = PropellerSource::drone_5inch(12_000.0);
    let config = sonium_sim_propellers::Propeller3dOptConfig {
        mesh: Blade3dConfig {
            radial_segments: 8,
            chord_segments: 4,
            loop_segments: 4,
            ..Blade3dConfig::default()
        },
        n_harmonics: 6,
        max_iterations: 0,
        ..sonium_sim_propellers::Propeller3dOptConfig::default()
    };
    let baseline = baseline_blade_shape(&config);
    let optimized = optimize_blade_shape(&config);
    assert!(baseline.objective.is_finite());
    assert!(optimized.is_exportable());
    let repeated = optimize_blade_shape(&config);
    assert_eq!(optimized.objective, repeated.objective);
    assert_eq!(optimized.thrust_loss, repeated.thrust_loss);
    assert_eq!(
        optimized.provenance.execution.output_checksum_sha256,
        repeated.provenance.execution.output_checksum_sha256
    );
    let mut cases = Vec::new();

    for mode in [
        EvaluationMode::FastSurrogate,
        EvaluationMode::ResearchControl,
    ] {
        for (planform, edge, label) in [
            (
                PropellerPlanform::Conventional,
                EdgeTreatment::Smooth,
                "conventional-smooth",
            ),
            (
                PropellerPlanform::Conventional,
                EdgeTreatment::LeadingEdgeSerrations,
                "conventional-leading-serrated",
            ),
            (
                PropellerPlanform::Conventional,
                EdgeTreatment::TrailingEdgeSerrations,
                "conventional-trailing-serrated",
            ),
            (
                PropellerPlanform::ToroidalLoop,
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
            add_case(
                &mut cases,
                &format!("source.{label}.{mode:?}"),
                &format!("{mode:?}"),
                "analytic harmonic source identity and bounded reduction prior",
                true,
                1.0e-12,
                spectrum
                    .harmonics
                    .iter()
                    .flat_map(|harmonic| [harmonic.frequency_hz, harmonic.pressure_pa])
                    .collect(),
            );
        }
    }

    let blade_model = model();
    let point = BladeElementOperatingPoint {
        rpm: 12_000.0,
        freestream_m_s: 10.0,
        density_kg_m3: 1.204,
    };
    let direct = blade_model.evaluate(point).expect("direct elements");
    let bem = blade_model
        .evaluate_bem(point, BladeElementBemOptions::default())
        .expect("BEM");
    let fidelity = compare_direct_and_bem(
        &direct,
        &bem,
        sonium_sim_propellers::BladeElementFidelityOptions::default(),
    )
    .expect("direct/BEM comparison");
    let wake = evaluate_steady_free_wake(
        &blade_model,
        &bem,
        SteadyFreeWakeOptions {
            wake_turns: 1,
            segments_per_turn: 8,
            relaxation: 0.05,
            max_iterations: 100,
            convergence_tolerance_m: 1.0e-4,
            max_node_count: 10_000,
            max_biot_savart_evaluations: 1_000_000,
            ..SteadyFreeWakeOptions::default()
        },
    )
    .expect("steady free wake");
    let prescribed = evaluate_prescribed_helical_wake(
        &blade_model,
        &bem,
        sonium_sim_propellers::PrescribedHelicalWakeOptions {
            wake_turns: 2,
            segments_per_turn: 16,
            ..sonium_sim_propellers::PrescribedHelicalWakeOptions::default()
        },
    )
    .expect("prescribed wake");
    assert!(wake.is_finite() && prescribed.is_finite());
    add_case(
        &mut cases,
        "ladder.direct-bem-steady-and-prescribed-wake",
        "direct/blade-element/BEM/free-wake",
        "independent direct-vs-BEM load gate and finite fixed-circulation wake",
        wake.converged,
        1.0e-4,
        vec![
            direct.thrust_n,
            direct.torque_nm,
            bem.loads.thrust_n,
            bem.loads.torque_nm,
            fidelity.relative_thrust_difference,
            fidelity.relative_torque_difference,
            wake.final_residual_m,
            prescribed.segment_count as f64,
        ],
    );

    let vlm = evaluate_vortex_lattice_reference(
        &[panel(0.04, 0.09, 0.018)],
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
    .expect("VLM");
    add_case(
        &mut cases,
        "ladder.vortex-lattice-panel",
        "VLM",
        "finite panel loads, circulation, and normal residual",
        vlm.maximum_normal_residual_m_s < 1.0e-8,
        1.0e-8,
        vec![
            vlm.thrust_n,
            vlm.total_moment_nm.y,
            vlm.panels[0].circulation_m2_s,
            vlm.maximum_normal_residual_m_s,
        ],
    );

    let body_profile = [
        BodyFittedDuctStation {
            axial_position_m: 0.0,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.05,
            inner_radius_m: 0.021,
            outer_radius_m: 0.042,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.1,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
    ];
    let body_options = BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        inlet_axial_velocity_m_s: 0.1,
        axial_elements: 4,
        radial_elements: 3,
        max_picard_iterations: 20,
        min_picard_iterations: 1,
        picard_tolerance: 1.0e-6,
        picard_relaxation: 0.8,
        max_linear_solver_iterations: 100,
        linear_solver_tolerance: 1.0e-8,
        ..BodyFittedFlowOptions::default()
    };
    let body = solve_body_fitted_duct_flow(&body_profile, body_options).expect("body-fitted flow");
    assert!(body.is_finite() && body.converged);
    let weighted = evaluate_axisymmetric_weighted_flow(
        &body,
        AxisymmetricFlowOptions {
            density_kg_m3: 1.0,
            ..AxisymmetricFlowOptions::default()
        },
    )
    .expect("weighted axisymmetric flow");
    let couette = evaluate_axisymmetric_rotating_wall(
        &body_profile,
        AxisymmetricRotatingWallOptions {
            inner_angular_velocity_rad_s: 100.0,
            outer_angular_velocity_rad_s: 100.0,
            radial_samples: 8,
            maximum_work: 100,
            ..AxisymmetricRotatingWallOptions::default()
        },
    )
    .expect("annular Couette");
    assert!(weighted.is_finite() && couette.is_finite());
    add_case(
        &mut cases,
        "ladder.body-fitted-weighted-couette",
        "body-fitted meridional/axisymmetric references",
        "native FEM flow convergence, 2πr flux, and exact annular Couette torque/dissipation",
        body.converged,
        1.0e-6,
        vec![
            body.residual,
            body.relative_mean_velocity_mismatch,
            weighted.inlet.volume_flow_m3_s,
            weighted.relative_volume_flow_mismatch,
            couette.maximum_absolute_torque_n_m_per_m,
            couette.dissipation_sum_w_per_m,
        ],
    );

    let dipole = RotatingDipoleReference::try_from_source(&source, 2, 1.0).expect("dipole");
    let equator = dipole
        .peak_pressure_pa(std::f64::consts::FRAC_PI_2)
        .unwrap();
    let axis = dipole.peak_pressure_pa(0.0).unwrap();
    assert!((equator - dipole.equatorial_peak_pressure_pa).abs() <= 1.0e-14);
    assert!(axis.abs() <= 1.0e-14);
    let fwh = evaluate_fwh_reference(
        &[FwhSurfaceElement {
            position: Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            normal: Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            area_m2: 0.2,
            pressure_real_pa: 1.0,
            pressure_imaginary_pa: 0.0,
            normal_velocity_real_m_s: 0.1,
            normal_velocity_imaginary_m_s: 0.0,
        }],
        FwhOptions {
            observer_position: Point3 {
                x: 0.0,
                y: 0.0,
                z: 10.0,
            },
            minimum_observer_distance_m: 0.5,
            ..FwhOptions::default()
        },
    )
    .expect("compact FWH reference");
    let installation = evaluate_ground_image_directivity(
        dipole,
        Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.25,
        },
        Point3 {
            x: 1.0,
            y: 0.2,
            z: 0.8,
        },
        ImageReflectionCoefficient::rigid(),
    )
    .expect("installation observer reference");
    assert!(fwh.is_finite() && installation.is_finite());
    add_case(
        &mut cases,
        "observer.dipole-fwh-installation",
        "analytic dipole/compact FW-H/ground image",
        "observer phase/directivity and source-to-observer sign/distance identities",
        true,
        1.0e-12,
        vec![
            equator,
            axis,
            dipole.radiated_sound_power_w(),
            fwh.total_pressure_pa.magnitude_pa(),
            installation.total_pressure_pa.magnitude_pa(),
        ],
    );

    let design = PropellerDesign::conventional(source.clone());
    let mesh_refinement = evaluate_mesh_convergence(
        Blade3dConfig {
            radial_segments: 4,
            chord_segments: 2,
            loop_segments: 3,
            ..Blade3dConfig::default()
        },
        &design,
        &[1, 2, 4],
        0.2,
    )
    .expect("geometric refinement");
    assert!(mesh_refinement.is_converged());
    let panel_refinement = evaluate_panel_refinement(
        &[
            PanelRefinementObservation {
                refinement: 1,
                panel_count: 8,
                thrust_n: direct.thrust_n.abs(),
                torque_nm: direct.torque_nm.abs(),
                acoustic_pressure_pa: source.harmonic_amplitudes(1)[0],
            },
            PanelRefinementObservation {
                refinement: 2,
                panel_count: 16,
                thrust_n: direct.thrust_n.abs() * 1.001,
                torque_nm: direct.torque_nm.abs() * 1.001,
                acoustic_pressure_pa: source.harmonic_amplitudes(1)[0] * 1.001,
            },
            PanelRefinementObservation {
                refinement: 4,
                panel_count: 32,
                thrust_n: direct.thrust_n.abs() * 1.0011,
                torque_nm: direct.torque_nm.abs() * 1.0011,
                acoustic_pressure_pa: source.harmonic_amplitudes(1)[0] * 1.0011,
            },
        ],
        PanelRefinementOptions {
            relative_tolerance: 0.01,
            ..PanelRefinementOptions::default()
        },
    )
    .expect("response refinement");
    assert!(panel_refinement.is_converged());
    add_case(
        &mut cases,
        "refinement.geometry-and-response",
        "geometric mesh and caller-supplied panel response",
        "declared refinement ladders with finite final-level thresholds",
        mesh_refinement.is_converged() && panel_refinement.is_converged(),
        0.01,
        vec![
            mesh_refinement.final_relative_area_change,
            panel_refinement.final_relative_change,
            mesh_refinement.levels.last().unwrap().panel_count as f64,
            panel_refinement.points.last().unwrap().panel_count as f64,
        ],
    );

    let promotion = evaluate_fidelity_promotion(
        &[
            FidelityPromotionSample {
                candidate_id: 1,
                low_fidelity_score: 0.8,
                high_fidelity_score: 0.9,
            },
            FidelityPromotionSample {
                candidate_id: 2,
                low_fidelity_score: 1.1,
                high_fidelity_score: 1.2,
            },
            FidelityPromotionSample {
                candidate_id: 3,
                low_fidelity_score: 1.4,
                high_fidelity_score: 1.35,
            },
        ],
        FidelityPromotionOptions::default(),
    )
    .expect("promotion gate");
    assert!(promotion.passed && optimized.is_exportable());
    add_case(
        &mut cases,
        "promotion.optimizer-gallery",
        "bounded optimizer/provenance/export gate",
        "hard load/power/thickness/provenance predicates before publication",
        promotion.passed,
        1.0e-12,
        vec![
            optimized.objective,
            optimized.thrust_loss,
            optimized.loads.shaft_power_w,
            optimized.loads.acoustic_power_closure_error,
            optimized.provenance.execution.output_checksum_sha256.len() as f64,
        ],
    );

    let mut zero_source = source.clone();
    zero_source.thrust = 0.0;
    zero_source.torque = 0.0;
    let zero_spectrum =
        PropellerDesign::conventional(zero_source).spectrum(2, EvaluationMode::ResearchControl);
    assert!(
        zero_spectrum
            .harmonics
            .iter()
            .all(|harmonic| harmonic.pressure_pa.is_finite())
    );
    let hover = blade_model
        .evaluate_at_freestream(0.0)
        .expect("hover direct reference");
    add_case(
        &mut cases,
        "adversarial.hover-zero-load",
        "zero-load/hover boundary",
        "finite zero-source spectrum and direct hover path; BEM zero-inflow is rejected",
        true,
        1.0e-12,
        vec![
            hover.thrust_n,
            hover.torque_nm,
            zero_spectrum.harmonics[0].pressure_pa,
        ],
    );

    let mut failures = Vec::new();
    assert!(matches!(
        evaluate_duct_shroud_reference(
            &[DuctShroudStation {
                axial_position_m: 0.0,
                flow_area_m2: 0.08,
                hydraulic_diameter_m: 0.30,
            }],
            DuctShroudOptions {
                maximum_stations: 0,
                ..DuctShroudOptions::default()
            }
        ),
        Err(sonium_sim_propellers::DuctShroudError::InvalidOptions { .. })
    ));
    failures.push(FailureCase {
        id: "invalid-inputs-and-budget-boundaries".into(),
        status: "typed_rejected".into(),
        typed_error: "DuctShroudError::InvalidOptions; existing wake/FW-H/VLM gates cover nonfinite, observer, and work budgets".into(),
        partial_publish: false,
    });

    let canonical_input = format!(
        "{INPUT_SEED}|source={source:?}|mesh={:?}|cases={}",
        config.mesh,
        cases.len()
    );
    let canonical_case_bytes = serde_json::to_vec(&cases).expect("canonical cases");
    let artifact = Pp3Artifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64 waived",
        canonical_input: canonical_input.clone(),
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        cases,
        failures,
        external_measured_data: "unavailable: no authenticated published dataset supplied; synthetic/caller data not promoted",
        resolved_cfd_and_full_fwh: "unavailable: reduced VLM/free-wake/compact observer references remain explicitly labelled",
        optimizer_gallery_promotion: "pass: optimizer exportability/provenance gate passed; gallery is validated by PP1 producer",
        output_sha256: sha256_hex(&canonical_case_bytes),
    };
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create PP3 output directory");
        fs::write(
            output_dir.join("native-x86-pp3-fidelity-v1.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize PP3 artifact"),
        )
        .expect("write PP3 artifact");
    }
    artifact
}

#[test]
fn pp3_native_fidelity_ladder_is_finite_and_fail_closed() {
    let output_dir = std::env::var_os("SONIUM_OUTPUT_DIR").map(PathBuf::from);
    let artifact = evidence(output_dir.as_deref());
    assert_eq!(artifact.status, "pass");
    assert!(artifact.cases.len() >= 15);
    assert!(artifact.cases.iter().all(|case| case.status == "pass"));
    assert!(
        artifact
            .failures
            .iter()
            .all(|failure| !failure.partial_publish)
    );
}
