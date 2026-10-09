//! Native PP3 production multi-rotor/off-design convergence evidence.
//!
//! This packet exercises the checked load-coupled multi-rotor BEM path with
//! two aligned rotors at different density, RPM, and freestream conditions.
//! The wake coupling is compared with an independent momentum/Gaussian-disk
//! calculation; this is not a duplicate source-spectrum surrogate and does
//! not claim resolved CFD, unsteady wake roll-up, or FW-H acoustics.

use serde_json::{Value, json};
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, MultiRotorBemError, MultiRotorBemOptions,
    MultiRotorBemRotor, evaluate_multi_rotor_bem, sha256_hex,
};
use std::fs;
use std::path::Path;

const SCHEMA: &str = "sonium.propellers.pp3-multirotor-offdesign-evidence.v1";
const CONVERGENCE_TOLERANCE: f64 = 1.0e-3;
const RADIAL_SAMPLES: usize = 3;
const AZIMUTHAL_SAMPLES: usize = 16;
const WAKE_RADIUS_GROWTH: f64 = 0.25;

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
    .expect("valid production polar");
    BladeElementModel::try_from_source(&source, stations, polar)
        .expect("valid production blade-element model")
}

fn rotor(
    center_m: [f64; 3],
    rpm: f64,
    freestream_m_s: f64,
    density_kg_m3: f64,
) -> MultiRotorBemRotor {
    MultiRotorBemRotor {
        center_m,
        axis: [0.0, 0.0, 1.0],
        model: model(),
        operating_point: BladeElementOperatingPoint {
            rpm,
            freestream_m_s,
            density_kg_m3,
        },
    }
}

fn options() -> MultiRotorBemOptions {
    MultiRotorBemOptions {
        bem_options: BladeElementBemOptions {
            max_iterations: 150,
            convergence_tolerance: 1.0e-8,
            relaxation: 0.35,
            ..BladeElementBemOptions::default()
        },
        max_iterations: 30,
        convergence_tolerance: CONVERGENCE_TOLERANCE,
        relaxation: 0.3,
        radial_samples: RADIAL_SAMPLES,
        azimuthal_samples: AZIMUTHAL_SAMPLES,
        wake_radius_growth: WAKE_RADIUS_GROWTH,
        max_bem_evaluations: 128,
        ..MultiRotorBemOptions::default()
    }
}

fn production_rotors() -> Vec<MultiRotorBemRotor> {
    vec![
        rotor([0.0, 0.0, 0.0], 12_000.0, 10.0, 1.204),
        rotor([0.0, 0.0, 0.12], 9_000.0, 8.0, 0.98),
    ]
}

fn independent_momentum_induced_velocity(
    thrust_n: f64,
    radius_m: f64,
    density_kg_m3: f64,
    freestream_m_s: f64,
) -> f64 {
    let area_m2 = std::f64::consts::PI * radius_m * radius_m;
    let loading_m2_s2 = thrust_n / (density_kg_m3 * area_m2);
    let root = freestream_m_s
        .mul_add(freestream_m_s, 2.0 * loading_m2_s2)
        .sqrt();
    let numerator = root - freestream_m_s;
    if numerator > 0.0 {
        0.5 * numerator
    } else {
        loading_m2_s2 / (root + freestream_m_s)
    }
}

fn independent_aligned_wake(
    source_thrust_n: f64,
    source_radius_m: f64,
    source_density_kg_m3: f64,
    source_freestream_m_s: f64,
    receiver_radius_m: f64,
    downstream_distance_m: f64,
    options: MultiRotorBemOptions,
) -> f64 {
    let source_induced = independent_momentum_induced_velocity(
        source_thrust_n,
        source_radius_m,
        source_density_kg_m3,
        source_freestream_m_s,
    );
    let wake_radius_m = source_radius_m
        * (1.0 + options.wake_radius_growth * downstream_distance_m / source_radius_m);
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for radial_index in 0..options.radial_samples {
        let radial_fraction = (radial_index as f64 + 0.5) / options.radial_samples as f64;
        let weight = 2.0 * radial_fraction / options.radial_samples as f64;
        let gaussian = (-(receiver_radius_m * radial_fraction).powi(2)
            / (wake_radius_m * wake_radius_m))
            .exp();
        for _azimuth_index in 0..options.azimuthal_samples {
            weighted_sum += weight * 2.0 * source_induced * gaussian;
            weight_sum += weight;
        }
    }
    weighted_sum / weight_sum
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * right.abs().max(left.abs()).max(1.0)
}

fn rotor_json(rotor: &sonium_sim_propellers::MultiRotorBemRotorResult) -> Value {
    json!({
        "center_m": rotor.center_m,
        "axis": rotor.axis,
        "external_induced_velocity_m_s": rotor.external_induced_velocity_m_s,
        "isolated_induced_velocity_m_s": rotor.isolated_induced_velocity_m_s,
        "effective_axial_velocity_m_s": rotor.effective_axial_velocity_m_s,
        "relative_load_change": rotor.relative_load_change,
        "operating_point": {
            "rpm": rotor.bem_result.loads.operating_point.rpm,
            "freestream_m_s": rotor.bem_result.loads.operating_point.freestream_m_s,
            "density_kg_m3": rotor.bem_result.loads.operating_point.density_kg_m3,
        },
        "loads": {
            "thrust_n": rotor.bem_result.loads.thrust_n,
            "torque_nm": rotor.bem_result.loads.torque_nm,
            "shaft_power_w": rotor.bem_result.loads.shaft_power_w,
        },
    })
}

fn failure_json(id: &str, error: &MultiRotorBemError) -> Value {
    json!({
        "id": id,
        "status": "typed_rejected",
        "publication": "none",
        "typed_error": format!("{error:?}"),
    })
}

fn write_artifact(
    path: &Path,
    result: &sonium_sim_propellers::MultiRotorBemResult,
    oracle: f64,
    repeated: bool,
    failures: Vec<Value>,
) {
    let source_radius_m = production_rotors()[0].model.tip_radius_m();
    let canonical_input = format!(
        "schema={SCHEMA};rotors=[(0,0,0,axis=(0,0,1),rpm=12000,V=10,rho=1.204),(0,0,0.12,axis=(0,0,1),rpm=9000,V=8,rho=0.98)];radial_samples={RADIAL_SAMPLES};azimuthal_samples={AZIMUTHAL_SAMPLES};wake_radius_growth={WAKE_RADIUS_GROWTH};coupling_tolerance={CONVERGENCE_TOLERANCE}"
    );
    let rotors: Vec<Value> = result.rotors.iter().map(rotor_json).collect();
    let canonical_output = serde_json::to_string(&json!({
        "rotors": rotors,
        "pairwise_induced_velocity_m_s": result.pairwise_induced_velocity_m_s,
        "total_thrust_n": result.total_thrust_n,
        "iterations": result.iterations,
        "maximum_relative_load_change": result.maximum_relative_load_change,
    }))
    .expect("serialize PP3 multi-rotor output");
    let artifact = json!({
        "schema": SCHEMA,
        "status": "pass",
        "scope": "native Linux x86_64 CPU/f64; Windows/Arm64/AVX512 waived",
        "claim_class": "production load-coupled multi-rotor BEM off-design convergence",
        "reference": "independent actuator/momentum induced-velocity identity plus independently recomputed aligned Gaussian receiving-disk average",
        "canonical_input": canonical_input,
        "canonical_input_sha256": sha256_hex(canonical_input.as_bytes()),
        "dimensions": ["rotor_count", "center_m", "axis", "rpm", "freestream_m_s", "density_kg_m3", "thrust_n", "torque_nm", "external_induced_velocity_m_s", "effective_axial_velocity_m_s", "relative_load_change"],
        "production_path": {
            "evaluator": "evaluate_multi_rotor_bem",
            "coupling": "load-coupled steady BEM with finite-core downstream disk averaging",
            "off_design": "rotor 0 (12000 rpm, 10 m/s, 1.204 kg/m3); rotor 1 (9000 rpm, 8 m/s, 0.98 kg/m3)",
            "rotor_count": result.rotors.len(),
            "tip_radius_m": source_radius_m,
        },
        "rotors": rotors,
        "pairwise_induced_velocity_m_s": result.pairwise_induced_velocity_m_s,
        "independent_oracle": {
            "source_rotor_induced_velocity_m_s": independent_momentum_induced_velocity(result.rotors[0].bem_result.loads.thrust_n, source_radius_m, 1.204, 10.0),
            "downstream_pairwise_velocity_m_s": oracle,
            "production_pairwise_relative_error": (result.pairwise_induced_velocity_m_s[1][0] - oracle).abs() / oracle.abs().max(1.0),
            "formula": "momentum disk induced velocity followed by radial midpoint/Gaussian wake average; azimuth integrates exactly by symmetry for aligned rotors",
        },
        "convergence": {
            "converged": result.converged,
            "iterations": result.iterations,
            "maximum_relative_load_change": result.maximum_relative_load_change,
            "tolerance": CONVERGENCE_TOLERANCE,
            "pair_evaluation_count": result.pair_evaluation_count,
            "bem_evaluation_count": result.bem_evaluation_count,
            "disk_sample_count_per_pair": result.disk_sample_count_per_pair,
        },
        "deterministic_repeat": repeated,
        "invalid_probes": failures,
        "canonical_output": canonical_output,
        "output_sha256": sha256_hex(canonical_output.as_bytes()),
        "unsupported_or_unexercised": [
            "resolved 3-D CFD, viscous wake roll-up, dynamic stall, aeroelastic deformation, and FW-H remain unsupported",
            "this packet is production CPU/f64 evidence; it does not claim CUDA/Metal execution",
            "external measured-data validation remains unavailable",
        ],
    });
    fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))
        .expect("create PP3 multi-rotor evidence directory");
    fs::write(
        path,
        serde_json::to_vec_pretty(&artifact).expect("serialize PP3 artifact"),
    )
    .expect("write PP3 multi-rotor artifact");
}

#[test]
fn native_pp3_multirotor_offdesign_convergence_evidence() {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );
    let rotors = production_rotors();
    let options = options();
    let result = evaluate_multi_rotor_bem(&rotors, options).expect("production multi-rotor BEM");
    assert!(result.is_finite());
    assert!(result.converged);
    assert!(result.maximum_relative_load_change <= CONVERGENCE_TOLERANCE);
    assert_eq!(result.rotors.len(), 2);
    assert!(result.rotors.iter().all(|rotor| {
        rotor.bem_result.loads.thrust_n.is_finite()
            && rotor.bem_result.loads.thrust_n > 0.0
            && rotor.bem_result.loads.torque_nm.is_finite()
            && rotor.bem_result.loads.torque_nm > 0.0
    }));
    assert!(result.pairwise_induced_velocity_m_s[1][0] > 0.0);
    assert_eq!(result.pairwise_induced_velocity_m_s[0][1], 0.0);
    assert_ne!(
        result.rotors[0].bem_result.loads.thrust_n,
        result.rotors[1].bem_result.loads.thrust_n
    );

    let source_radius_m = rotors[0].model.tip_radius_m();
    let oracle = independent_aligned_wake(
        result.rotors[0].bem_result.loads.thrust_n,
        source_radius_m,
        1.204,
        10.0,
        source_radius_m,
        0.12,
        options,
    );
    assert!(close(
        result.pairwise_induced_velocity_m_s[1][0],
        oracle,
        1.0e-12
    ));
    let source_induced = independent_momentum_induced_velocity(
        result.rotors[0].bem_result.loads.thrust_n,
        source_radius_m,
        1.204,
        10.0,
    );
    assert!(close(
        result.rotors[0].isolated_induced_velocity_m_s,
        source_induced,
        1.0e-12
    ));

    let repeated = evaluate_multi_rotor_bem(&rotors, options).expect("deterministic replay");
    assert_eq!(result, repeated);

    let invalid_axis = evaluate_multi_rotor_bem(
        &[MultiRotorBemRotor {
            axis: [0.0, 0.0, 0.0],
            ..rotors[0].clone()
        }],
        options,
    )
    .expect_err("zero axis must be typed-rejected");
    assert!(matches!(
        invalid_axis,
        MultiRotorBemError::InvalidRotor { .. }
    ));
    let rotor_budget = evaluate_multi_rotor_bem(
        &rotors,
        MultiRotorBemOptions {
            max_rotors: 1,
            ..options
        },
    )
    .expect_err("rotor budget must be typed-rejected");
    assert!(matches!(
        rotor_budget,
        MultiRotorBemError::TooManyRotors { .. }
    ));
    let pair_budget = evaluate_multi_rotor_bem(
        &rotors,
        MultiRotorBemOptions {
            max_pair_evaluations: 1,
            ..options
        },
    )
    .expect_err("pair budget must be typed-rejected");
    assert!(matches!(
        pair_budget,
        MultiRotorBemError::PairEvaluationBudgetExceeded { .. }
    ));
    let bem_budget = evaluate_multi_rotor_bem(
        &rotors,
        MultiRotorBemOptions {
            max_bem_evaluations: 2,
            ..options
        },
    )
    .expect_err("BEM budget must be typed-rejected");
    assert!(matches!(
        bem_budget,
        MultiRotorBemError::BemEvaluationBudgetExceeded { .. }
    ));
    let failures = vec![
        failure_json("zero-axis", &invalid_axis),
        failure_json("rotor-budget", &rotor_budget),
        failure_json("pair-budget", &pair_budget),
        failure_json("bem-budget", &bem_budget),
    ];

    if let Some(path) = std::env::var_os("SONIUM_PP3_MULTI_ROTOR_ARTIFACT") {
        write_artifact(
            Path::new(&path),
            &result,
            oracle,
            result == repeated,
            failures,
        );
    }
}
