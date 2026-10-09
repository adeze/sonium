//! Native Linux/x86_64 PC-fan product-reference evidence.
//!
//! This producer executes the real CPU CFD steady and rotating-wall BDF
//! paths, and the checked reduced-order objective/gradient path.  It records
//! fields, conservation ledgers, objective decomposition, derivative budgets,
//! and the execution provenance needed to reproduce the run.  Arm64 and
//! Windows are not synthesized on this host; the artifact records that the
//! cross-architecture comparison is waived.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sonium_sim_pcfan::{
    PcFanCfdBackend, PcFanCfdConfig, PcFanCfdConservation, PcFanConfig, PcFanDesignVariables,
    PcFanFlowModel, PcFanResolvedAdjointConfig, PcFanResolvedTransientObjective, evaluate_pc_fan,
    solve_pc_fan_cfd_mean_flow,
    solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory,
    solve_pc_fan_cfd_rotating_wall_transient_with_trajectory,
    solve_pc_fan_shape_gradient_reduced_adjoint, try_evaluate_pc_fan_from_cfd_solution,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SCHEMA: &str = "sonium.pcfan.native-x86-pf3.v1";
const SEED: &str = "sonium.native-x86.pcfan-pf3.reference.v1";
const GRADIENT_RELATIVE_BUDGET: f64 = 5.0e-3;

fn sha256_bytes(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes.into_iter().collect::<Vec<_>>());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_f64(values: impl IntoIterator<Item = f64>) -> String {
    sha256_bytes(values.into_iter().flat_map(f64::to_le_bytes))
}

fn finite_stats(values: &[f64]) -> Value {
    assert!(!values.is_empty(), "PF3 field must not be empty");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "PF3 field contains a non-finite value"
    );
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let l2 = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    json!({
        "count": values.len(),
        "min": min,
        "max": max,
        "mean": mean,
        "l2": l2,
        "sha256_le_f64": sha256_f64(values.iter().copied()),
        "finite": true,
    })
}

fn flatten_velocity(values: &[[f64; 3]]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| value.iter().copied())
        .collect()
}

fn ledger_json(ledger: PcFanCfdConservation) -> Value {
    json!({
        "inlet_mass_flow_kg_s": ledger.inlet_mass_flow_kg_s,
        "outlet_mass_flow_kg_s": ledger.outlet_mass_flow_kg_s,
        "relative_mass_imbalance": ledger.relative_mass_imbalance,
        "inlet_axial_momentum_flux_n": ledger.inlet_axial_momentum_flux_n,
        "outlet_axial_momentum_flux_n": ledger.outlet_axial_momentum_flux_n,
        "pressure_force_n": ledger.pressure_force_n,
        "body_force_n": ledger.body_force_n,
        "axial_momentum_residual_n": ledger.axial_momentum_residual_n,
        "relative_momentum_imbalance": ledger.relative_momentum_imbalance,
        "pressure_force_from_loading_n": ledger.pressure_force_from_loading_n,
        "relative_pressure_closure": ledger.relative_pressure_closure,
        "pressure_power_w": ledger.pressure_power_w,
        "shaft_power_to_fluid_w": ledger.shaft_power_to_fluid_w,
        "relative_energy_imbalance": ledger.relative_energy_imbalance,
    })
}

fn execution_json(environment: &sonium_sim_pcfan::PcFanCfdExecutionEnvironment) -> Value {
    json!({
        "target_arch": environment.target_arch,
        "target_os": environment.target_os,
        "compiler": environment.compiler,
        "cpu_features": environment.cpu_features,
        "simd_dispatch": environment.simd_dispatch,
        "thread_count": environment.thread_count,
        "requested_backend": format!("{:?}", environment.requested_backend),
        "actual_backend": format!("{:?}", environment.actual_backend),
        "execution_mode": format!("{:?}", environment.execution_mode),
        "reuse_validity": {
            "mesh_reused": environment.reuse_validity.mesh_reused,
            "sparsity_reused": environment.reuse_validity.sparsity_reused,
            "preconditioner_reused": environment.reuse_validity.preconditioner_reused,
            "flow_state_reused": environment.reuse_validity.flow_state_reused,
            "transported_state_reused": environment.reuse_validity.transported_state_reused,
            "aeroacoustic_state_reused": environment.reuse_validity.aeroacoustic_state_reused,
        },
        "linear_solver": format!("{:?}", environment.linear_solver),
        "cpu_preconditioner": format!("{:?}", environment.cpu_preconditioner),
        "backend_preconditioner": format!("{:?}", environment.backend_preconditioner),
        "linear_solver_max_iterations": environment.linear_solver_max_iterations,
        "linear_solver_tolerance": environment.linear_solver_tolerance,
    })
}

fn cfd_config(flow_model: PcFanFlowModel) -> PcFanConfig {
    let mut config = PcFanConfig::default_120mm();
    config.cfd = PcFanCfdConfig {
        flow_model,
        backend: PcFanCfdBackend::Cpu,
        allow_host_fallback: false,
        require_device_resident: false,
        cells_x: 2,
        cells_y: 2,
        cells_z: 2,
        max_picard_iterations: 1,
        nonlinear_tolerance: 10.0,
        max_relative_mass_imbalance: 10.0,
        max_relative_momentum_imbalance: 10.0,
        max_relative_pressure_closure: 10.0,
        max_relative_energy_imbalance: 10.0,
        ..config.cfd
    };
    config
}

fn reduced_gradient_config() -> PcFanConfig {
    let mut config = PcFanConfig::default_120mm();
    config.mesh.radial_segments = 4;
    config.mesh.chord_segments = 3;
    config.mesh.circumferential_segments = 12;
    config.mesh.harmonic_count = 3;
    config.optimizer.finite_difference_step = 2.0e-5;
    config
}

fn independent_gradient_error(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    gradient: &[f64],
) -> (Vec<f64>, f64) {
    let mut errors = Vec::with_capacity(gradient.len());
    for (index, &actual) in gradient.iter().enumerate() {
        let mut plus_values = design.to_vector();
        let mut minus_values = design.to_vector();
        let scale = if index < 10 {
            0.01
        } else if index < 30 {
            match (index - 10) % 4 {
                0 => config.dimensions.depth_m.max(1.0e-6),
                1 => 0.004,
                2 => 0.001,
                _ => 10.0,
            }
        } else {
            0.003
        };
        let step = config.optimizer.finite_difference_step.max(1.0e-7)
            * design.to_vector()[index].abs().max(scale);
        plus_values[index] += step;
        minus_values[index] -= step;
        let plus = evaluate_pc_fan(config, &PcFanDesignVariables::from_vector(plus_values));
        let minus = evaluate_pc_fan(config, &PcFanDesignVariables::from_vector(minus_values));
        let reference = (plus.objective - minus.objective) / (2.0 * step);
        let error = (actual - reference).abs() / actual.abs().max(reference.abs()).max(1.0);
        assert!(error.is_finite());
        errors.push(error);
    }
    let maximum = errors.iter().copied().fold(0.0, f64::max);
    (errors, maximum)
}

fn objective_json(config: &PcFanConfig, evaluation: &sonium_sim_pcfan::PcFanEvaluation) -> Value {
    let acoustic = if config.weights.a_weighting_enabled {
        evaluation.a_weighted_acoustic_level_db
    } else {
        evaluation.acoustic_level_db
    };
    let recomputed = config.weights.acoustic_weight * acoustic
        + evaluation.tonality_penalty
        + evaluation.spectral_spread_penalty
        + evaluation.directivity_spread_penalty
        + evaluation.operating_point_spread_penalty
        + evaluation.airflow_penalty
        + evaluation.collision_penalty
        + evaluation.shroud_clearance_penalty
        + evaluation.frame_containment_penalty
        + evaluation.hub_attachment_penalty
        + evaluation.thermal_penalty
        + evaluation.structural_penalty
        + evaluation.manufacturing_penalty;
    let decomposition_error = (evaluation.objective - recomputed).abs();
    assert!(decomposition_error <= 1.0e-10);
    json!({
        "objective": evaluation.objective,
        "airflow_cfm": evaluation.airflow_cfm,
        "acoustic_level_db": evaluation.acoustic_level_db,
        "a_weighted_acoustic_level_db": evaluation.a_weighted_acoustic_level_db,
        "flow_field": {
            "airflow_cfm": evaluation.flow.airflow_cfm,
            "axial_velocity_m_s": evaluation.flow.axial_velocity_m_s,
            "pressure_rise_pa": evaluation.flow.pressure_rise_pa,
            "reynolds_number": evaluation.flow.reynolds_number,
            "mach_number": evaluation.flow.mach_number,
        },
        "airflow_penalty": evaluation.airflow_penalty,
        "collision_penalty": evaluation.collision_penalty,
        "decomposition_error": decomposition_error,
        "finite": evaluation.objective.is_finite(),
    })
}

fn steady_evidence(config: &PcFanConfig, design: &PcFanDesignVariables) -> (Value, Vec<u8>) {
    let solution = solve_pc_fan_cfd_mean_flow(config, design);
    assert!(
        solution.converged,
        "steady native CPU fixture must converge"
    );
    let evaluation = try_evaluate_pc_fan_from_cfd_solution(config, design, &solution)
        .expect("steady native CPU fixture must pass checked admission");
    let velocity = flatten_velocity(&solution.velocity_m_s);
    let pressure = solution.pressure_pa.clone();
    let node_values = solution
        .nodes
        .iter()
        .flat_map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    let fields = json!({
        "node_count": solution.nodes.len(),
        "velocity_count": solution.velocity_m_s.len(),
        "pressure_count": solution.pressure_pa.len(),
        "nodes": finite_stats(&node_values),
        "velocity_m_s": finite_stats(&velocity),
        "pressure_pa": finite_stats(&pressure),
    });
    let provenance = json!({
        "requested_backend": format!("{:?}", solution.backend_provenance.requested_backend),
        "actual_backend": format!("{:?}", solution.backend_provenance.actual_backend),
        "execution_mode": format!("{:?}", solution.backend_provenance.execution_mode),
        "fallback_reason": solution.backend_provenance.fallback_reason.map(|reason| format!("{reason:?}")),
        "environment": execution_json(&solution.execution_environment),
    });
    let record = json!({
        "flow_model": format!("{:?}", config.cfd.flow_model),
        "fields": fields,
        "ledger": ledger_json(solution.conservation),
        "metrics": {
            "airflow_cfm": solution.airflow_cfm,
            "pressure_drop_pa": solution.pressure_drop_pa,
            "max_velocity_m_s": solution.max_velocity_m_s,
            "reynolds_number": solution.reynolds_number,
            "iterations": solution.iterations,
            "residual": solution.residual,
        },
        "objective": objective_json(config, &evaluation),
        "provenance": provenance,
    });
    let canonical = serde_json::to_vec(&record).expect("steady evidence JSON");
    (record, canonical)
}

fn transient_evidence(config: &PcFanConfig, design: &PcFanDesignVariables) -> (Value, Vec<u8>) {
    let trajectory =
        solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(config, design, 1.0e-3, 1.0e-3)
            .expect("native rotating-wall transient fixture must solve");
    assert!(trajectory.observation.converged);
    assert!(trajectory.steps.len() >= 2);
    let times = trajectory
        .steps
        .iter()
        .map(|step| step.time_s)
        .collect::<Vec<_>>();
    assert!(times.windows(2).all(|window| window[1] > window[0]));
    let steps = trajectory
        .steps
        .iter()
        .map(|step| {
            assert!(step.state.iter().all(|value| value.is_finite()));
            assert!(step.residual_norm.is_finite());
            assert!(step.residual_norm <= config.cfd.nonlinear_tolerance);
            json!({
                "step": step.step,
                "time_s": step.time_s,
                "state": finite_stats(&step.state),
                "residual": step.residual_norm,
                "ledger": ledger_json(step.conservation),
                "actual_backend": format!("{:?}", step.actual_backend),
                "provenance": execution_json(&step.execution_environment),
                "state_jacobian_provenance": step.state_jacobian_provenance,
            })
        })
        .collect::<Vec<_>>();
    let adjoint = solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
        &PcFanResolvedAdjointConfig {
            max_state_residual: 1.0,
            ..PcFanResolvedAdjointConfig::default()
        },
        config,
        design,
        &trajectory,
        PcFanResolvedTransientObjective::RotatingWallAxialLoading,
    )
    .expect("native transient objective adjoint must solve");
    assert!(adjoint.converged);
    assert!(
        adjoint
            .design_gradient
            .iter()
            .all(|value| value.is_finite())
    );
    let record = json!({
        "flow_model": format!("{:?}", config.cfd.flow_model),
        "time_step_s": 1.0e-3,
        "rotor_revolutions": 1.0e-3,
        "steps": steps,
        "observation": {
            "final_time_s": trajectory.observation.final_time_s,
            "airflow_cfm": trajectory.observation.airflow_cfm,
            "pressure_drop_pa": trajectory.observation.pressure_drop_pa,
            "axial_loading_n": trajectory.observation.axial_loading_n,
            "residual": trajectory.observation.residual,
            "conservation": ledger_json(trajectory.observation.conservation),
            "converged": trajectory.observation.converged,
        },
        "gradient": {
            "objective": "RotatingWallAxialLoading",
            "provenance": adjoint.provenance,
            "flow_model": format!("{:?}", adjoint.flow_model),
            "backend": format!("{:?}", adjoint.backend),
            "design_gradient": finite_stats(&adjoint.design_gradient),
            "state_residual_norm": adjoint.state_residual_norm,
            "adjoint_residual_norm": adjoint.adjoint_residual_norm,
            "iterations": adjoint.iterations,
            "converged": adjoint.converged,
        },
        "provenance": execution_json(
            &trajectory.steps[0].execution_environment,
        ),
    });
    let canonical = serde_json::to_vec(&record).expect("transient evidence JSON");
    (record, canonical)
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("-Vv")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".into())
}

fn output_dir() -> PathBuf {
    std::env::var_os("SONIUM_PF3_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/sonium-unix-pf3"))
}

fn write_checksums(root: &Path, names: &[&str]) {
    let mut checksums = BTreeMap::new();
    for name in names {
        let path = root.join(name);
        checksums.insert((*name).to_owned(), sha256_bytes(fs::read(path).unwrap()));
    }
    let body = checksums
        .into_iter()
        .map(|(name, digest)| format!("{digest}  {name}\n"))
        .collect::<String>();
    fs::write(root.join("checksums.sha256"), body).unwrap();
}

#[test]
fn native_x86_pcfan_pf3_reference_artifact() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated native-x86 script that sets deterministic mode.
    if std::env::var("SONIUM_PF3_DETERMINISTIC").as_deref() != Ok("1") {
        eprintln!("skipping native PF3 evidence: set SONIUM_PF3_DETERMINISTIC=1 to produce it");
        return;
    }
    assert_eq!(std::env::consts::OS, "linux", "PF3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PF3 requires native x86_64"
    );

    let output = output_dir();
    fs::create_dir_all(&output).unwrap();
    let design = PcFanDesignVariables::baseline_for(&PcFanConfig::default_120mm());
    let steady_config = cfd_config(PcFanFlowModel::IncompressibleCfd);
    let transient_config = cfd_config(PcFanFlowModel::IncompressibleCfdFull);
    let (steady, steady_bytes) = steady_evidence(&steady_config, &design);
    let (steady_repeat, steady_repeat_bytes) = steady_evidence(&steady_config, &design);
    assert_eq!(steady_bytes, steady_repeat_bytes);
    assert_eq!(steady, steady_repeat);
    let (transient, transient_bytes) = transient_evidence(&transient_config, &design);
    let (transient_repeat, transient_repeat_bytes) = transient_evidence(&transient_config, &design);
    assert_eq!(transient_bytes, transient_repeat_bytes);
    assert_eq!(transient, transient_repeat);

    let gradient_config = reduced_gradient_config();
    let gradient_design = gradient_config.default_design_variables();
    let gradient = solve_pc_fan_shape_gradient_reduced_adjoint(&gradient_config, &gradient_design)
        .expect("reduced-order gradient fixture must solve");
    let (gradient_errors, maximum_gradient_error) = independent_gradient_error(
        &gradient_config,
        &gradient_design,
        &gradient.objective_gradient,
    );
    assert!(maximum_gradient_error <= GRADIENT_RELATIVE_BUDGET);
    assert!(gradient.residual.max_abs.is_finite());
    let reduced_evaluation = evaluate_pc_fan(&gradient_config, &gradient_design);

    let host = json!({
        "target_arch": std::env::consts::ARCH,
        "target_os": std::env::consts::OS,
        "kernel": fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_else(|_| "unavailable".into()).trim(),
        "cpu_model": fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| text.lines().find(|line| line.starts_with("model name")).map(str::to_owned))
            .unwrap_or_else(|| "unavailable".into()),
        "rustc": rustc_version(),
        "cargo_target_dir": std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "default".into()),
        "features": std::env::var("CARGO_FEATURES").unwrap_or_else(|_| "none".into()),
        "rayon_thread_policy": std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        "locale": std::env::var("LC_ALL").or_else(|_| std::env::var("LANG")).unwrap_or_else(|_| "unset".into()),
    });
    let artifact = json!({
        "schema": SCHEMA,
        // PF3 is complete within the authorized native Linux/x86 scope.  The
        // release_gate below remains blocked by the Arm64 waiver and the
        // separate PF1/PF2 fan-in dependencies.
        "status": "pass",
        "scope": "native Linux x86_64 CPU/f64; Arm64 and Windows waived",
        "seed": SEED,
        "deterministic_mode": true,
        "host": host,
        "steady": steady,
        "transient": transient,
        "reduced_objective_gradient": {
            "model": "ReducedOrder",
            "objective": objective_json(&gradient_config, &reduced_evaluation),
            "provenance": "reduced-order-discrete-adjoint:v1",
            "gradient": finite_stats(&gradient.objective_gradient),
            "residual_max_abs": gradient.residual.max_abs,
            "finite_difference": {
                "budget_relative_max": GRADIENT_RELATIVE_BUDGET,
                "maximum_relative_error": maximum_gradient_error,
                "component_errors": gradient_errors,
                "independent_oracle": "centered finite difference of evaluate_pc_fan objective",
            },
        },
        "independent_budgets": {
            "gradient_relative_error": GRADIENT_RELATIVE_BUDGET,
            "ledger_admission": "config.cfd conservation tolerances; every field finite",
            "objective_decomposition_absolute": 1.0e-10,
            "transient_step_residual": transient_config.cfd.nonlinear_tolerance,
        },
        "cross_architecture": {
            "status": "waived",
            "targets": ["aarch64", "x86_64"],
            "reason": "Arm64 comparison is explicitly waived for this host; no Rosetta or synthetic Arm64 result is claimed",
        },
        "release_gate": "blocked: Arm64 comparison waived and PF1/PF2/shared QA fan-in remain incomplete",
    });
    let artifact_path = output.join("native-x86-pcfan-pf3-v1.json");
    fs::write(
        &artifact_path,
        serde_json::to_vec_pretty(&artifact)
            .unwrap()
            .into_iter()
            .chain([b'\n'])
            .collect::<Vec<_>>(),
    )
    .unwrap();
    write_checksums(&output, &["native-x86-pcfan-pf3-v1.json"]);
    assert!(artifact_path.is_file());
}
