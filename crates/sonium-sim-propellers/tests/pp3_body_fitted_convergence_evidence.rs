//! Native PP3 body-fitted meridional-flow mesh-convergence evidence.
//!
//! This packet runs the production body-fitted FEM adapter on a constant-area
//! annulus at six resolutions.  The constant-area, incompressible identity is
//! that the outlet mean axial velocity approaches the prescribed inlet mean;
//! the artifact records the actual mismatch rather than manufacturing an
//! exact answer.  This is bounded native FEM evidence, not resolved rotor,
//! CFD, or externally validated propeller data.

use serde::Serialize;
use sonium_sim_propellers::{
    BodyFittedDuctStation, BodyFittedFlowError, BodyFittedFlowOptions, sha256_hex,
    solve_body_fitted_duct_flow,
};
use std::fs;
use std::path::Path;

const SCHEMA: &str = "sonium.propellers.body-fitted-convergence-evidence.v1";
const INLET_VELOCITY_M_S: f64 = 0.1;
const FINAL_MISMATCH_TOLERANCE: f64 = 2.0e-2;
const LEVELS: [(usize, usize); 6] = [(2, 2), (4, 3), (8, 6), (16, 8), (32, 16), (64, 24)];

#[derive(Debug, Serialize)]
struct MeshLevel {
    axial_elements: usize,
    radial_elements: usize,
    expected_mesh_node_count: usize,
    mesh_node_count: usize,
    expected_mesh_element_count: usize,
    mesh_element_count: usize,
    iterations: usize,
    residual: f64,
    converged: bool,
    inlet_mean_axial_velocity_m_s: f64,
    outlet_mean_axial_velocity_m_s: f64,
    relative_mean_velocity_mismatch: f64,
    maximum_speed_m_s: f64,
    minimum_pressure_pa: f64,
    maximum_pressure_pa: f64,
}

#[derive(Debug, Serialize)]
struct FailureProbe {
    id: &'static str,
    status: &'static str,
    publication: &'static str,
    typed_error: String,
}

#[derive(Debug, Serialize)]
struct BodyFittedConvergenceArtifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    claim_class: &'static str,
    reference: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    dimensions: Vec<&'static str>,
    levels: Vec<MeshLevel>,
    continuity_identity: &'static str,
    final_mismatch_tolerance: f64,
    improvement_factors: Vec<f64>,
    minimum_improvement_factor: f64,
    final_relative_mean_velocity_mismatch: f64,
    converged: bool,
    invalid_probes: Vec<FailureProbe>,
    canonical_output: String,
    output_sha256: String,
}

fn profile() -> [BodyFittedDuctStation; 2] {
    [
        BodyFittedDuctStation {
            axial_position_m: 0.0,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
        BodyFittedDuctStation {
            axial_position_m: 0.1,
            inner_radius_m: 0.02,
            outer_radius_m: 0.05,
        },
    ]
}

fn options(axial_elements: usize, radial_elements: usize) -> BodyFittedFlowOptions {
    BodyFittedFlowOptions {
        density_kg_m3: 1.0,
        dynamic_viscosity_pa_s: 1.0,
        inlet_axial_velocity_m_s: INLET_VELOCITY_M_S,
        axial_elements,
        radial_elements,
        max_picard_iterations: 20,
        min_picard_iterations: 1,
        picard_tolerance: 1.0e-8,
        picard_relaxation: 0.8,
        max_linear_solver_iterations: 100,
        linear_solver_tolerance: 1.0e-10,
        ..BodyFittedFlowOptions::default()
    }
}

fn evidence(output_dir: Option<&Path>) -> BodyFittedConvergenceArtifact {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );

    let stations = profile();
    let mut levels = Vec::with_capacity(LEVELS.len());
    for (axial_elements, radial_elements) in LEVELS {
        let result =
            solve_body_fitted_duct_flow(&stations, options(axial_elements, radial_elements))
                .expect("body-fitted FEM solve");
        assert!(result.is_finite());
        let expected_nodes = (axial_elements + 1) * (radial_elements + 1);
        let expected_elements = 2 * axial_elements * radial_elements;
        assert_eq!(result.mesh_node_count, expected_nodes);
        assert_eq!(result.mesh_element_count, expected_elements);
        assert!(result.converged);
        assert!(result.residual <= 1.0e-8);
        assert!((result.inlet_mean_axial_velocity_m_s - INLET_VELOCITY_M_S).abs() <= 1.0e-14);
        assert!(result.relative_mean_velocity_mismatch.is_finite());
        levels.push(MeshLevel {
            axial_elements,
            radial_elements,
            expected_mesh_node_count: expected_nodes,
            mesh_node_count: result.mesh_node_count,
            expected_mesh_element_count: expected_elements,
            mesh_element_count: result.mesh_element_count,
            iterations: result.iterations,
            residual: result.residual,
            converged: result.converged,
            inlet_mean_axial_velocity_m_s: result.inlet_mean_axial_velocity_m_s,
            outlet_mean_axial_velocity_m_s: result.outlet_mean_axial_velocity_m_s,
            relative_mean_velocity_mismatch: result.relative_mean_velocity_mismatch,
            maximum_speed_m_s: result.maximum_speed_m_s,
            minimum_pressure_pa: result.minimum_pressure_pa,
            maximum_pressure_pa: result.maximum_pressure_pa,
        });
    }

    let improvement_factors = levels
        .windows(2)
        .map(|pair| {
            pair[0].relative_mean_velocity_mismatch / pair[1].relative_mean_velocity_mismatch
        })
        .collect::<Vec<_>>();
    assert!(
        improvement_factors
            .iter()
            .all(|factor| factor.is_finite() && *factor > 1.0)
    );
    let minimum_improvement_factor = improvement_factors
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let final_mismatch = levels
        .last()
        .expect("body-fitted convergence ladder has levels")
        .relative_mean_velocity_mismatch;
    assert!(final_mismatch <= FINAL_MISMATCH_TOLERANCE);

    let mut malformed = stations.to_vec();
    malformed[1].axial_position_m = malformed[0].axial_position_m;
    let invalid_profile = match solve_body_fitted_duct_flow(&malformed, options(2, 2)) {
        Err(error @ BodyFittedFlowError::NonMonotonicProfile { .. }) => FailureProbe {
            id: "non-monotonic-profile",
            status: "typed_rejected",
            publication: "none",
            typed_error: format!("{error:?}"),
        },
        other => panic!("malformed profile was not typed-rejected: {other:?}"),
    };
    let mut over_budget = options(2, 2);
    over_budget.maximum_nodes = 1;
    let invalid_budget = match solve_body_fitted_duct_flow(&stations, over_budget) {
        Err(error @ BodyFittedFlowError::MeshBudgetExceeded { .. }) => FailureProbe {
            id: "mesh-budget-exceeded",
            status: "typed_rejected",
            publication: "none",
            typed_error: format!("{error:?}"),
        },
        other => panic!("mesh budget was not typed-rejected: {other:?}"),
    };

    let canonical_input = format!(
        "schema={SCHEMA};density_kg_m3=1;dynamic_viscosity_pa_s=1;inlet_axial_velocity_m_s={INLET_VELOCITY_M_S};profile=[(0,0.02,0.05),(0.1,0.02,0.05)];levels={LEVELS:?};picard_tolerance=1e-8;linear_solver_tolerance=1e-10;final_mismatch_tolerance={FINAL_MISMATCH_TOLERANCE}"
    );
    let canonical_output = serde_json::to_string(&levels).expect("serialize body-fitted levels");
    let artifact = BodyFittedConvergenceArtifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64/AVX512 waived",
        claim_class: "production body-fitted FEM meridional-flow mesh convergence",
        reference: "constant-area incompressible continuity: outlet mean axial velocity approaches the prescribed inlet mean under mesh refinement",
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        canonical_input,
        dimensions: vec![
            "axial_elements",
            "radial_elements",
            "mesh_node_count",
            "mesh_element_count",
            "residual",
            "inlet_mean_axial_velocity_m_s",
            "outlet_mean_axial_velocity_m_s",
            "relative_mean_velocity_mismatch",
            "maximum_speed_m_s",
            "pressure_pa",
        ],
        levels,
        continuity_identity: "constant-area annulus with zero body force; outlet mean axial velocity converges toward inlet mean",
        final_mismatch_tolerance: FINAL_MISMATCH_TOLERANCE,
        minimum_improvement_factor,
        improvement_factors,
        final_relative_mean_velocity_mismatch: final_mismatch,
        converged: true,
        invalid_probes: vec![invalid_profile, invalid_budget],
        output_sha256: sha256_hex(canonical_output.as_bytes()),
        canonical_output,
    };
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create PP3 body-fitted output directory");
        fs::write(
            output_dir.join("native-x86-pp3-body-fitted-convergence-v1.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize body-fitted artifact"),
        )
        .expect("write PP3 body-fitted artifact");
    }
    artifact
}

#[test]
fn native_pp3_body_fitted_convergence_evidence_v1() {
    let artifact = evidence(
        std::env::var_os("SONIUM_OUTPUT_DIR")
            .map(|path| Path::new(&path).to_path_buf())
            .as_deref(),
    );
    assert_eq!(artifact.status, "pass");
    assert_eq!(artifact.levels.len(), LEVELS.len());
    assert!(artifact.converged);
    assert_eq!(artifact.invalid_probes.len(), 2);
}
