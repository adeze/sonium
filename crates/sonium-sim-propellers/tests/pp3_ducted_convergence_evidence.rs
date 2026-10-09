//! Native PP3 ducted/shroud station-resolution convergence evidence.
//!
//! This packet exercises the production quasi-one-dimensional duct/shroud
//! evaluator on a smooth area transition at six station resolutions.  The
//! independent oracle is the closed-form laminar Darcy integral plus the
//! Bernoulli pressure identity.  It is intentionally reduced-order duct
//! evidence, not resolved three-dimensional CFD or externally measured data.

use serde::Serialize;
use sonium_sim_propellers::{
    DuctShroudError, DuctShroudOptions, DuctShroudStation, evaluate_duct_shroud_reference,
    sha256_hex,
};
use std::fs;
use std::path::Path;

const SCHEMA: &str = "sonium.propellers.ducted-convergence-evidence.v1";
const SEGMENT_LEVELS: [usize; 6] = [4, 8, 16, 32, 64, 128];
const LENGTH_M: f64 = 0.8;
const INLET_AREA_M2: f64 = 0.01;
const OUTLET_AREA_M2: f64 = 0.025;
const HYDRAULIC_DIAMETER_M: f64 = 0.04;
const MASS_FLOW_KG_S: f64 = 0.01;
const DENSITY_KG_M3: f64 = 1.2;
const DYNAMIC_VISCOSITY_PA_S: f64 = 1.0e-3;
const INLET_STATIC_PRESSURE_PA: f64 = 101_325.0;
const SPEED_OF_SOUND_M_S: f64 = 343.0;
const FINAL_RELATIVE_TOLERANCE: f64 = 1.0e-2;

#[derive(Debug, Serialize)]
struct ResolutionLevel {
    axial_segments: usize,
    station_count: usize,
    axial_step_m: f64,
    expected_wall_loss_pa: f64,
    observed_wall_loss_pa: f64,
    relative_wall_loss_error: f64,
    expected_outlet_static_pressure_pa: f64,
    observed_outlet_static_pressure_pa: f64,
    relative_outlet_pressure_error: f64,
    maximum_continuity_residual_kg_s: f64,
    maximum_velocity_m_s: f64,
    maximum_mach: f64,
    reynolds_number_range: [f64; 2],
    finite: bool,
}

#[derive(Debug, Serialize)]
struct FailureProbe {
    id: &'static str,
    status: &'static str,
    publication: &'static str,
    typed_error: String,
}

#[derive(Debug, Serialize)]
struct DuctedConvergenceArtifact {
    schema: &'static str,
    status: &'static str,
    scope: &'static str,
    claim_class: &'static str,
    reference: &'static str,
    canonical_input: String,
    canonical_input_sha256: String,
    dimensions: Vec<&'static str>,
    levels: Vec<ResolutionLevel>,
    independent_oracle: &'static str,
    observed_orders: Vec<f64>,
    minimum_observed_order: f64,
    final_relative_wall_loss_error: f64,
    final_relative_pressure_error: f64,
    final_tolerance: f64,
    converged: bool,
    deterministic_repeat: bool,
    invalid_probes: Vec<FailureProbe>,
    canonical_output: String,
    output_sha256: String,
}

fn options(maximum_stations: usize) -> DuctShroudOptions {
    DuctShroudOptions {
        mass_flow_kg_s: MASS_FLOW_KG_S,
        density_kg_m3: DENSITY_KG_M3,
        dynamic_viscosity_pa_s: DYNAMIC_VISCOSITY_PA_S,
        inlet_static_pressure_pa: INLET_STATIC_PRESSURE_PA,
        speed_of_sound_m_s: SPEED_OF_SOUND_M_S,
        minor_loss_coefficient: 0.0,
        maximum_velocity_m_s: 200.0,
        maximum_mach: 0.6,
        maximum_stations,
    }
}

fn stations(axial_segments: usize) -> Vec<DuctShroudStation> {
    (0..=axial_segments)
        .map(|index| {
            let fraction = index as f64 / axial_segments as f64;
            DuctShroudStation {
                axial_position_m: LENGTH_M * fraction,
                flow_area_m2: INLET_AREA_M2 + (OUTLET_AREA_M2 - INLET_AREA_M2) * fraction,
                hydraulic_diameter_m: HYDRAULIC_DIAMETER_M,
            }
        })
        .collect()
}

/// Closed-form laminar Darcy loss for the linearly varying area profile.
fn independent_wall_loss_pa() -> f64 {
    let coefficient = 32.0 * DYNAMIC_VISCOSITY_PA_S * MASS_FLOW_KG_S
        / (DENSITY_KG_M3 * HYDRAULIC_DIAMETER_M.powi(2));
    coefficient * LENGTH_M / (OUTLET_AREA_M2 - INLET_AREA_M2)
        * (OUTLET_AREA_M2 / INLET_AREA_M2).ln()
}

fn independent_outlet_pressure_pa() -> f64 {
    let inlet_velocity = MASS_FLOW_KG_S / (DENSITY_KG_M3 * INLET_AREA_M2);
    let outlet_velocity = MASS_FLOW_KG_S / (DENSITY_KG_M3 * OUTLET_AREA_M2);
    let inlet_dynamic = 0.5 * DENSITY_KG_M3 * inlet_velocity.powi(2);
    let outlet_dynamic = 0.5 * DENSITY_KG_M3 * outlet_velocity.powi(2);
    INLET_STATIC_PRESSURE_PA + inlet_dynamic - outlet_dynamic - independent_wall_loss_pa()
}

fn relative_error(observed: f64, expected: f64) -> f64 {
    (observed - expected).abs() / expected.abs().max(1.0)
}

fn evidence(output_dir: Option<&Path>) -> DuctedConvergenceArtifact {
    assert_eq!(std::env::consts::OS, "linux", "PP3 requires native Linux");
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "PP3 requires native x86_64"
    );

    let expected_wall_loss = independent_wall_loss_pa();
    let expected_outlet_pressure = independent_outlet_pressure_pa();
    let mut levels = Vec::with_capacity(SEGMENT_LEVELS.len());
    for axial_segments in SEGMENT_LEVELS {
        let station_history = stations(axial_segments);
        let result = evaluate_duct_shroud_reference(&station_history, options(axial_segments + 1))
            .expect("valid ducted station ladder");
        assert!(result.is_finite());
        assert_eq!(result.stations.len(), axial_segments + 1);
        assert!(result.maximum_continuity_residual_kg_s <= 1.0e-12);
        let reynolds = result
            .stations
            .iter()
            .map(|station| station.reynolds_number)
            .collect::<Vec<_>>();
        let level = ResolutionLevel {
            axial_segments,
            station_count: station_history.len(),
            axial_step_m: LENGTH_M / axial_segments as f64,
            expected_wall_loss_pa: expected_wall_loss,
            observed_wall_loss_pa: result.total_wall_loss_pa,
            relative_wall_loss_error: relative_error(result.total_wall_loss_pa, expected_wall_loss),
            expected_outlet_static_pressure_pa: expected_outlet_pressure,
            observed_outlet_static_pressure_pa: result.outlet_static_pressure_pa,
            relative_outlet_pressure_error: relative_error(
                result.outlet_static_pressure_pa,
                expected_outlet_pressure,
            ),
            maximum_continuity_residual_kg_s: result.maximum_continuity_residual_kg_s,
            maximum_velocity_m_s: result.maximum_velocity_m_s,
            maximum_mach: result.maximum_mach,
            reynolds_number_range: [
                reynolds.iter().copied().fold(f64::INFINITY, f64::min),
                reynolds.iter().copied().fold(0.0, f64::max),
            ],
            finite: result.is_finite(),
        };
        assert!(level.relative_wall_loss_error.is_finite());
        assert!(level.relative_outlet_pressure_error.is_finite());
        assert!(level.relative_wall_loss_error > 0.0);
        assert!(level.relative_outlet_pressure_error > 0.0);
        assert!(level.reynolds_number_range[1] < 2_300.0);
        levels.push(level);
    }

    let observed_orders = levels
        .windows(2)
        .map(|pair| {
            (pair[0].relative_wall_loss_error / pair[1].relative_wall_loss_error).ln()
                / 2.0_f64.ln()
        })
        .collect::<Vec<_>>();
    assert!(
        observed_orders
            .iter()
            .all(|order| order.is_finite() && *order > 0.7)
    );
    let minimum_observed_order = observed_orders
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let final_wall_error = levels
        .last()
        .expect("ducted convergence ladder has levels")
        .relative_wall_loss_error;
    let final_pressure_error = levels
        .last()
        .expect("ducted convergence ladder has levels")
        .relative_outlet_pressure_error;
    assert!(final_wall_error <= FINAL_RELATIVE_TOLERANCE);
    assert!(final_pressure_error <= FINAL_RELATIVE_TOLERANCE);

    let repeat = evaluate_duct_shroud_reference(
        &stations(*SEGMENT_LEVELS.last().unwrap()),
        options(*SEGMENT_LEVELS.last().unwrap() + 1),
    )
    .expect("deterministic repeat");
    let deterministic_repeat = repeat.total_wall_loss_pa
        == levels.last().unwrap().observed_wall_loss_pa
        && repeat.outlet_static_pressure_pa
            == levels.last().unwrap().observed_outlet_static_pressure_pa;
    assert!(deterministic_repeat);

    let mut malformed = stations(4);
    malformed[2].axial_position_m = malformed[1].axial_position_m;
    let invalid_axial = match evaluate_duct_shroud_reference(&malformed, options(5)) {
        Err(error @ DuctShroudError::NonMonotonicAxial { .. }) => FailureProbe {
            id: "non-monotonic-axial-station",
            status: "typed_rejected",
            publication: "none",
            typed_error: format!("{error:?}"),
        },
        other => panic!("malformed axial station was not typed-rejected: {other:?}"),
    };
    let invalid_budget = match evaluate_duct_shroud_reference(&stations(4), options(4)) {
        Err(error @ DuctShroudError::TooManyStations { .. }) => FailureProbe {
            id: "station-budget-exceeded",
            status: "typed_rejected",
            publication: "none",
            typed_error: format!("{error:?}"),
        },
        other => panic!("station budget was not typed-rejected: {other:?}"),
    };

    let canonical_input = format!(
        "schema={SCHEMA};mass_flow_kg_s={MASS_FLOW_KG_S};density_kg_m3={DENSITY_KG_M3};dynamic_viscosity_pa_s={DYNAMIC_VISCOSITY_PA_S};inlet_static_pressure_pa={INLET_STATIC_PRESSURE_PA};speed_of_sound_m_s={SPEED_OF_SOUND_M_S};length_m={LENGTH_M};area_m2=[{INLET_AREA_M2},{OUTLET_AREA_M2}];hydraulic_diameter_m={HYDRAULIC_DIAMETER_M};minor_loss_coefficient=0;levels={SEGMENT_LEVELS:?};oracle=laminar-darcy-integral-plus-bernoulli;final_tolerance={FINAL_RELATIVE_TOLERANCE}"
    );
    let canonical_output = serde_json::to_string(&levels).expect("serialize ducted levels");
    let artifact = DuctedConvergenceArtifact {
        schema: SCHEMA,
        status: "pass",
        scope: "native Linux x86_64 CPU/f64; Windows/Arm64/AVX512 waived",
        claim_class: "production reduced-order duct/shroud station-resolution convergence",
        reference: "independent laminar Darcy wall-loss integral and Bernoulli static-pressure identity for a linearly varying-area streamtube",
        canonical_input_sha256: sha256_hex(canonical_input.as_bytes()),
        canonical_input,
        dimensions: vec![
            "axial_segments",
            "station_count",
            "axial_step_m",
            "flow_area_m2",
            "hydraulic_diameter_m",
            "mass_flow_kg_s",
            "density_kg_m3",
            "dynamic_viscosity_pa_s",
            "wall_loss_pa",
            "static_pressure_pa",
            "continuity_residual_kg_s",
            "reynolds_number",
            "mach_number",
        ],
        levels,
        independent_oracle: "closed-form laminar Darcy integral plus Bernoulli pressure balance; oracle does not call the production evaluator",
        observed_orders,
        minimum_observed_order,
        final_relative_wall_loss_error: final_wall_error,
        final_relative_pressure_error: final_pressure_error,
        final_tolerance: FINAL_RELATIVE_TOLERANCE,
        converged: true,
        deterministic_repeat,
        invalid_probes: vec![invalid_axial, invalid_budget],
        output_sha256: sha256_hex(canonical_output.as_bytes()),
        canonical_output,
    };
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(output_dir).expect("create PP3 ducted output directory");
        fs::write(
            output_dir.join("native-x86-pp3-ducted-convergence-v1.json"),
            serde_json::to_vec_pretty(&artifact).expect("serialize ducted artifact"),
        )
        .expect("write PP3 ducted artifact");
    }
    artifact
}

#[test]
fn native_pp3_ducted_convergence_evidence_v1() {
    let artifact = evidence(
        std::env::var_os("SONIUM_OUTPUT_DIR")
            .map(|path| Path::new(&path).to_path_buf())
            .as_deref(),
    );
    assert_eq!(artifact.status, "pass");
    assert_eq!(artifact.levels.len(), SEGMENT_LEVELS.len());
    assert!(artifact.converged);
    assert!(artifact.deterministic_repeat);
    assert_eq!(artifact.invalid_probes.len(), 2);
}
