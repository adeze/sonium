//! Release-facing rigid-sphere reciprocity/passivity reference.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/sphere_reciprocity_passivity.rs"]
mod sphere_reciprocity_passivity;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.sphere-reciprocity-passivity.v1";
const MAX_RECIPROCITY_RELATIVE_ERROR: f64 = 2.0e-2;
const MAX_SCATTERED_POWER_RELATIVE_ERROR: f64 = 8.0e-2;
const MAX_FIELD_PATTERN_L2_RELATIVE_ERROR: f64 = 1.5e-1;
const MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR: f64 = 2.5e-1;

fn main() {
    let started = Instant::now();
    match sphere_reciprocity_passivity::release_metrics() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("independent rigid-sphere Mie scattering series")
                .reference_checksum("analytic/deterministic-fixture:bem-sphere-mie-v1")
                .input_checksum("analytic/deterministic-fixture:bem-sphere-mie-v1")
                .dimension("3D rigid sphere; ka=0.5; constant-panel TBEM; reciprocal +/-z plane waves")
                .formulation("production dense TBEM BiCGSTAB solve, exterior field evaluation, Mie comparison")
                .tolerance("reciprocity <= 2e-2; scattered-power <= 8e-2; field L2 <= 1.5e-1; pointwise <= 2.5e-1")
                .metric("reciprocity_relative_error", metrics.reciprocity_relative_error)
                .metric("scattered_power_w", metrics.scattered_power_w)
                .metric("scattered_power_relative_error", metrics.scattered_power_relative_error)
                .metric("field_pattern_l2_relative_error", metrics.field_pattern_l2_relative_error)
                .metric(
                    "field_pattern_pointwise_relative_error",
                    metrics.field_pattern_pointwise_relative_error,
                )
                .maximum("reciprocity_relative_error", MAX_RECIPROCITY_RELATIVE_ERROR)
                .maximum(
                    "scattered_power_relative_error",
                    MAX_SCATTERED_POWER_RELATIVE_ERROR,
                )
                .maximum(
                    "field_pattern_l2_relative_error",
                    MAX_FIELD_PATTERN_L2_RELATIVE_ERROR,
                )
                .maximum(
                    "field_pattern_pointwise_relative_error",
                    MAX_FIELD_PATTERN_POINTWISE_RELATIVE_ERROR,
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("BEM sphere QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone());
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("BEM sphere QA failure must serialize")
            );
            eprintln!("BEM SPHERE RECIPROCITY/PASSIVITY FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
