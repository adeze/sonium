//! Release-facing H-matrix versus dense rigid-sphere reference.

#[allow(dead_code)]
#[path = "../../sonium-bem/tests/sphere_reciprocity_passivity.rs"]
mod sphere_reciprocity_passivity;

use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.sphere-hmatrix-backend-parity.v1";
const MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR: f64 = 2.0e-5;
const MAX_RECIPROCITY_RELATIVE_ERROR: f64 = 2.0e-2;
const MAX_SCATTERED_POWER_RELATIVE_ERROR: f64 = 8.0e-2;

fn main() {
    let started = Instant::now();
    match sphere_reciprocity_passivity::hmatrix_release_metrics() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("dense TBEM and independent rigid-sphere Mie scattering series")
                .reference_checksum("analytic/deterministic-fixture:bem-sphere-mie-v1")
                .input_checksum("analytic/deterministic-fixture:bem-sphere-mie-v1")
                .dimension("3D rigid sphere; ka=0.5; H-matrix GMRES versus dense TBEM; reciprocal +/-z plane waves")
                .formulation("production H-matrix GMRES exterior field versus dense TBEM and Mie power")
                .tolerance("H-matrix backscatter <= 2e-5; reciprocity <= 2e-2; Mie scattered-power <= 8e-2")
                .metric("dense_backscatter_relative_error", metrics.dense_backscatter_relative_error)
                .metric("reciprocity_relative_error", metrics.reciprocity_relative_error)
                .metric("scattered_power_w", metrics.scattered_power_w)
                .metric("scattered_power_relative_error", metrics.scattered_power_relative_error)
                .maximum(
                    "dense_backscatter_relative_error",
                    MAX_HMATRIX_BACKSCATTER_RELATIVE_ERROR,
                )
                .maximum("reciprocity_relative_error", MAX_RECIPROCITY_RELATIVE_ERROR)
                .maximum(
                    "scattered_power_relative_error",
                    MAX_SCATTERED_POWER_RELATIVE_ERROR,
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("BEM H-matrix QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone());
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("BEM H-matrix QA failure must serialize")
            );
            eprintln!("BEM H-MATRIX REFERENCE FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
