//! Regression tests for `reviews/spark-sonium-wasm-20260904.md`:
//! binaural per-source accumulation, slice solver-method provenance,
//! BEM-plus-impedance rejection, damping floor, and hybrid determinism.

#![cfg(not(target_arch = "wasm32"))]

use serde_json::{Value, json};
use sonium_roomsim::{RoomSimulatorWasm, create_default_config, validate_config};

fn base_config() -> Value {
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"] = json!({"type":"rectangular", "width":5.0, "depth":4.0, "height":2.8});
    config["sources"][0]["position"] = json!({"x":4.5, "y":0.5, "z":1.2});
    config["sources"]
        .as_array_mut()
        .expect("sources")
        .truncate(1);
    config["listening_positions"] = json!([{"x":2.5, "y":2.0, "z":1.2}]);
    config["frequencies"]["min_freq"] = json!(500.0);
    config["frequencies"]["max_freq"] = json!(501.0);
    config["frequencies"]["num_points"] = json!(2);
    config["solver"]["method"] = json!("direct");
    config["visualization"]["binaural"]["enabled"] = json!(true);
    config
}

fn simulator(config: &Value) -> RoomSimulatorWasm {
    RoomSimulatorWasm::new(&serde_json::to_string(config).expect("config JSON"))
        .expect("simulator from valid config")
}

#[test]
fn binaural_two_identical_sources_preserve_ild() {
    // Linearity invariant: duplicating a source scales both ears equally,
    // so ILD is unchanged. The old total-field-times-every-gain code scaled
    // the ears by different gain powers and broke this invariant.
    let single = base_config();
    let single_out: Value =
        serde_json::from_str(&simulator(&single).run_simulation().expect("run"))
            .expect("result JSON");
    let ild_single = single_out["binaural_response"]["ild"]
        .as_f64()
        .expect("single-source ILD");

    let mut doubled = base_config();
    let second = doubled["sources"][0].clone();
    doubled["sources"]
        .as_array_mut()
        .expect("sources")
        .push(second);
    let double_out: Value =
        serde_json::from_str(&simulator(&doubled).run_simulation().expect("run"))
            .expect("result JSON");
    let ild_double = double_out["binaural_response"]["ild"]
        .as_f64()
        .expect("two-source ILD");

    assert!(
        ild_single.is_finite() && ild_double.is_finite(),
        "ILD must be finite: {ild_single} vs {ild_double}"
    );
    assert!(
        (ild_single - ild_double).abs() < 1e-9,
        "duplicating a source must preserve ILD: {ild_single} vs {ild_double}"
    );
}

#[test]
fn slices_carry_the_configured_solver_method() {
    // Slices evaluate the same method-aware field as the main response
    // (`calculate_direct_field` dispatches on method); provenance must agree.
    for method in ["modal", "image-source-2"] {
        let mut config = base_config();
        config["solver"]["method"] = json!(method);
        config["visualization"]["binaural"]["enabled"] = json!(false);
        let out: Value = serde_json::from_str(&simulator(&config).run_simulation().expect("run"))
            .expect("result JSON");
        let slices = out["horizontal_slices"]
            .as_array()
            .expect("horizontal slices present");
        assert!(!slices.is_empty(), "method {method} must emit slices");
        assert_eq!(
            slices[0]["provenance"]["method"].as_str(),
            Some(method),
            "slice provenance must name the configured solver method"
        );
        assert_eq!(out["provenance"]["method"].as_str(), Some(method));
    }
}

#[test]
fn bem_with_impedance_table_is_rejected_loudly() {
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["solver"]["method"] = json!("bem");
    config["wall_materials"]["left"] = json!({
        "type": "custom",
        "name": "impedance-wall",
        "absorption": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "impedance": {
            "real": [820.0, 820.0, 820.0, 820.0, 820.0, 820.0],
            "imag": [205.0, 205.0, 205.0, 205.0, 205.0, 205.0]
        }
    });
    let report: Value = serde_json::from_str(&validate_config(
        &serde_json::to_string(&config).expect("JSON"),
    ))
    .expect("validation JSON");
    assert_eq!(
        report["valid"], false,
        "BEM + impedance must be rejected, not silently defaulted"
    );
    assert!(RoomSimulatorWasm::new(&serde_json::to_string(&config).expect("JSON")).is_err());
}

#[test]
fn zero_modal_damping_is_rejected() {
    // A zero Q-factor denominator is a pole (inf/NaN pressures); validation
    // requires strictly positive damping.
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["solver"]["modal_damping"] = json!(0.0);
    let report: Value = serde_json::from_str(&validate_config(
        &serde_json::to_string(&config).expect("JSON"),
    ))
    .expect("validation JSON");
    assert_eq!(report["valid"], false, "damping 0 must be rejected");
}

#[test]
fn hybrid_run_is_finite_and_deterministic() {
    // The broadband Schroeder crossover must not depend on evaluation order:
    // two identical hybrid runs agree exactly (serial and parallel sweep
    // paths share one closure), and every SPL is finite.
    let mut config = base_config();
    config["solver"]["method"] = json!("hybrid");
    config["visualization"]["binaural"]["enabled"] = json!(false);
    let first: Value = serde_json::from_str(&simulator(&config).run_simulation().expect("run"))
        .expect("result JSON");
    let second: Value = serde_json::from_str(&simulator(&config).run_simulation().expect("run"))
        .expect("result JSON");
    assert_eq!(first["frequency_response"], second["frequency_response"]);
    for spl in first["frequency_response"]
        .as_array()
        .expect("frequency response")
    {
        assert!(
            spl.as_f64().is_some_and(|v| v.is_finite()),
            "hybrid SPL must be finite: {spl}"
        );
    }
}
