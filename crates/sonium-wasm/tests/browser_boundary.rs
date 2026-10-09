#![cfg(target_arch = "wasm32")]

use serde_json::Value;
use sonium_roomsim::{
    RoomSimulatorWasm, compute_webgpu_preview, create_default_config, get_wasm_info,
    init_thread_pool_if_supported, interpolate_measurement, parse_measurement_bundle_json,
    parse_measurement_csv, probe_webgpu_adapter, shared_wasm_memory_available, supports_threading,
    validate_config,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::window;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn default_config_and_capability_report_are_browser_safe() {
    let config = create_default_config();
    let validation: Value =
        serde_json::from_str(&validate_config(&config)).expect("validation JSON");
    assert_eq!(validation["valid"], true);
    assert!(validation["estimated_working_set_bytes"].as_u64().is_some());

    let capabilities: Value = serde_json::from_str(&get_wasm_info()).expect("capability JSON");
    for field in [
        "threading_supported",
        "shared_array_buffer",
        "shared_wasm_memory",
        "cross_origin_isolated",
        "simd_supported",
        "webgpu_supported",
        "webgpu_execution_mode",
        "webgpu_adapter_probe",
        "webgpu_backend_active",
        "webgpu_preview_available",
        "execution_backend",
        "fallback_reason",
        "execution_mode",
        "threads_available",
    ] {
        assert!(
            capabilities.get(field).is_some(),
            "missing capability field {field}"
        );
    }
}

#[wasm_bindgen_test]
async fn webgpu_probe_is_distinct_from_backend_selection() {
    let capabilities: Value = serde_json::from_str(&get_wasm_info()).expect("capability JSON");
    let adapter_available = probe_webgpu_adapter().await.expect("WebGPU probe");
    assert_eq!(capabilities["webgpu_backend_active"], false);
    assert!(matches!(
        capabilities["execution_backend"].as_str(),
        Some("wasm-cpu-threaded") | Some("wasm-cpu-single-threaded")
    ));
    if !capabilities["webgpu_supported"].as_bool().unwrap_or(false) {
        assert!(!adapter_available);
    }
}

#[wasm_bindgen_test]
async fn webgpu_preview_executes_or_reports_the_documented_fallback() {
    let input = r#"{
        "frequencies": [1000.0],
        "distances": [0.0],
        "amplitudes": [1.0],
        "speed_of_sound": 343.0
    }"#;
    let adapter_available = probe_webgpu_adapter().await.expect("WebGPU probe");
    match compute_webgpu_preview(input).await {
        Ok(result) => {
            let result: Value = serde_json::from_str(&result).expect("WebGPU preview JSON");
            assert_eq!(result["backend"], "webgpu");
            assert!((result["complex_response"][0][0].as_f64().unwrap() - 1.0).abs() < 1e-5);
            assert!(result["complex_response"][0][1].as_f64().unwrap().abs() < 1e-5);
        }
        Err(error) => {
            assert!(
                !adapter_available,
                "WebGPU preview failed despite an available adapter: {error:?}"
            );
        }
    }
}

#[wasm_bindgen_test]
async fn thread_pool_initialization_reports_safe_fallback() {
    let memory_shared = js_sys::Reflect::get(
        &wasm_bindgen::memory(),
        &wasm_bindgen::JsValue::from_str("buffer"),
    )
    .expect("WASM memory buffer")
    .is_instance_of::<js_sys::SharedArrayBuffer>();
    assert_eq!(shared_wasm_memory_available(), memory_shared);
    assert_eq!(
        supports_threading(),
        memory_shared
            && sonium_roomsim::shared_array_buffer_available()
            && sonium_roomsim::cross_origin_isolated()
    );
    assert!(init_thread_pool_if_supported(0).await.is_err());
    let initialized = init_thread_pool_if_supported(2)
        .await
        .expect("thread-pool initialization result");
    assert_eq!(initialized, supports_threading());
    let active_workers = rayon::current_num_threads();
    assert_eq!(active_workers >= 2, initialized);
    if initialized {
        // Keep two jobs in flight until both have started. This proves the
        // configured browser Rayon pool actually schedules work on two
        // distinct workers, rather than only reporting threading capability.
        let started = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_indices = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        // `std::time::Instant` is unsupported in this wasm target. Use the
        // browser clock so the bounded worker rendezvous can run in Chrome.
        let deadline_ms = js_sys::Date::now() + 5_000.0;
        rayon::scope(|scope| {
            for _ in 0..2 {
                let started = std::sync::Arc::clone(&started);
                let worker_indices = std::sync::Arc::clone(&worker_indices);
                scope.spawn(move |_| {
                    worker_indices
                        .lock()
                        .expect("worker index lock")
                        .push(rayon::current_thread_index().expect("Rayon worker index"));
                    started.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    while started.load(std::sync::atomic::Ordering::Acquire) < 2
                        && js_sys::Date::now() < deadline_ms
                    {
                        std::thread::yield_now();
                    }
                });
            }
        });
        let mut worker_indices = worker_indices.lock().expect("worker index lock").clone();
        worker_indices.sort_unstable();
        worker_indices.dedup();
        assert_eq!(
            worker_indices.len(),
            2,
            "distinct browser workers executed jobs"
        );
    }
}

#[wasm_bindgen_test]
fn measurement_loaders_reject_malformed_shapes_at_js_boundary() {
    assert!(parse_measurement_csv("frequency,SPL\n-1,-2").is_err());
    assert!(parse_measurement_bundle_json(r#"{"measurements":["Missing"],"data":{}}"#).is_err());
    assert!(interpolate_measurement(
        r#"{"measurement":{"name":"bad","frequency":[20,100],"spl":[-1]},"target_frequencies":[50]}"#
    )
    .is_err());
}

#[wasm_bindgen_test]
fn malformed_config_is_rejected_before_simulation() {
    let invalid_json: Value =
        serde_json::from_str(&validate_config("{")).expect("invalid JSON validation result");
    assert_eq!(invalid_json["valid"], false);
    assert!(RoomSimulatorWasm::new("{").is_err());

    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"]["width"] = Value::from(-1.0);
    let config_json = serde_json::to_string(&config).expect("config JSON");
    let validation: Value =
        serde_json::from_str(&validate_config(&config_json)).expect("validation JSON");
    assert_eq!(validation["valid"], false);
    assert!(RoomSimulatorWasm::new(&config_json).is_err());

    let mut density_config: Value =
        serde_json::from_str(&create_default_config()).expect("default JSON");
    density_config["solver"]["air_density"] = Value::from(-0.1);
    let density_json = serde_json::to_string(&density_config).expect("density config JSON");
    let density_validation: Value =
        serde_json::from_str(&validate_config(&density_json)).expect("density validation JSON");
    assert_eq!(density_validation["valid"], false);
}

#[wasm_bindgen_test]
fn frequency_result_exposes_validity_and_fallback_provenance() {
    let simulator = RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    let result: Value = serde_json::from_str(
        &simulator
            .compute_frequency_point(0)
            .expect("frequency result"),
    )
    .expect("frequency result JSON");
    assert!(result["provenance"]["validity"].is_array());
    assert!(result["provenance"]["bem_fallback_count"].is_number());
    assert!(result["provenance"]["uncertainty"].is_string());
    assert!(result["provenance"]["error_estimate"]["relative"].is_number());
    assert!(result["provenance"]["error_estimate"]["basis"].is_string());
}

#[wasm_bindgen_test]
fn impulse_response_exposes_time_sample_sensitivity() {
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["frequencies"]["min_freq"] = Value::from(100.0);
    config["frequencies"]["max_freq"] = Value::from(2_000.0);
    config["frequencies"]["num_points"] = Value::from(8);
    config["visualization"]["generate_impulse_response"] = Value::from(true);
    config["visualization"]["impulse_response"]["duration"] = Value::from(0.05);
    config["visualization"]["impulse_response"]["fft_size"] = Value::from(1024);
    let simulator = RoomSimulatorWasm::new(&serde_json::to_string(&config).expect("config JSON"))
        .expect("impulse-response simulator");
    let full_start = window()
        .expect("browser window")
        .performance()
        .expect("performance API")
        .now();
    let result: Value = serde_json::from_str(&simulator.run_simulation().expect("full result"))
        .expect("full result JSON");
    let full_result_ms = window()
        .expect("browser window")
        .performance()
        .expect("performance API")
        .now()
        - full_start;
    assert!(
        full_result_ms < 30_000.0,
        "complete-result budget exceeded: {full_result_ms:.1} ms"
    );
    assert!(result["impulse_response"]["uncertainty"].is_array());
    assert_eq!(
        result["impulse_response"]["uncertainty"]
            .as_array()
            .unwrap()
            .len(),
        result["impulse_response"]["amplitude"]
            .as_array()
            .unwrap()
            .len()
    );
}

#[wasm_bindgen_test]
fn first_result_stays_within_browser_performance_budget() {
    let performance = window()
        .expect("browser window")
        .performance()
        .expect("performance API");
    let start = performance.now();
    let simulator = RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    let startup_ms = performance.now() - start;
    assert!(
        startup_ms < 5_000.0,
        "startup budget exceeded: {startup_ms:.1} ms"
    );

    let first_start = performance.now();
    simulator.compute_frequency_point(0).expect("first result");
    let first_result_ms = performance.now() - first_start;
    assert!(
        first_result_ms < 10_000.0,
        "first-result budget exceeded: {first_result_ms:.1} ms"
    );
}

#[wasm_bindgen_test]
fn chunked_results_report_cancellation_without_false_completion() {
    let simulator = RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    assert_eq!(simulator.num_frequencies(), 100);

    let mut simulator = simulator;
    simulator.init_chunked_compute();
    simulator.cancel_chunked_compute();
    let progress: Value =
        serde_json::from_str(&simulator.get_chunk_progress()).expect("progress JSON");
    assert_eq!(progress["cancelled"], true);
    assert_eq!(progress["in_progress"], false);

    let result: Value =
        serde_json::from_str(&simulator.finalize_chunked_compute()).expect("result JSON");
    assert_eq!(result["cancelled"], true);
    assert_eq!(result["done"], false);
}

#[wasm_bindgen_test]
fn cancelled_chunked_results_retain_the_safe_partial_prefix() {
    let mut simulator =
        RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    simulator.init_chunked_compute();
    simulator.compute_chunk(3);
    simulator.cancel_chunked_compute();

    let result: Value =
        serde_json::from_str(&simulator.finalize_chunked_compute()).expect("result JSON");
    assert_eq!(result["cancelled"], true);
    assert_eq!(result["done"], false);
    assert_eq!(result["frequencies_completed"], 3);
    assert_eq!(result["partial_spl"].as_array().unwrap().len(), 3);
    assert!(result["provenance"].is_object());
}

#[wasm_bindgen_test]
fn cancelling_a_completed_chunked_result_preserves_completion() {
    let mut simulator =
        RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    simulator.init_chunked_compute();
    let total = simulator.num_frequencies();
    simulator.compute_chunk(total);
    simulator.cancel_chunked_compute();

    let result: Value =
        serde_json::from_str(&simulator.finalize_chunked_compute()).expect("result JSON");
    assert_eq!(result["cancelled"], Value::Null);
    assert_eq!(result["done"], true);
    assert_eq!(result["frequencies_completed"], total);
}

#[wasm_bindgen_test]
fn chunked_results_expose_partial_prefix_before_completion() {
    let mut simulator =
        RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    simulator.init_chunked_compute();
    let first: Value = serde_json::from_str(&simulator.compute_chunk(3)).expect("chunk JSON");
    assert_eq!(first["done"], false);
    assert_eq!(first["frequencies_completed"], 3);
    assert_eq!(first["partial_spl"].as_array().unwrap().len(), 3);
}

#[wasm_bindgen_test]
fn incomplete_chunked_result_cannot_be_reported_as_final() {
    let mut simulator =
        RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    simulator.init_chunked_compute();
    simulator.compute_chunk(3);

    let final_result: Value =
        serde_json::from_str(&simulator.finalize_chunked_compute()).expect("final JSON");
    assert_eq!(final_result["done"], false);
    assert_eq!(final_result["frequencies_completed"], 3);
    assert_eq!(final_result["partial_spl"].as_array().unwrap().len(), 3);
    assert!(
        final_result["error"]
            .as_str()
            .unwrap()
            .contains("incomplete")
    );
}

#[wasm_bindgen_test]
fn canonical_direct_field_matches_golden_fixture() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/canonical_direct_field.json"))
        .expect("canonical fixture JSON");
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"] = fixture["room"].clone();
    config["sources"] = serde_json::json!([{
        "name": "golden-source",
        "position": {"x": fixture["source"][0], "y": fixture["source"][1], "z": fixture["source"][2]},
        "amplitude": 1.0,
        "orientation": {"x": 0.0, "y": 1.0, "z": 0.0},
        "directivity": {"type": "omnidirectional"}
    }]);
    config["listening_positions"] = serde_json::json!([{
        "x": fixture["listener"][0],
        "y": fixture["listener"][1],
        "z": fixture["listener"][2]
    }]);
    config["frequencies"] = serde_json::json!({
        "min_freq": fixture["frequency_hz"],
        "max_freq": fixture["frequency_hz"].as_f64().unwrap() + 1.0,
        "num_points": 2,
        "spacing": "linear"
    });
    config["solver"]["method"] = "direct".into();
    config["solver"]["speed_of_sound"] = fixture["speed_of_sound"].clone();
    config["solver"]["air_absorption"] = false.into();
    config["visualization"]["generate_slices"] = false.into();

    let simulator = RoomSimulatorWasm::new(&serde_json::to_string(&config).expect("config JSON"))
        .expect("golden simulator");
    let result: Value = serde_json::from_str(
        &simulator
            .compute_frequency_point(0)
            .expect("golden frequency point"),
    )
    .expect("golden result JSON");
    let real_error = (result["pressure_real"].as_f64().unwrap()
        - fixture["expected_pressure"]["real"].as_f64().unwrap())
    .abs();
    let imag_error = (result["pressure_imag"].as_f64().unwrap()
        - fixture["expected_pressure"]["imag"].as_f64().unwrap())
    .abs();
    assert!(real_error < 1.0e-14, "real golden mismatch: {real_error}");
    assert!(
        imag_error < 1.0e-14,
        "imaginary golden mismatch: {imag_error}"
    );
}
