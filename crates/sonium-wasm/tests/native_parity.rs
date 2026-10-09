#![cfg(not(target_arch = "wasm32"))]

use num_complex::Complex64;
use serde_json::{Value, json};
use sonium_compute::HostMatrix;
use sonium_roomsim::{RoomSimulatorWasm, create_default_config, validate_config};
use sonium_sim_room::fast_room::{FastRoomConfig, run_fast_hybrid_with_config};
use sonium_xem::{
    BoundaryConfig, Directivity, DirectivityGrid, Point3D, RectangularRoom, RoomGeometry,
    RoomSimulation, Source, SurfaceConfig,
};

fn rigid_material() -> Value {
    json!({
        "type": "custom",
        "name": "rigid-parity-wall",
        "absorption": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
}

fn fully_absorbing_material() -> Value {
    json!({
        "type": "custom",
        "name": "fully-absorbing-parity-wall",
        "absorption": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
}

fn directional_grid_config() -> Value {
    json!({"type": "cardioid", "front_back_ratio": 4.0})
}

fn impedance_material() -> Value {
    json!({
        "type": "custom",
        "name": "constant-impedance-parity-wall",
        "absorption": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "impedance": {
            "real": [820.0, 820.0, 820.0, 820.0, 820.0, 820.0],
            "imag": [205.0, 205.0, 205.0, 205.0, 205.0, 205.0]
        }
    })
}

#[test]
fn wasm_rectangular_ism_matches_native_fast_room_at_high_frequency() {
    let source_position = Point3D::new(1.0, 1.2, 1.1);
    let listener_position = Point3D::new(3.4, 2.7, 1.2);
    let frequency = 1_000.0;

    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"] = json!({"type":"rectangular", "width":5.0, "depth":4.0, "height":2.8});
    config["sources"][0]["position"] =
        json!({"x":source_position.x, "y":source_position.y, "z":source_position.z});
    config["sources"]
        .as_array_mut()
        .expect("sources")
        .truncate(1);
    config["listening_positions"] =
        json!([{"x":listener_position.x, "y":listener_position.y, "z":listener_position.z}]);
    config["frequencies"]["min_freq"] = json!(frequency);
    config["frequencies"]["max_freq"] = json!(frequency + 1.0);
    config["frequencies"]["num_points"] = json!(2);
    config["solver"]["method"] = json!("image-source-2");
    config["solver"]["air_absorption"] = json!(false);
    for surface in ["left", "right", "front", "back", "floor", "ceiling"] {
        config["wall_materials"][surface] = rigid_material();
    }
    config["visualization"]["generate_impulse_response"] = json!(true);
    config["visualization"]["impulse_response"]["duration"] = json!(0.1);
    config["visualization"]["impulse_response"]["fft_size"] = json!(2048);

    let wasm = RoomSimulatorWasm::new(&serde_json::to_string(&config).expect("WASM config JSON"))
        .expect("WASM simulator");
    let point: Value = serde_json::from_str(
        &wasm
            .compute_frequency_point(0)
            .expect("WASM frequency point"),
    )
    .expect("WASM point JSON");
    let wasm_pressure = num_complex::Complex64::new(
        point["pressure_real"].as_f64().expect("WASM real pressure"),
        point["pressure_imag"]
            .as_f64()
            .expect("WASM imaginary pressure"),
    );
    assert!(
        point["provenance"]["error_estimate"]["relative"]
            .as_f64()
            .is_some()
    );
    assert!(
        point["provenance"]["error_estimate"]["basis"]
            .as_str()
            .is_some_and(|basis| basis.contains("image-source order 2 versus 1"))
    );

    let full: Value = serde_json::from_str(&wasm.run_simulation().expect("WASM full result"))
        .expect("WASM full result JSON");
    assert_eq!(full["frequency_uncertainty"].as_array().unwrap().len(), 2);
    assert!(full["frequency_uncertainty"][0]["basis"].is_string());
    assert!(full["impulse_response"]["uncertainty"].is_array());
    assert_eq!(
        full["impulse_response"]["uncertainty"]
            .as_array()
            .unwrap()
            .len(),
        full["impulse_response"]["amplitude"]
            .as_array()
            .unwrap()
            .len()
    );

    let native_simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(5.0, 4.0, 2.8)),
        vec![Source::omnidirectional(source_position, 1.0)],
        vec![listener_position],
        frequency,
        frequency + 1.0,
        2,
    );
    let mut native_config = FastRoomConfig::default();
    native_config.image_order = 2;
    native_config.max_path_time = 1.0;
    native_config.min_transition_hz = 100.0;
    native_config.max_transition_hz = 500.0;
    let native = run_fast_hybrid_with_config(&native_simulation, &native_config)
        .expect("native fast-room result");
    let native_pressure = native.pressures[0][0][0];

    let error = (wasm_pressure - native_pressure).norm();
    let scale = native_pressure.norm().max(1.0e-12);
    assert!(
        error / scale < 1.0e-11,
        "WASM/native ISM mismatch: wasm={wasm_pressure:?}, native={native_pressure:?}, relative_error={}",
        error / scale
    );
}

#[test]
fn modal_result_reports_adjacent_order_sensitivity() {
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["solver"]["method"] = json!("modal");
    config["solver"]["max_mode_order"] = json!(4);
    config["frequencies"]["min_freq"] = json!(100.0);
    config["frequencies"]["max_freq"] = json!(101.0);
    config["frequencies"]["num_points"] = json!(2);
    let simulator = RoomSimulatorWasm::new(&serde_json::to_string(&config).expect("config"))
        .expect("modal simulator");
    let point: Value =
        serde_json::from_str(&simulator.compute_frequency_point(0).expect("modal point"))
            .expect("modal point JSON");
    assert_eq!(
        point["provenance"]["error_estimate"]["basis"],
        "modal order 4 versus 3"
    );
}

#[test]
fn wasm_directional_ism_matches_native_fast_room() {
    let source_position = Point3D::new(1.8, 1.4, 1.1);
    let listener_position = Point3D::new(2.0, 3.1, 1.2);
    let frequency = 1_000.0;
    let directivity_json = directional_grid_config();

    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"] = json!({"type":"rectangular", "width":5.0, "depth":4.0, "height":2.8});
    config["sources"][0]["position"] =
        json!({"x":source_position.x, "y":source_position.y, "z":source_position.z});
    config["sources"][0]["directivity"] = directivity_json.clone();
    config["sources"][0]["orientation"] = json!({"x":1.0, "y":0.0, "z":0.0});
    config["sources"]
        .as_array_mut()
        .expect("sources")
        .truncate(1);
    config["listening_positions"] =
        json!([{"x":listener_position.x, "y":listener_position.y, "z":listener_position.z}]);
    config["frequencies"]["min_freq"] = json!(frequency);
    config["frequencies"]["max_freq"] = json!(frequency + 1.0);
    config["frequencies"]["num_points"] = json!(2);
    config["solver"]["method"] = json!("image-source-2");
    config["solver"]["air_absorption"] = json!(false);
    for surface in ["left", "right", "front", "back", "floor", "ceiling"] {
        config["wall_materials"][surface] = rigid_material();
    }

    let config_json = serde_json::to_string(&config).expect("config JSON");
    let validation: Value =
        serde_json::from_str(&validate_config(&config_json)).expect("validation JSON");
    assert_eq!(
        validation["valid"], true,
        "invalid directional config: {validation}"
    );
    let wasm = RoomSimulatorWasm::new(&config_json).expect("WASM simulator");
    let point: Value = serde_json::from_str(
        &wasm
            .compute_frequency_point(0)
            .expect("WASM frequency point"),
    )
    .expect("WASM point JSON");
    let wasm_pressure = num_complex::Complex64::new(
        point["pressure_real"].as_f64().expect("WASM real pressure"),
        point["pressure_imag"]
            .as_f64()
            .expect("WASM imaginary pressure"),
    );

    let front_back_ratio = 4.0;
    let a = (front_back_ratio + 1.0) / (2.0 * front_back_ratio);
    let b = (front_back_ratio - 1.0) / (2.0 * front_back_ratio);
    let horizontal_angles: Vec<f64> = (0..36).map(|index| index as f64 * 10.0).collect();
    let vertical_angles: Vec<f64> = (0..19).map(|index| index as f64 * 10.0).collect();
    let magnitude = HostMatrix::from_fn(vertical_angles.len(), horizontal_angles.len(), |v, h| {
        let theta = vertical_angles[v].to_radians();
        let phi = horizontal_angles[h].to_radians();
        (a + b * theta.sin() * phi.cos()).max(0.0)
    });
    let grid = DirectivityGrid::try_new(horizontal_angles, vertical_angles, magnitude)
        .expect("valid native directivity grid");
    let native_source = Source::new(source_position, Directivity::Grid(grid), 1.0)
        .with_orientation(Point3D::new(1.0, 0.0, 0.0));
    let native_simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(5.0, 4.0, 2.8)),
        vec![native_source],
        vec![listener_position],
        frequency,
        frequency + 1.0,
        2,
    );
    let mut native_config = FastRoomConfig::default();
    native_config.image_order = 2;
    native_config.max_path_time = 1.0;
    native_config.min_transition_hz = 100.0;
    native_config.max_transition_hz = 500.0;
    let native = run_fast_hybrid_with_config(&native_simulation, &native_config)
        .expect("native fast-room result");
    let native_pressure = native.pressures[0][0][0];

    let error = (wasm_pressure - native_pressure).norm();
    let scale = native_pressure.norm().max(1.0e-12);
    assert!(
        error / scale < 1.0e-11,
        "WASM/native directional ISM mismatch: wasm={wasm_pressure:?}, native={native_pressure:?}, relative_error={}",
        error / scale
    );
}

#[test]
fn wasm_constant_impedance_ism_matches_native_fast_room() {
    let source_position = Point3D::new(1.0, 1.2, 1.1);
    let listener_position = Point3D::new(3.4, 2.7, 1.2);
    let frequency = 1_000.0;

    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"] = json!({"type":"rectangular", "width":5.0, "depth":4.0, "height":2.8});
    config["sources"][0]["position"] =
        json!({"x":source_position.x, "y":source_position.y, "z":source_position.z});
    config["sources"]
        .as_array_mut()
        .expect("sources")
        .truncate(1);
    config["listening_positions"] =
        json!([{"x":listener_position.x, "y":listener_position.y, "z":listener_position.z}]);
    config["frequencies"]["min_freq"] = json!(frequency);
    config["frequencies"]["max_freq"] = json!(frequency + 1.0);
    config["frequencies"]["num_points"] = json!(2);
    config["solver"]["method"] = json!("image-source-2");
    config["solver"]["air_absorption"] = json!(false);
    config["solver"]["air_density"] = json!(1.204);
    for surface in ["left", "right", "front", "back", "floor", "ceiling"] {
        config["wall_materials"][surface] = impedance_material();
    }

    let config_json = serde_json::to_string(&config).expect("config JSON");
    let validation: Value =
        serde_json::from_str(&validate_config(&config_json)).expect("validation JSON");
    assert_eq!(
        validation["valid"], true,
        "invalid impedance config: {validation}"
    );
    let wasm = RoomSimulatorWasm::new(&config_json).expect("WASM simulator");
    let point: Value = serde_json::from_str(
        &wasm
            .compute_frequency_point(0)
            .expect("WASM frequency point"),
    )
    .expect("WASM point JSON");
    let wasm_pressure = num_complex::Complex64::new(
        point["pressure_real"].as_f64().expect("WASM real pressure"),
        point["pressure_imag"]
            .as_f64()
            .expect("WASM imaginary pressure"),
    );

    let mut native_simulation = RoomSimulation::with_frequencies(
        RoomGeometry::Rectangular(RectangularRoom::new(5.0, 4.0, 2.8)),
        vec![Source::omnidirectional(source_position, 1.0)],
        vec![listener_position],
        frequency,
        frequency + 1.0,
        2,
    );
    let surface = || SurfaceConfig::Impedance {
        real: 820.0,
        imag: 205.0,
    };
    native_simulation.boundaries = BoundaryConfig {
        floor: surface(),
        ceiling: surface(),
        walls: surface(),
        front_wall: Some(surface()),
        back_wall: Some(surface()),
        left_wall: Some(surface()),
        right_wall: Some(surface()),
    };
    let mut native_config = FastRoomConfig::default();
    native_config.image_order = 2;
    native_config.max_path_time = 1.0;
    native_config.min_transition_hz = 100.0;
    native_config.max_transition_hz = 500.0;
    let native = run_fast_hybrid_with_config(&native_simulation, &native_config)
        .expect("native fast-room result");
    let native_pressure = native.pressures[0][0][0];

    let error = (wasm_pressure - native_pressure).norm();
    let scale = native_pressure.norm().max(1.0e-12);
    assert!(
        error / scale < 1.0e-11,
        "WASM/native impedance ISM mismatch: wasm={wasm_pressure:?}, native={native_pressure:?}, relative_error={}",
        error / scale
    );
}

#[test]
fn wasm_ism_with_fully_absorbing_walls_reduces_to_direct_field() {
    let mut direct_config: Value =
        serde_json::from_str(&create_default_config()).expect("default JSON");
    direct_config["solver"]["method"] = json!("direct");
    direct_config["solver"]["air_absorption"] = json!(true);
    direct_config["frequencies"]["min_freq"] = json!(1_250.0);
    direct_config["frequencies"]["max_freq"] = json!(1_251.0);
    direct_config["frequencies"]["num_points"] = json!(2);
    direct_config["sources"][0]["position"] = json!({"x": 1.4, "y": 1.1, "z": 1.0});
    direct_config["listening_positions"] = json!([{"x": 3.2, "y": 2.7, "z": 1.3}]);
    for surface in ["left", "right", "front", "back", "floor", "ceiling"] {
        direct_config["wall_materials"][surface] = fully_absorbing_material();
    }

    let mut ism_config = direct_config.clone();
    ism_config["solver"]["method"] = json!("image-source-3");

    let direct =
        RoomSimulatorWasm::new(&serde_json::to_string(&direct_config).expect("direct config JSON"))
            .expect("direct simulator")
            .compute_frequency_point(0)
            .expect("direct result JSON");
    let ism = RoomSimulatorWasm::new(&serde_json::to_string(&ism_config).expect("ISM config JSON"))
        .expect("ISM simulator")
        .compute_frequency_point(0)
        .expect("ISM result JSON");

    let direct_pressure = Complex64::new(
        serde_json::from_str::<Value>(&direct).expect("direct JSON")["pressure_real"]
            .as_f64()
            .expect("direct real"),
        serde_json::from_str::<Value>(&direct).expect("direct JSON")["pressure_imag"]
            .as_f64()
            .expect("direct imaginary"),
    );
    let ism_json: Value = serde_json::from_str(&ism).expect("ISM JSON");
    let ism_pressure = Complex64::new(
        ism_json["pressure_real"].as_f64().expect("ISM real"),
        ism_json["pressure_imag"].as_f64().expect("ISM imaginary"),
    );

    assert!(
        (direct_pressure - ism_pressure).norm() < 1.0e-12,
        "direct={direct_pressure:?}, ism={ism_pressure:?}, error={}",
        (direct_pressure - ism_pressure).norm()
    );
}

#[test]
fn canonical_direct_field_matches_golden_fixture() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/canonical_direct_field.json"))
        .expect("canonical fixture JSON");
    let mut config: Value = serde_json::from_str(&create_default_config()).expect("default JSON");
    config["room"] = fixture["room"].clone();
    config["sources"] = json!([{
        "name": "golden-source",
        "position": {"x": fixture["source"][0], "y": fixture["source"][1], "z": fixture["source"][2]},
        "amplitude": 1.0,
        "orientation": {"x": 0.0, "y": 1.0, "z": 0.0},
        "directivity": {"type": "omnidirectional"}
    }]);
    config["listening_positions"] = json!([{
        "x": fixture["listener"][0],
        "y": fixture["listener"][1],
        "z": fixture["listener"][2]
    }]);
    config["frequencies"] = json!({
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
    assert!(
        (result["pressure_real"].as_f64().unwrap()
            - fixture["expected_pressure"]["real"].as_f64().unwrap())
        .abs()
            < 1.0e-14
    );
    assert!(
        (result["pressure_imag"].as_f64().unwrap()
            - fixture["expected_pressure"]["imag"].as_f64().unwrap())
        .abs()
            < 1.0e-14
    );
}
