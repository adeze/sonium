//! Produce a deterministic RoomSim gallery without a manual browser session.
//!
//! The JSON exports are the data contract used by the web client. The SVGs are
//! fixed-viewport presentation assets, deliberately accompanied by the exact
//! response and slice arrays that generated them.

use sonium_roomsim::{
    BrowserCaptureContract, BrowserCaptureScreenshot, BrowserCaptureViewport, RoomSimulatorWasm,
    create_default_config,
};
use sonium_xem::io::{GalleryRenderer, validate_gallery_manifest};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: web_gallery <output-dir> [slice-resolution]".to_owned())?;
    // Optional slice grid side length (default 50 preserves smoke output).
    let slice_resolution: usize = match args.next() {
        None => 50,
        Some(raw) => raw
            .parse()
            .map_err(|_| "slice-resolution must be a positive integer".to_string())?,
    };
    if !(8..=1024).contains(&slice_resolution) {
        return Err("slice-resolution must be within 8..=1024".to_string());
    }
    std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let mut config_value: serde_json::Value =
        serde_json::from_str(&create_default_config()).map_err(|e| e.to_string())?;
    config_value["visualization"]["generate_impulse_response"] = serde_json::Value::Bool(true);
    config_value["visualization"]["binaural"]["enabled"] = serde_json::Value::Bool(true);
    config_value["visualization"]["slice_resolution"] =
        serde_json::Value::from(slice_resolution as u64);
    let config = serde_json::to_string_pretty(&config_value).map_err(|e| e.to_string())?;
    let simulator = RoomSimulatorWasm::new(&config).map_err(js_error)?;
    let results = simulator.run_simulation().map_err(js_error)?;
    let slice = simulator
        .compute_horizontal_slice(250.0)
        .map_err(js_error)?;
    std::fs::write(directory.join("scenario.json"), &config).map_err(|e| e.to_string())?;
    std::fs::write(directory.join("results.json"), &results).map_err(|e| e.to_string())?;
    std::fs::write(directory.join("pressure_slice_250hz.json"), &slice)
        .map_err(|e| e.to_string())?;
    // Capture the same progressive contract consumed by the browser.  This
    // stays separate from the full simulation above because chunked mode
    // intentionally returns a displayable direct-field prefix while solving.
    let mut progressive_simulator = RoomSimulatorWasm::new(&config).map_err(js_error)?;
    let first_chunk = progressive_simulator.compute_chunk(1);
    let completed_chunk = progressive_simulator.compute_chunk(10_000);
    let first_progress: serde_json::Value =
        serde_json::from_str(&first_chunk).map_err(|e| e.to_string())?;
    let completed_progress: serde_json::Value =
        serde_json::from_str(&completed_chunk).map_err(|e| e.to_string())?;
    let progressive_state = serde_json::json!({
        "first_chunk": first_progress,
        "completed_chunk": completed_progress,
        "contract": "progress is percentage complete; partial_spl is a displayable prefix; done is the completion signal"
    });
    std::fs::write(
        directory.join("progressive_state.json"),
        serde_json::to_vec_pretty(&progressive_state).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let result_value: serde_json::Value =
        serde_json::from_str(&results).map_err(|e| e.to_string())?;
    let slice_value: serde_json::Value = serde_json::from_str(&slice).map_err(|e| e.to_string())?;
    let response = finite_array(&result_value["frequency_response"])?;
    let frequencies = finite_array(&result_value["frequencies"])?;
    let slice_spl = finite_array(&slice_value["spl"])?;
    let slice_columns = slice_value["shape"]
        .get(1)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0 && slice_spl.len().is_multiple_of(value))
        .ok_or_else(|| "RoomSim slice has no valid rectangular shape".to_owned())?;
    let ir_time = finite_array(&result_value["impulse_response"]["time"])?;
    let ir_amplitude = finite_array(&result_value["impulse_response"]["amplitude"])?;
    let energy_decay = finite_array(&result_value["impulse_response"]["energy_decay"])?;
    let binaural_left_time = finite_array(&result_value["binaural_response"]["left"]["time"])?;
    let binaural_left = finite_array(&result_value["binaural_response"]["left"]["amplitude"])?;
    let binaural_right_time = finite_array(&result_value["binaural_response"]["right"]["time"])?;
    let binaural_right = finite_array(&result_value["binaural_response"]["right"]["amplitude"])?;
    let progressive_x = vec![
        first_progress["frequencies_completed"]
            .as_f64()
            .ok_or_else(|| "chunked RoomSim result lacks frequencies_completed".to_owned())?,
        completed_progress["frequencies_completed"]
            .as_f64()
            .ok_or_else(|| "completed RoomSim chunk lacks frequencies_completed".to_owned())?,
    ];
    let progressive_y = vec![
        first_progress["progress"]
            .as_f64()
            .ok_or_else(|| "chunked RoomSim result lacks progress".to_owned())?,
        completed_progress["progress"]
            .as_f64()
            .ok_or_else(|| "completed RoomSim chunk lacks progress".to_owned())?,
    ];
    write_room_plan_svg(&directory.join("room_plan.svg"), &config_value)?;
    write_room_isometric_svg(&directory.join("room_isometric.svg"), &config_value)?;
    let mut rendered = GalleryRenderer::new(&directory).map_err(|e| e.to_string())?;
    rendered
        .existing_figure(
            "room_isometric.svg",
            "svg",
            "scenario.json",
            "Fixed-camera isometric room view with scenario wall materials, sources, and listener",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .existing_figure(
            "room_plan.svg",
            "svg",
            "scenario.json",
            "Fixed-viewport room plan with source and listener coordinates from the scenario",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .smooth_grid_heatmap_png(
            "pressure_slice_250hz.png",
            &slice_spl,
            slice_columns,
            "pressure_slice_250hz.json",
            format!(
                "SPL field at 250 Hz; row-major {slice_columns}×{slice_columns} grid in dB re 20 uPa"
            ),
        )
        .map_err(|e| e.to_string())?;
    rendered
        .line_svg(
            "progressive_state_plot.svg",
            &progressive_x,
            &progressive_y,
            "RoomSim progressive-state contract",
            "progressive_state.json",
            "Chunked UI progress (%) versus completed frequency points; completion is explicit in the source JSON",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .line_svg(
            "binaural_left_response.svg",
            &binaural_left_time,
            &binaural_left,
            "RoomSim binaural left-ear impulse response",
            "results.json",
            "Left-ear normalized impulse-response amplitude versus time (s)",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .line_svg(
            "binaural_right_response.svg",
            &binaural_right_time,
            &binaural_right,
            "RoomSim binaural right-ear impulse response",
            "results.json",
            "Right-ear normalized impulse-response amplitude versus time (s)",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .line_svg(
            "impulse_response_plot.svg",
            &ir_time,
            &ir_amplitude,
            "RoomSim impulse response",
            "results.json",
            "Normalized impulse-response amplitude versus time (s)",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .line_svg(
            "energy_decay_plot.svg",
            &ir_time,
            &energy_decay,
            "RoomSim energy-decay curve",
            "results.json",
            "Energy decay (dB) versus time (s), derived from the exported impulse response",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .line_svg(
            "frequency_response_plot.svg",
            &frequencies,
            &response,
            "RoomSim frequency response",
            "results.json",
            "SPL (dB re 20 uPa) versus frequency (Hz)",
        )
        .map_err(|e| e.to_string())?;
    rendered
        .finish(
            "RoomSim default-room web gallery",
            vec![
                "fixed 1920x1080 viewport; device scale 1".into(),
                "results.json and pressure_slice_250hz.json are authoritative arrays".into(),
            ],
        )
        .map_err(|e| e.to_string())?;
    let gallery_manifest =
        validate_gallery_manifest(directory.join("gallery.json")).map_err(|e| e.to_string())?;
    let manifest = serde_json::json!({
        "schema": "sonium-roomsim-web-gallery/v1",
        "git_sha": git_sha(), "scenario": "default-room", "frequency_hz": 250.0,
        "viewport": {"width": 1920, "height": 1080}, "device_scale": 1,
        "camera": "fixed isometric presentation camera", "colour_scale": "fixed gallery palette",
        "assets": ["room_isometric.svg", "room_plan.svg", "pressure_slice_250hz.png", "frequency_response_plot.svg", "impulse_response_plot.svg", "energy_decay_plot.svg", "binaural_left_response.svg", "binaural_right_response.svg", "progressive_state_plot.svg"],
        "data": ["scenario.json", "results.json", "pressure_slice_250hz.json", "progressive_state.json"],
        "provenance": "all display arrays are exported directly from RoomSimulatorWasm; inspect results.json and slice JSON before using a visual for numerical claims"
    });
    std::fs::write(
        directory.join("web-gallery-provenance.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let capture_contract = BrowserCaptureContract {
        schema: "sonium-roomsim-browser-capture/v1".into(),
        entrypoint: "index.html".into(),
        viewport: BrowserCaptureViewport {
            width: 1920,
            height: 1080,
        },
        device_scale: 1,
        ready_selector: r#"#integrity[data-status="ok"]"#.into(),
        failure_selector: r#"#integrity[data-status="failed"]"#.into(),
        required_figures: [
            "room_isometric.svg",
            "room_plan.svg",
            "pressure_slice_250hz.png",
            "frequency_response_plot.svg",
            "impulse_response_plot.svg",
            "energy_decay_plot.svg",
            "binaural_left_response.svg",
            "binaural_right_response.svg",
            "progressive_state_plot.svg",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        screenshots: vec![
            BrowserCaptureScreenshot {
                name: "completed".into(),
                wait_for: Some(r#"#integrity[data-status="ok"]"#.into()),
                selector: None,
            },
            BrowserCaptureScreenshot {
                name: "provenance".into(),
                wait_for: None,
                selector: Some("#provenance".into()),
            },
            BrowserCaptureScreenshot {
                name: "figures".into(),
                wait_for: None,
                selector: Some("#figures".into()),
            },
        ],
        note:
            "A browser runner must fail on failure_selector and capture only after ready_selector."
                .into(),
    };
    capture_contract.validate_against_figure_names(
        gallery_manifest
            .figures
            .iter()
            .map(|figure| figure.filename.as_str()),
    )?;
    std::fs::write(
        directory.join("web-gallery-capture.json"),
        capture_contract.to_json()?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        directory.join("index.html"),
        include_str!("../assets/web_gallery.html"),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn js_error(error: wasm_bindgen::JsValue) -> String {
    error
        .as_string()
        .unwrap_or_else(|| "RoomSim JavaScript error".to_owned())
}

fn finite_array(value: &serde_json::Value) -> Result<Vec<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "expected a numeric array in RoomSim output".to_owned())?;
    let values: Vec<f64> = values
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "RoomSim output has a non-numeric sample".to_owned())
        })
        .collect::<Result<_, _>>()?;
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err("RoomSim output has empty or non-finite gallery samples".to_owned());
    }
    Ok(values)
}

fn write_room_plan_svg(path: &std::path::Path, config: &serde_json::Value) -> Result<(), String> {
    let width = config["room"]["width"]
        .as_f64()
        .ok_or_else(|| "default gallery room has no width".to_owned())?;
    let depth = config["room"]["depth"]
        .as_f64()
        .ok_or_else(|| "default gallery room has no depth".to_owned())?;
    let sx = |x: f64| 180.0 + x / width * 1560.0;
    let sy = |y: f64| 900.0 - y / depth * 720.0;
    let sources = config["sources"]
        .as_array()
        .ok_or_else(|| "default gallery has no sources".to_owned())?;
    let mut marks = String::new();
    for source in sources {
        let x = source["position"]["x"]
            .as_f64()
            .ok_or_else(|| "source x is missing".to_owned())?;
        let y = source["position"]["y"]
            .as_f64()
            .ok_or_else(|| "source y is missing".to_owned())?;
        marks.push_str(&format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"18\" fill=\"#f9c74f\"/>",
            sx(x),
            sy(y)
        ));
    }
    let listener = &config["listening_positions"][0];
    let listener_x = listener["x"]
        .as_f64()
        .ok_or_else(|| "listener x is missing".to_owned())?;
    let listener_y = listener["y"]
        .as_f64()
        .ok_or_else(|| "listener y is missing".to_owned())?;
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\"><rect width=\"100%\" height=\"100%\" fill=\"#101820\"/><rect x=\"180\" y=\"180\" width=\"1560\" height=\"720\" fill=\"none\" stroke=\"#5bc0eb\" stroke-width=\"12\"/>{marks}<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"20\" fill=\"#f94144\"/><text x=\"120\" y=\"100\" fill=\"white\" font-size=\"44\">Room plan: yellow = sources, red = listener</text><text x=\"120\" y=\"1030\" fill=\"#ccd6dd\" font-size=\"24\">coordinates scaled directly from scenario.json; room dimensions {:.3} m × {:.3} m</text></svg>",
        sx(listener_x),
        sy(listener_y),
        width,
        depth
    );
    std::fs::write(path, svg).map_err(|e| e.to_string())
}

fn write_room_isometric_svg(
    path: &std::path::Path,
    config: &serde_json::Value,
) -> Result<(), String> {
    let dimension = |name| {
        config["room"][name]
            .as_f64()
            .ok_or_else(|| format!("default gallery room has no {name}"))
    };
    let (width, depth, height) = (
        dimension("width")?,
        dimension("depth")?,
        dimension("height")?,
    );
    let project = |x: f64, y: f64, z: f64| {
        (
            400.0 + x / width * 900.0 - y / depth * 350.0,
            850.0 - x / width * 190.0 - y / depth * 160.0 - z / height * 470.0,
        )
    };
    let polygon = |points: &[(f64, f64)]| {
        points
            .iter()
            .map(|(x, y)| format!("{x:.1},{y:.1}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let p000 = project(0.0, 0.0, 0.0);
    let p100 = project(width, 0.0, 0.0);
    let p010 = project(0.0, depth, 0.0);
    let p001 = project(0.0, 0.0, height);
    let p101 = project(width, 0.0, height);
    let p011 = project(0.0, depth, height);
    let material = |name| {
        config["wall_materials"][name]["name"]
            .as_str()
            .unwrap_or("unspecified")
    };
    let mut markers = String::new();
    for source in config["sources"]
        .as_array()
        .ok_or_else(|| "default gallery has no sources".to_owned())?
    {
        let (x, y, z) = (
            source["position"]["x"]
                .as_f64()
                .ok_or("source x is missing")?,
            source["position"]["y"]
                .as_f64()
                .ok_or("source y is missing")?,
            source["position"]["z"]
                .as_f64()
                .ok_or("source z is missing")?,
        );
        let (px, py) = project(x, y, z);
        markers.push_str(&format!(
            "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"18\" fill=\"#f9c74f\" stroke=\"#101820\" stroke-width=\"5\"/>"
        ));
    }
    let listener = &config["listening_positions"][0];
    let (listener_x, listener_y, listener_z) = (
        listener["x"].as_f64().ok_or("listener x is missing")?,
        listener["y"].as_f64().ok_or("listener y is missing")?,
        listener["z"].as_f64().ok_or("listener z is missing")?,
    );
    let (listener_px, listener_py) = project(listener_x, listener_y, listener_z);
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\"><rect width=\"100%\" height=\"100%\" fill=\"#101820\"/><polygon points=\"{}\" fill=\"#8a6f47\" stroke=\"#dfe7ef\" stroke-width=\"4\"/><polygon points=\"{}\" fill=\"#436b83\" stroke=\"#dfe7ef\" stroke-width=\"4\"/><polygon points=\"{}\" fill=\"#385b70\" stroke=\"#dfe7ef\" stroke-width=\"4\"/>{markers}<circle cx=\"{listener_px:.1}\" cy=\"{listener_py:.1}\" r=\"20\" fill=\"#f94144\" stroke=\"#101820\" stroke-width=\"5\"/><text x=\"100\" y=\"100\" fill=\"white\" font-size=\"44\" font-family=\"sans-serif\">Fixed isometric room view: yellow = sources, red = listener</text><text x=\"100\" y=\"975\" fill=\"#dfe7ef\" font-size=\"28\" font-family=\"sans-serif\">floor: {} · front: {} · left: {} · dimensions: {width:.2} × {depth:.2} × {height:.2} m</text></svg>",
        polygon(&[p000, p100, project(width, depth, 0.0), p010]),
        polygon(&[p000, p010, p011, p001]),
        polygon(&[p000, p100, p101, p001]),
        material("floor"),
        material("front"),
        material("left"),
    );
    std::fs::write(path, svg).map_err(|e| e.to_string())
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned())
}
