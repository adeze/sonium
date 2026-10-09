//! WASM environment detection for threading support
//!
//! This module provides utilities to detect browser capabilities
//! and determine the best execution strategy.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Navigator, window};

/// Detect if the browser supports SharedArrayBuffer (required for wasm-bindgen-rayon)
///
/// The browser must be isolated and the loaded WASM module must use shared
/// memory. An atomics-enabled build can still be instantiated with ordinary
/// ArrayBuffer memory by a test runner or host; attempting to start Rayon in
/// that state throws inside JavaScript's Atomics.wait.
#[wasm_bindgen]
pub fn supports_threading() -> bool {
    shared_array_buffer_available() && cross_origin_isolated() && shared_wasm_memory_available()
}

/// Whether the memory backing this exact WASM module is shared.
#[wasm_bindgen]
pub fn shared_wasm_memory_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Reflect::get(&wasm_bindgen::memory(), &JsValue::from_str("buffer"))
            .map(|buffer| buffer.is_instance_of::<js_sys::SharedArrayBuffer>())
            .unwrap_or(false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

/// Whether the JavaScript runtime exposes SharedArrayBuffer.
#[wasm_bindgen]
pub fn shared_array_buffer_available() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("SharedArrayBuffer"))
        .map(|value| !value.is_undefined() && !value.is_null())
        .unwrap_or(false)
}

/// Whether the page opted into the isolation required by wasm-bindgen-rayon.
#[wasm_bindgen]
pub fn cross_origin_isolated() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("crossOriginIsolated"))
        .ok()
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

/// Whether this WASM binary was compiled with the SIMD extension enabled.
#[wasm_bindgen]
pub fn simd_supported() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}

/// Whether the browser exposes a WebGPU adapter entry point.
///
/// The current roomsim kernels remain CPU/WASM implementations; this signal
/// lets a caller choose a future WebGPU preview without treating availability
/// as proof that a WebGPU roomsim backend is active.
#[wasm_bindgen]
pub fn webgpu_supported() -> bool {
    window()
        .map(|window| {
            js_sys::Reflect::get(window.navigator().as_ref(), &JsValue::from_str("gpu"))
                .map(|value| !value.is_null() && !value.is_undefined())
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// Probe whether the browser can actually acquire a WebGPU adapter.
///
/// `webgpu_supported` only checks for the navigator API. Adapter acquisition
/// can still fail because the device lacks a compatible adapter, the browser
/// is configured without GPU access, or the request is denied. This probe does
/// not execute roomsim kernels; the selected roomsim backend remains the CPU
/// implementation until a validated WebGPU backend is available.
#[wasm_bindgen]
pub async fn probe_webgpu_adapter() -> Result<bool, JsValue> {
    let Some(window) = window() else {
        return Ok(false);
    };
    let gpu = js_sys::Reflect::get(window.navigator().as_ref(), &JsValue::from_str("gpu"))?;
    if gpu.is_null() || gpu.is_undefined() {
        return Ok(false);
    }
    let request_adapter = match js_sys::Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))
        .ok()
        .and_then(|value| value.dyn_into::<js_sys::Function>().ok())
    {
        Some(function) => function,
        None => return Ok(false),
    };
    let promise = match request_adapter
        .call0(&gpu)
        .ok()
        .and_then(|value| value.dyn_into::<js_sys::Promise>().ok())
    {
        Some(promise) => promise,
        None => return Ok(false),
    };
    let adapter = match JsFuture::from(promise).await {
        Ok(adapter) => adapter,
        Err(_) => return Ok(false),
    };
    Ok(!adapter.is_null() && !adapter.is_undefined())
}

/// Name of the backend currently executing roomsim calculations.
#[wasm_bindgen]
pub fn selected_execution_backend() -> String {
    if supports_threading() {
        "wasm-cpu-threaded".to_string()
    } else {
        "wasm-cpu-single-threaded".to_string()
    }
}

/// Get the number of hardware threads available
///
/// In browser environment, this uses navigator.hardwareConcurrency
/// Falls back to 1 if unavailable
#[wasm_bindgen]
pub fn num_threads_available() -> usize {
    if !supports_threading() {
        return 1;
    }
    if let Some(window) = web_sys::window() {
        let navigator: Navigator = window.navigator();
        return (navigator.hardware_concurrency() as usize).max(1);
    }
    1
}

/// Get recommended chunk size based on available threads
///
/// Larger chunks = fewer Web Worker round-trips (better for threading)
/// Smaller chunks = more frequent UI updates (better for single-threaded)
#[wasm_bindgen]
pub fn recommended_chunk_size() -> usize {
    if supports_threading() {
        // With threading, use larger chunks for efficiency
        64
    } else {
        // Without threading, use smaller chunks for responsive UI
        16
    }
}

/// Get recommended resolution for spatial slices
///
/// Returns (horizontal_resolution, vertical_resolution) tuple
#[wasm_bindgen]
pub fn recommended_slice_resolution() -> Vec<usize> {
    let threads = num_threads_available();
    if threads >= 4 {
        vec![64, 64] // High quality
    } else if threads >= 2 {
        vec![32, 32] // Medium quality
    } else {
        vec![16, 16] // Low quality for slow devices
    }
}

/// WASM environment info for debugging
#[wasm_bindgen]
pub fn get_wasm_info() -> String {
    let info = serde_json::json!({
        "threading_supported": supports_threading(),
        "shared_array_buffer": shared_array_buffer_available(),
        "shared_wasm_memory": shared_wasm_memory_available(),
        "cross_origin_isolated": cross_origin_isolated(),
        "simd_supported": simd_supported(),
        "webgpu_supported": webgpu_supported(),
        "webgpu_adapter_probe": "async_probe_webgpu_adapter",
        "webgpu_backend_active": false,
        "webgpu_preview_available": webgpu_supported(),
        "webgpu_execution_mode": "cpu-wasm-fallback",
        "execution_backend": selected_execution_backend(),
        "execution_mode": if supports_threading() { "threaded" } else { "single-threaded" },
        "fallback_reason": "webgpu_roomsim_kernels_not_implemented",
        "threads_available": num_threads_available(),
        "recommended_chunk_size": recommended_chunk_size(),
        "slice_resolution": recommended_slice_resolution(),
    });
    info.to_string()
}
