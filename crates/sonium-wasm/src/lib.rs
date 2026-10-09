//! WASM bindings for Room Acoustics Simulator
//!
//! This module provides WebAssembly bindings to run room acoustics
//! simulations in the browser. It supports:
//! - Rectangular and L-shaped room geometries
//! - Multiple sources with directivity and crossover filters
//! - Multiple solver methods:
//!   - Direct field computation (free-field propagation)
//!   - Image Source Method (ISM) with 1st, 2nd, 3rd order reflections
//!   - Modal analysis for low frequencies
//!   - Hybrid modal/ISM with Schroeder frequency crossover
//!   - BEM (Boundary Element Method) for accurate wave-based simulation
//! - Frequency response and spatial slice visualization
//! - Impulse response generation
//! - Binaural rendering
//!
//! The BEM solver uses pure Rust linear algebra (no BLAS) for WASM compatibility.

use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sonium_compute::HostMatrix;
use sonium_xem::FrequencyConfig as NativeFrequencyConfig;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use wasm_bindgen::prelude::*;

const MAX_CONFIG_JSON_BYTES: usize = 16 * 1024 * 1024;
const MAX_IMPULSE_INPUT_JSON_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const MAX_FREQUENCY_POINTS: usize = 1_000_000;

/// Machine-adaptive rayon gates: a fixed work-item threshold cannot be right
/// on every machine (pool dispatch cost, worker round-trips in-browser, and
/// per-item ISM/modal evaluation all vary), so every gate resolves through
/// thread count with environment overrides:
///
/// - default: `work_items >= items_per_thread * threads`, where `threads`
///   is the active Rayon pool's worker count. Single-threaded processes
///   (including the deterministic single-threaded browser fallback) always
///   run serially;
/// - `SONIUM_ROOMSIM_PAR_THRESHOLD_SWEEP=<n>` / `SONIUM_ROOMSIM_PAR_THRESHOLD_SLICES=<n>`:
///   absolute work-item overrides for tuning, benches, and forced-path tests;
/// - `SONIUM_SERIAL=1`: process-wide serial escape hatch for deterministic
///   QA runs.
///
/// The per-thread grains below are starting points, not calibrated values.
pub(crate) const SWEEP_ITEMS_PER_THREAD: usize = 4;
pub(crate) const SLICE_CELLS_PER_THREAD: usize = 2048;

/// Machine-adaptive parallelization policy (see above).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParallelPolicy {
    threads: usize,
    force_serial: bool,
    sweep_override: Option<usize>,
    slices_override: Option<usize>,
}

impl ParallelPolicy {
    pub(crate) fn from_env() -> Self {
        Self {
            threads: rayon::current_num_threads(),
            force_serial: std::env::var("SONIUM_SERIAL")
                .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
                .unwrap_or(false),
            sweep_override: std::env::var("SONIUM_ROOMSIM_PAR_THRESHOLD_SWEEP")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
            slices_override: std::env::var("SONIUM_ROOMSIM_PAR_THRESHOLD_SLICES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        }
    }

    fn decide(&self, work_items: usize, items_per_thread: usize, override_: Option<usize>) -> bool {
        if self.force_serial || self.threads < 2 {
            return false;
        }
        if let Some(threshold) = override_ {
            return work_items >= threshold;
        }
        work_items >= items_per_thread.saturating_mul(self.threads).max(1)
    }

    pub(crate) fn parallel_sweep(&self, work_items: usize) -> bool {
        self.decide(work_items, SWEEP_ITEMS_PER_THREAD, self.sweep_override)
    }

    pub(crate) fn parallel_slices(&self, work_items: usize) -> bool {
        self.decide(work_items, SLICE_CELLS_PER_THREAD, self.slices_override)
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_policy_tests {
    use super::{ParallelPolicy, SLICE_CELLS_PER_THREAD, SWEEP_ITEMS_PER_THREAD};
    use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn parallel_policy_tracks_active_browser_pool() {
        let policy = ParallelPolicy::from_env();
        let active = rayon::current_num_threads();
        assert_eq!(policy.threads, active);

        let without_overrides = ParallelPolicy {
            force_serial: false,
            sweep_override: None,
            slices_override: None,
            ..policy
        };
        let sweep_threshold = SWEEP_ITEMS_PER_THREAD.saturating_mul(active);
        let slice_threshold = SLICE_CELLS_PER_THREAD.saturating_mul(active);
        assert!(!without_overrides.parallel_sweep(sweep_threshold.saturating_sub(1)));
        assert!(!without_overrides.parallel_slices(slice_threshold.saturating_sub(1)));
        assert_eq!(
            without_overrides.parallel_sweep(sweep_threshold),
            active >= 2
        );
        assert_eq!(
            without_overrides.parallel_slices(slice_threshold),
            active >= 2
        );
    }

    #[wasm_bindgen_test]
    fn parallel_sweep_path_uses_browser_workers() {
        use rayon::prelude::*;
        use std::collections::HashSet;
        use std::sync::Mutex;

        let active = rayon::current_num_threads();
        // Work far above the parallel grain so the parallel branch is taken
        // whenever the pool has workers.
        let work_items: Vec<usize> = (0..1024).collect();
        let seen = Mutex::new(HashSet::new());
        let sum: usize = work_items
            .par_iter()
            .map(|item| {
                seen.lock()
                    .expect("worker id set")
                    .insert(format!("{:?}", std::thread::current().id()));
                item * 2
            })
            .sum();
        assert_eq!(sum, 1023 * 1024);
        let workers = seen.lock().expect("worker id set").len();
        if active >= 2 {
            assert!(
                workers >= 2,
                "parallel sweep used {workers} worker(s) with {active} in the pool"
            );
        } else {
            // Single-threaded fallback (no cross-origin isolation): the work
            // still completes, on exactly one thread.
            assert_eq!(workers, 1);
        }
    }
}

// Re-export the low-level initializer for callers that need direct control.
pub use wasm_bindgen_rayon::init_thread_pool;

/// Build a `JsValue` error carrying `message` on both targets.
///
/// `JsValue::from_str` routes through wasm ABI type descriptors and aborts
/// with "function not implemented on non-wasm32 targets" under wasm-bindgen
/// 0.2.127, which would make every fallible entry point untestable natively.
/// On wasm32 this is exactly `JsValue::from_str`; elsewhere the message is
/// dropped and `JsValue::NULL` marks the failure so native callers (and
/// tests) still observe `Err`. Prefer this helper over `JsValue::from_str`
/// in all fallible entry points.
fn js_error(message: impl Into<String>) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    {
        JsValue::from_str(&message.into())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = message.into();
        JsValue::NULL
    }
}

/// Initialize the Rayon worker pool when the browser can safely support it.
///
/// Returns `Ok(false)` when the page is not cross-origin isolated (or does not
/// expose `SharedArrayBuffer`). In that case roomsim remains usable through its
/// deterministic single-threaded fallback. A positive thread count is
/// required when initialization is requested; failures from worker creation
/// are returned to JavaScript instead of becoming an implicit panic.
#[wasm_bindgen]
pub async fn init_thread_pool_if_supported(num_threads: usize) -> Result<bool, JsValue> {
    if num_threads == 0 {
        return Err(js_error(
            "thread pool initialization requires at least one thread",
        ));
    }
    if !supports_threading() {
        return Ok(false);
    }

    init_thread_pool(num_threads).await?;
    Ok(true)
}

// BEM solver from sonium-bem (with pure Rust fallbacks for WASM)
mod bem_solver;
mod scattering_objects;

// WASM utilities
mod adaptive;
mod impulse_response;
mod ism;
mod loaders;
#[cfg(feature = "meshlang")]
mod meshlang;
mod serialization;
mod validation;
mod webgpu_preview;
mod worker_detect;

// Computation modules
pub mod acoustics;
pub mod room_modes;

// Re-export BEM types for external use
pub use bem_solver::{BemAssemblyMethod, BemConfig, BemResult, BemSolverMethod, FmmConfig};
pub use scattering_objects::{BoxObject, CylinderObject, ScatteringObjectConfig, SphereObject};

// Re-export WASM utilities
pub use adaptive::{AdaptiveConfig, AdaptiveState, QualityMode, detect_optimal_quality};
pub use impulse_response::{
    ImpulseResponse, ImpulseResponseConfig, ImpulseResponseError, calculate_impulse_response,
};
pub(crate) use impulse_response::{
    MAX_IMPULSE_RESPONSE_SAMPLES, calculate_impulse_response_with_uncertainty,
};
pub use loaders::{
    MeasurementBundle, MeasurementData, interpolate_measurement, parse_measurement_bundle_json,
    parse_measurement_csv,
};
#[cfg(feature = "meshlang")]
pub use meshlang::{
    MESHLANG_FRONTEND_API_NAME, MESHLANG_FRONTEND_API_SCHEMA_VERSION,
    mesh_meshlang_projection_json, meshlang_frontend_contract_json,
    parse_meshlang_cache_stats_json, parse_meshlang_cached_json,
    parse_meshlang_cached_subtree_json, parse_meshlang_json, parse_meshlang_subtree_json,
    preview_meshlang_json, preview_meshlang_with_registry_json,
    preview_meshlang_with_signed_registry_json, preview_meshlang_with_trusted_package_json,
    preview_meshlang_with_trusted_registry_json, validate_meshlang_grammar_json,
};
#[cfg(all(feature = "meshlang", target_arch = "wasm32"))]
pub use meshlang::{
    MeshLangEditor, mesh_meshlang_projection_wasm, meshlang_frontend_contract_wasm,
    parse_meshlang_wasm, preview_meshlang_wasm, preview_meshlang_with_registry_wasm,
    preview_meshlang_with_signed_registry_wasm, preview_meshlang_with_trusted_package_wasm,
    preview_meshlang_with_trusted_registry_wasm, validate_meshlang_grammar_wasm,
};
pub use validation::{
    BROWSER_CAPTURE_CONTRACT_SCHEMA_V1, BrowserCaptureContract, BrowserCaptureScreenshot,
    BrowserCaptureViewport,
};
use validation::{estimated_working_set_bytes, validate_simulation_config};
pub use webgpu_preview::compute_webgpu_preview;
pub use worker_detect::{
    cross_origin_isolated, get_wasm_info, num_threads_available, probe_webgpu_adapter,
    recommended_chunk_size, recommended_slice_resolution, selected_execution_backend,
    shared_array_buffer_available, shared_wasm_memory_available, simd_supported,
    supports_threading, webgpu_supported,
};

// ============================================================================
// Wall Materials and Absorption Coefficients
// ============================================================================

/// Standard octave band center frequencies for absorption coefficients (Hz)
pub const ABSORPTION_FREQUENCIES: [f64; 6] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];

fn default_reflection_phase() -> [f64; 6] {
    [0.0; 6]
}

fn interpolate_log_frequency(values: &[f64; 6], frequency: f64) -> f64 {
    if frequency <= ABSORPTION_FREQUENCIES[0] {
        return values[0];
    }
    if frequency >= ABSORPTION_FREQUENCIES[5] {
        return values[5];
    }
    for i in 0..5 {
        if frequency >= ABSORPTION_FREQUENCIES[i] && frequency < ABSORPTION_FREQUENCIES[i + 1] {
            let log_f = frequency.ln();
            let log_f1 = ABSORPTION_FREQUENCIES[i].ln();
            let log_f2 = ABSORPTION_FREQUENCIES[i + 1].ln();
            let t = (log_f - log_f1) / (log_f2 - log_f1);
            return values[i] * (1.0 - t) + values[i + 1] * t;
        }
    }
    values[5]
}

/// Six-band complex surface impedance in acoustic Pascal-second/meter units.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImpedanceBands {
    pub real: [f64; 6],
    pub imag: [f64; 6],
}

/// Wall material with frequency-dependent absorption coefficients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallMaterial {
    pub name: String,
    /// Absorption coefficients at 125, 250, 500, 1000, 2000, 4000 Hz
    pub absorption: [f64; 6],
    /// Reflection phase in radians at the same octave-band frequencies.
    #[serde(default = "default_reflection_phase")]
    pub reflection_phase: [f64; 6],
    /// Optional complex surface impedance; when present it supersedes phase.
    #[serde(default)]
    pub impedance: Option<ImpedanceBands>,
}

impl WallMaterial {
    /// Create a new wall material with given absorption coefficients
    pub fn new(name: &str, absorption: [f64; 6]) -> Self {
        Self::new_with_phase(name, absorption, [0.0; 6])
    }

    /// Create a material with absorption and frequency-dependent reflection phase.
    pub fn new_with_phase(name: &str, absorption: [f64; 6], reflection_phase: [f64; 6]) -> Self {
        Self {
            name: name.to_string(),
            absorption,
            reflection_phase,
            impedance: None,
        }
    }

    /// Create a material using a frequency-dependent complex impedance.
    pub fn new_with_impedance(
        name: &str,
        absorption: [f64; 6],
        reflection_phase: [f64; 6],
        impedance: ImpedanceBands,
    ) -> Self {
        Self {
            name: name.to_string(),
            absorption,
            reflection_phase,
            impedance: Some(impedance),
        }
    }

    /// Get absorption coefficient at a given frequency (interpolated)
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        interpolate_log_frequency(&self.absorption, frequency)
    }

    /// Convert absorption coefficient to reflection coefficient: R = sqrt(1 - α)
    pub fn reflection_at_frequency(&self, frequency: f64) -> f64 {
        self.reflection_at_frequency_complex(frequency).norm()
    }

    /// Complex pressure reflection coefficient, including configured phase.
    pub fn reflection_at_frequency_complex(&self, frequency: f64) -> Complex64 {
        self.reflection_at_frequency_complex_with_reference_impedance(frequency, 1.2 * 343.0)
    }

    /// Complex reflection coefficient using the caller's ambient impedance.
    pub fn reflection_at_frequency_complex_with_reference_impedance(
        &self,
        frequency: f64,
        reference_impedance: f64,
    ) -> Complex64 {
        if let Some(impedance) = &self.impedance {
            let real = interpolate_log_frequency(&impedance.real, frequency);
            let imag = interpolate_log_frequency(&impedance.imag, frequency);
            let z = Complex64::new(real, imag);
            let z0 = reference_impedance.max(f64::MIN_POSITIVE);
            return (z - Complex64::new(z0, 0.0)) / (z + Complex64::new(z0, 0.0));
        }
        // α = 1 is a valid fully absorbing boundary and must produce zero
        // pressure reflection. Keep the logarithm guard in RT60 calculations
        // separate; reflection conversion itself has no singularity at 1.
        let alpha = self.absorption_at_frequency(frequency).clamp(0.0, 1.0);
        let phase = interpolate_log_frequency(&self.reflection_phase, frequency);
        Complex64::from_polar((1.0 - alpha).sqrt(), phase)
    }

    // ========== Material Presets ==========

    /// Concrete or brick (painted)
    pub fn concrete() -> Self {
        Self::new("Concrete", [0.01, 0.01, 0.02, 0.02, 0.02, 0.03])
    }

    /// Unpainted brick
    pub fn brick() -> Self {
        Self::new("Brick", [0.03, 0.03, 0.03, 0.04, 0.05, 0.07])
    }

    /// Gypsum board (drywall) on studs
    pub fn drywall() -> Self {
        Self::new("Drywall", [0.29, 0.10, 0.05, 0.04, 0.07, 0.09])
    }

    /// Plaster on brick
    pub fn plaster() -> Self {
        Self::new("Plaster", [0.01, 0.02, 0.02, 0.03, 0.04, 0.05])
    }

    /// Glass (large pane)
    pub fn glass() -> Self {
        Self::new("Glass", [0.18, 0.06, 0.04, 0.03, 0.02, 0.02])
    }

    /// Wood paneling (thin)
    pub fn wood_thin() -> Self {
        Self::new("Wood (thin)", [0.42, 0.21, 0.10, 0.08, 0.06, 0.06])
    }

    /// Heavy wooden door
    pub fn wood_thick() -> Self {
        Self::new("Wood (thick)", [0.14, 0.10, 0.06, 0.08, 0.10, 0.10])
    }

    /// Carpet on concrete
    pub fn carpet_thin() -> Self {
        Self::new("Carpet (thin)", [0.02, 0.06, 0.14, 0.37, 0.60, 0.65])
    }

    /// Heavy carpet on underlay
    pub fn carpet_thick() -> Self {
        Self::new("Carpet (thick)", [0.08, 0.24, 0.57, 0.69, 0.71, 0.73])
    }

    /// Acoustic ceiling tiles
    pub fn acoustic_tile() -> Self {
        Self::new("Acoustic Tile", [0.50, 0.70, 0.60, 0.70, 0.70, 0.50])
    }

    /// Curtains (medium weight, draped)
    pub fn curtains() -> Self {
        Self::new("Curtains", [0.07, 0.31, 0.49, 0.75, 0.70, 0.60])
    }

    /// Acoustic foam panels
    pub fn acoustic_foam() -> Self {
        Self::new("Acoustic Foam", [0.08, 0.25, 0.60, 0.90, 0.95, 0.90])
    }

    /// Hardwood floor
    pub fn hardwood() -> Self {
        Self::new("Hardwood", [0.15, 0.11, 0.10, 0.07, 0.06, 0.07])
    }

    /// Concrete floor (sealed)
    pub fn concrete_floor() -> Self {
        Self::new("Concrete Floor", [0.01, 0.01, 0.02, 0.02, 0.02, 0.02])
    }

    /// Default material (moderate absorption, like plaster walls)
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::plaster()
    }
}

/// Wall surfaces in a rectangular room
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallSurface {
    /// Left wall (x = 0)
    Left,
    /// Right wall (x = width)
    Right,
    /// Front wall (y = 0)
    Front,
    /// Back wall (y = depth)
    Back,
    /// Floor (z = 0)
    Floor,
    /// Ceiling (z = height)
    Ceiling,
}

impl WallSurface {
    pub fn all() -> [WallSurface; 6] {
        [
            WallSurface::Left,
            WallSurface::Right,
            WallSurface::Front,
            WallSurface::Back,
            WallSurface::Floor,
            WallSurface::Ceiling,
        ]
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! console_log {
    ($($t:tt)*) => {{
        let _ = format_args!($($t)*);
    }}
}

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();

        // Log threading status
        if supports_threading() {
            console_log!(
                "RoomSimulator WASM initialized: {} threads available, {} freq points/chunk",
                num_threads_available(),
                recommended_chunk_size()
            );
        } else {
            console_log!(
                "WASM: Running in single-threaded mode - {} threads, chunked computation",
                num_threads_available()
            );
        }
    }
}

// ============================================================================
// Room Geometry Types (local implementation for WASM)
// ============================================================================

/// 3D point in space
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Rectangular room
pub struct RectangularRoom {
    pub width: f64,
    pub depth: f64,
    pub height: f64,
}

impl RectangularRoom {
    pub fn new(width: f64, depth: f64, height: f64) -> Self {
        Self {
            width,
            depth,
            height,
        }
    }

    pub fn get_edges(&self) -> Vec<(Point3D, Point3D)> {
        vec![
            // Floor edges
            (
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(self.width, 0.0, 0.0),
            ),
            (
                Point3D::new(self.width, 0.0, 0.0),
                Point3D::new(self.width, self.depth, 0.0),
            ),
            (
                Point3D::new(self.width, self.depth, 0.0),
                Point3D::new(0.0, self.depth, 0.0),
            ),
            (
                Point3D::new(0.0, self.depth, 0.0),
                Point3D::new(0.0, 0.0, 0.0),
            ),
            // Ceiling edges
            (
                Point3D::new(0.0, 0.0, self.height),
                Point3D::new(self.width, 0.0, self.height),
            ),
            (
                Point3D::new(self.width, 0.0, self.height),
                Point3D::new(self.width, self.depth, self.height),
            ),
            (
                Point3D::new(self.width, self.depth, self.height),
                Point3D::new(0.0, self.depth, self.height),
            ),
            (
                Point3D::new(0.0, self.depth, self.height),
                Point3D::new(0.0, 0.0, self.height),
            ),
            // Vertical edges
            (
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(0.0, 0.0, self.height),
            ),
            (
                Point3D::new(self.width, 0.0, 0.0),
                Point3D::new(self.width, 0.0, self.height),
            ),
            (
                Point3D::new(self.width, self.depth, 0.0),
                Point3D::new(self.width, self.depth, self.height),
            ),
            (
                Point3D::new(0.0, self.depth, 0.0),
                Point3D::new(0.0, self.depth, self.height),
            ),
        ]
    }
}

/// L-shaped room
pub struct LShapedRoom {
    pub width1: f64,
    pub depth1: f64,
    pub width2: f64,
    pub depth2: f64,
    pub height: f64,
}

impl LShapedRoom {
    pub fn new(width1: f64, depth1: f64, width2: f64, depth2: f64, height: f64) -> Self {
        Self {
            width1,
            depth1,
            width2,
            depth2,
            height,
        }
    }

    /// Check if a 2D point (x, y) is inside the L-shaped room footprint
    /// The L-shape consists of two rectangular regions:
    /// - Region 1: x in [0, width1], y in [0, depth1]
    /// - Region 2: x in [0, width2], y in [depth1, depth1 + depth2]
    pub fn contains_xy(&self, x: f64, y: f64) -> bool {
        let total_depth = self.depth1 + self.depth2;

        // Check region 1 (front section, wider)
        if x >= 0.0 && x <= self.width1 && y >= 0.0 && y <= self.depth1 {
            return true;
        }

        // Check region 2 (back section, narrower)
        if x >= 0.0 && x <= self.width2 && y >= self.depth1 && y <= total_depth {
            return true;
        }

        false
    }

    /// Check if a 3D point is inside the L-shaped room
    pub fn contains(&self, p: &Point3D) -> bool {
        self.contains_xy(p.x, p.y) && p.z >= 0.0 && p.z <= self.height
    }

    /// Validate if an image source produces a valid reflection path
    /// The reflection path from source to listener via the image source must:
    /// 1. Cross the intended wall
    /// 2. Not pass through any internal walls or outside the room
    pub fn is_valid_image_source(
        &self,
        image: &Point3D,
        listener: &Point3D,
        _source: &Point3D,
    ) -> bool {
        ism::is_valid_l_shaped_image_source(self, image, listener)
    }

    /// Generate first-order image sources for L-shaped room
    /// L-shaped room has 8 walls: 6 exterior + 2 interior (at the step)
    ///
    /// Walls:
    /// - Left: x = 0 (full depth)
    /// - Right (section 1): x = width1, y in [0, depth1]
    /// - Right (section 2): x = width2, y in [depth1, total_depth]
    /// - Front: y = 0, x in [0, width1]
    /// - Back: y = total_depth, x in [0, width2]
    /// - Floor: z = 0
    /// - Ceiling: z = height
    /// - Interior wall 1 (step): x in [width2, width1], y = depth1
    /// - Interior wall 2 (step): y in [depth1, depth1], x = width2
    ///
    /// Returns: Vec of (image_position, wall_name)
    pub fn get_first_order_images(&self, source: &Point3D) -> Vec<(Point3D, &'static str)> {
        ism::l_shaped_first_order_images(self, source)
    }

    pub fn get_edges(&self) -> Vec<(Point3D, Point3D)> {
        let total_depth = self.depth1 + self.depth2;
        vec![
            // Floor edges - Main section
            (
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(self.width1, 0.0, 0.0),
            ),
            (
                Point3D::new(self.width1, 0.0, 0.0),
                Point3D::new(self.width1, self.depth1, 0.0),
            ),
            (
                Point3D::new(self.width1, self.depth1, 0.0),
                Point3D::new(self.width2, self.depth1, 0.0),
            ),
            (
                Point3D::new(self.width2, self.depth1, 0.0),
                Point3D::new(self.width2, total_depth, 0.0),
            ),
            (
                Point3D::new(self.width2, total_depth, 0.0),
                Point3D::new(0.0, total_depth, 0.0),
            ),
            (
                Point3D::new(0.0, total_depth, 0.0),
                Point3D::new(0.0, 0.0, 0.0),
            ),
            // Ceiling edges
            (
                Point3D::new(0.0, 0.0, self.height),
                Point3D::new(self.width1, 0.0, self.height),
            ),
            (
                Point3D::new(self.width1, 0.0, self.height),
                Point3D::new(self.width1, self.depth1, self.height),
            ),
            (
                Point3D::new(self.width1, self.depth1, self.height),
                Point3D::new(self.width2, self.depth1, self.height),
            ),
            (
                Point3D::new(self.width2, self.depth1, self.height),
                Point3D::new(self.width2, total_depth, self.height),
            ),
            (
                Point3D::new(self.width2, total_depth, self.height),
                Point3D::new(0.0, total_depth, self.height),
            ),
            (
                Point3D::new(0.0, total_depth, self.height),
                Point3D::new(0.0, 0.0, self.height),
            ),
            // Vertical edges
            (
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(0.0, 0.0, self.height),
            ),
            (
                Point3D::new(self.width1, 0.0, 0.0),
                Point3D::new(self.width1, 0.0, self.height),
            ),
            (
                Point3D::new(self.width1, self.depth1, 0.0),
                Point3D::new(self.width1, self.depth1, self.height),
            ),
            (
                Point3D::new(self.width2, self.depth1, 0.0),
                Point3D::new(self.width2, self.depth1, self.height),
            ),
            (
                Point3D::new(self.width2, total_depth, 0.0),
                Point3D::new(self.width2, total_depth, self.height),
            ),
            (
                Point3D::new(0.0, total_depth, 0.0),
                Point3D::new(0.0, total_depth, self.height),
            ),
        ]
    }
}

/// Room geometry enum
pub enum RoomGeometry {
    Rectangular(RectangularRoom),
    LShaped(LShapedRoom),
}

impl RoomGeometry {
    pub fn get_edges(&self) -> Vec<(Point3D, Point3D)> {
        match self {
            RoomGeometry::Rectangular(r) => r.get_edges(),
            RoomGeometry::LShaped(r) => r.get_edges(),
        }
    }
}

/// Crossover filter
pub enum CrossoverFilter {
    FullRange,
    Lowpass {
        cutoff_freq: f64,
        order: u32,
    },
    Highpass {
        cutoff_freq: f64,
        order: u32,
    },
    Bandpass {
        low_cutoff: f64,
        high_cutoff: f64,
        order: u32,
    },
}

impl CrossoverFilter {
    pub fn amplitude_at_frequency(&self, frequency: f64) -> f64 {
        match self {
            CrossoverFilter::FullRange => 1.0,
            CrossoverFilter::Lowpass { cutoff_freq, order } => {
                let ratio = frequency / cutoff_freq;
                1.0 / (1.0 + ratio.powi(*order as i32 * 2)).sqrt()
            }
            CrossoverFilter::Highpass { cutoff_freq, order } => {
                let ratio = cutoff_freq / frequency;
                1.0 / (1.0 + ratio.powi(*order as i32 * 2)).sqrt()
            }
            CrossoverFilter::Bandpass {
                low_cutoff,
                high_cutoff,
                order,
            } => {
                let hp = 1.0 / (1.0 + (low_cutoff / frequency).powi(*order as i32 * 2)).sqrt();
                let lp = 1.0 / (1.0 + (frequency / high_cutoff).powi(*order as i32 * 2)).sqrt();
                hp * lp
            }
        }
    }
}

/// Directivity pattern
pub struct DirectivityPattern {
    pub horizontal_angles: Vec<f64>,
    pub vertical_angles: Vec<f64>,
    pub magnitude: HostMatrix<f64>,
}

impl DirectivityPattern {
    pub fn omnidirectional() -> Self {
        let horizontal_angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let vertical_angles: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();
        let magnitude = HostMatrix::from_elem(vertical_angles.len(), horizontal_angles.len(), 1.0);

        Self {
            horizontal_angles,
            vertical_angles,
            magnitude,
        }
    }

    pub fn interpolate(&self, theta: f64, phi: f64) -> f64 {
        if self.horizontal_angles.is_empty() || self.vertical_angles.is_empty() {
            // An empty pattern can only arise through direct construction of
            // this public Rust type; validated JS configuration always
            // produces populated axes. Treat it as an absent directivity
            // pattern instead of allowing `len() - 1` to panic.
            return 0.0;
        }
        let theta_deg = theta.to_degrees();
        let mut phi_deg = phi.to_degrees();

        while phi_deg < 0.0 {
            phi_deg += 360.0;
        }
        while phi_deg >= 360.0 {
            phi_deg -= 360.0;
        }

        let h_idx = (phi_deg / 10.0).floor() as usize;
        let v_idx = (theta_deg / 10.0).floor() as usize;

        let h_idx = h_idx.min(self.horizontal_angles.len() - 1);
        let v_idx = v_idx.min(self.vertical_angles.len() - 1);

        let h_next = (h_idx + 1) % self.horizontal_angles.len();
        let v_next = (v_idx + 1).min(self.vertical_angles.len() - 1);

        let h_frac = (phi_deg / 10.0) - h_idx as f64;
        let v_frac = (theta_deg / 10.0) - v_idx as f64;

        let m00 = self.magnitude[[v_idx, h_idx]];
        let m01 = self.magnitude[[v_idx, h_next]];
        let m10 = self.magnitude[[v_next, h_idx]];
        let m11 = self.magnitude[[v_next, h_next]];

        let m0 = m00 * (1.0 - h_frac) + m01 * h_frac;
        let m1 = m10 * (1.0 - h_frac) + m11 * h_frac;

        m0 * (1.0 - v_frac) + m1 * v_frac
    }
}

/// Sound source
pub struct Source {
    pub position: Point3D,
    /// Forward axis (acoustic axis). Need not be unit length; normalized internally.
    pub orientation: Point3D,
    pub directivity: DirectivityPattern,
    pub amplitude: f64,
    pub crossover: CrossoverFilter,
    pub name: String,
    /// Time delay in seconds
    pub delay_sec: f64,
    /// Phase inversion flag (true = 180 degree phase shift)
    pub invert_phase: bool,
}

impl Source {
    pub fn new(position: Point3D, directivity: DirectivityPattern, amplitude: f64) -> Self {
        Self {
            position,
            orientation: Point3D::new(0.0, 1.0, 0.0),
            directivity,
            amplitude,
            crossover: CrossoverFilter::FullRange,
            name: String::from("Source"),
            delay_sec: 0.0,
            invert_phase: false,
        }
    }

    pub fn with_orientation(mut self, orientation: Point3D) -> Self {
        self.orientation = orientation;
        self
    }

    pub fn with_crossover(mut self, crossover: CrossoverFilter) -> Self {
        self.crossover = crossover;
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn with_delay_ms(mut self, delay_ms: f64) -> Self {
        self.delay_sec = delay_ms / 1000.0;
        self
    }

    pub fn with_phase_inversion(mut self, invert: bool) -> Self {
        self.invert_phase = invert;
        self
    }

    /// Get the phase factor for this source at a given frequency
    /// Includes both time delay and phase inversion
    pub fn phase_factor(&self, frequency: f64) -> Complex64 {
        let omega = 2.0 * PI * frequency;
        // Phase shift from delay: e^(-i * omega * delay)
        let delay_phase = Complex64::new(0.0, -omega * self.delay_sec).exp();
        // Phase inversion: multiply by -1
        if self.invert_phase {
            -delay_phase
        } else {
            delay_phase
        }
    }

    pub fn amplitude_towards(&self, point: &Point3D, frequency: f64) -> f64 {
        let dx = point.x - self.position.x;
        let dy = point.y - self.position.y;
        let dz = point.z - self.position.z;

        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 1e-10 {
            return self.amplitude * self.crossover.amplitude_at_frequency(frequency);
        }

        // Directivity patterns use a local frame with +Y as the acoustic axis,
        // +Z as up, and +X to the right.  Keep the legacy default orientation
        // (+Y) exactly unchanged while rotating arbitrary source directions
        // into that frame before interpolation.
        let orientation_norm = (self.orientation.x * self.orientation.x
            + self.orientation.y * self.orientation.y
            + self.orientation.z * self.orientation.z)
            .sqrt();
        let forward = if orientation_norm.is_finite() && orientation_norm > 1e-12 {
            [
                self.orientation.x / orientation_norm,
                self.orientation.y / orientation_norm,
                self.orientation.z / orientation_norm,
            ]
        } else {
            [0.0, 1.0, 0.0]
        };
        let world_up = [0.0, 0.0, 1.0];
        let up_projection = [
            world_up[0] - forward[0] * forward[2],
            world_up[1] - forward[1] * forward[2],
            world_up[2] - forward[2] * forward[2],
        ];
        let up_norm = (up_projection[0] * up_projection[0]
            + up_projection[1] * up_projection[1]
            + up_projection[2] * up_projection[2])
            .sqrt();
        let up = if up_norm > 1e-12 {
            [
                up_projection[0] / up_norm,
                up_projection[1] / up_norm,
                up_projection[2] / up_norm,
            ]
        } else {
            // A source pointing vertically has no unique world-up projection.
            // Use world +X as the stable fallback reference axis.
            let reference = [1.0, 0.0, 0.0];
            let projection = [
                reference[0] - forward[0] * forward[0],
                reference[1] - forward[1] * forward[0],
                reference[2] - forward[2] * forward[0],
            ];
            let norm = (projection[0] * projection[0]
                + projection[1] * projection[1]
                + projection[2] * projection[2])
                .sqrt();
            [
                projection[0] / norm,
                projection[1] / norm,
                projection[2] / norm,
            ]
        };
        let right = [
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        ];
        let local_x = dx * right[0] + dy * right[1] + dz * right[2];
        let local_y = dx * forward[0] + dy * forward[1] + dz * forward[2];
        let local_z = dx * up[0] + dy * up[1] + dz * up[2];
        let theta = (local_z / r).clamp(-1.0, 1.0).acos();
        let phi = local_y.atan2(local_x);

        let directivity_factor = self.directivity.interpolate(theta, phi);
        let crossover_factor = self.crossover.amplitude_at_frequency(frequency);
        self.amplitude * directivity_factor * crossover_factor
    }
}

// ============================================================================
// Configuration types (JSON-serializable)
// ============================================================================

/// Room geometry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoomGeometryConfig {
    #[serde(rename = "rectangular")]
    Rectangular { width: f64, depth: f64, height: f64 },
    #[serde(rename = "lshaped")]
    LShaped {
        width1: f64,
        depth1: f64,
        width2: f64,
        depth2: f64,
        height: f64,
    },
}

/// 3D point configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3DConfig {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<Point3DConfig> for Point3D {
    fn from(p: Point3DConfig) -> Self {
        Point3D::new(p.x, p.y, p.z)
    }
}

impl From<Point3DConfig> for sonium_bem::room_acoustics::Point3D {
    fn from(p: Point3DConfig) -> Self {
        sonium_bem::room_acoustics::Point3D::new(p.x, p.y, p.z)
    }
}

/// Crossover filter configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CrossoverConfig {
    #[serde(rename = "fullrange")]
    #[default]
    FullRange,
    #[serde(rename = "lowpass")]
    Lowpass { cutoff_freq: f64, order: u32 },
    #[serde(rename = "highpass")]
    Highpass { cutoff_freq: f64, order: u32 },
    #[serde(rename = "bandpass")]
    Bandpass {
        low_cutoff: f64,
        high_cutoff: f64,
        order: u32,
    },
}

impl CrossoverConfig {
    fn to_filter(&self) -> CrossoverFilter {
        match self {
            CrossoverConfig::FullRange => CrossoverFilter::FullRange,
            CrossoverConfig::Lowpass { cutoff_freq, order } => CrossoverFilter::Lowpass {
                cutoff_freq: *cutoff_freq,
                order: *order,
            },
            CrossoverConfig::Highpass { cutoff_freq, order } => CrossoverFilter::Highpass {
                cutoff_freq: *cutoff_freq,
                order: *order,
            },
            CrossoverConfig::Bandpass {
                low_cutoff,
                high_cutoff,
                order,
            } => CrossoverFilter::Bandpass {
                low_cutoff: *low_cutoff,
                high_cutoff: *high_cutoff,
                order: *order,
            },
        }
    }
}

/// Directivity pattern configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DirectivityConfig {
    #[serde(rename = "omnidirectional")]
    #[default]
    Omnidirectional,
    #[serde(rename = "cardioid")]
    Cardioid { front_back_ratio: f64 },
    /// Spinorama directivity data with horizontal and vertical SPL curves at different angles
    #[serde(rename = "spinorama")]
    Spinorama {
        /// Horizontal SPL data at different angles
        horizontal: Vec<SpinoramaCurve>,
        /// Vertical SPL data at different angles
        vertical: Vec<SpinoramaCurve>,
    },
}

/// A single SPL curve at a specific angle (for spinorama directivity data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpinoramaCurve {
    /// Angle in degrees (e.g., -60, -50, ..., 0, ..., 50, 60)
    pub angle: f64,
    /// Frequency points in Hz
    pub freq: Vec<f64>,
    /// SPL values in dB
    pub spl: Vec<f64>,
}

/// Wall material configuration (JSON-serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WallMaterialConfig {
    /// Use a preset material by name
    #[serde(rename = "preset")]
    Preset { name: String },
    /// Custom absorption coefficients at 125, 250, 500, 1000, 2000, 4000 Hz
    #[serde(rename = "custom")]
    Custom {
        name: String,
        #[serde(default = "default_absorption")]
        absorption: [f64; 6],
        /// Complex pressure-reflection phase in radians at the octave bands.
        #[serde(default = "default_reflection_phase")]
        reflection_phase: [f64; 6],
        /// Optional six-band complex surface impedance.
        #[serde(default)]
        impedance: Option<ImpedanceBands>,
    },
}

fn default_absorption() -> [f64; 6] {
    [0.02, 0.02, 0.03, 0.03, 0.04, 0.05] // Default plaster-like
}

impl Default for WallMaterialConfig {
    fn default() -> Self {
        WallMaterialConfig::Preset {
            name: "plaster".to_string(),
        }
    }
}

impl WallMaterialConfig {
    /// True when this surface carries an explicit complex-impedance band
    /// table. The room-acoustics BEM path uses its native default boundary
    /// model and does not apply it (see validation).
    pub fn has_impedance(&self) -> bool {
        matches!(
            self,
            WallMaterialConfig::Custom {
                impedance: Some(_),
                ..
            }
        )
    }

    /// Convert config to WallMaterial
    pub fn to_material(&self) -> WallMaterial {
        match self {
            WallMaterialConfig::Preset { name } => {
                match name.to_lowercase().as_str() {
                    "concrete" => WallMaterial::concrete(),
                    "brick" => WallMaterial::brick(),
                    "drywall" | "gypsum" => WallMaterial::drywall(),
                    "plaster" => WallMaterial::plaster(),
                    "glass" => WallMaterial::glass(),
                    "wood_thin" | "wood-thin" | "thin_wood" => WallMaterial::wood_thin(),
                    "wood_thick" | "wood-thick" | "thick_wood" | "wood" => {
                        WallMaterial::wood_thick()
                    }
                    "carpet_thin" | "carpet-thin" | "thin_carpet" => WallMaterial::carpet_thin(),
                    "carpet_thick" | "carpet-thick" | "thick_carpet" | "carpet" => {
                        WallMaterial::carpet_thick()
                    }
                    "acoustic_tile" | "acoustic-tile" | "ceiling_tile" => {
                        WallMaterial::acoustic_tile()
                    }
                    "curtains" | "drapes" => WallMaterial::curtains(),
                    "acoustic_foam" | "acoustic-foam" | "foam" => WallMaterial::acoustic_foam(),
                    "hardwood" | "wood_floor" | "wood-floor" => WallMaterial::hardwood(),
                    "concrete_floor" | "concrete-floor" => WallMaterial::concrete_floor(),
                    _ => WallMaterial::plaster(), // Default fallback
                }
            }
            WallMaterialConfig::Custom {
                name,
                absorption,
                reflection_phase,
                impedance,
            } => match impedance {
                Some(impedance) => WallMaterial::new_with_impedance(
                    name,
                    *absorption,
                    *reflection_phase,
                    impedance.clone(),
                ),
                None => WallMaterial::new_with_phase(name, *absorption, *reflection_phase),
            },
        }
    }

    /// Get absorption coefficient at a given frequency
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        self.to_material().absorption_at_frequency(frequency)
    }
}

/// Wall materials configuration for all 6 surfaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallMaterialsConfig {
    /// Left wall (x = 0)
    #[serde(default)]
    pub left: WallMaterialConfig,
    /// Right wall (x = width)
    #[serde(default)]
    pub right: WallMaterialConfig,
    /// Front wall (y = 0)
    #[serde(default)]
    pub front: WallMaterialConfig,
    /// Back wall (y = depth)
    #[serde(default)]
    pub back: WallMaterialConfig,
    /// Floor (z = 0)
    #[serde(default = "default_floor_material")]
    pub floor: WallMaterialConfig,
    /// Ceiling (z = height)
    #[serde(default = "default_ceiling_material")]
    pub ceiling: WallMaterialConfig,
}

fn default_floor_material() -> WallMaterialConfig {
    WallMaterialConfig::Preset {
        name: "hardwood".to_string(),
    }
}

fn default_ceiling_material() -> WallMaterialConfig {
    WallMaterialConfig::Preset {
        name: "plaster".to_string(),
    }
}

impl Default for WallMaterialsConfig {
    fn default() -> Self {
        Self {
            left: WallMaterialConfig::default(),
            right: WallMaterialConfig::default(),
            front: WallMaterialConfig::default(),
            back: WallMaterialConfig::default(),
            floor: default_floor_material(),
            ceiling: default_ceiling_material(),
        }
    }
}

impl WallMaterialsConfig {
    /// Get the material for a specific wall surface
    pub fn get_material(&self, surface: WallSurface) -> WallMaterial {
        match surface {
            WallSurface::Left => self.left.to_material(),
            WallSurface::Right => self.right.to_material(),
            WallSurface::Front => self.front.to_material(),
            WallSurface::Back => self.back.to_material(),
            WallSurface::Floor => self.floor.to_material(),
            WallSurface::Ceiling => self.ceiling.to_material(),
        }
    }

    /// Get reflection coefficient for a wall at a given frequency
    pub fn reflection_at(&self, surface: WallSurface, frequency: f64) -> f64 {
        self.get_material(surface)
            .reflection_at_frequency(frequency)
    }

    /// Get the complex pressure-reflection coefficient for a wall.
    pub fn reflection_at_complex(&self, surface: WallSurface, frequency: f64) -> Complex64 {
        self.get_material(surface)
            .reflection_at_frequency_complex(frequency)
    }

    /// Get complex reflection using the configured speed of sound and air density.
    pub fn reflection_at_complex_with_speed(
        &self,
        surface: WallSurface,
        frequency: f64,
        speed_of_sound: f64,
    ) -> Complex64 {
        self.reflection_at_complex_with_medium(surface, frequency, speed_of_sound, 1.2)
    }

    /// Get complex reflection using the configured ambient medium.
    pub fn reflection_at_complex_with_medium(
        &self,
        surface: WallSurface,
        frequency: f64,
        speed_of_sound: f64,
        air_density: f64,
    ) -> Complex64 {
        self.get_material(surface)
            .reflection_at_frequency_complex_with_reference_impedance(
                frequency,
                air_density * speed_of_sound,
            )
    }

    /// Get average absorption coefficient across all walls at a given frequency
    pub fn average_absorption_at(&self, frequency: f64) -> f64 {
        let surfaces = WallSurface::all();
        let total: f64 = surfaces
            .iter()
            .map(|&s| self.get_material(s).absorption_at_frequency(frequency))
            .sum();
        total / surfaces.len() as f64
    }
}

/// Source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub name: String,
    pub position: Point3DConfig,
    /// Acoustic axis in world coordinates; defaults to +Y for compatibility.
    #[serde(default = "default_source_orientation")]
    pub orientation: Point3DConfig,
    #[serde(default = "default_amplitude")]
    pub amplitude: f64,
    #[serde(default)]
    pub directivity: DirectivityConfig,
    #[serde(default)]
    pub crossover: CrossoverConfig,
    /// Time delay in milliseconds (for driver alignment)
    #[serde(default)]
    pub delay_ms: f64,
    /// Phase inversion (180 degree phase shift)
    #[serde(default)]
    pub invert_phase: bool,
}

fn default_amplitude() -> f64 {
    1.0
}

fn default_source_orientation() -> Point3DConfig {
    Point3DConfig {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    }
}

/// Frequency range configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyConfig {
    pub min_freq: f64,
    pub max_freq: f64,
    pub num_points: usize,
    #[serde(default = "default_spacing")]
    pub spacing: String,
}

fn default_spacing() -> String {
    "logarithmic".to_string()
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Solver method: "direct", "image-source-1/2/3", "modal", "hybrid", "bem", or "hybrid-bem"
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_mesh_resolution")]
    pub mesh_resolution: usize,
    #[serde(default = "default_speed_of_sound")]
    pub speed_of_sound: f64,
    /// Ambient fluid density in kg/m³ used by impedance boundary models.
    /// The legacy default is retained for JSON compatibility.
    #[serde(default = "default_air_density")]
    pub air_density: f64,
    /// Temperature in Celsius (affects speed of sound and air absorption)
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Relative humidity in percent (0-100, affects air absorption)
    #[serde(default = "default_humidity")]
    pub humidity: f64,
    /// Enable air absorption modeling
    #[serde(default = "default_air_absorption")]
    pub air_absorption: bool,
    /// Enable edge diffraction modeling (adds diffraction contributions at room corners)
    #[serde(default = "default_edge_diffraction")]
    pub edge_diffraction: bool,
    /// Crossover width for hybrid solver (in octaves, default 0.5)
    #[serde(default = "default_hybrid_crossover_width")]
    pub hybrid_crossover_width: f64,
    /// Maximum mode order for modal analysis (default 20)
    #[serde(default = "default_max_mode_order")]
    pub max_mode_order: u32,
    /// Modal damping factor (Q factor, default 50)
    #[serde(default = "default_modal_damping")]
    pub modal_damping: f64,
    /// BEM solver configuration (used when method is "bem" or "hybrid-bem")
    #[serde(default)]
    pub bem_config: BemConfig,
}

fn default_method() -> String {
    "direct".to_string()
}
fn default_mesh_resolution() -> usize {
    2
}
fn default_speed_of_sound() -> f64 {
    343.0
}
fn default_air_density() -> f64 {
    1.2
}
fn default_temperature() -> f64 {
    20.0
} // 20°C
fn default_humidity() -> f64 {
    50.0
} // 50% relative humidity
fn default_air_absorption() -> bool {
    true
}
fn default_edge_diffraction() -> bool {
    false
} // Disabled by default (computationally expensive)
fn default_hybrid_crossover_width() -> f64 {
    0.5
} // 0.5 octaves
fn default_max_mode_order() -> u32 {
    20
}
fn default_modal_damping() -> f64 {
    10.0
} // Q factor (typical for residential rooms)

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            method: default_method(),
            mesh_resolution: default_mesh_resolution(),
            speed_of_sound: default_speed_of_sound(),
            air_density: default_air_density(),
            temperature: default_temperature(),
            humidity: default_humidity(),
            air_absorption: default_air_absorption(),
            edge_diffraction: default_edge_diffraction(),
            hybrid_crossover_width: default_hybrid_crossover_width(),
            max_mode_order: default_max_mode_order(),
            modal_damping: default_modal_damping(),
            bem_config: BemConfig::default(),
        }
    }
}

/// Calculate air absorption coefficient (Np/m) using ISO 9613-1 approximation
///
/// This is a simplified model based on ISO 9613-1 that provides reasonable
/// accuracy for typical room acoustics conditions (15-30°C, 20-80% RH).
///
/// Returns the absorption coefficient in Nepers per meter (Np/m).
/// To convert to dB/m: multiply by 8.686
///
/// # Arguments
/// * `frequency` - Sound frequency in Hz
/// * `temperature` - Temperature in Celsius
/// * `humidity` - Relative humidity in percent (0-100)
///
/// # Reference
/// ISO 9613-1:1993 - Acoustics -- Attenuation of sound during propagation outdoors
pub fn calculate_air_absorption(frequency: f64, temperature: f64, humidity: f64) -> f64 {
    // Simplified air absorption model based on empirical data (Np/m)
    // This provides reasonable accuracy for typical room acoustics conditions.
    //
    // Reference values at 20°C, 50% RH:
    // ~0.0001 Np/m at 500 Hz
    // ~0.001 Np/m at 1 kHz
    // ~0.004 Np/m at 4 kHz
    // ~0.010 Np/m at 8 kHz
    //
    // For more accurate results, implement full ISO 9613-1 calculation.

    let base_absorption = match frequency {
        f if f < 500.0 => 0.0001 * (f / 500.0).powi(2),
        f if f < 2000.0 => 0.0001 + 0.0009 * ((f - 500.0) / 1500.0),
        f if f < 8000.0 => 0.001 + 0.009 * ((f - 2000.0) / 6000.0),
        _ => 0.01 + 0.005 * ((frequency - 8000.0) / 8000.0),
    };

    // Temperature correction (absorption increases ~2%/°C deviation from 20°C)
    let temp_factor = 1.0 + 0.02 * (temperature - 20.0).abs();

    // Humidity correction (absorption decreases with humidity up to ~40%, then increases)
    let humidity_factor = if humidity < 40.0 {
        1.0 + 0.01 * (40.0 - humidity)
    } else {
        1.0 + 0.005 * (humidity - 40.0)
    };

    base_absorption * temp_factor * humidity_factor
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    #[serde(default = "default_generate_slices")]
    pub generate_slices: bool,
    #[serde(default = "default_slice_resolution")]
    pub slice_resolution: usize,
    #[serde(default)]
    pub slice_frequency_indices: Vec<usize>,
    /// Generate impulse response output
    #[serde(default = "default_generate_impulse_response")]
    pub generate_impulse_response: bool,
    /// Impulse response configuration
    #[serde(default)]
    pub impulse_response: ImpulseResponseConfig,
    /// Binaural rendering configuration
    #[serde(default)]
    pub binaural: BinauralConfig,
}

fn default_generate_slices() -> bool {
    true
}
fn default_slice_resolution() -> usize {
    50
}
fn default_generate_impulse_response() -> bool {
    false
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            generate_slices: true,
            slice_resolution: 50,
            slice_frequency_indices: Vec::new(),
            generate_impulse_response: false,
            impulse_response: ImpulseResponseConfig::default(),
            binaural: BinauralConfig::default(),
        }
    }
}

/// Complete simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub room: RoomGeometryConfig,
    pub sources: Vec<SourceConfig>,
    pub listening_positions: Vec<Point3DConfig>,
    pub frequencies: FrequencyConfig,
    /// Acoustic phasor, coordinate-frame, normal-orientation, and amplitude
    /// conventions carried with this configuration.
    ///
    /// The room simulator consumes canonical peak `exp(-iωt)` phasors.  Keep
    /// a WASM-specific serde default so legacy configuration documents retain
    /// that historical interpretation even though XEM's generic default uses
    /// RMS amplitudes.
    #[serde(default = "default_wasm_conventions")]
    pub conventions: sonium_xem::ConventionRegistry,
    #[serde(default)]
    pub solver: SolverConfig,
    #[serde(default)]
    pub visualization: VisualizationConfig,
    #[serde(default)]
    pub wall_materials: WallMaterialsConfig,
    #[serde(default)]
    pub metadata: MetadataConfig,
    /// Scattering objects inside the room (furniture, equipment, etc.)
    /// Only used with BEM solver methods ("bem" or "hybrid-bem")
    #[serde(default)]
    pub scattering_objects: Vec<ScatteringObjectConfig>,
}

fn default_wasm_conventions() -> sonium_xem::ConventionRegistry {
    sonium_xem::ConventionRegistry {
        rms_amplitudes: false,
        ..Default::default()
    }
}

/// Simulation metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataConfig {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub notes: String,
}

// ============================================================================
// Output types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct SourceResponse {
    pub source_name: String,
    pub spl: Vec<f64>,
}

/// Frequency response and uncertainty for one listener position.
#[derive(Debug, Clone, Serialize)]
pub struct ListenerResponse {
    /// Listener position in room coordinates, in metres.
    pub position: [f64; 3],
    /// SPL response aligned with the top-level frequency axis.
    pub frequency_response: Vec<f64>,
    /// A posteriori uncertainty estimates aligned with the frequency axis.
    pub frequency_uncertainty: Vec<NumericalErrorEstimate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SliceOutput {
    pub frequency: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<Vec<f64>>,
    pub spl: Vec<f64>,
    /// Whether each flattened SPL sample lies inside the room geometry.
    /// Cutout samples in an L-shaped room are retained for rectangular grid
    /// compatibility but are explicitly marked false.
    pub valid: Vec<bool>,
    pub provenance: SimulationProvenance,
    pub shape: [usize; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct RoomOutput {
    pub width: f64,
    pub depth: f64,
    pub height: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room_type: Option<String>,
    pub edges: Vec<[[f64; 3]; 2]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationResults {
    pub room: RoomOutput,
    pub sources: Vec<SourceOutputInfo>,
    pub listening_position: [f64; 3],
    /// Responses for every validated listener position. The legacy
    /// `listening_position` and `frequency_response` fields retain the first
    /// listener for compatibility with existing consumers.
    pub listener_responses: Vec<ListenerResponse>,
    pub frequencies: Vec<f64>,
    pub frequency_response: Vec<f64>,
    /// A posteriori numerical estimates aligned with `frequencies`.  These
    /// estimate truncation/model-discretisation sensitivity; they are not a
    /// substitute for measured-room uncertainty or native validation.
    pub frequency_uncertainty: Vec<NumericalErrorEstimate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_responses: Option<Vec<SourceResponse>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub horizontal_slices: Option<Vec<SliceOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertical_slices: Option<Vec<SliceOutput>>,
    pub solver: String,
    /// Explicit provenance for browser consumers. These results must not be
    /// mistaken for native validation data or for a uniformly high-fidelity
    /// wave solve.
    pub provenance: SimulationProvenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh_nodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh_elements: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room_modes: Option<Vec<RoomMode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub room_acoustics: Option<RoomAcoustics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub impulse_response: Option<ImpulseResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binaural_response: Option<BinauralResponse>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationProvenance {
    /// Stable hash of the validated simulation configuration.
    pub model_hash: String,
    /// The selected computation method.
    pub method: String,
    /// `approximate`, `blended`, or `wave_solved`.
    pub fidelity: String,
    /// Human-readable contributions included in the response.
    pub components: Vec<String>,
    /// Uncertainty is not estimated unless a future calibrated backend adds it.
    pub uncertainty: String,
    /// Method-specific estimate for a single-frequency result, when one can
    /// be computed without claiming a solver residual that is unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_estimate: Option<NumericalErrorEstimate>,
    /// Native parity is intentionally explicit rather than implied by the API.
    pub native_validation: String,
    /// Assumptions and validity envelope for the selected approximation.
    pub validity: Vec<String>,
    /// Number of BEM evaluations that used the documented ISM fallback.
    pub bem_fallback_count: usize,
    /// First BEM fallback reason, if a fallback occurred during this result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bem_fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NumericalErrorEstimate {
    /// Relative complex-pressure difference between adjacent approximations.
    pub relative: f64,
    /// Absolute difference in the metric named by `metric`.
    pub absolute: f64,
    /// Metric used for `absolute` and `relative`.
    pub metric: String,
    /// The comparison used to obtain the estimate.
    pub basis: String,
    /// This is a sensitivity estimate, not a calibrated confidence interval.
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceOutputInfo {
    pub name: String,
    pub position: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crossover: Option<String>,
}

// ============================================================================
// Room Modes
// ============================================================================

/// Represents a room resonant mode
#[derive(Debug, Clone, Serialize)]
pub struct RoomMode {
    /// Resonant frequency in Hz
    pub frequency: f64,
    /// Mode indices (n, m, p) for x, y, z dimensions
    pub indices: [u32; 3],
    /// Mode type: "axial", "tangential", or "oblique"
    pub mode_type: String,
    /// Description of the mode (e.g., "1,0,0 - Length mode")
    pub description: String,
}

/// Calculate room modes for a rectangular room
///
/// Room modes occur at frequencies where standing waves form between parallel surfaces.
/// - Axial modes: Standing waves between one pair of parallel walls (n,0,0), (0,m,0), or (0,0,p)
/// - Tangential modes: Standing waves between two pairs of walls (n,m,0), (n,0,p), or (0,m,p)
/// - Oblique modes: Standing waves between all three pairs of walls (n,m,p)
///
/// Formula: f = (c/2) * sqrt((n/Lx)² + (m/Ly)² + (p/Lz)²)
/// Calculate room modes (implementation in room_modes module)
pub fn calculate_room_modes(
    length_x: f64,
    length_y: f64,
    length_z: f64,
    speed_of_sound: f64,
    max_frequency: f64,
    max_order: u32,
) -> Vec<RoomMode> {
    room_modes::calculate_room_modes(
        length_x,
        length_y,
        length_z,
        speed_of_sound,
        max_frequency,
        max_order,
    )
}

/// Calculate modal pressure at a point (implementation in room_modes module)
#[allow(clippy::too_many_arguments)]
pub fn calculate_modal_pressure(
    source: &Point3D,
    listener: &Point3D,
    frequency: f64,
    room_width: f64,
    room_depth: f64,
    room_height: f64,
    speed_of_sound: f64,
    max_mode_order: u32,
    modal_damping: f64,
) -> Complex64 {
    room_modes::calculate_modal_pressure(
        source,
        listener,
        frequency,
        room_width,
        room_depth,
        room_height,
        speed_of_sound,
        max_mode_order,
        modal_damping,
    )
}

/// Calculate hybrid crossover weight (implementation in room_modes module)
pub fn hybrid_crossover_weight(
    frequency: f64,
    schroeder_frequency: f64,
    crossover_width_octaves: f64,
) -> f64 {
    room_modes::hybrid_crossover_weight(frequency, schroeder_frequency, crossover_width_octaves)
}

// ============================================================================
// RT60 and Reverberation Time Calculations
// ============================================================================

/// Room acoustics metrics including RT60
#[derive(Debug, Clone, Serialize)]
pub struct RoomAcoustics {
    /// Sabine RT60 in seconds
    pub rt60_sabine: f64,
    /// Eyring RT60 in seconds (more accurate for absorptive rooms)
    pub rt60_eyring: f64,
    /// Room volume in cubic meters
    pub volume: f64,
    /// Total surface area in square meters
    pub surface_area: f64,
    /// Average absorption coefficient
    pub average_alpha: f64,
    /// Total absorption in sabins (m²)
    pub total_absorption: f64,
    /// Schroeder frequency (transition from modal to statistical behavior)
    pub schroeder_frequency: f64,
    /// Critical distance (where direct and reverberant fields are equal)
    pub critical_distance: f64,
}

/// Calculate RT60 using Sabine's formula (implementation in acoustics module)
pub fn rt60_sabine(volume: f64, total_absorption: f64) -> f64 {
    acoustics::rt60_sabine(volume, total_absorption)
}

/// Calculate RT60 using Eyring's formula
///
/// RT60 = 0.161 * V / (-S * ln(1 - α_avg))
///
/// Where:
/// - V = room volume (m³)
/// - S = total surface area (m²)
/// - α_avg = average absorption coefficient
///
/// More accurate than Sabine for rooms with higher absorption (α > 0.2)
pub fn rt60_eyring(volume: f64, surface_area: f64, average_alpha: f64) -> f64 {
    if average_alpha > 0.0 && average_alpha < 1.0 {
        let clamped_alpha = average_alpha.min(0.99); // Prevent ln(0)
        0.161 * volume / (-surface_area * (1.0 - clamped_alpha).ln())
    } else if average_alpha >= 1.0 {
        0.0 // Perfect absorption = no reverberation
    } else {
        f64::INFINITY // No absorption = infinite reverberation
    }
}

/// Calculate critical distance (where direct and reverberant fields are equal)
///
/// r_c = 0.057 * sqrt(V / RT60)
///
/// At distances less than r_c, direct sound dominates.
/// At distances greater than r_c, reverberant sound dominates.
pub fn critical_distance(volume: f64, rt60: f64) -> f64 {
    if rt60 > 0.0 {
        0.057 * (volume / rt60).sqrt()
    } else {
        f64::INFINITY // No reverberation = direct sound always dominates
    }
}

/// Calculate room acoustics for a rectangular room with given wall materials
pub fn calculate_room_acoustics(
    width: f64,
    depth: f64,
    height: f64,
    materials: &WallMaterialsConfig,
    frequency: f64,
) -> RoomAcoustics {
    // Calculate room geometry
    let volume = width * depth * height;

    // Surface areas for each wall pair
    let area_left_right = depth * height; // Left and right walls (perpendicular to X)
    let area_front_back = width * height; // Front and back walls (perpendicular to Y)
    let area_floor_ceiling = width * depth; // Floor and ceiling (perpendicular to Z)

    let surface_area = 2.0 * (area_left_right + area_front_back + area_floor_ceiling);

    // Get absorption coefficients at the specified frequency
    let alpha_left = materials.left.absorption_at_frequency(frequency);
    let alpha_right = materials.right.absorption_at_frequency(frequency);
    let alpha_front = materials.front.absorption_at_frequency(frequency);
    let alpha_back = materials.back.absorption_at_frequency(frequency);
    let alpha_floor = materials.floor.absorption_at_frequency(frequency);
    let alpha_ceiling = materials.ceiling.absorption_at_frequency(frequency);

    // Calculate total absorption (sabins)
    let total_absorption = alpha_left * area_left_right
        + alpha_right * area_left_right
        + alpha_front * area_front_back
        + alpha_back * area_front_back
        + alpha_floor * area_floor_ceiling
        + alpha_ceiling * area_floor_ceiling;

    // Calculate average absorption coefficient
    let average_alpha = total_absorption / surface_area;

    // Calculate RT60 using both formulas
    let rt60_sab = rt60_sabine(volume, total_absorption);
    let rt60_eyr = rt60_eyring(volume, surface_area, average_alpha);

    // Use Eyring RT60 for Schroeder frequency (more accurate)
    let schroeder_freq = 2000.0 * (rt60_eyr / volume).sqrt();
    let crit_dist = critical_distance(volume, rt60_eyr);

    RoomAcoustics {
        rt60_sabine: rt60_sab,
        rt60_eyring: rt60_eyr,
        volume,
        surface_area,
        average_alpha,
        total_absorption,
        schroeder_frequency: schroeder_freq,
        critical_distance: crit_dist,
    }
}

// ============================================================================
// Binaural Rendering
// ============================================================================

/// Default head radius (meters) - average human head
const DEFAULT_HEAD_RADIUS: f64 = 0.0875; // 8.75 cm

/// Default ear spacing (meters) - typical ear-to-ear distance
const DEFAULT_EAR_SPACING: f64 = 0.175; // 17.5 cm

/// Binaural output containing left and right ear responses
#[derive(Debug, Clone, Serialize)]
pub struct BinauralResponse {
    /// Left ear impulse response
    pub left: ImpulseResponse,
    /// Right ear impulse response
    pub right: ImpulseResponse,
    /// Interaural time difference (seconds) - positive means left ear leads
    pub itd: f64,
    /// Interaural level difference (dB) - positive means left ear is louder
    pub ild: f64,
    /// Head position used
    pub head_position: [f64; 3],
    /// Head orientation (yaw in degrees, 0 = facing +Y)
    pub head_yaw: f64,
}

/// Configuration for binaural rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinauralConfig {
    /// Enable binaural rendering
    #[serde(default)]
    pub enabled: bool,
    /// Head center position (defaults to listening position)
    #[serde(default)]
    pub head_position: Option<Point3DConfig>,
    /// Head yaw angle in degrees (0 = facing +Y, 90 = facing -X)
    #[serde(default)]
    pub head_yaw: f64,
    /// Head radius in meters (default 0.0875m)
    #[serde(default = "default_head_radius")]
    pub head_radius: f64,
    /// Ear spacing in meters (default 0.175m)
    #[serde(default = "default_ear_spacing")]
    pub ear_spacing: f64,
    /// Impulse response configuration for binaural output
    #[serde(default)]
    pub ir_config: ImpulseResponseConfig,
}

fn default_head_radius() -> f64 {
    DEFAULT_HEAD_RADIUS
}
fn default_ear_spacing() -> f64 {
    DEFAULT_EAR_SPACING
}

impl Default for BinauralConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            head_position: None,
            head_yaw: 0.0,
            head_radius: DEFAULT_HEAD_RADIUS,
            ear_spacing: DEFAULT_EAR_SPACING,
            ir_config: ImpulseResponseConfig::default(),
        }
    }
}

/// Calculate ear positions from head center and orientation
pub fn calculate_ear_positions(
    head_center: &Point3D,
    head_yaw_deg: f64,
    ear_spacing: f64,
) -> (Point3D, Point3D) {
    let yaw_rad = head_yaw_deg.to_radians();
    let half_spacing = ear_spacing / 2.0;

    // Ears are perpendicular to facing direction
    // If facing +Y (yaw=0), left ear is at -X, right ear is at +X
    let ear_dx = -yaw_rad.sin() * half_spacing;
    let ear_dy = yaw_rad.cos() * half_spacing;

    // Left ear (positive perpendicular direction when facing forward)
    let left_ear = Point3D::new(
        head_center.x - ear_dx,
        head_center.y - ear_dy,
        head_center.z,
    );

    // Right ear (negative perpendicular direction)
    let right_ear = Point3D::new(
        head_center.x + ear_dx,
        head_center.y + ear_dy,
        head_center.z,
    );

    (left_ear, right_ear)
}

/// Calculate interaural time difference (ITD) using Woodworth's formula
///
/// ITD = (r/c) * (theta + sin(theta))
///
/// Where:
/// - r = head radius
/// - c = speed of sound
/// - theta = azimuth angle from median plane
pub fn calculate_itd(
    source: &Point3D,
    head_center: &Point3D,
    head_yaw_deg: f64,
    head_radius: f64,
    speed_of_sound: f64,
) -> f64 {
    // Calculate azimuth angle relative to head orientation
    let dx = source.x - head_center.x;
    let dy = source.y - head_center.y;

    // Source azimuth in world coordinates
    let source_azimuth = dx.atan2(dy);

    // Head facing direction
    let head_facing = head_yaw_deg.to_radians();

    // Relative azimuth (positive = source to the left)
    let theta = source_azimuth - head_facing;

    // Woodworth's formula
    (head_radius / speed_of_sound) * (theta + theta.sin())
}

/// Simplified HRTF magnitude approximation
///
/// This uses a simplified head-shadowing model based on the angle to the source.
/// For more accurate results, measured HRTF databases should be used.
///
/// Returns (left_gain, right_gain) as linear multipliers
pub fn approximate_hrtf_magnitude(
    source: &Point3D,
    head_center: &Point3D,
    head_yaw_deg: f64,
    frequency: f64,
) -> (f64, f64) {
    // Calculate azimuth relative to head
    let dx = source.x - head_center.x;
    let dy = source.y - head_center.y;
    let source_azimuth = dx.atan2(dy);
    let head_facing = head_yaw_deg.to_radians();
    let theta = source_azimuth - head_facing;

    // Head shadowing is more pronounced at higher frequencies
    // Simple model: use sinusoidal ILD that increases with frequency
    let freq_factor = (frequency / 1000.0).clamp(0.1, 4.0);

    // Maximum ILD is about 10-20 dB at high frequencies for sources at 90 degrees
    let max_ild_db = 6.0 * freq_factor; // ~6dB at 1kHz, ~24dB at 4kHz+

    // ILD varies sinusoidally with azimuth
    let ild_db = max_ild_db * theta.sin();

    // Convert to linear gains
    let left_gain = 10.0_f64.powf(-ild_db / 20.0);
    let right_gain = 10.0_f64.powf(ild_db / 20.0);

    // Normalize so average is 1
    let avg = (left_gain + right_gain) / 2.0;
    (left_gain / avg, right_gain / avg)
}

/// Calculate binaural response for a room simulation
pub fn calculate_binaural_response(
    frequencies: &[f64],
    left_pressures: &[Complex64],
    right_pressures: &[Complex64],
    config: &BinauralConfig,
    _speed_of_sound: f64,
    head_position: Point3D,
) -> Result<BinauralResponse, ImpulseResponseError> {
    // Calculate impulse responses for both ears
    let left_ir = calculate_impulse_response(frequencies, left_pressures, &config.ir_config)?;
    let right_ir = calculate_impulse_response(frequencies, right_pressures, &config.ir_config)?;

    // Calculate average ITD from magnitude-weighted phases
    let mut itd_sum = 0.0;
    let mut weight_sum = 0.0;

    for i in 0..frequencies
        .len()
        .min(left_pressures.len())
        .min(right_pressures.len())
    {
        let left_phase = left_pressures[i].arg();
        let right_phase = right_pressures[i].arg();
        let freq = frequencies[i];

        if freq > 20.0 && freq < 1500.0 {
            // Phase-based ITD is most reliable below 1.5kHz
            let phase_diff = left_phase - right_phase;
            let itd_sample = phase_diff / (2.0 * PI * freq);
            let weight = (left_pressures[i].norm() + right_pressures[i].norm()) / 2.0;
            itd_sum += itd_sample * weight;
            weight_sum += weight;
        }
    }

    let itd = if weight_sum > 1e-10 {
        itd_sum / weight_sum
    } else {
        0.0
    };

    // Calculate ILD from average magnitude difference
    let left_energy: f64 = left_pressures.iter().map(|p| p.norm_sqr()).sum();
    let right_energy: f64 = right_pressures.iter().map(|p| p.norm_sqr()).sum();

    let ild = if right_energy > 1e-10 {
        10.0 * (left_energy / right_energy).log10()
    } else if left_energy > 1e-10 {
        20.0 // Left much louder
    } else {
        0.0
    };

    Ok(BinauralResponse {
        left: left_ir,
        right: right_ir,
        itd,
        ild,
        head_position: [head_position.x, head_position.y, head_position.z],
        head_yaw: config.head_yaw,
    })
}

// ============================================================================
// Edge Diffraction (Simplified Biot-Tolstoy model)
// ============================================================================

/// Calculate edge diffraction coefficient using simplified UTD (Uniform Theory of Diffraction)
///
/// This is a simplified approximation based on the Biot-Tolstoy-Medwin model,
/// commonly used in room acoustics for diffraction around edges.
///
/// The diffraction coefficient depends on:
/// - Wedge angle (for room corners, typically 90 degrees = pi/2)
/// - Source and receiver angles relative to the edge
/// - Frequency (through wavenumber k)
/// - Distances from source/receiver to edge
///
/// Reference: Svensson, U. P., Fred, R. I., & Vanderkooy, J. (1999).
/// "An analytic secondary source model of edge diffraction impulse responses."
pub fn edge_diffraction_coefficient(
    wedge_angle: f64,  // Interior wedge angle (radians), typically PI/2 for room corners
    r_source: f64,     // Distance from source to edge
    r_receiver: f64,   // Distance from receiver to edge
    theta_source: f64, // Angle from source to edge face (radians)
    theta_receiver: f64, // Angle from receiver to edge face (radians)
    k: f64,            // Wavenumber (2 * PI * f / c)
) -> Complex64 {
    // Wedge index (n = PI / wedge_angle)
    let n = PI / wedge_angle;

    // Total path length from source around edge to receiver
    let path_length = r_source + r_receiver;

    // Diffraction attenuation factor
    // This is a simplified version - full BTM involves integral over edge length
    let nu = n;

    // Cotangent diffraction functions for each term
    let beta_plus = theta_source + theta_receiver;
    let beta_minus = theta_source - theta_receiver;

    // Fresnel-like diffraction function (simplified)
    let d_plus = cot_diffraction_term(beta_plus, nu);
    let d_minus = cot_diffraction_term(beta_minus, nu);

    // Distance factor (geometric spreading from edge)
    let distance_factor = 1.0 / (4.0 * PI * (r_source * r_receiver).sqrt());

    // Frequency-dependent phase
    let phase = Complex64::new(0.0, k * path_length).exp();

    // Combined diffraction coefficient
    let diffraction_amplitude = distance_factor * (d_plus + d_minus).abs();

    phase * diffraction_amplitude
}

/// Helper function for cotangent diffraction term
fn cot_diffraction_term(beta: f64, nu: f64) -> f64 {
    // Simplified cotangent function for diffraction calculation
    // cot((PI + beta) / (2 * nu))
    let arg = (PI + beta) / (2.0 * nu);

    // Handle singularities
    if arg.sin().abs() < 1e-10 {
        return 0.0;
    }

    arg.cos() / arg.sin()
}

/// Represents an edge in the room for diffraction calculation
#[derive(Debug, Clone)]
pub struct DiffractionEdge {
    /// Start point of the edge
    pub start: Point3D,
    /// End point of the edge
    pub end: Point3D,
    /// Interior wedge angle (radians)
    pub wedge_angle: f64,
}

impl DiffractionEdge {
    pub fn new(start: Point3D, end: Point3D, wedge_angle: f64) -> Self {
        Self {
            start,
            end,
            wedge_angle,
        }
    }

    /// Get the edge direction vector
    pub fn direction(&self) -> Point3D {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let dz = self.end.z - self.start.z;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        Point3D::new(dx / len, dy / len, dz / len)
    }

    /// Get the edge length
    pub fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    /// Find closest point on edge to a given point
    pub fn closest_point(&self, point: &Point3D) -> Point3D {
        let edge_vec = Point3D::new(
            self.end.x - self.start.x,
            self.end.y - self.start.y,
            self.end.z - self.start.z,
        );
        let point_vec = Point3D::new(
            point.x - self.start.x,
            point.y - self.start.y,
            point.z - self.start.z,
        );

        let edge_len_sq =
            edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y + edge_vec.z * edge_vec.z;
        if edge_len_sq < 1e-10 {
            return self.start;
        }

        let dot = point_vec.x * edge_vec.x + point_vec.y * edge_vec.y + point_vec.z * edge_vec.z;
        let t = (dot / edge_len_sq).clamp(0.0, 1.0);

        Point3D::new(
            self.start.x + t * edge_vec.x,
            self.start.y + t * edge_vec.y,
            self.start.z + t * edge_vec.z,
        )
    }

    /// Calculate diffraction contribution from this edge
    pub fn diffraction_contribution(
        &self,
        source: &Point3D,
        receiver: &Point3D,
        k: f64,
    ) -> Complex64 {
        // Find the diffraction point (closest point on edge to midpoint of source-receiver)
        let midpoint = Point3D::new(
            (source.x + receiver.x) / 2.0,
            (source.y + receiver.y) / 2.0,
            (source.z + receiver.z) / 2.0,
        );
        let diff_point = self.closest_point(&midpoint);

        // Distances from source/receiver to diffraction point
        let r_source = source.distance_to(&diff_point);
        let r_receiver = receiver.distance_to(&diff_point);

        // Skip if distances are too small
        if r_source < 0.01 || r_receiver < 0.01 {
            return Complex64::new(0.0, 0.0);
        }

        // Simplified angles (assume perpendicular to edge)
        // In a full implementation, we'd calculate the actual angles relative to edge faces
        let theta_source = PI / 4.0; // 45 degrees - typical value
        let theta_receiver = PI / 4.0;

        edge_diffraction_coefficient(
            self.wedge_angle,
            r_source,
            r_receiver,
            theta_source,
            theta_receiver,
            k,
        )
    }
}

/// Get diffraction edges for a rectangular room (12 edges at corners)
pub fn get_rectangular_room_edges(width: f64, depth: f64, height: f64) -> Vec<DiffractionEdge> {
    let wedge_90 = PI / 2.0; // 90-degree corners

    vec![
        // Floor edges
        DiffractionEdge::new(
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(width, 0.0, 0.0),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(width, 0.0, 0.0),
            Point3D::new(width, depth, 0.0),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(width, depth, 0.0),
            Point3D::new(0.0, depth, 0.0),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(0.0, depth, 0.0),
            Point3D::new(0.0, 0.0, 0.0),
            wedge_90,
        ),
        // Ceiling edges
        DiffractionEdge::new(
            Point3D::new(0.0, 0.0, height),
            Point3D::new(width, 0.0, height),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(width, 0.0, height),
            Point3D::new(width, depth, height),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(width, depth, height),
            Point3D::new(0.0, depth, height),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(0.0, depth, height),
            Point3D::new(0.0, 0.0, height),
            wedge_90,
        ),
        // Vertical edges
        DiffractionEdge::new(
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(0.0, 0.0, height),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(width, 0.0, 0.0),
            Point3D::new(width, 0.0, height),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(width, depth, 0.0),
            Point3D::new(width, depth, height),
            wedge_90,
        ),
        DiffractionEdge::new(
            Point3D::new(0.0, depth, 0.0),
            Point3D::new(0.0, depth, height),
            wedge_90,
        ),
    ]
}

// ============================================================================
// Core computation functions
// ============================================================================

/// Green's function for 3D Helmholtz equation: G(r) = exp(ikr) / (4 pi r)
fn greens_function_3d(r: f64, k: f64) -> Complex64 {
    if r < 1e-10 {
        return Complex64::new(0.0, 0.0);
    }
    let ikr = Complex64::new(0.0, k * r);
    ikr.exp() / (4.0 * PI * r)
}

/// Convert complex pressure to SPL (dB re 20 uPa)
fn pressure_to_spl(pressure: Complex64) -> f64 {
    let magnitude = pressure.norm();
    let p_ref = 20e-6;
    20.0 * (magnitude / p_ref).max(1e-10).log10()
}

/// Generate linearly spaced values
fn lin_space(start: f64, end: f64, num: usize) -> Vec<f64> {
    if num <= 1 {
        return vec![start];
    }
    (0..num)
        .map(|i| start + (end - start) * i as f64 / (num - 1) as f64)
        .collect()
}

/// Create a simple cardioid directivity pattern
fn create_cardioid_pattern(front_back_ratio: f64) -> DirectivityPattern {
    let horizontal_angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
    let vertical_angles: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();

    let mut magnitude = HostMatrix::<f64>::zeros(vertical_angles.len(), horizontal_angles.len());

    // Cardioid pattern: (a + b*cos(theta)) where a + b = 1 (front), a - b = 1/ratio (back)
    // Solving: a = (ratio + 1) / (2 * ratio), b = (ratio - 1) / (2 * ratio)
    let a = (front_back_ratio + 1.0) / (2.0 * front_back_ratio);
    let b = (front_back_ratio - 1.0) / (2.0 * front_back_ratio);

    for (v_idx, &theta_deg) in vertical_angles.iter().enumerate() {
        for (h_idx, &phi_deg) in horizontal_angles.iter().enumerate() {
            let theta = theta_deg.to_radians();
            let phi = phi_deg.to_radians();
            let cos_angle = theta.sin() * phi.cos();
            let response = (a + b * cos_angle).max(0.0);
            magnitude[[v_idx, h_idx]] = response;
        }
    }

    DirectivityPattern {
        horizontal_angles,
        vertical_angles,
        magnitude,
    }
}

/// Create a DirectivityPattern from spinorama.org directivity data
///
/// The spinorama data contains SPL curves at different angles for both horizontal
/// and vertical planes. This function:
/// 1. Normalizes all SPL values relative to the on-axis response
/// 2. Converts dB differences to linear magnitude ratios
/// 3. Creates a 2D interpolation grid for arbitrary angle lookups
///
/// # Arguments
/// * `horizontal` - Horizontal plane SPL curves at different angles
/// * `vertical` - Vertical plane SPL curves at different angles
/// * `frequency` - The frequency at which to sample the directivity
fn create_spinorama_pattern(
    horizontal: &[SpinoramaCurve],
    vertical: &[SpinoramaCurve],
    frequency: f64,
) -> DirectivityPattern {
    // Standard 10-degree resolution for the pattern
    let horizontal_angles: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
    let vertical_angles: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();

    let mut magnitude =
        HostMatrix::from_elem(vertical_angles.len(), horizontal_angles.len(), 1.0_f64);

    // Find the on-axis SPL for normalization (0° in horizontal data)
    let on_axis_spl = horizontal
        .iter()
        .find(|c| (c.angle - 0.0).abs() < 0.5)
        .map(|c| interpolate_spl_at_frequency(&c.freq, &c.spl, frequency))
        .unwrap_or(0.0);

    // Helper to get SPL at a specific angle from spinorama data
    let get_spl_at_angle = |curves: &[SpinoramaCurve], angle: f64| -> f64 {
        // Handle the fact that spinorama uses -60 to +60 range
        // while our pattern uses 0 to 360 for horizontal
        let search_angle = if angle > 180.0 { angle - 360.0 } else { angle };

        // Find the two closest angles and interpolate
        let mut closest_below: Option<&SpinoramaCurve> = None;
        let mut closest_above: Option<&SpinoramaCurve> = None;

        for curve in curves {
            if curve.angle <= search_angle
                && (closest_below.is_none() || curve.angle > closest_below.unwrap().angle)
            {
                closest_below = Some(curve);
            }
            if curve.angle >= search_angle
                && (closest_above.is_none() || curve.angle < closest_above.unwrap().angle)
            {
                closest_above = Some(curve);
            }
        }

        match (closest_below, closest_above) {
            (Some(below), Some(above)) if (below.angle - above.angle).abs() < 0.5 => {
                // Exact match
                interpolate_spl_at_frequency(&below.freq, &below.spl, frequency)
            }
            (Some(below), Some(above)) => {
                // Interpolate between two angles
                let spl_below = interpolate_spl_at_frequency(&below.freq, &below.spl, frequency);
                let spl_above = interpolate_spl_at_frequency(&above.freq, &above.spl, frequency);
                let t = (search_angle - below.angle) / (above.angle - below.angle);
                spl_below * (1.0 - t) + spl_above * t
            }
            (Some(only), None) | (None, Some(only)) => {
                interpolate_spl_at_frequency(&only.freq, &only.spl, frequency)
            }
            (None, None) => on_axis_spl, // Fallback to on-axis
        }
    };

    // Fill the magnitude array
    // Horizontal plane (theta = 90°, phi varies) - use horizontal data
    // Vertical plane (phi = 0°, theta varies) - use vertical data
    // Other angles: interpolate between horizontal and vertical

    for (v_idx, &theta_deg) in vertical_angles.iter().enumerate() {
        for (h_idx, &phi_deg) in horizontal_angles.iter().enumerate() {
            // Get SPL from horizontal and vertical data
            let h_spl = get_spl_at_angle(horizontal, phi_deg);
            let v_spl = get_spl_at_angle(vertical, theta_deg);

            // Blend based on angle: at theta=90° use horizontal, at theta=0° or 180° use vertical
            let theta_rad = theta_deg.to_radians();
            let blend = theta_rad.sin(); // 1 at 90°, 0 at 0° and 180°

            let combined_spl = h_spl * blend + v_spl * (1.0 - blend);

            // Convert dB difference to linear magnitude
            let db_diff = combined_spl - on_axis_spl;
            let linear_mag = 10.0_f64.powf(db_diff / 20.0);

            magnitude[[v_idx, h_idx]] = linear_mag.clamp(0.0, 10.0);
        }
    }

    DirectivityPattern {
        horizontal_angles,
        vertical_angles,
        magnitude,
    }
}

/// Interpolate SPL value at a specific frequency from spinorama frequency/SPL arrays
fn interpolate_spl_at_frequency(freq: &[f64], spl: &[f64], target_freq: f64) -> f64 {
    if freq.is_empty() || spl.is_empty() {
        return 0.0;
    }

    // Handle edge cases
    if target_freq <= freq[0] {
        return spl[0];
    }
    if target_freq >= freq[freq.len() - 1] {
        return spl[spl.len() - 1];
    }

    // Find bracketing indices and interpolate logarithmically
    for i in 0..freq.len() - 1 {
        if target_freq >= freq[i] && target_freq < freq[i + 1] {
            let log_f = target_freq.ln();
            let log_f1 = freq[i].ln();
            let log_f2 = freq[i + 1].ln();
            let t = (log_f - log_f1) / (log_f2 - log_f1);
            return spl[i] * (1.0 - t) + spl[i + 1] * t;
        }
    }

    spl[spl.len() - 1]
}

// ============================================================================
// WASM-exported Room Simulator
// ============================================================================

/// Room Acoustics Simulator - WASM interface
#[wasm_bindgen]
pub struct RoomSimulatorWasm {
    config: SimulationConfig,
    room_geometry: RoomGeometry,
    sources: Vec<Source>,
    listening_positions: Vec<Point3D>,
    listening_position: Point3D,
    frequencies: Vec<f64>,
    speed_of_sound: f64,
    wall_materials: WallMaterialsConfig,
    /// Temperature in Celsius
    temperature: f64,
    /// Relative humidity (0-100%)
    humidity: f64,
    /// Whether to model air absorption
    air_absorption_enabled: bool,
    /// Whether to model edge diffraction
    edge_diffraction_enabled: bool,
    /// Pre-computed diffraction edges for the room
    diffraction_edges: Vec<DiffractionEdge>,
    /// Chunked computation state
    chunked_frequencies_completed: usize,
    chunked_pressures: Option<Vec<f64>>,
    chunked_in_progress: bool,
    chunked_cancelled: bool,
    /// Listener-major SPL prefixes for progressive multi-listener output.
    chunked_listener_pressures: Option<Vec<Vec<f64>>>,
    model_hash: String,
    bem_fallback_count: AtomicUsize,
    bem_fallback_reason: Mutex<Option<String>>,
    bem_diagnostics: Mutex<HashMap<u64, (f64, usize, bool)>>,
    bem_workspace_budget: bem_solver::BemWorkspaceBudget,
}

#[wasm_bindgen]
impl RoomSimulatorWasm {
    /// Create a new simulator from JSON configuration
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<RoomSimulatorWasm, JsValue> {
        init_panic_hook();

        if config_json.len() > MAX_CONFIG_JSON_BYTES {
            return Err(js_error("Configuration JSON exceeds the 16 MiB limit"));
        }

        let config: SimulationConfig = serde_json::from_str(config_json)
            .map_err(|e| js_error(format!("Config parse error: {}", e)))?;
        validate_simulation_config(&config).map_err(|errors| {
            js_error(format!("Invalid simulation config: {}", errors.join("; ")))
        })?;
        let model_hash = simulation_model_hash(&config);

        // Build room geometry
        let room_geometry = match &config.room {
            RoomGeometryConfig::Rectangular {
                width,
                depth,
                height,
            } => {
                console_log!(
                    "Creating rectangular room: {}x{}x{} m",
                    width,
                    depth,
                    height
                );
                RoomGeometry::Rectangular(RectangularRoom::new(*width, *depth, *height))
            }
            RoomGeometryConfig::LShaped {
                width1,
                depth1,
                width2,
                depth2,
                height,
            } => {
                console_log!(
                    "Creating L-shaped room: {}x{} + {}x{} x {} m",
                    width1,
                    depth1,
                    width2,
                    depth2,
                    height
                );
                RoomGeometry::LShaped(LShapedRoom::new(
                    *width1, *depth1, *width2, *depth2, *height,
                ))
            }
        };

        // Build sources
        let sources: Vec<Source> = config
            .sources
            .iter()
            .map(|s| {
                let directivity = match &s.directivity {
                    DirectivityConfig::Omnidirectional => DirectivityPattern::omnidirectional(),
                    DirectivityConfig::Cardioid { front_back_ratio } => {
                        create_cardioid_pattern(*front_back_ratio)
                    }
                    DirectivityConfig::Spinorama {
                        horizontal,
                        vertical,
                    } => {
                        // Use 1kHz as the representative frequency for the directivity pattern
                        // This is a common choice for speaker directivity visualization
                        create_spinorama_pattern(horizontal, vertical, 1000.0)
                    }
                };

                Source::new(s.position.into(), directivity, s.amplitude)
                    .with_orientation(s.orientation.into())
                    .with_name(s.name.clone())
                    .with_crossover(s.crossover.to_filter())
                    .with_delay_ms(s.delay_ms)
                    .with_phase_inversion(s.invert_phase)
            })
            .collect();

        console_log!("Created {} sources", sources.len());
        for source in &sources {
            if source.delay_sec > 0.0 || source.invert_phase {
                console_log!(
                    "  {}: delay={:.2}ms, invert={}",
                    source.name,
                    source.delay_sec * 1000.0,
                    source.invert_phase
                );
            }
        }

        let listening_positions: Vec<Point3D> = config
            .listening_positions
            .iter()
            .map(|point| (*point).into())
            .collect();
        let listening_position = listening_positions
            .first()
            .copied()
            .unwrap_or(Point3D::new(0.0, 0.0, 0.0));

        // Use the native exchange-model frequency convention so browser and
        // native results have identical grids for the same configuration.
        let frequencies = NativeFrequencyConfig {
            min_freq: config.frequencies.min_freq,
            max_freq: config.frequencies.max_freq,
            num_points: config.frequencies.num_points,
            spacing: config.frequencies.spacing.clone(),
        }
        .generate_frequencies();

        console_log!(
            "Frequency range: {:.1} - {:.1} Hz ({} points)",
            frequencies.first().unwrap_or(&0.0),
            frequencies.last().unwrap_or(&0.0),
            frequencies.len()
        );

        let speed_of_sound = config.solver.speed_of_sound;
        let wall_materials = config.wall_materials.clone();
        let temperature = config.solver.temperature;
        let humidity = config.solver.humidity;
        let air_absorption_enabled = config.solver.air_absorption;

        // Log wall materials
        console_log!(
            "Wall materials: Left={}, Right={}, Front={}, Back={}, Floor={}, Ceiling={}",
            wall_materials.get_material(WallSurface::Left).name,
            wall_materials.get_material(WallSurface::Right).name,
            wall_materials.get_material(WallSurface::Front).name,
            wall_materials.get_material(WallSurface::Back).name,
            wall_materials.get_material(WallSurface::Floor).name,
            wall_materials.get_material(WallSurface::Ceiling).name,
        );

        // Log environmental conditions
        if air_absorption_enabled {
            console_log!(
                "Air absorption enabled: T={:.1}°C, RH={:.0}%",
                temperature,
                humidity
            );
        }

        let edge_diffraction_enabled = config.solver.edge_diffraction;

        // Pre-compute diffraction edges for the room
        let diffraction_edges = if edge_diffraction_enabled {
            match &room_geometry {
                RoomGeometry::Rectangular(r) => {
                    console_log!("Edge diffraction enabled: computing 12 room edges");
                    get_rectangular_room_edges(r.width, r.depth, r.height)
                }
                RoomGeometry::LShaped(l) => {
                    // For L-shaped rooms, we have more edges (18 total)
                    // For now, use a simplified approximation with outer bounding box
                    console_log!("Edge diffraction enabled for L-shaped room (simplified)");
                    get_rectangular_room_edges(l.width1, l.depth1 + l.depth2, l.height)
                }
            }
        } else {
            Vec::new()
        };

        let bem_workspace_limit = validation::MAX_ESTIMATED_WORKING_SET_BYTES
            .saturating_sub(validation::estimated_working_set_bytes(&config))
            as usize;
        Ok(RoomSimulatorWasm {
            config,
            room_geometry,
            sources,
            listening_positions,
            listening_position,
            frequencies,
            speed_of_sound,
            wall_materials,
            temperature,
            humidity,
            air_absorption_enabled,
            edge_diffraction_enabled,
            diffraction_edges,
            chunked_frequencies_completed: 0,
            chunked_pressures: None,
            chunked_in_progress: false,
            chunked_cancelled: false,
            chunked_listener_pressures: None,
            model_hash,
            bem_fallback_count: AtomicUsize::new(0),
            bem_fallback_reason: Mutex::new(None),
            bem_diagnostics: Mutex::new(HashMap::new()),
            bem_workspace_budget: bem_solver::BemWorkspaceBudget::new(bem_workspace_limit),
        })
    }

    fn record_bem_fallback(&self, frequency: f64, fallback: &str, reason: impl std::fmt::Display) {
        let count = self.bem_fallback_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count == 1
            && let Ok(mut first_reason) = self.bem_fallback_reason.lock()
        {
            *first_reason = Some(format!(
                "BEM failed at {frequency:.6} Hz; used {fallback}: {reason}"
            ));
        }
    }

    fn record_bem_diagnostics(&self, result: &bem_solver::BemResult) {
        if let Ok(mut diagnostics) = self.bem_diagnostics.lock() {
            diagnostics.insert(
                result.frequency.to_bits(),
                (result.residual, result.iterations, result.converged),
            );
        }
    }

    fn provenance(&self) -> SimulationProvenance {
        let fallback_reason = self
            .bem_fallback_reason
            .lock()
            .ok()
            .and_then(|reason| reason.clone());
        simulation_provenance(
            &self.config.solver.method,
            &self.model_hash,
            self.bem_fallback_count.load(Ordering::Relaxed),
            fallback_reason,
            self.config.scattering_objects.len(),
        )
    }

    fn provenance_with_error(
        &self,
        estimate: Option<NumericalErrorEstimate>,
    ) -> SimulationProvenance {
        let mut provenance = self.provenance();
        if estimate.is_some() && provenance.bem_fallback_count == 0 {
            provenance.uncertainty = "a_posteriori_truncation_sensitivity".to_string();
        }
        provenance.error_estimate = estimate;
        provenance
    }

    fn numerical_error_estimate(
        &self,
        point: &Point3D,
        frequency: f64,
        pressure: Complex64,
    ) -> Option<NumericalErrorEstimate> {
        if matches!(self.config.solver.method.as_str(), "bem" | "hybrid-bem")
            && let Ok(diagnostics) = self.bem_diagnostics.lock()
            && let Some(&(residual, iterations, converged)) = diagnostics.get(&frequency.to_bits())
            && residual.is_finite()
            && residual > 0.0
        {
            let scale = pressure.norm().max(1.0e-12);
            return Some(NumericalErrorEstimate {
                relative: residual / scale,
                absolute: residual,
                metric: "solver_residual_norm".to_string(),
                basis: format!(
                    "BEM solver residual after {} iteration(s), converged={converged}",
                    iterations
                ),
                confidence: "solver_diagnostic_not_calibrated".to_string(),
            });
        }

        let (comparison, basis) = match self.config.solver.method.as_str() {
            "modal" => {
                let order = self.config.solver.max_mode_order;
                if order == 0 {
                    return None;
                }
                let (width, depth, height) = self.get_room_dimensions();
                let mut lower = Complex64::new(0.0, 0.0);
                for source in &self.sources {
                    lower += calculate_modal_pressure(
                        &source.position,
                        point,
                        frequency,
                        width,
                        depth,
                        height,
                        self.speed_of_sound,
                        order - 1,
                        self.config.solver.modal_damping,
                    ) * source.amplitude_towards(point, frequency)
                        * source.phase_factor(frequency);
                }
                (lower, format!("modal order {} versus {}", order, order - 1))
            }
            "direct" => return None,
            "image-source-1" | "image-source-2" | "image-source-3" => {
                let order = match self.config.solver.method.as_str() {
                    "image-source-1" => 1,
                    "image-source-2" => 2,
                    "image-source-3" => 3,
                    _ => 2,
                };
                if order == 0 {
                    return None;
                }
                (
                    self.calculate_ism_field_with_order(point, frequency, Some(order - 1)),
                    format!("image-source order {} versus {}", order, order - 1),
                )
            }
            "hybrid" => {
                if !matches!(&self.room_geometry, RoomGeometry::Rectangular(_)) {
                    return None;
                }
                let order = 2_usize;
                let (width, depth, height) = self.get_room_dimensions();
                let schroeder = self.broadband_schroeder_frequency();
                let weight = hybrid_crossover_weight(
                    frequency,
                    schroeder,
                    self.config.solver.hybrid_crossover_width,
                );
                let mut modal = Complex64::new(0.0, 0.0);
                for source in &self.sources {
                    modal += calculate_modal_pressure(
                        &source.position,
                        point,
                        frequency,
                        width,
                        depth,
                        height,
                        self.speed_of_sound,
                        self.config.solver.max_mode_order,
                        self.config.solver.modal_damping,
                    ) * source.amplitude_towards(point, frequency)
                        * source.phase_factor(frequency);
                }
                let lower_ism =
                    self.calculate_ism_field_with_order(point, frequency, Some(order - 1));
                (
                    modal * (1.0 - weight) + lower_ism * weight,
                    format!("hybrid image-source order {} versus {}", order, order - 1),
                )
            }
            _ => return None,
        };

        let absolute = (pressure - comparison).norm();
        let relative = absolute / pressure.norm().max(1.0e-12);
        Some(NumericalErrorEstimate {
            relative,
            absolute,
            metric: "relative_complex_pressure".to_string(),
            basis,
            confidence: "sensitivity_only_uncalibrated".to_string(),
        })
    }

    fn reset_bem_fallbacks(&self) {
        self.bem_fallback_count.store(0, Ordering::Relaxed);
        if let Ok(mut reason) = self.bem_fallback_reason.lock() {
            *reason = None;
        }
        if let Ok(mut diagnostics) = self.bem_diagnostics.lock() {
            diagnostics.clear();
        }
    }

    fn wavenumber(&self, frequency: f64) -> f64 {
        2.0 * PI * frequency / self.speed_of_sound
    }

    /// Single broadband Schroeder frequency for all modal/ISM crossover
    /// blends: `2000·√(RT60/V)` with Eyring RT60 from the absorption
    /// averaged over `self.frequencies`.
    ///
    /// The per-frequency Sabine RT60 previously recomputed here made the
    /// crossover point itself frequency-dependent, which contradicts the
    /// Schroeder construction (one broadband RT60). Eyring matches the
    /// room-data path in `calculate_room_acoustics`.
    fn broadband_schroeder_frequency(&self) -> f64 {
        let (room_width, room_depth, room_height) = self.get_room_dimensions();
        let volume = room_width * room_depth * room_height;
        let surface_area =
            2.0 * (room_width * room_depth + room_width * room_height + room_depth * room_height);
        let average_alpha = if self.frequencies.is_empty() {
            self.wall_materials.average_absorption_at(500.0)
        } else {
            self.frequencies
                .iter()
                .map(|&f| self.wall_materials.average_absorption_at(f))
                .sum::<f64>()
                / self.frequencies.len() as f64
        };
        let rt60 = rt60_eyring(volume, surface_area, average_alpha);
        2000.0 * (rt60 / volume).sqrt()
    }

    /// Check if a point is inside the room geometry
    fn point_inside_room(&self, point: &Point3D) -> bool {
        match &self.room_geometry {
            RoomGeometry::Rectangular(r) => {
                point.x >= 0.0
                    && point.x <= r.width
                    && point.y >= 0.0
                    && point.y <= r.depth
                    && point.z >= 0.0
                    && point.z <= r.height
            }
            RoomGeometry::LShaped(l) => l.contains(point),
        }
    }

    /// Calculate air absorption attenuation factor for a given distance
    /// Returns a multiplier in the range (0, 1] where 1 = no attenuation
    fn air_absorption_factor(&self, distance: f64, frequency: f64) -> f64 {
        if !self.air_absorption_enabled || distance < 1e-6 {
            return 1.0;
        }

        let alpha = calculate_air_absorption(frequency, self.temperature, self.humidity);
        // Attenuation: exp(-alpha * distance)
        (-alpha * distance).exp()
    }

    fn calculate_direct_field(&self, point: &Point3D, frequency: f64) -> Complex64 {
        let (room_width, room_depth, room_height) = self.get_room_dimensions();

        // Handle different solver methods
        let method = self.config.solver.method.as_str();

        // BEM-based methods
        if method == "bem" || method == "hybrid-bem" {
            return self.calculate_bem_or_hybrid_bem(point, frequency, method);
        }

        // Modal and modal-ISM hybrid methods
        if method == "modal" || method == "hybrid" {
            // Modal analysis is only valid for rectangular rooms
            if let RoomGeometry::Rectangular(_) = &self.room_geometry {
                // Calculate modal pressure for all sources
                let mut modal_pressure = Complex64::new(0.0, 0.0);
                for source in &self.sources {
                    let amplitude = source.amplitude_towards(point, frequency);
                    let phase_factor = source.phase_factor(frequency);

                    let modal = calculate_modal_pressure(
                        &source.position,
                        point,
                        frequency,
                        room_width,
                        room_depth,
                        room_height,
                        self.config.solver.speed_of_sound,
                        self.config.solver.max_mode_order,
                        self.config.solver.modal_damping,
                    );

                    modal_pressure += modal * amplitude * phase_factor;
                }

                if method == "modal" {
                    // Pure modal: return modal pressure only
                    return modal_pressure;
                }

                // Hybrid: blend modal and ISM at the single broadband
                // Schroeder crossover (see `broadband_schroeder_frequency`).
                let schroeder_freq = self.broadband_schroeder_frequency();

                let ism_weight = hybrid_crossover_weight(
                    frequency,
                    schroeder_freq,
                    self.config.solver.hybrid_crossover_width,
                );

                // If fully in modal region, return modal pressure
                if ism_weight < 1e-6 {
                    return modal_pressure;
                }

                // Calculate ISM pressure and blend
                let ism_pressure = self.calculate_ism_field(point, frequency);

                // Blend: p = (1-w) * modal + w * ISM
                return modal_pressure * (1.0 - ism_weight) + ism_pressure * ism_weight;
            }
        }

        // Standard ISM calculation for non-hybrid modes
        self.calculate_ism_field(point, frequency)
    }

    /// Calculate field using BEM or hybrid BEM+ISM
    ///
    /// For "bem" mode: Uses modal analysis for rectangular rooms (captures room modes),
    /// falls back to BEM direct field + scattering for non-rectangular.
    ///
    /// For "hybrid-bem" mode: Blends modal/BEM at low frequencies with ISM at high
    /// frequencies using Schroeder frequency crossover.
    fn calculate_bem_or_hybrid_bem(
        &self,
        point: &Point3D,
        frequency: f64,
        method: &str,
    ) -> Complex64 {
        let (room_width, room_depth, room_height) = self.get_room_dimensions();

        // Calculate low-frequency pressure using modal analysis (rectangular) or BEM (other)
        let low_freq_pressure = if let RoomGeometry::Rectangular(_) = &self.room_geometry {
            // Modal analysis for rectangular rooms - captures room resonances accurately
            let mut modal_pressure = Complex64::new(0.0, 0.0);
            for source in &self.sources {
                let amplitude = source.amplitude_towards(point, frequency);
                let phase_factor = source.phase_factor(frequency);

                let modal = calculate_modal_pressure(
                    &source.position,
                    point,
                    frequency,
                    room_width,
                    room_depth,
                    room_height,
                    self.config.solver.speed_of_sound,
                    self.config.solver.max_mode_order,
                    self.config.solver.modal_damping,
                );

                modal_pressure += modal * amplitude * phase_factor;
            }

            // Add scattering object contributions if present
            if !self.config.scattering_objects.is_empty() {
                let scattering_result = bem_solver::solve_bem_with_budget(
                    &self.room_geometry,
                    &self.sources,
                    &self.config.scattering_objects,
                    point,
                    frequency,
                    self.config.solver.speed_of_sound,
                    &self.config.solver.bem_config,
                    &self.bem_workspace_budget,
                );
                match scattering_result {
                    Ok(result) => {
                        self.record_bem_diagnostics(&result);
                        // Add scattered field contribution (subtract direct to avoid double counting)
                        let k = 2.0 * std::f64::consts::PI * frequency
                            / self.config.solver.speed_of_sound;
                        let mut direct_field = Complex64::new(0.0, 0.0);
                        for source in &self.sources {
                            let amp = source.amplitude_towards(point, frequency);
                            let dist = source.position.distance_to(point);
                            direct_field += bem_solver::greens_function(dist, k)
                                * amp
                                * source.phase_factor(frequency);
                        }
                        // Scattered contribution = total BEM - direct
                        let scattered = result.pressure - direct_field;
                        modal_pressure += scattered;
                    }
                    Err(error) => self.record_bem_fallback(frequency, "modal-only result", error),
                }
            }

            modal_pressure
        } else {
            // Non-rectangular rooms: use full BEM solver
            let bem_result = bem_solver::solve_bem_with_budget(
                &self.room_geometry,
                &self.sources,
                &self.config.scattering_objects,
                point,
                frequency,
                self.config.solver.speed_of_sound,
                &self.config.solver.bem_config,
                &self.bem_workspace_budget,
            );

            match bem_result {
                Ok(result) => {
                    self.record_bem_diagnostics(&result);
                    result.pressure
                }
                Err(error) => {
                    self.record_bem_fallback(frequency, "ISM fallback", error);
                    return self.calculate_ism_field(point, frequency);
                }
            }
        };

        if method == "bem" {
            // Pure BEM/modal: return low frequency pressure only
            return low_freq_pressure;
        }

        // Hybrid-BEM: blend modal/BEM at low frequencies with ISM at high
        // frequencies at the single broadband Schroeder crossover (see
        // `broadband_schroeder_frequency`).
        let schroeder_freq = self.broadband_schroeder_frequency();

        // ISM weight increases above Schroeder frequency
        let ism_weight = hybrid_crossover_weight(
            frequency,
            schroeder_freq,
            self.config.solver.hybrid_crossover_width,
        );

        // If fully in modal/BEM region (below Schroeder), return low freq pressure
        if ism_weight < 1e-6 {
            return low_freq_pressure;
        }

        // If fully in ISM region (above Schroeder), return ISM pressure
        if ism_weight > 1.0 - 1e-6 {
            return self.calculate_ism_field(point, frequency);
        }

        // Blend: p = (1-w) * modal/BEM + w * ISM
        let ism_pressure = self.calculate_ism_field(point, frequency);
        low_freq_pressure * (1.0 - ism_weight) + ism_pressure * ism_weight
    }

    /// Calculate field using Image Source Method (ISM)
    fn calculate_ism_field(&self, point: &Point3D, frequency: f64) -> Complex64 {
        self.calculate_ism_field_with_order(point, frequency, None)
    }

    fn calculate_ism_field_with_order(
        &self,
        point: &Point3D,
        frequency: f64,
        order_override: Option<usize>,
    ) -> Complex64 {
        let k = self.wavenumber(frequency);
        let mut total_pressure = Complex64::new(0.0, 0.0);

        // Determine reflection order from solver method
        let configured_order = ism::configured_reflection_order(self.config.solver.method.as_str());
        let reflection_order = order_override.unwrap_or(configured_order);

        // Get room dimensions for image source calculation
        let (room_width, room_depth, room_height) = self.get_room_dimensions();

        // Get frequency-dependent reflection coefficients for each wall
        let r_left = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Left,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_right = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Right,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_front = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Front,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_back = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Back,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_floor = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Floor,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_ceiling = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Ceiling,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );

        for source in &self.sources {
            let amplitude = source.amplitude_towards(point, frequency);
            // Get phase factor for delay and phase inversion
            let phase_factor = source.phase_factor(frequency);

            // Direct sound (always included) with air absorption and phase
            let r_direct = source.position.distance_to(point);
            let air_atten_direct = self.air_absorption_factor(r_direct, frequency);
            total_pressure +=
                greens_function_3d(r_direct, k) * amplitude * air_atten_direct * phase_factor;

            // Keep the rectangular ISM path aligned with the native solver:
            // enumerate all axis-image combinations up to the requested
            // reflection order, including repeated reflections on one axis.
            if matches!(&self.room_geometry, RoomGeometry::Rectangular(_)) {
                let images = room_modes::rectangular_image_sources(
                    &source.position,
                    [room_width, room_depth, room_height],
                    [r_left, r_right, r_front, r_back, r_floor, r_ceiling],
                    reflection_order,
                );
                for (image, reflection, order) in images {
                    if order == 0 {
                        continue;
                    }
                    let distance = image.distance_to(point);
                    if distance > 1e-6 {
                        let air_atten = self.air_absorption_factor(distance, frequency);
                        total_pressure += greens_function_3d(distance, k)
                            * amplitude
                            * reflection
                            * air_atten
                            * phase_factor;
                    }
                }
                continue;
            }

            if reflection_order >= 1 {
                // First-order image sources - handle L-shaped rooms specially
                let first_order_images: Vec<(Point3D, Complex64)> = match &self.room_geometry {
                    RoomGeometry::LShaped(l_room) => {
                        // Use L-shaped room's proper image source generation
                        l_room
                            .get_first_order_images(&source.position)
                            .into_iter()
                            .filter_map(|(image, wall_name)| {
                                // Validate the image source path
                                if !l_room.is_valid_image_source(&image, point, &source.position) {
                                    return None;
                                }
                                // Map wall name to reflection coefficient
                                let refl = match wall_name {
                                    "left" => r_left,
                                    "right" => r_right,
                                    "front" => r_front,
                                    "back" => r_back,
                                    "floor" => r_floor,
                                    "ceiling" => r_ceiling,
                                    "step_horizontal" | "step_vertical" => {
                                        // Interior step walls - use average of nearby walls
                                        (r_right + r_back) / 2.0
                                    }
                                    _ => r_right, // Default fallback
                                };
                                Some((image, refl))
                            })
                            .collect()
                    }
                    RoomGeometry::Rectangular(_) => {
                        // Standard rectangular room image sources (6 walls)
                        vec![
                            // Left wall (x=0)
                            (
                                Point3D::new(
                                    -source.position.x,
                                    source.position.y,
                                    source.position.z,
                                ),
                                r_left,
                            ),
                            // Right wall (x=width)
                            (
                                Point3D::new(
                                    2.0 * room_width - source.position.x,
                                    source.position.y,
                                    source.position.z,
                                ),
                                r_right,
                            ),
                            // Front wall (y=0)
                            (
                                Point3D::new(
                                    source.position.x,
                                    -source.position.y,
                                    source.position.z,
                                ),
                                r_front,
                            ),
                            // Back wall (y=depth)
                            (
                                Point3D::new(
                                    source.position.x,
                                    2.0 * room_depth - source.position.y,
                                    source.position.z,
                                ),
                                r_back,
                            ),
                            // Floor (z=0)
                            (
                                Point3D::new(
                                    source.position.x,
                                    source.position.y,
                                    -source.position.z,
                                ),
                                r_floor,
                            ),
                            // Ceiling (z=height)
                            (
                                Point3D::new(
                                    source.position.x,
                                    source.position.y,
                                    2.0 * room_height - source.position.z,
                                ),
                                r_ceiling,
                            ),
                        ]
                    }
                };

                for (image_src, refl_coeff) in &first_order_images {
                    let r_image = image_src.distance_to(point);
                    if r_image > 1e-6 {
                        let air_atten = self.air_absorption_factor(r_image, frequency);
                        total_pressure += greens_function_3d(r_image, k)
                            * amplitude
                            * refl_coeff
                            * air_atten
                            * phase_factor;
                    }
                }
            }

            // Higher-order reflections only for rectangular rooms
            // L-shaped room higher-order ISM is complex and would need proper path validation
            if let RoomGeometry::Rectangular(_) = &self.room_geometry {
                if reflection_order >= 2 {
                    // Second-order image sources (edges - 12 combinations)
                    // Each involves reflection off two walls, so multiply their coefficients
                    let second_order_images = [
                        // Left + Front (x=0, y=0)
                        (
                            Point3D::new(-source.position.x, -source.position.y, source.position.z),
                            r_left * r_front,
                        ),
                        // Left + Back (x=0, y=depth)
                        (
                            Point3D::new(
                                -source.position.x,
                                2.0 * room_depth - source.position.y,
                                source.position.z,
                            ),
                            r_left * r_back,
                        ),
                        // Right + Front (x=width, y=0)
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                -source.position.y,
                                source.position.z,
                            ),
                            r_right * r_front,
                        ),
                        // Right + Back (x=width, y=depth)
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                2.0 * room_depth - source.position.y,
                                source.position.z,
                            ),
                            r_right * r_back,
                        ),
                        // Left + Floor (x=0, z=0)
                        (
                            Point3D::new(-source.position.x, source.position.y, -source.position.z),
                            r_left * r_floor,
                        ),
                        // Left + Ceiling (x=0, z=height)
                        (
                            Point3D::new(
                                -source.position.x,
                                source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_left * r_ceiling,
                        ),
                        // Right + Floor (x=width, z=0)
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                source.position.y,
                                -source.position.z,
                            ),
                            r_right * r_floor,
                        ),
                        // Right + Ceiling (x=width, z=height)
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_right * r_ceiling,
                        ),
                        // Front + Floor (y=0, z=0)
                        (
                            Point3D::new(source.position.x, -source.position.y, -source.position.z),
                            r_front * r_floor,
                        ),
                        // Front + Ceiling (y=0, z=height)
                        (
                            Point3D::new(
                                source.position.x,
                                -source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_front * r_ceiling,
                        ),
                        // Back + Floor (y=depth, z=0)
                        (
                            Point3D::new(
                                source.position.x,
                                2.0 * room_depth - source.position.y,
                                -source.position.z,
                            ),
                            r_back * r_floor,
                        ),
                        // Back + Ceiling (y=depth, z=height)
                        (
                            Point3D::new(
                                source.position.x,
                                2.0 * room_depth - source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_back * r_ceiling,
                        ),
                    ];

                    for (image_src, refl_coeff) in &second_order_images {
                        let r_image = image_src.distance_to(point);
                        if r_image > 1e-6 {
                            let air_atten = self.air_absorption_factor(r_image, frequency);
                            total_pressure += greens_function_3d(r_image, k)
                                * amplitude
                                * refl_coeff
                                * air_atten
                                * phase_factor;
                        }
                    }
                }

                if reflection_order >= 3 {
                    // Third-order image sources (corners - 8 combinations)
                    // Each involves reflection off three walls
                    let third_order_images = [
                        // Left + Front + Floor
                        (
                            Point3D::new(
                                -source.position.x,
                                -source.position.y,
                                -source.position.z,
                            ),
                            r_left * r_front * r_floor,
                        ),
                        // Left + Front + Ceiling
                        (
                            Point3D::new(
                                -source.position.x,
                                -source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_left * r_front * r_ceiling,
                        ),
                        // Left + Back + Floor
                        (
                            Point3D::new(
                                -source.position.x,
                                2.0 * room_depth - source.position.y,
                                -source.position.z,
                            ),
                            r_left * r_back * r_floor,
                        ),
                        // Left + Back + Ceiling
                        (
                            Point3D::new(
                                -source.position.x,
                                2.0 * room_depth - source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_left * r_back * r_ceiling,
                        ),
                        // Right + Front + Floor
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                -source.position.y,
                                -source.position.z,
                            ),
                            r_right * r_front * r_floor,
                        ),
                        // Right + Front + Ceiling
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                -source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_right * r_front * r_ceiling,
                        ),
                        // Right + Back + Floor
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                2.0 * room_depth - source.position.y,
                                -source.position.z,
                            ),
                            r_right * r_back * r_floor,
                        ),
                        // Right + Back + Ceiling
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                2.0 * room_depth - source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_right * r_back * r_ceiling,
                        ),
                    ];

                    for (image_src, refl_coeff) in &third_order_images {
                        let r_image = image_src.distance_to(point);
                        if r_image > 1e-6 {
                            let air_atten = self.air_absorption_factor(r_image, frequency);
                            total_pressure += greens_function_3d(r_image, k)
                                * amplitude
                                * refl_coeff
                                * air_atten
                                * phase_factor;
                        }
                    }
                }
            }

            // Add edge diffraction contributions
            if self.edge_diffraction_enabled && !self.diffraction_edges.is_empty() {
                for edge in &self.diffraction_edges {
                    let diff_contrib = edge.diffraction_contribution(&source.position, point, k);
                    // Scale diffraction by source amplitude and phase
                    total_pressure += diff_contrib * amplitude * phase_factor;
                }
            }
        }

        total_pressure
    }

    /// Calculate field from a single source using the configured solver method
    /// This mirrors calculate_direct_field but for a single source only
    fn calculate_source_field(
        &self,
        source_idx: usize,
        point: &Point3D,
        frequency: f64,
    ) -> Complex64 {
        if source_idx >= self.sources.len() {
            return Complex64::new(0.0, 0.0);
        }

        let (room_width, room_depth, room_height) = self.get_room_dimensions();
        let method = self.config.solver.method.as_str();
        let source = &self.sources[source_idx];

        // BEM-based methods - use modal for single source
        if method == "bem" || method == "hybrid-bem" {
            return self
                .calculate_single_source_bem_or_hybrid(source_idx, point, frequency, method);
        }

        // Modal and modal-ISM hybrid methods
        if (method == "modal" || method == "hybrid")
            && let RoomGeometry::Rectangular(_) = &self.room_geometry
        {
            let amplitude = source.amplitude_towards(point, frequency);
            let phase_factor = source.phase_factor(frequency);

            let modal_pressure = calculate_modal_pressure(
                &source.position,
                point,
                frequency,
                room_width,
                room_depth,
                room_height,
                self.config.solver.speed_of_sound,
                self.config.solver.max_mode_order,
                self.config.solver.modal_damping,
            ) * amplitude
                * phase_factor;

            if method == "modal" {
                return modal_pressure;
            }

            // Hybrid: blend modal and ISM at the single broadband Schroeder
            // crossover (see `broadband_schroeder_frequency`).
            let schroeder_freq = self.broadband_schroeder_frequency();

            let ism_weight = hybrid_crossover_weight(
                frequency,
                schroeder_freq,
                self.config.solver.hybrid_crossover_width,
            );

            if ism_weight < 1e-6 {
                return modal_pressure;
            }

            let ism_pressure = self.calculate_single_source_ism(source_idx, point, frequency);
            return modal_pressure * (1.0 - ism_weight) + ism_pressure * ism_weight;
        }

        // Standard ISM calculation
        self.calculate_single_source_ism(source_idx, point, frequency)
    }

    /// Calculate field from a single source using BEM or hybrid BEM+ISM
    fn calculate_single_source_bem_or_hybrid(
        &self,
        source_idx: usize,
        point: &Point3D,
        frequency: f64,
        method: &str,
    ) -> Complex64 {
        let (room_width, room_depth, room_height) = self.get_room_dimensions();
        let source = &self.sources[source_idx];

        // For rectangular rooms, use modal analysis for the single source
        let low_freq_pressure = if let RoomGeometry::Rectangular(_) = &self.room_geometry {
            let amplitude = source.amplitude_towards(point, frequency);
            let phase_factor = source.phase_factor(frequency);

            calculate_modal_pressure(
                &source.position,
                point,
                frequency,
                room_width,
                room_depth,
                room_height,
                self.config.solver.speed_of_sound,
                self.config.solver.max_mode_order,
                self.config.solver.modal_damping,
            ) * amplitude
                * phase_factor
        } else {
            // Non-rectangular: use direct field approximation
            let k = self.wavenumber(frequency);
            let amplitude = source.amplitude_towards(point, frequency);
            let dist = source.position.distance_to(point);
            let phase_factor = source.phase_factor(frequency);
            greens_function_3d(dist, k) * amplitude * phase_factor
        };

        if method == "bem" {
            return low_freq_pressure;
        }

        // Hybrid-BEM: blend with ISM at high frequencies at the single
        // broadband Schroeder crossover (see `broadband_schroeder_frequency`).
        let schroeder_freq = self.broadband_schroeder_frequency();

        let ism_weight = hybrid_crossover_weight(
            frequency,
            schroeder_freq,
            self.config.solver.hybrid_crossover_width,
        );

        if ism_weight < 1e-6 {
            return low_freq_pressure;
        }

        if ism_weight > 1.0 - 1e-6 {
            return self.calculate_single_source_ism(source_idx, point, frequency);
        }

        let ism_pressure = self.calculate_single_source_ism(source_idx, point, frequency);
        low_freq_pressure * (1.0 - ism_weight) + ism_pressure * ism_weight
    }

    /// Calculate field from a single source using ISM
    fn calculate_single_source_ism(
        &self,
        source_idx: usize,
        point: &Point3D,
        frequency: f64,
    ) -> Complex64 {
        let k = self.wavenumber(frequency);
        let source = &self.sources[source_idx];
        let amplitude = source.amplitude_towards(point, frequency);
        let phase_factor = source.phase_factor(frequency);

        // Determine reflection order from solver method
        let reflection_order = match self.config.solver.method.as_str() {
            "direct" => 0,
            "image-source-1" => 1,
            "image-source-2" => 2,
            "image-source-3" => 3,
            "modal" => 0,
            "bem" => 0,
            "hybrid" => 2,
            "hybrid-bem" => 2,
            _ => 2,
        };

        let (room_width, room_depth, room_height) = self.get_room_dimensions();

        // Get frequency-dependent reflection coefficients
        let r_left = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Left,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_right = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Right,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_front = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Front,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_back = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Back,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_floor = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Floor,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );
        let r_ceiling = self.wall_materials.reflection_at_complex_with_medium(
            WallSurface::Ceiling,
            frequency,
            self.config.solver.speed_of_sound,
            self.config.solver.air_density,
        );

        let mut total_pressure = Complex64::new(0.0, 0.0);

        // Direct sound
        let r_direct = source.position.distance_to(point);
        let air_atten_direct = self.air_absorption_factor(r_direct, frequency);
        total_pressure +=
            greens_function_3d(r_direct, k) * amplitude * air_atten_direct * phase_factor;

        if reflection_order >= 1 {
            // First-order image sources - handle L-shaped rooms
            let first_order_images: Vec<(Point3D, Complex64)> = match &self.room_geometry {
                RoomGeometry::LShaped(l_room) => l_room
                    .get_first_order_images(&source.position)
                    .into_iter()
                    .filter_map(|(image, wall_name)| {
                        if !l_room.is_valid_image_source(&image, point, &source.position) {
                            return None;
                        }
                        let refl = match wall_name {
                            "left" => r_left,
                            "right" => r_right,
                            "front" => r_front,
                            "back" => r_back,
                            "floor" => r_floor,
                            "ceiling" => r_ceiling,
                            "step_horizontal" | "step_vertical" => (r_right + r_back) / 2.0,
                            _ => r_right,
                        };
                        Some((image, refl))
                    })
                    .collect(),
                RoomGeometry::Rectangular(_) => {
                    vec![
                        (
                            Point3D::new(-source.position.x, source.position.y, source.position.z),
                            r_left,
                        ),
                        (
                            Point3D::new(
                                2.0 * room_width - source.position.x,
                                source.position.y,
                                source.position.z,
                            ),
                            r_right,
                        ),
                        (
                            Point3D::new(source.position.x, -source.position.y, source.position.z),
                            r_front,
                        ),
                        (
                            Point3D::new(
                                source.position.x,
                                2.0 * room_depth - source.position.y,
                                source.position.z,
                            ),
                            r_back,
                        ),
                        (
                            Point3D::new(source.position.x, source.position.y, -source.position.z),
                            r_floor,
                        ),
                        (
                            Point3D::new(
                                source.position.x,
                                source.position.y,
                                2.0 * room_height - source.position.z,
                            ),
                            r_ceiling,
                        ),
                    ]
                }
            };

            for (image_src, refl_coeff) in first_order_images {
                let r_image = image_src.distance_to(point);
                if r_image > 1e-6 {
                    let air_atten = self.air_absorption_factor(r_image, frequency);
                    total_pressure += greens_function_3d(r_image, k)
                        * amplitude
                        * refl_coeff
                        * air_atten
                        * phase_factor;
                }
            }

            // Higher-order reflections for rectangular rooms only
            if reflection_order >= 2
                && let RoomGeometry::Rectangular(_) = &self.room_geometry
            {
                for nx in -2i32..=2 {
                    for ny in -2i32..=2 {
                        for nz in -2i32..=2 {
                            let order = nx.abs() + ny.abs() + nz.abs();
                            if order < 2 || order > reflection_order {
                                continue;
                            }

                            let image_x = if nx % 2 == 0 {
                                source.position.x + (nx as f64) * room_width
                            } else {
                                -source.position.x + (nx as f64 + 1.0) * room_width
                            };
                            let image_y = if ny % 2 == 0 {
                                source.position.y + (ny as f64) * room_depth
                            } else {
                                -source.position.y + (ny as f64 + 1.0) * room_depth
                            };
                            let image_z = if nz % 2 == 0 {
                                source.position.z + (nz as f64) * room_height
                            } else {
                                -source.position.z + (nz as f64 + 1.0) * room_height
                            };

                            let image_pos = Point3D::new(image_x, image_y, image_z);
                            let r_image = image_pos.distance_to(point);

                            if r_image > 1e-6 {
                                let refl_coeff = r_left.powi((nx.abs() + 1) / 2)
                                    * r_right.powi(nx.abs() / 2)
                                    * r_front.powi((ny.abs() + 1) / 2)
                                    * r_back.powi(ny.abs() / 2)
                                    * r_floor.powi((nz.abs() + 1) / 2)
                                    * r_ceiling.powi(nz.abs() / 2);

                                let air_atten = self.air_absorption_factor(r_image, frequency);
                                total_pressure += greens_function_3d(r_image, k)
                                    * amplitude
                                    * refl_coeff
                                    * air_atten
                                    * phase_factor;
                            }
                        }
                    }
                }
            }
        }

        total_pressure
    }

    fn listener_frequency_samples(
        &self,
        frequency: f64,
    ) -> Vec<(Complex64, f64, NumericalErrorEstimate)> {
        self.listening_positions
            .iter()
            .map(|position| {
                let pressure = self.calculate_direct_field(position, frequency);
                let spl = pressure_to_spl(pressure);
                let uncertainty = self
                    .numerical_error_estimate(position, frequency, pressure)
                    .unwrap_or_else(|| NumericalErrorEstimate {
                        relative: 0.0,
                        absolute: 0.0,
                        metric: "not_available".to_string(),
                        basis: "not_available_for_selected_method".to_string(),
                        confidence: "not_calibrated".to_string(),
                    });
                (pressure, spl, uncertainty)
            })
            .collect()
    }

    /// Run the full simulation and return JSON results
    #[wasm_bindgen]
    pub fn run_simulation(&self) -> Result<String, JsValue> {
        self.reset_bem_fallbacks();
        console_log!("Starting simulation...");

        // Hoisted listener gate: the per-frequency closures below index the
        // first listener sample. Validation normally guarantees a listener,
        // but chunked/direct reuse paths must fail as JsValue here rather
        // than panicking inside a worker thread.
        if self.listening_positions.is_empty() {
            return Err(js_error("simulation has no listening positions"));
        }

        let compute_ir = self.config.visualization.generate_impulse_response;
        let compute_binaural = self.config.visualization.binaural.enabled;

        // Precompute ear positions for binaural
        let (left_ear_pos, right_ear_pos) = if compute_binaural {
            let binaural_config = &self.config.visualization.binaural;
            let head_center = binaural_config
                .head_position
                .as_ref()
                .map(|p| Point3D::new(p.x, p.y, p.z))
                .unwrap_or(self.listening_position);
            calculate_ear_positions(
                &head_center,
                binaural_config.head_yaw,
                binaural_config.ear_spacing,
            )
        } else {
            (self.listening_position, self.listening_position)
        };

        // Precompute head center and yaw for HRTF
        let head_center_for_hrtf = self
            .config
            .visualization
            .binaural
            .head_position
            .as_ref()
            .map(|p| Point3D::new(p.x, p.y, p.z))
            .unwrap_or(self.listening_position);
        let head_yaw = self.config.visualization.binaural.head_yaw;

        console_log!(
            "Computing frequency response using {} threads...",
            rayon::current_num_threads()
        );

        // Frequency response: each frequency is independent, results
        // collected in order. The machine-adaptive gate parallelizes only
        // when the sweep merits pool dispatch on this machine; small sweeps
        // run the identical closure serially.
        let parallel_sweep = ParallelPolicy::from_env().parallel_sweep(
            self.frequencies
                .len()
                .saturating_mul(self.listening_positions.len().max(1))
                .saturating_mul(self.sources.len().max(1)),
        );
        let compute_point = |(idx, &freq): (usize, &f64)| {
            let listener_samples = self.listener_frequency_samples(freq);
            let (pressure, spl, uncertainty) = listener_samples
                .first()
                .cloned()
                .expect("listening positions checked non-empty before the sweep");
            let listener_spls = listener_samples
                .iter()
                .map(|(_, listener_spl, _)| *listener_spl)
                .collect::<Vec<_>>();
            let listener_uncertainties = listener_samples
                .iter()
                .map(|(_, _, listener_uncertainty)| listener_uncertainty.clone())
                .collect::<Vec<_>>();

            // Per-source SPL values
            let source_spls: Vec<f64> = (0..self.sources.len())
                .map(|src_idx| {
                    let p = self.calculate_source_field(src_idx, &self.listening_position, freq);
                    pressure_to_spl(p)
                })
                .collect();

            // Binaural computation if enabled
            //
            // Each source's HRTF gain applies to that source's own ear
            // pressure *before* summation (and via the solver-method
            // aware `calculate_source_field`, matching the main
            // response). Multiplying every gain onto the summed field
            // would raise single-source output to the power of the
            // source count.
            let binaural = if compute_binaural {
                let mut left_total = Complex64::new(0.0, 0.0);
                let mut right_total = Complex64::new(0.0, 0.0);
                for (src_idx, source) in self.sources.iter().enumerate() {
                    let (left_gain, right_gain) = approximate_hrtf_magnitude(
                        &source.position,
                        &head_center_for_hrtf,
                        head_yaw,
                        freq,
                    );
                    left_total += self.calculate_source_field(src_idx, &left_ear_pos, freq)
                        * Complex64::new(left_gain, 0.0);
                    right_total += self.calculate_source_field(src_idx, &right_ear_pos, freq)
                        * Complex64::new(right_gain, 0.0);
                }
                Some((left_total, right_total))
            } else {
                None
            };

            (
                idx,
                listener_spls,
                listener_uncertainties,
                spl,
                pressure,
                source_spls,
                binaural,
                uncertainty,
            )
        };
        let freq_results: Vec<_> = if parallel_sweep {
            self.frequencies
                .par_iter()
                .enumerate()
                .map(&compute_point)
                .collect()
        } else {
            self.frequencies
                .iter()
                .enumerate()
                .map(compute_point)
                .collect()
        };

        // Unpack parallel results into separate vectors (maintaining order)
        let mut combined_spl = Vec::with_capacity(self.frequencies.len());
        let mut complex_pressures = Vec::with_capacity(self.frequencies.len());
        let mut listener_spls: Vec<Vec<f64>> = self
            .listening_positions
            .iter()
            .map(|_| Vec::with_capacity(self.frequencies.len()))
            .collect();
        let mut listener_uncertainties: Vec<Vec<NumericalErrorEstimate>> = self
            .listening_positions
            .iter()
            .map(|_| Vec::with_capacity(self.frequencies.len()))
            .collect();
        let mut source_responses: Vec<Vec<f64>> = self
            .sources
            .iter()
            .map(|_| Vec::with_capacity(self.frequencies.len()))
            .collect();
        let mut left_ear_pressures = Vec::with_capacity(self.frequencies.len());
        let mut right_ear_pressures = Vec::with_capacity(self.frequencies.len());
        let mut frequency_uncertainty = Vec::with_capacity(self.frequencies.len());

        for (
            _idx,
            frequency_listener_spls,
            frequency_listener_uncertainties,
            spl,
            pressure,
            source_spls,
            binaural,
            uncertainty,
        ) in freq_results
        {
            combined_spl.push(spl);
            frequency_uncertainty.push(uncertainty);

            for (listener_index, listener_spl) in frequency_listener_spls.into_iter().enumerate() {
                listener_spls[listener_index].push(listener_spl);
            }
            for (listener_index, listener_uncertainty) in
                frequency_listener_uncertainties.into_iter().enumerate()
            {
                listener_uncertainties[listener_index].push(listener_uncertainty);
            }

            if compute_ir {
                complex_pressures.push(pressure);
            }

            for (src_idx, src_spl) in source_spls.into_iter().enumerate() {
                source_responses[src_idx].push(src_spl);
            }

            if let Some((left, right)) = binaural {
                left_ear_pressures.push(left);
                right_ear_pressures.push(right);
            }
        }

        console_log!("Frequency response computed");

        let source_responses_output: Vec<SourceResponse> = self
            .sources
            .iter()
            .zip(source_responses)
            .map(|(source, spl)| SourceResponse {
                source_name: source.name.clone(),
                spl,
            })
            .collect();

        let (horizontal_slices, vertical_slices) = if self.config.visualization.generate_slices {
            console_log!("Computing spatial slices...");
            self.compute_slices()
        } else {
            (None, None)
        };

        let room_output = self.build_room_output();

        // Calculate room modes and acoustics for rectangular rooms only
        let (room_modes, room_acoustics) = match &self.config.room {
            RoomGeometryConfig::Rectangular {
                width,
                depth,
                height,
            } => {
                let max_freq = self.frequencies.last().copied().unwrap_or(500.0);
                let modes = calculate_room_modes(
                    *width,
                    *depth,
                    *height,
                    self.config.solver.speed_of_sound,
                    max_freq,
                    10, // max order
                );
                // Calculate room acoustics at 1kHz (mid-frequency reference)
                let acoustics =
                    calculate_room_acoustics(*width, *depth, *height, &self.wall_materials, 1000.0);
                (Some(modes), Some(acoustics))
            }
            _ => (None, None), // L-shaped and complex rooms don't have simple calculations
        };

        // Calculate impulse response if enabled
        let impulse_response = if compute_ir && !complex_pressures.is_empty() {
            console_log!("Computing impulse response...");
            let ir_config = &self.config.visualization.impulse_response;

            // If duration not set, use RT60 + 100ms if available, otherwise default
            let ir_duration = ir_config
                .duration
                .or_else(|| room_acoustics.as_ref().map(|ra| ra.rt60_eyring + 0.1));

            let config_with_duration = ImpulseResponseConfig {
                duration: ir_duration,
                ..ir_config.clone()
            };
            let relative_uncertainty = frequency_uncertainty
                .iter()
                .any(|estimate| estimate.metric != "not_available")
                .then(|| {
                    frequency_uncertainty
                        .iter()
                        .map(|estimate| estimate.relative)
                        .collect::<Vec<_>>()
                });

            Some(
                calculate_impulse_response_with_uncertainty(
                    &self.frequencies,
                    &complex_pressures,
                    relative_uncertainty.as_deref(),
                    &config_with_duration,
                )
                .map_err(|error| js_error(error.to_string()))?,
            )
        } else {
            None
        };

        // Calculate binaural response if enabled
        let binaural_response = if compute_binaural && !left_ear_pressures.is_empty() {
            console_log!("Computing binaural response...");
            let binaural_config = &self.config.visualization.binaural;
            let head_center = binaural_config
                .head_position
                .as_ref()
                .map(|p| Point3D::new(p.x, p.y, p.z))
                .unwrap_or(self.listening_position);

            Some(
                calculate_binaural_response(
                    &self.frequencies,
                    &left_ear_pressures,
                    &right_ear_pressures,
                    binaural_config,
                    self.config.solver.speed_of_sound,
                    head_center,
                )
                .map_err(|error| js_error(error.to_string()))?,
            )
        } else {
            None
        };

        let sources_output: Vec<SourceOutputInfo> = self
            .sources
            .iter()
            .zip(self.config.sources.iter())
            .map(|(source, config)| {
                let crossover_str = match &config.crossover {
                    CrossoverConfig::FullRange => None,
                    CrossoverConfig::Lowpass { cutoff_freq, order } => {
                        Some(format!("LP {}Hz {}nd", cutoff_freq, order))
                    }
                    CrossoverConfig::Highpass { cutoff_freq, order } => {
                        Some(format!("HP {}Hz {}nd", cutoff_freq, order))
                    }
                    CrossoverConfig::Bandpass {
                        low_cutoff,
                        high_cutoff,
                        order,
                    } => Some(format!("BP {}-{}Hz {}nd", low_cutoff, high_cutoff, order)),
                };

                SourceOutputInfo {
                    name: source.name.clone(),
                    position: [source.position.x, source.position.y, source.position.z],
                    crossover: crossover_str,
                }
            })
            .collect();

        let listener_responses = self
            .listening_positions
            .iter()
            .enumerate()
            .map(|(listener_index, position)| ListenerResponse {
                position: [position.x, position.y, position.z],
                frequency_response: listener_spls[listener_index].clone(),
                frequency_uncertainty: listener_uncertainties[listener_index].clone(),
            })
            .collect();

        let results = SimulationResults {
            room: room_output,
            sources: sources_output,
            listening_position: [
                self.listening_position.x,
                self.listening_position.y,
                self.listening_position.z,
            ],
            listener_responses,
            frequencies: self.frequencies.clone(),
            frequency_response: combined_spl,
            frequency_uncertainty,
            source_responses: Some(source_responses_output),
            horizontal_slices,
            vertical_slices,
            solver: self.config.solver.method.clone(),
            provenance: self.provenance(),
            mesh_nodes: None,
            mesh_elements: None,
            metadata: Some(self.config.metadata.clone()),
            room_modes,
            room_acoustics,
            impulse_response,
            binaural_response,
        };

        console_log!("Simulation complete!");

        serde_json::to_string(&results)
            .map_err(|e| js_error(format!("JSON serialization error: {}", e)))
    }

    /// Compute a single frequency response (for progressive updates)
    #[wasm_bindgen]
    pub fn compute_frequency_point(&self, freq_index: usize) -> Result<String, JsValue> {
        if freq_index >= self.frequencies.len() {
            return Err(js_error("Frequency index out of bounds"));
        }

        let freq = self.frequencies[freq_index];
        self.reset_bem_fallbacks();
        if self.listening_positions.is_empty() {
            return Err(js_error("simulation has no listening positions"));
        }
        let listener_samples = self.listener_frequency_samples(freq);
        let (pressure, spl, uncertainty) = listener_samples
            .first()
            .cloned()
            .expect("listening positions checked non-empty above");
        let listeners = listener_samples
            .iter()
            .zip(self.listening_positions.iter())
            .map(
                |((listener_pressure, listener_spl, listener_uncertainty), position)| {
                    serde_json::json!({
                        "position": [position.x, position.y, position.z],
                        "spl": listener_spl,
                        "pressure_real": listener_pressure.re,
                        "pressure_imag": listener_pressure.im,
                        "uncertainty": listener_uncertainty,
                    })
                },
            )
            .collect::<Vec<_>>();

        let result = serde_json::json!({
            "frequency": freq,
            "spl": spl,
            "pressure_real": pressure.re,
            "pressure_imag": pressure.im,
            "listeners": listeners,
            "index": freq_index,
            "total": self.frequencies.len(),
            "provenance": self.provenance_with_error(Some(uncertainty)),
        });

        serde_json::to_string(&result).map_err(|e| js_error(format!("JSON error: {}", e)))
    }

    /// Compute a horizontal slice at a specific frequency
    #[wasm_bindgen]
    pub fn compute_horizontal_slice(&self, frequency: f64) -> Result<String, JsValue> {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(js_error("Slice frequency must be finite and positive"));
        }
        console_log!("Computing horizontal slice at {:.1} Hz", frequency);

        self.reset_bem_fallbacks();
        let resolution = self.config.visualization.slice_resolution;
        let (room_width, room_depth, _) = self.get_room_dimensions();

        let x_points = lin_space(0.0, room_width, resolution);
        let y_points = lin_space(0.0, room_depth, resolution);

        let mut spl_values = Vec::with_capacity(resolution * resolution);
        let mut valid = Vec::with_capacity(resolution * resolution);

        for &y in &y_points {
            for &x in &x_points {
                let point = Point3D::new(x, y, self.listening_position.z);
                let is_valid = self.point_inside_room(&point);
                let pressure = if is_valid {
                    self.calculate_direct_field(&point, frequency)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                spl_values.push(if is_valid {
                    pressure_to_spl(pressure)
                } else {
                    -100.0
                });
                valid.push(is_valid);
            }
        }

        let result = SliceOutput {
            frequency,
            x: x_points,
            y: y_points,
            z: None,
            spl: spl_values,
            valid,
            provenance: self.provenance(),
            shape: [resolution, resolution],
        };

        serde_json::to_string(&result).map_err(|e| js_error(format!("JSON error: {}", e)))
    }

    /// Compute a vertical slice at a specific frequency
    #[wasm_bindgen]
    pub fn compute_vertical_slice(&self, frequency: f64) -> Result<String, JsValue> {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(js_error("Slice frequency must be finite and positive"));
        }
        console_log!("Computing vertical slice at {:.1} Hz", frequency);

        self.reset_bem_fallbacks();
        let resolution = self.config.visualization.slice_resolution;
        let (room_width, _, room_height) = self.get_room_dimensions();

        let x_points = lin_space(0.0, room_width, resolution);
        let z_points = lin_space(0.0, room_height, resolution);

        let mut spl_values = Vec::with_capacity(resolution * resolution);
        let mut valid = Vec::with_capacity(resolution * resolution);

        for &z in &z_points {
            for &x in &x_points {
                let point = Point3D::new(x, self.listening_position.y, z);
                let is_valid = self.point_inside_room(&point);
                let pressure = if is_valid {
                    self.calculate_direct_field(&point, frequency)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                spl_values.push(if is_valid {
                    pressure_to_spl(pressure)
                } else {
                    -100.0
                });
                valid.push(is_valid);
            }
        }

        let result = SliceOutput {
            frequency,
            x: x_points,
            y: Vec::new(),
            z: Some(z_points),
            spl: spl_values,
            valid,
            provenance: self.provenance(),
            shape: [resolution, resolution],
        };

        serde_json::to_string(&result).map_err(|e| js_error(format!("JSON error: {}", e)))
    }

    /// Get room info as JSON
    #[wasm_bindgen]
    pub fn get_room_info(&self) -> String {
        let room_output = self.build_room_output();
        serde_json::to_string(&room_output).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get configuration as JSON
    #[wasm_bindgen]
    pub fn get_config(&self) -> String {
        serde_json::to_string(&self.config).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get number of frequencies
    #[wasm_bindgen]
    pub fn num_frequencies(&self) -> usize {
        self.frequencies.len()
    }

    /// Get number of sources
    #[wasm_bindgen]
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Get the stable hash of the validated configuration used by this
    /// simulator. Consumers can use it as a cache key for geometry, modes,
    /// slices, and frequency-response results.
    #[wasm_bindgen]
    pub fn get_model_hash(&self) -> String {
        self.model_hash.clone()
    }

    /// Initialize chunked computation
    ///
    /// Call this before using compute_chunk() to reset state
    #[wasm_bindgen]
    pub fn init_chunked_compute(&mut self) {
        self.reset_bem_fallbacks();
        self.chunked_frequencies_completed = 0;
        self.chunked_pressures = Some(Vec::with_capacity(self.frequencies.len()));
        self.chunked_listener_pressures = Some(
            self.listening_positions
                .iter()
                .map(|_| Vec::with_capacity(self.frequencies.len()))
                .collect(),
        );
        self.chunked_in_progress = true;
        self.chunked_cancelled = false;
        console_log!(
            "Chunked compute initialized: {} frequencies",
            self.frequencies.len()
        );
    }

    /// Compute a chunk of frequencies for progressive updates
    ///
    /// Returns JSON with progress and partial results:
    /// ```json
    /// {
    ///   "progress": 25.0,
    ///   "frequencies_completed": 32,
    ///   "total_frequencies": 128,
    ///   "done": false,
    ///   "partial_spl": [...]  // Accumulated prefix available every chunk
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn compute_chunk(&mut self, chunk_size: usize) -> String {
        if chunk_size == 0 {
            return serde_json::json!({
                "error": "Chunk size must be positive",
                "done": false
            })
            .to_string();
        }
        // Keep a hostile JS request from turning one incremental call into an
        // unbounded allocation. The caller can continue with another chunk.
        let chunk_size = chunk_size.min(10_000);
        if self.chunked_cancelled {
            return serde_json::json!({
                "error": "Chunked compute was cancelled",
                "cancelled": true,
                "done": false
            })
            .to_string();
        }
        if !self.chunked_in_progress {
            // Auto-initialize if not done
            self.init_chunked_compute();
        }

        let start_idx = self.chunked_frequencies_completed;
        let end_idx = (start_idx + chunk_size).min(self.frequencies.len());

        if start_idx >= self.frequencies.len() {
            // Already done
            return self.finalize_chunked_compute();
        }

        if self.chunked_pressures.is_none() || self.chunked_listener_pressures.is_none() {
            return serde_json::json!({
                "error": "Chunked compute not initialized",
                "done": true
            })
            .to_string();
        }

        console_log!(
            "Computing frequencies {}-{} of {}",
            start_idx,
            end_idx,
            self.frequencies.len()
        );

        // Compute the chunk - collect results first to avoid borrow issues
        let mut computed_listener_spl: Vec<Vec<f64>> = self
            .listening_positions
            .iter()
            .map(|_| Vec::with_capacity(end_idx - start_idx))
            .collect();

        // Clone the needed data to avoid borrow issues
        let listening_positions = self.listening_positions.clone();
        let frequencies = self.frequencies.clone();

        for freq in frequencies.iter().take(end_idx).skip(start_idx).copied() {
            for (listener_index, listening_pos) in listening_positions.iter().enumerate() {
                let pressure = self.calculate_direct_field(listening_pos, freq);
                computed_listener_spl[listener_index].push(pressure_to_spl(pressure));
            }
        }

        // Now update the state
        if let Some(pressures) = &mut self.chunked_pressures {
            pressures.extend(computed_listener_spl.first().cloned().unwrap_or_default());
        }
        if let Some(listener_pressures) = &mut self.chunked_listener_pressures {
            for (listener_index, computed_spl) in computed_listener_spl.into_iter().enumerate() {
                listener_pressures[listener_index].extend(computed_spl);
            }
        }

        self.chunked_frequencies_completed = end_idx;

        let total = self.frequencies.len();
        let progress = (end_idx as f64 / total as f64) * 100.0;
        let done = end_idx >= total;

        if done {
            self.chunked_in_progress = false;
        }

        let result = serde_json::json!({
            "progress": progress,
            "frequencies_completed": end_idx,
            "total_frequencies": total,
            "done": done,
            // Return the accumulated prefix so callers can render a first
            // result while later chunks are still being computed. The
            // `done` flag remains the sole completion signal.
            "partial_spl": self.chunked_pressures.as_deref().unwrap_or(&[]),
            "partial_listener_spl": self
                .chunked_listener_pressures
                .as_deref()
                .unwrap_or(&[]),
            "provenance": self.provenance(),
        });

        result.to_string()
    }

    /// Get final results after all chunks complete
    ///
    /// Returns the full frequency response as JSON
    #[wasm_bindgen]
    pub fn finalize_chunked_compute(&self) -> String {
        let total = self.frequencies.len();

        if self.chunked_cancelled {
            return serde_json::json!({
                "error": "Chunked compute was cancelled",
                "cancelled": true,
                "done": false,
                "frequencies_completed": self.chunked_frequencies_completed,
                "total_frequencies": total,
                // A cancellation preserves the prefix already computed. This
                // lets the browser retain its partial rendering while making
                // it unambiguous that the response is not final.
                "partial_spl": self.chunked_pressures.as_deref().unwrap_or(&[]),
                "partial_listener_spl": self
                    .chunked_listener_pressures
                    .as_deref()
                    .unwrap_or(&[]),
                "provenance": self.provenance(),
            })
            .to_string();
        }

        // If we have computed pressures, return them
        if let Some(pressures) = &self.chunked_pressures {
            if pressures.len() != self.chunked_frequencies_completed
                || self.chunked_frequencies_completed > total
            {
                return serde_json::json!({
                    "error": "Chunked compute state is inconsistent",
                    "done": false,
                    "frequencies_completed": self.chunked_frequencies_completed,
                    "total_frequencies": total,
                })
                .to_string();
            }
            let listener_pressures = match &self.chunked_listener_pressures {
                Some(listener_pressures)
                    if listener_pressures.len() == self.listening_positions.len()
                        && listener_pressures
                            .iter()
                            .all(|values| values.len() == self.chunked_frequencies_completed) =>
                {
                    listener_pressures
                }
                _ => {
                    return serde_json::json!({
                        "error": "Chunked listener state is inconsistent",
                        "done": false,
                        "frequencies_completed": self.chunked_frequencies_completed,
                        "total_frequencies": total,
                    })
                    .to_string();
                }
            };
            if self.chunked_frequencies_completed < total {
                return serde_json::json!({
                    "error": "Chunked compute is incomplete; call compute_chunk until done",
                    "done": false,
                    "frequencies_completed": self.chunked_frequencies_completed,
                    "total_frequencies": total,
                    "partial_spl": pressures,
                    "partial_listener_spl": listener_pressures,
                    "provenance": self.provenance(),
                })
                .to_string();
            }
            let progress = if total > 0 {
                (pressures.len() as f64 / total as f64) * 100.0
            } else {
                100.0
            };

            let result = serde_json::json!({
                "progress": progress,
                "frequencies_completed": pressures.len(),
                "total_frequencies": total,
                "done": true,
                "frequencies": self.frequencies,
                "spl": pressures,
                "listener_spl": listener_pressures,
                "provenance": self.provenance(),
            });

            result.to_string()
        } else {
            // No chunked data - compute synchronously
            let mut listener_spl: Vec<Vec<f64>> = self
                .listening_positions
                .iter()
                .map(|_| Vec::with_capacity(self.frequencies.len()))
                .collect();
            for freq in &self.frequencies {
                for (listener_index, position) in self.listening_positions.iter().enumerate() {
                    let pressure = self.calculate_direct_field(position, *freq);
                    listener_spl[listener_index].push(pressure_to_spl(pressure));
                }
            }

            let result = serde_json::json!({
                "progress": 100.0,
                "frequencies_completed": total,
                "total_frequencies": total,
                "done": true,
                "frequencies": self.frequencies,
                "spl": listener_spl.first().cloned().unwrap_or_default(),
                "listener_spl": listener_spl,
                "provenance": self.provenance(),
            });

            result.to_string()
        }
    }

    /// Get chunked computation progress
    #[wasm_bindgen]
    pub fn get_chunk_progress(&self) -> String {
        let completed = self.chunked_frequencies_completed;
        let total = self.frequencies.len();
        let progress = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            100.0
        };

        serde_json::json!({
            "progress": progress,
            "frequencies_completed": completed,
            "total_frequencies": total,
            "in_progress": self.chunked_in_progress,
            "cancelled": self.chunked_cancelled,
        })
        .to_string()
    }

    /// Cancel the current progressive computation. Partial results remain
    /// available for inspection, but are never reported as complete.
    #[wasm_bindgen]
    pub fn cancel_chunked_compute(&mut self) {
        // Completion is terminal: a late cancellation request must not turn
        // an already complete result into an error response.
        if self.chunked_in_progress {
            self.chunked_cancelled = true;
            self.chunked_in_progress = false;
        }
    }

    fn get_room_dimensions(&self) -> (f64, f64, f64) {
        match &self.room_geometry {
            RoomGeometry::Rectangular(r) => (r.width, r.depth, r.height),
            RoomGeometry::LShaped(r) => (r.width1, r.depth1 + r.depth2, r.height),
        }
    }

    fn build_room_output(&self) -> RoomOutput {
        let edges = self.room_geometry.get_edges();
        let edges_arrays: Vec<[[f64; 3]; 2]> = edges
            .iter()
            .map(|(p1, p2)| [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])
            .collect();

        match &self.room_geometry {
            RoomGeometry::Rectangular(r) => RoomOutput {
                width: r.width,
                depth: r.depth,
                height: r.height,
                room_type: Some("rectangular".to_string()),
                edges: edges_arrays,
            },
            RoomGeometry::LShaped(r) => RoomOutput {
                width: r.width1,
                depth: r.depth1 + r.depth2,
                height: r.height,
                room_type: Some("lshaped".to_string()),
                edges: edges_arrays,
            },
        }
    }

    fn compute_slices(&self) -> (Option<Vec<SliceOutput>>, Option<Vec<SliceOutput>>) {
        let resolution = self.config.visualization.slice_resolution;
        let (room_width, room_depth, room_height) = self.get_room_dimensions();

        let x_points = lin_space(0.0, room_width, resolution);
        let y_points = lin_space(0.0, room_depth, resolution);
        let z_points = lin_space(0.0, room_height, resolution);

        let freq_indices: Vec<usize> =
            if self.config.visualization.slice_frequency_indices.is_empty() {
                let step = (self.frequencies.len() / 10).max(1);
                (0..self.frequencies.len()).step_by(step).collect()
            } else {
                self.config
                    .visualization
                    .slice_frequency_indices
                    .iter()
                    .filter(|&&i| i < self.frequencies.len())
                    .copied()
                    .collect()
            };

        console_log!(
            "Computing {} spatial slices in parallel...",
            freq_indices.len()
        );

        // Slices share the method-aware `calculate_direct_field` with the
        // main response (not a direct-only path); each frequency slice is
        // independent. The machine-adaptive gate parallelizes only when the
        // cell count merits pool dispatch on this machine; small grids run
        // the identical closure serially.
        let parallel_slices = ParallelPolicy::from_env().parallel_slices(
            freq_indices
                .len()
                .saturating_mul(resolution)
                .saturating_mul(resolution),
        );
        let compute_slice = |&freq_idx: &usize| {
            let freq = self.frequencies[freq_idx];

            // Horizontal slice - parallelize the grid computation
            // For L-shaped rooms, points outside the room boundary get very low SPL
            let h_values: Vec<(f64, bool)> = y_points
                .iter()
                .flat_map(|&y| {
                    x_points.iter().map(move |&x| {
                        let point = Point3D::new(x, y, self.listening_position.z);
                        // Check if point is inside the room
                        if !self.point_inside_room(&point) {
                            return (-100.0, false); // Grid placeholder outside room
                        }
                        let pressure = self.calculate_direct_field(&point, freq);
                        (pressure_to_spl(pressure), true)
                    })
                })
                .collect();
            let (h_spl, h_valid): (Vec<_>, Vec<_>) = h_values.into_iter().unzip();

            let h_slice = SliceOutput {
                frequency: freq,
                x: x_points.clone(),
                y: y_points.clone(),
                z: None,
                spl: h_spl,
                valid: h_valid,
                provenance: self.provenance(),
                shape: [resolution, resolution],
            };

            // Vertical slice - parallelize the grid computation
            // For L-shaped rooms, points outside the room boundary get very low SPL
            let v_values: Vec<(f64, bool)> = z_points
                .iter()
                .flat_map(|&z| {
                    x_points.iter().map(move |&x| {
                        let point = Point3D::new(x, self.listening_position.y, z);
                        // Check if point is inside the room
                        if !self.point_inside_room(&point) {
                            return (-100.0, false); // Grid placeholder outside room
                        }
                        let pressure = self.calculate_direct_field(&point, freq);
                        (pressure_to_spl(pressure), true)
                    })
                })
                .collect();
            let (v_spl, v_valid): (Vec<_>, Vec<_>) = v_values.into_iter().unzip();

            let v_slice = SliceOutput {
                frequency: freq,
                x: x_points.clone(),
                y: Vec::new(),
                z: Some(z_points.clone()),
                spl: v_spl,
                valid: v_valid,
                provenance: self.provenance(),
                shape: [resolution, resolution],
            };

            (h_slice, v_slice)
        };
        let slices: Vec<(SliceOutput, SliceOutput)> = if parallel_slices {
            freq_indices.par_iter().map(&compute_slice).collect()
        } else {
            freq_indices.iter().map(compute_slice).collect()
        };

        // Unzip into separate vectors
        let (horizontal_slices, vertical_slices): (Vec<_>, Vec<_>) = slices.into_iter().unzip();

        console_log!("Spatial slices computed");

        (Some(horizontal_slices), Some(vertical_slices))
    }
}

fn simulation_provenance(
    method: &str,
    model_hash: &str,
    bem_fallback_count: usize,
    bem_fallback_reason: Option<String>,
    scattering_object_count: usize,
) -> SimulationProvenance {
    serialization::simulation_provenance(
        method,
        model_hash,
        bem_fallback_count,
        bem_fallback_reason,
        scattering_object_count,
    )
}

fn simulation_model_hash(config: &SimulationConfig) -> String {
    serialization::simulation_model_hash(config)
}

// ============================================================================
// Standalone utility functions for JS
// ============================================================================

/// Create a default configuration JSON
#[wasm_bindgen]
pub fn create_default_config() -> String {
    serialization::create_default_config()
}

/// Get a list of available wall material presets
#[wasm_bindgen]
pub fn get_material_presets() -> String {
    let presets = vec![
        (
            "concrete",
            "Concrete/Brick (painted)",
            [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
        ),
        (
            "brick",
            "Unpainted Brick",
            [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],
        ),
        (
            "drywall",
            "Drywall/Gypsum",
            [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
        ),
        (
            "plaster",
            "Plaster on Brick",
            [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
        ),
        (
            "glass",
            "Glass (large pane)",
            [0.18, 0.06, 0.04, 0.03, 0.02, 0.02],
        ),
        (
            "wood_thin",
            "Wood Paneling (thin)",
            [0.42, 0.21, 0.10, 0.08, 0.06, 0.06],
        ),
        (
            "wood_thick",
            "Heavy Wood/Door",
            [0.14, 0.10, 0.06, 0.08, 0.10, 0.10],
        ),
        (
            "carpet_thin",
            "Carpet (thin)",
            [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
        ),
        (
            "carpet_thick",
            "Carpet (heavy)",
            [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
        ),
        (
            "acoustic_tile",
            "Acoustic Ceiling Tiles",
            [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
        ),
        (
            "curtains",
            "Curtains/Drapes",
            [0.07, 0.31, 0.49, 0.75, 0.70, 0.60],
        ),
        (
            "acoustic_foam",
            "Acoustic Foam",
            [0.08, 0.25, 0.60, 0.90, 0.95, 0.90],
        ),
        (
            "hardwood",
            "Hardwood Floor",
            [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
        ),
        (
            "concrete_floor",
            "Concrete Floor",
            [0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
        ),
    ];

    let result: Vec<serde_json::Value> = presets
        .iter()
        .map(|(id, name, absorption)| {
            serde_json::json!({
                "id": id,
                "name": name,
                "absorption": {
                    "125": absorption[0],
                    "250": absorption[1],
                    "500": absorption[2],
                    "1000": absorption[3],
                    "2000": absorption[4],
                    "4000": absorption[5],
                }
            })
        })
        .collect();

    serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string())
}

/// Calculate room modes for a rectangular room (standalone WASM function)
///
/// Returns JSON with mode information:
/// - frequency: Resonant frequency in Hz
/// - indices: [n, m, p] mode indices
/// - mode_type: "axial", "tangential", or "oblique"
/// - description: Human-readable description
///
/// # Arguments
/// * `width` - Room width (X dimension) in meters
/// * `depth` - Room depth (Y dimension) in meters
/// * `height` - Room height (Z dimension) in meters
/// * `speed_of_sound` - Speed of sound in m/s (typically 343)
/// * `max_frequency` - Maximum frequency to calculate modes up to (Hz)
#[wasm_bindgen]
pub fn get_room_modes(
    width: f64,
    depth: f64,
    height: f64,
    speed_of_sound: f64,
    max_frequency: f64,
) -> String {
    let modes = calculate_room_modes(width, depth, height, speed_of_sound, max_frequency, 10);
    serde_json::to_string(&modes).unwrap_or_else(|_| "[]".to_string())
}

/// Get the Schroeder frequency for a room
///
/// The Schroeder frequency marks the transition between modal behavior (below)
/// and statistical behavior (above). Below this frequency, individual room modes
/// dominate the response. Above it, the modal density is high enough for
/// statistical acoustics methods to be valid.
///
/// Formula: f_s = 2000 * sqrt(RT60 / V)
/// Where RT60 is reverberation time (seconds) and V is room volume (m³)
///
/// For rooms with unknown RT60, a rough estimate can be made:
/// f_s ≈ 2000 * sqrt(0.5 / V) for typical furnished rooms
#[wasm_bindgen]
pub fn get_schroeder_frequency(volume: f64, rt60: f64) -> f64 {
    2000.0 * (rt60 / volume).sqrt()
}

/// Calculate RT60 and other room acoustics metrics
///
/// Returns JSON with:
/// - rt60_sabine: Sabine reverberation time (seconds)
/// - rt60_eyring: Eyring reverberation time (seconds) - more accurate for absorptive rooms
/// - volume: Room volume (m³)
/// - surface_area: Total surface area (m²)
/// - average_alpha: Average absorption coefficient
/// - total_absorption: Total absorption in sabins (m²)
/// - schroeder_frequency: Transition frequency from modal to statistical behavior (Hz)
/// - critical_distance: Distance where direct and reverberant fields are equal (m)
///
/// # Arguments
/// * `width` - Room width (X dimension) in meters
/// * `depth` - Room depth (Y dimension) in meters
/// * `height` - Room height (Z dimension) in meters
/// * `average_absorption` - Average absorption coefficient (0.0 to 1.0)
/// * `frequency` - Reference frequency for absorption (typically 1000 Hz)
#[wasm_bindgen]
pub fn get_rt60(width: f64, depth: f64, height: f64, average_absorption: f64) -> String {
    let volume = width * depth * height;
    let surface_area = 2.0 * (width * depth + width * height + depth * height);
    let total_absorption = average_absorption * surface_area;

    let rt60_sab = rt60_sabine(volume, total_absorption);
    let rt60_eyr = rt60_eyring(volume, surface_area, average_absorption);
    let schroeder_freq = 2000.0 * (rt60_eyr / volume).sqrt();
    let crit_dist = critical_distance(volume, rt60_eyr);

    let result = RoomAcoustics {
        rt60_sabine: rt60_sab,
        rt60_eyring: rt60_eyr,
        volume,
        surface_area,
        average_alpha: average_absorption,
        total_absorption,
        schroeder_frequency: schroeder_freq,
        critical_distance: crit_dist,
    };

    serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
}

/// Compute impulse response from frequency response data
///
/// Takes JSON with:
/// - frequencies: array of frequency values (Hz)
/// - magnitudes: array of magnitude values (linear, not dB)
/// - phases: array of phase values (radians)
/// - config: optional ImpulseResponseConfig object
///
/// Returns JSON with impulse response data
#[wasm_bindgen]
pub fn compute_impulse_response(input_json: &str) -> String {
    #[derive(Deserialize)]
    struct IrInput {
        frequencies: Vec<f64>,
        magnitudes: Vec<f64>,
        phases: Vec<f64>,
        #[serde(default)]
        config: Option<ImpulseResponseConfig>,
    }

    if input_json.len() > MAX_IMPULSE_INPUT_JSON_BYTES {
        return serde_json::json!({
            "error": "Impulse-response JSON exceeds the 32 MiB limit"
        })
        .to_string();
    }

    let input: IrInput = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => return format!(r#"{{"error": "Invalid JSON: {}"}}"#, e),
    };

    if input.frequencies.len() != input.magnitudes.len()
        || input.frequencies.len() != input.phases.len()
    {
        return r#"{"error": "frequencies, magnitudes, and phases arrays must have same length"}"#
            .to_string();
    }

    if input.frequencies.len() > MAX_FREQUENCY_POINTS {
        return serde_json::json!({
            "error": format!(
                "frequency response contains too many points (maximum {MAX_FREQUENCY_POINTS})"
            )
        })
        .to_string();
    }
    if input
        .magnitudes
        .iter()
        .any(|magnitude| !magnitude.is_finite() || *magnitude < 0.0)
        || input.phases.iter().any(|phase| !phase.is_finite())
    {
        return r#"{"error": "magnitudes must be finite and non-negative; phases must be finite"}"#
            .to_string();
    }

    // Build complex frequency response from magnitude and phase
    let complex_response: Vec<Complex64> = input
        .magnitudes
        .iter()
        .zip(input.phases.iter())
        .map(|(&mag, &phase)| Complex64::from_polar(mag, phase))
        .collect();

    let config = input.config.unwrap_or_default();
    let ir = match calculate_impulse_response(&input.frequencies, &complex_response, &config) {
        Ok(ir) => ir,
        Err(error) => return serde_json::json!({ "error": error.to_string() }).to_string(),
    };

    serde_json::to_string(&ir)
        .unwrap_or_else(|_| r#"{"error": "Serialization failed"}"#.to_string())
}

/// Validate a configuration JSON and return any errors
#[wasm_bindgen]
pub fn validate_config(config_json: &str) -> String {
    if config_json.len() > MAX_CONFIG_JSON_BYTES {
        return serde_json::json!({
            "valid": false,
            "error": "Configuration JSON exceeds the 16 MiB limit"
        })
        .to_string();
    }
    match serde_json::from_str::<SimulationConfig>(config_json) {
        Ok(config) => match validate_simulation_config(&config) {
            Ok(()) => serde_json::json!({
                "valid": true,
                "warnings": [],
                "estimated_working_set_bytes": estimated_working_set_bytes(&config),
            })
            .to_string(),
            Err(errors) => serde_json::json!({"valid": false, "warnings": errors}).to_string(),
        },
        Err(e) => serde_json::json!({"valid": false, "error": e.to_string()}).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_policy_uses_active_pool_size() {
        for threads in [1, 2] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut policy = ParallelPolicy::from_env();
                    assert_eq!(policy.threads, threads);
                    // Keep the dispatch assertion independent of process env.
                    policy.force_serial = false;
                    policy.sweep_override = None;
                    assert_eq!(policy.parallel_sweep(16), threads > 1);
                });
        }
    }

    #[test]
    fn parallel_policy_branch_selection() {
        // A single-threaded pool is always serial, even far above threshold.
        let serial = ParallelPolicy {
            threads: 1,
            force_serial: false,
            sweep_override: None,
            slices_override: None,
        };
        assert!(!serial.parallel_sweep(usize::MAX));
        assert!(!serial.parallel_slices(usize::MAX));

        // The serial escape hatch wins over worker count and overrides.
        let forced = ParallelPolicy {
            threads: 8,
            force_serial: true,
            sweep_override: Some(1),
            slices_override: Some(1),
        };
        assert!(!forced.parallel_sweep(usize::MAX));
        assert!(!forced.parallel_slices(usize::MAX));

        // Absolute overrides replace the per-thread grain on both paths.
        let overridden = ParallelPolicy {
            threads: 8,
            force_serial: false,
            sweep_override: Some(100),
            slices_override: Some(100),
        };
        assert!(!overridden.parallel_sweep(99));
        assert!(overridden.parallel_sweep(100));
        assert!(!overridden.parallel_slices(99));
        assert!(overridden.parallel_slices(100));

        // Default grain: threshold is items_per_thread * threads.
        let default = ParallelPolicy {
            threads: 4,
            force_serial: false,
            sweep_override: None,
            slices_override: None,
        };
        assert!(!default.parallel_sweep(SWEEP_ITEMS_PER_THREAD * 4 - 1));
        assert!(default.parallel_sweep(SWEEP_ITEMS_PER_THREAD * 4));
        assert!(!default.parallel_slices(SLICE_CELLS_PER_THREAD * 4 - 1));
        assert!(default.parallel_slices(SLICE_CELLS_PER_THREAD * 4));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn js_error_surfaces_failure_as_err_off_wasm() {
        // Native builds cannot carry JS strings through wasm-bindgen, and even
        // comparing JsValue payloads requires the JS `===` import (which
        // panics off-wasm). The only natively observable contract is therefore
        // `Err` versus `Ok`: invalid input must surface as `Err` carrying the
        // `js_error` sentinel, never as a panic or a silent success.
        assert!(RoomSimulatorWasm::new("not json").is_err());
        assert!(RoomSimulatorWasm::new("{}").is_err());
        RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
    }

    #[test]
    fn complex_material_reflection_preserves_legacy_magnitude_and_adds_phase() {
        let legacy = WallMaterial::new("legacy", [0.25; 6]);
        let phased = WallMaterial::new_with_phase("phased", [0.25; 6], [PI / 2.0; 6]);
        let legacy_reflection = legacy.reflection_at_frequency_complex(500.0);
        let phased_reflection = phased.reflection_at_frequency_complex(500.0);

        assert!((legacy_reflection.re - 0.75_f64.sqrt()).abs() < 1e-12);
        assert!(legacy_reflection.im.abs() < 1e-12);
        assert!(phased_reflection.re.abs() < 1e-12);
        assert!((phased_reflection.im - 0.75_f64.sqrt()).abs() < 1e-12);
        assert!((phased.reflection_at_frequency(500.0) - legacy_reflection.norm()).abs() < 1e-12);
    }

    #[test]
    fn impedance_material_uses_native_normal_incidence_coefficient() {
        let material = WallMaterial::new_with_impedance(
            "impedance",
            [0.2; 6],
            [0.0; 6],
            ImpedanceBands {
                real: [2.0 * 1.2 * 343.0; 6],
                imag: [1.0 * 1.2 * 343.0; 6],
            },
        );
        let expected = {
            let z0 = 1.2 * 343.0;
            let z = Complex64::new(2.0 * z0, z0);
            (z - Complex64::new(z0, 0.0)) / (z + Complex64::new(z0, 0.0))
        };
        let actual = material.reflection_at_frequency_complex(1_000.0);
        assert!((actual - expected).norm() < 1e-12);
        let alternate = material
            .reflection_at_frequency_complex_with_reference_impedance(1_000.0, 1.21 * 344.0);
        assert!((actual - alternate).norm() > 1.0e-6);
    }

    #[test]
    fn provenance_distinguishes_bem_fallback_occurrence() {
        let solved = simulation_provenance("bem", "model", 0, None, 0);
        assert_eq!(solved.bem_fallback_count, 0);
        assert!(solved.bem_fallback_reason.is_none());
        assert!(
            solved
                .validity
                .iter()
                .any(|entry| entry.contains("mesh quality"))
        );
        let solved_json = serde_json::to_value(&solved).expect("provenance JSON");
        assert_eq!(solved_json["bem_fallback_count"], 0);
        assert!(solved_json.get("bem_fallback_reason").is_none());

        let fallback = simulation_provenance(
            "bem",
            "model",
            3,
            Some("BEM failed at 100 Hz; used ISM fallback: singular system".to_string()),
            0,
        );
        assert_eq!(fallback.bem_fallback_count, 3);
        assert_eq!(fallback.fidelity, "fallback_approximate");
        assert_eq!(fallback.uncertainty, "fallback_observed");
        assert!(
            fallback
                .bem_fallback_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("ISM fallback"))
        );
    }

    #[test]
    fn provenance_discloses_non_bem_scattering_object_limitation() {
        let ism = simulation_provenance("image-source-2", "model", 0, None, 2);
        assert!(ism.validity.iter().any(|entry| {
            entry.contains("2 configured scattering object(s)") && entry.contains("BEM")
        }));

        let bem = simulation_provenance("bem", "model", 0, None, 2);
        assert!(
            bem.components
                .iter()
                .any(|entry| { entry.contains("2 interior scattering object(s) included") })
        );
    }

    fn impulse_response_fixture(size: usize) -> Result<ImpulseResponse, ImpulseResponseError> {
        let config = ImpulseResponseConfig {
            fft_size: Some(size),
            ..Default::default()
        };
        calculate_impulse_response(
            &[20.0, 1_000.0, 20_000.0],
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, -0.1),
                Complex64::new(0.1, 0.0),
            ],
            &config,
        )
    }

    #[test]
    fn impulse_response_validates_fft_size_boundary_matrix() {
        for invalid_size in [0, 4097, 5000, MAX_IMPULSE_RESPONSE_SAMPLES * 2] {
            assert!(
                impulse_response_fixture(invalid_size).is_err(),
                "FFT size {invalid_size} must be rejected before entering the FFT"
            );
        }
        for valid_size in [4096, 8192] {
            let response = impulse_response_fixture(valid_size)
                .unwrap_or_else(|error| panic!("FFT size {valid_size}: {error}"));
            assert_eq!(response.amplitude.len(), valid_size);
            assert_eq!(response.time.len(), valid_size);
        }
    }

    #[test]
    fn impulse_response_propagates_frequency_sensitivity_to_time_samples() {
        let config = ImpulseResponseConfig {
            fft_size: Some(1024),
            ..Default::default()
        };
        let frequencies = [20.0, 1_000.0, 20_000.0];
        let response = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.1),
            Complex64::new(0.1, 0.0),
        ];
        let sensitivity = [0.1, 0.2, 0.3];
        let result = calculate_impulse_response_with_uncertainty(
            &frequencies,
            &response,
            Some(&sensitivity),
            &config,
        )
        .expect("valid uncertainty spectrum");
        let uncertainty = result.uncertainty.expect("propagated uncertainty");
        assert_eq!(uncertainty.len(), result.amplitude.len());
        assert!(
            uncertainty
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(uncertainty.iter().any(|value| *value > 0.0));

        assert!(
            calculate_impulse_response_with_uncertainty(
                &frequencies,
                &response,
                Some(&sensitivity[..2]),
                &config,
            )
            .is_err()
        );
    }

    #[test]
    fn impulse_response_rejects_all_malformed_sample_contracts() {
        let config = ImpulseResponseConfig {
            fft_size: Some(16),
            ..Default::default()
        };
        let response = [Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];

        for frequencies in [
            vec![],
            vec![20.0, 20.0],
            vec![1000.0, 20.0],
            vec![0.0, 20.0],
            vec![-20.0, 20.0],
            vec![20.0, f64::NAN],
            vec![20.0, f64::INFINITY],
        ] {
            assert!(calculate_impulse_response(&frequencies, &response, &config).is_err());
        }

        assert!(calculate_impulse_response(&[20.0, 1000.0], &response[..1], &config).is_err());
        for invalid_value in [
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.0, f64::NAN),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
        ] {
            assert!(
                calculate_impulse_response(
                    &[20.0, 1000.0],
                    &[Complex64::new(1.0, 0.0), invalid_value],
                    &config,
                )
                .is_err()
            );
        }

        for sample_rate in [0.0, -48_000.0, f64::NAN, f64::INFINITY] {
            let invalid_config = ImpulseResponseConfig {
                sample_rate,
                ..config.clone()
            };
            assert!(
                calculate_impulse_response(&[20.0, 1000.0], &response, &invalid_config).is_err()
            );
        }
    }

    #[test]
    fn impulse_response_rejects_non_power_of_two_fft_size() {
        let config = ImpulseResponseConfig {
            fft_size: Some(5000),
            ..Default::default()
        };
        let result = calculate_impulse_response(
            &[20.0, 1000.0],
            &[Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)],
            &config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn impulse_response_validates_frequency_response_shape_and_order() {
        let config = ImpulseResponseConfig {
            fft_size: Some(16),
            ..Default::default()
        };
        assert!(
            calculate_impulse_response(&[20.0, 1000.0], &[Complex64::new(1.0, 0.0)], &config)
                .is_err()
        );
        assert!(
            calculate_impulse_response(
                &[1000.0, 20.0],
                &[Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)],
                &config
            )
            .is_err()
        );
    }

    #[test]
    fn impulse_response_accepts_valid_power_of_two_size() {
        let config = ImpulseResponseConfig {
            fft_size: Some(16),
            ..Default::default()
        };
        let response = calculate_impulse_response(
            &[20.0, 1000.0],
            &[Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)],
            &config,
        )
        .expect("valid impulse-response input");
        assert_eq!(response.amplitude.len(), 16);
    }

    #[test]
    fn lshaped_room_contains_xy() {
        let room = LShapedRoom::new(5.0, 4.0, 3.0, 3.0, 2.5);
        // Inside region 1 (front, wider)
        assert!(room.contains_xy(2.0, 2.0));
        // Inside region 2 (back, narrower)
        assert!(room.contains_xy(1.0, 5.0));
        // Outside: in the cut-out area
        assert!(!room.contains_xy(4.0, 5.0));
        // Outside: beyond bounds
        assert!(!room.contains_xy(6.0, 2.0));
    }

    #[test]
    fn lshaped_room_contains_3d() {
        let room = LShapedRoom::new(5.0, 4.0, 3.0, 3.0, 2.5);
        assert!(room.contains(&Point3D::new(2.0, 2.0, 1.0)));
        assert!(!room.contains(&Point3D::new(2.0, 2.0, 3.0))); // above ceiling
    }

    #[test]
    fn lshaped_room_first_order_images() {
        let room = LShapedRoom::new(5.0, 4.0, 3.0, 3.0, 2.5);
        let source = Point3D::new(1.0, 1.0, 1.0);
        let images = room.get_first_order_images(&source);
        // Should have at least: left, right, front, back, floor, ceiling
        assert!(
            images.len() >= 6,
            "Expected at least 6 images, got {}",
            images.len()
        );

        // Verify left wall image
        let left = images.iter().find(|(_, name)| *name == "left");
        assert!(left.is_some());
        assert_eq!(left.unwrap().0.x, -1.0);

        // Verify front wall image
        let front = images.iter().find(|(_, name)| *name == "front");
        assert!(front.is_some());
        assert_eq!(front.unwrap().0.y, -1.0);
    }

    #[test]
    fn lshaped_room_image_source_validation_rejects_forbidden_region() {
        let room = LShapedRoom::new(5.0, 4.0, 3.0, 3.0, 2.5);
        let source = Point3D::new(1.0, 1.0, 1.0);
        let listener = Point3D::new(2.0, 2.0, 1.0);

        // Image in the forbidden (cut-out) region should be rejected
        let forbidden_image = Point3D::new(4.0, 5.0, 1.0);
        assert!(!room.is_valid_image_source(&forbidden_image, &listener, &source));

        // Valid image outside the room but not in forbidden region
        let valid_image = Point3D::new(-1.0, 1.0, 1.0);
        assert!(room.is_valid_image_source(&valid_image, &listener, &source));
    }

    #[test]
    fn crossover_filter_amplitude_at_frequency() {
        let lp = CrossoverFilter::Lowpass {
            cutoff_freq: 100.0,
            order: 2,
        };
        // Well below cutoff → ~1
        assert!((lp.amplitude_at_frequency(10.0) - 1.0).abs() < 0.1);
        // At cutoff (-3 dB)
        let at_cutoff = lp.amplitude_at_frequency(100.0);
        assert!(at_cutoff > 0.6 && at_cutoff < 0.8);
        // Well above cutoff → attenuated
        assert!(lp.amplitude_at_frequency(1000.0) < 0.1);

        let hp = CrossoverFilter::Highpass {
            cutoff_freq: 100.0,
            order: 2,
        };
        // Well above cutoff → ~1
        assert!((hp.amplitude_at_frequency(1000.0) - 1.0).abs() < 0.1);
        // Well below cutoff → attenuated
        assert!(hp.amplitude_at_frequency(10.0) < 0.1);

        let bp = CrossoverFilter::Bandpass {
            low_cutoff: 100.0,
            high_cutoff: 1000.0,
            order: 2,
        };
        // In passband → ~1
        assert!((bp.amplitude_at_frequency(300.0) - 1.0).abs() < 0.1);
        // Below low cutoff → attenuated
        assert!(bp.amplitude_at_frequency(10.0) < 0.1);
        // Above high cutoff → attenuated
        assert!(bp.amplitude_at_frequency(5000.0) < 0.1);
    }

    #[test]
    fn source_orientation_rotates_directivity_into_local_frame() {
        let horizontal_angles: Vec<f64> = (0..36).map(|index| index as f64 * 10.0).collect();
        let vertical_angles: Vec<f64> = (0..19).map(|index| index as f64 * 10.0).collect();
        let mut magnitude = sonium_compute::HostMatrix::zeros(19, 36);
        // A single lobe along the local +Y acoustic axis.
        magnitude[[9, 9]] = 1.0;
        let directivity = DirectivityPattern {
            horizontal_angles,
            vertical_angles,
            magnitude,
        };
        let source = Source::new(Point3D::new(0.0, 0.0, 0.0), directivity, 1.0);

        let forward_y = source.amplitude_towards(&Point3D::new(0.0, 1.0, 0.0), 1_000.0);
        let side_y = source.amplitude_towards(&Point3D::new(1.0, 0.0, 0.0), 1_000.0);
        assert!(forward_y > 0.99);
        assert!(side_y < 0.01);

        let rotated = source.with_orientation(Point3D::new(1.0, 0.0, 0.0));
        let forward_x = rotated.amplitude_towards(&Point3D::new(1.0, 0.0, 0.0), 1_000.0);
        let side_x = rotated.amplitude_towards(&Point3D::new(0.0, 1.0, 0.0), 1_000.0);
        assert!(forward_x > 0.99);
        assert!(side_x < 0.01);
    }

    #[test]
    fn wall_material_conversion_covers_presets_and_aliases() {
        let names = [
            "concrete",
            "brick",
            "drywall",
            "gypsum",
            "plaster",
            "glass",
            "wood-thin",
            "thin_wood",
            "wood",
            "thick_wood",
            "carpet-thin",
            "thin_carpet",
            "carpet",
            "thick_carpet",
            "acoustic-tile",
            "ceiling_tile",
            "curtains",
            "drapes",
            "acoustic-foam",
            "foam",
            "hardwood",
            "wood-floor",
            "concrete-floor",
        ];

        for name in names {
            let material = WallMaterialConfig::Preset {
                name: name.to_string(),
            }
            .to_material();
            assert!(
                !material.name.is_empty(),
                "empty material for preset {name}"
            );
            assert!(material.absorption_at_frequency(1_000.0).is_finite());
        }
    }

    #[test]
    fn wall_material_conversion_preserves_complex_impedance() {
        let impedance = ImpedanceBands {
            real: [820.0; 6],
            imag: [205.0; 6],
        };
        let material = WallMaterialConfig::Custom {
            name: "impedance-test".to_string(),
            absorption: [0.2; 6],
            reflection_phase: [0.0; 6],
            impedance: Some(impedance.clone()),
        }
        .to_material();

        assert_eq!(material.name, "impedance-test");
        assert_eq!(material.impedance, Some(impedance));
        let reflection =
            material.reflection_at_frequency_complex_with_reference_impedance(1_000.0, 1.2 * 343.0);
        assert!(reflection.re.is_finite() && reflection.im.is_finite());
    }

    #[test]
    fn empty_directivity_axes_are_a_safe_neutral_fallback() {
        let pattern = DirectivityPattern {
            horizontal_angles: Vec::new(),
            vertical_angles: vec![0.0],
            magnitude: sonium_compute::HostMatrix::zeros(1, 0),
        };
        assert_eq!(pattern.interpolate(0.0, 0.0), 0.0);
    }

    #[test]
    fn chunked_compute_returns_partial_results_before_completion() {
        let mut simulator =
            RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
        simulator.init_chunked_compute();
        let result: serde_json::Value =
            serde_json::from_str(&simulator.compute_chunk(3)).expect("chunk JSON");
        assert_eq!(result["done"], false);
        assert_eq!(result["frequencies_completed"], 3);
        assert_eq!(result["partial_spl"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn multi_listener_results_are_serialized_for_full_and_chunked_paths() {
        let mut config: serde_json::Value =
            serde_json::from_str(&create_default_config()).expect("default config JSON");
        config["solver"]["method"] = serde_json::json!("image-source-1");
        config["frequencies"]["min_freq"] = serde_json::json!(100.0);
        config["frequencies"]["max_freq"] = serde_json::json!(200.0);
        config["frequencies"]["num_points"] = serde_json::json!(2);
        config["listening_positions"] = serde_json::json!([
            {"x": 2.0, "y": 2.0, "z": 1.2},
            {"x": 2.5, "y": 2.0, "z": 1.2}
        ]);
        let config_json = serde_json::to_string(&config).expect("multi-listener config");
        let mut simulator = RoomSimulatorWasm::new(&config_json).expect("multi-listener simulator");

        let full: serde_json::Value =
            serde_json::from_str(&simulator.run_simulation().expect("full result"))
                .expect("full result JSON");
        assert_eq!(full["listener_responses"].as_array().unwrap().len(), 2);
        assert_eq!(
            full["listener_responses"][0]["frequency_response"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        simulator.init_chunked_compute();
        let partial: serde_json::Value =
            serde_json::from_str(&simulator.compute_chunk(1)).expect("partial result JSON");
        assert_eq!(partial["partial_listener_spl"].as_array().unwrap().len(), 2);
        assert_eq!(
            partial["partial_listener_spl"][0].as_array().unwrap().len(),
            1
        );

        let complete: serde_json::Value =
            serde_json::from_str(&simulator.compute_chunk(1)).expect("complete result JSON");
        assert_eq!(complete["done"], true);
        assert_eq!(
            complete["partial_listener_spl"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            complete["partial_listener_spl"][1]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let final_result: serde_json::Value =
            serde_json::from_str(&simulator.finalize_chunked_compute()).expect("final result JSON");
        assert_eq!(final_result["listener_spl"].as_array().unwrap().len(), 2);
        assert_eq!(final_result["listener_spl"][1].as_array().unwrap().len(), 2);
    }

    #[test]
    fn incomplete_chunked_compute_cannot_be_finalized_as_complete() {
        let mut simulator =
            RoomSimulatorWasm::new(&create_default_config()).expect("default simulator");
        simulator.init_chunked_compute();
        simulator.compute_chunk(3);

        let result: serde_json::Value =
            serde_json::from_str(&simulator.finalize_chunked_compute()).expect("finalize JSON");
        assert_eq!(result["done"], false);
        assert_eq!(result["frequencies_completed"], 3);
        assert_eq!(result["total_frequencies"], simulator.num_frequencies());
        assert_eq!(result["partial_spl"].as_array().unwrap().len(), 3);
        assert!(result["error"].as_str().unwrap().contains("incomplete"));
    }

    #[test]
    fn impulse_response_js_boundary_rejects_invalid_magnitude_and_phase() {
        let negative = compute_impulse_response(
            r#"{"frequencies":[100.0],"magnitudes":[-1.0],"phases":[0.0]}"#,
        );
        assert!(negative.contains("magnitudes must be finite"));

        let nonfinite = compute_impulse_response(
            r#"{"frequencies":[100.0],"magnitudes":[1.0],"phases":[null]}"#,
        );
        assert!(nonfinite.contains("Invalid JSON"));
    }
}
