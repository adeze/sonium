use super::MAX_FREQUENCY_POINTS;
use super::{
    CrossoverConfig, DirectivityConfig, Point3DConfig, RoomGeometryConfig, ScatteringObjectConfig,
    SimulationConfig, WallMaterialConfig,
};

use serde::{Deserialize, Serialize};

/// Versioned browser capture contract emitted alongside a static RoomSim
/// gallery. Native tooling validates it before a browser runner opens a page.
pub const BROWSER_CAPTURE_CONTRACT_SCHEMA_V1: &str = "sonium-roomsim-browser-capture/v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserCaptureViewport {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserCaptureScreenshot {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait_for: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserCaptureContract {
    pub schema: String,
    pub entrypoint: String,
    pub viewport: BrowserCaptureViewport,
    pub device_scale: u32,
    pub ready_selector: String,
    pub failure_selector: String,
    pub required_figures: Vec<String>,
    pub screenshots: Vec<BrowserCaptureScreenshot>,
    pub note: String,
}

impl BrowserCaptureContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != BROWSER_CAPTURE_CONTRACT_SCHEMA_V1 {
            return Err(format!(
                "unsupported browser capture schema {:?}",
                self.schema
            ));
        }
        if !safe_relative_asset_path(&self.entrypoint) {
            return Err("browser capture entrypoint must be a safe relative path".into());
        }
        if self.viewport.width == 0 || self.viewport.height == 0 {
            return Err("browser capture viewport must be non-zero".into());
        }
        if self.device_scale == 0 {
            return Err("browser capture device scale must be non-zero".into());
        }
        if self.ready_selector.trim().is_empty()
            || self.failure_selector.trim().is_empty()
            || self.ready_selector == self.failure_selector
        {
            return Err("browser capture ready/failure selectors must be distinct".into());
        }
        if self.required_figures.is_empty() {
            return Err("browser capture requires at least one figure".into());
        }
        let mut figures = std::collections::HashSet::with_capacity(self.required_figures.len());
        for figure in &self.required_figures {
            if !safe_relative_asset_path(figure) {
                return Err(format!("unsafe required browser figure path {figure:?}"));
            }
            if !figures.insert(figure) {
                return Err(format!("duplicate required browser figure {figure:?}"));
            }
        }
        if self.screenshots.is_empty() {
            return Err("browser capture requires at least one screenshot".into());
        }
        let mut names = std::collections::HashSet::with_capacity(self.screenshots.len());
        for screenshot in &self.screenshots {
            if screenshot.name.trim().is_empty() || !names.insert(&screenshot.name) {
                return Err(format!(
                    "browser screenshot names must be non-empty and unique: {:?}",
                    screenshot.name
                ));
            }
            let has_wait = screenshot
                .wait_for
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            let has_selector = screenshot
                .selector
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            if has_wait == has_selector {
                return Err(format!(
                    "browser screenshot {:?} must specify exactly one non-empty wait_for or selector",
                    screenshot.name
                ));
            }
        }
        if self.note.trim().is_empty() {
            return Err("browser capture note must not be empty".into());
        }
        Ok(())
    }

    /// Ensure every figure promised to a browser capture exists in the
    /// already-validated native gallery manifest.
    pub fn validate_against_figure_names<'a, I>(&self, figure_names: I) -> Result<(), String>
    where
        I: IntoIterator<Item = &'a str>,
    {
        self.validate()?;
        let available: std::collections::HashSet<&str> = figure_names.into_iter().collect();
        for required in &self.required_figures {
            if !available.contains(required.as_str()) {
                return Err(format!(
                    "browser capture requires missing gallery figure {:?}",
                    required
                ));
            }
        }
        Ok(())
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        let contract: Self = serde_json::from_str(json)
            .map_err(|error| format!("invalid browser capture contract JSON: {error}"))?;
        contract.validate()?;
        Ok(contract)
    }

    pub fn to_json(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        serde_json::to_vec_pretty(self)
            .map_err(|error| format!("browser capture contract serialization failed: {error}"))
    }
}

fn safe_relative_asset_path(path: &str) -> bool {
    !path.is_empty() && !path.starts_with('/') && !path.contains("..") && !path.contains('\\')
}

const MAX_SLICE_RESOLUTION: usize = 2_000;
const MAX_MESH_RESOLUTION: usize = 1_000;
const MAX_DIRECTIVITY_CURVES: usize = 720;
const MAX_DIRECTIVITY_POINTS: usize = 100_000;
const MAX_SCATTERING_OBJECTS: usize = 10_000;
const MAX_OBJECT_SUBDIVISIONS: usize = 8_192;
const MAX_SOURCES: usize = 1_024;
const MAX_LISTENING_POSITIONS: usize = 1_024;
pub(crate) const MAX_ESTIMATED_WORKING_SET_BYTES: u128 = 256 * 1024 * 1024;

pub(crate) fn validate_simulation_config(config: &SimulationConfig) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    validate_acoustic_conventions(config, &mut errors);
    validate_room(&config.room, &mut errors);
    let room_contains = |point: &Point3DConfig| point_is_inside(&config.room, point);

    if config.sources.is_empty() {
        errors.push("at least one source is required".into());
    }
    if config.sources.len() > MAX_SOURCES {
        errors.push(format!("source count must not exceed {MAX_SOURCES}"));
    }
    for (index, source) in config.sources.iter().enumerate() {
        validate_point(&source.position, &format!("source {index}"), &mut errors);
        let orientation_norm_squared = source.orientation.x * source.orientation.x
            + source.orientation.y * source.orientation.y
            + source.orientation.z * source.orientation.z;
        if !source.orientation.x.is_finite()
            || !source.orientation.y.is_finite()
            || !source.orientation.z.is_finite()
            || !orientation_norm_squared.is_finite()
            || orientation_norm_squared <= 1e-24
        {
            errors.push(format!(
                "source {index} orientation must be finite and non-zero"
            ));
        }
        if !room_contains(&source.position) {
            errors.push(format!("source {index} is outside the room"));
        }
        if !source.amplitude.is_finite() || source.amplitude < 0.0 {
            errors.push(format!(
                "source {index} amplitude must be finite and non-negative"
            ));
        }
        if !source.delay_ms.is_finite() || source.delay_ms < 0.0 {
            errors.push(format!(
                "source {index} delay_ms must be finite and non-negative"
            ));
        }
        validate_directivity(&source.directivity, index, &mut errors);
        validate_crossover(&source.crossover, index, &mut errors);
    }

    if config.listening_positions.is_empty() {
        errors.push("at least one listening position is required".into());
    }
    if config.listening_positions.len() > MAX_LISTENING_POSITIONS {
        errors.push(format!(
            "listening-position count must not exceed {MAX_LISTENING_POSITIONS}"
        ));
    }
    for (index, point) in config.listening_positions.iter().enumerate() {
        validate_point(point, &format!("listening position {index}"), &mut errors);
        if !room_contains(point) {
            errors.push(format!("listening position {index} is outside the room"));
        }
    }

    let frequencies = &config.frequencies;
    if !frequencies.min_freq.is_finite() || frequencies.min_freq <= 0.0 {
        errors.push("minimum frequency must be finite and positive".into());
    }
    if !frequencies.max_freq.is_finite() || frequencies.max_freq <= frequencies.min_freq {
        errors.push("maximum frequency must be finite and greater than minimum frequency".into());
    }
    if !(2..=MAX_FREQUENCY_POINTS).contains(&frequencies.num_points) {
        errors.push(format!(
            "frequency point count must be between 2 and {MAX_FREQUENCY_POINTS}"
        ));
    }
    if !matches!(
        frequencies.spacing.to_ascii_lowercase().as_str(),
        "linear" | "logarithmic"
    ) {
        errors.push("frequency spacing must be either linear or logarithmic".into());
    }

    let solver = &config.solver;
    if !matches!(
        solver.method.as_str(),
        "direct"
            | "image-source-1"
            | "image-source-2"
            | "image-source-3"
            | "modal"
            | "hybrid"
            | "bem"
            | "hybrid-bem"
    ) {
        errors.push(format!("unsupported solver method `{}`", solver.method));
    }
    finite_positive(solver.speed_of_sound, "speed_of_sound", &mut errors);
    finite_positive(solver.air_density, "air_density", &mut errors);
    finite_in_range(solver.humidity, 0.0, 100.0, "humidity", &mut errors);
    finite(solver.temperature, "temperature", &mut errors);
    finite_positive(
        solver.hybrid_crossover_width,
        "hybrid_crossover_width",
        &mut errors,
    );
    finite_positive(solver.modal_damping, "modal_damping", &mut errors);
    if solver.mesh_resolution == 0 || solver.mesh_resolution > MAX_MESH_RESOLUTION {
        errors.push(format!(
            "mesh resolution must be between 1 and {MAX_MESH_RESOLUTION}"
        ));
    }
    if solver.max_mode_order == 0 {
        errors.push("max_mode_order must be positive".into());
    }
    validate_bem_config(config, &mut errors);

    if config.visualization.slice_resolution == 0
        || config.visualization.slice_resolution > MAX_SLICE_RESOLUTION
    {
        errors.push(format!(
            "slice resolution must be between 1 and {MAX_SLICE_RESOLUTION}"
        ));
    }
    for &index in &config.visualization.slice_frequency_indices {
        if index >= frequencies.num_points {
            errors.push(format!("slice frequency index {index} is out of range"));
        }
    }
    validate_impulse_config(
        &config.visualization.impulse_response,
        "impulse response",
        &mut errors,
    );
    let binaural = &config.visualization.binaural;
    if let Some(point) = &binaural.head_position {
        validate_point(point, "binaural head position", &mut errors);
        if !room_contains(point) {
            errors.push("binaural head position is outside the room".into());
        }
    }
    finite_positive(binaural.head_radius, "binaural head radius", &mut errors);
    finite_positive(binaural.ear_spacing, "binaural ear spacing", &mut errors);
    finite(binaural.head_yaw, "binaural head yaw", &mut errors);
    validate_impulse_config(
        &binaural.ir_config,
        "binaural impulse response",
        &mut errors,
    );

    validate_materials(config, &mut errors);
    validate_scattering_objects(config, &room_contains, &mut errors);
    if !config.scattering_objects.is_empty()
        && !matches!(solver.method.as_str(), "bem" | "hybrid-bem")
    {
        errors.push("scattering objects require the bem or hybrid-bem solver".into());
    }

    validate_working_set_estimate(config, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify that the exchange convention attached to browser input matches the
/// phasor convention implemented by the room-BEM, modal, and image-source
/// paths.  Rejecting unsupported-but-valid exchange data here is preferable
/// to returning a numerically plausible result under silently different
/// phase or amplitude semantics.
fn validate_acoustic_conventions(config: &SimulationConfig, errors: &mut Vec<String>) {
    let convention = match sonium_bem::BemConvention::from_exchange(&config.conventions) {
        Ok(convention) => convention,
        Err(error) => {
            errors.push(format!("incompatible BEM convention: {error}"));
            return;
        }
    };

    if convention != sonium_bem::ACOUSTIC_CONVENTION {
        errors.push(
            "room simulator supports only the canonical exp(-iωt) acoustic phasor convention"
                .into(),
        );
    }
    if config.conventions.rms_amplitudes {
        errors.push("room simulator expects peak, not RMS, source amplitudes".into());
    }
}

/// Reject combinations whose worst-case output buffers would exceed a
/// conservative browser working-set budget. This is intentionally an
/// estimate rather than a claim about the browser's total memory: the room
/// solver, JSON serialization, and WASM runtime have additional overhead.
/// Estimate the peak browser working set for a validated configuration.
///
/// This is a conservative planning value, not a measurement of the browser's
/// allocator or total process memory.
pub(crate) fn estimated_working_set_bytes(config: &SimulationConfig) -> u128 {
    let frequency_count = config.frequencies.num_points as u128;
    let source_count = config.sources.len() as u128;
    let listener_count = config.listening_positions.len() as u128;
    // run_simulation retains the per-frequency result tuple, source SPLs,
    // unpacked output vectors, and the chunked prefix can retain another
    // SPL vector. Use a deliberately conservative per-frequency allowance.
    // Listener SPL, uncertainty and progressive prefixes coexist with the
    // collected tuples; source SPLs coexist with the unpacked responses.
    let frequency_bytes = frequency_count.saturating_mul(
        256u128
            .saturating_add(source_count.saturating_mul(16))
            .saturating_add(listener_count.saturating_mul(128)),
    );

    let resolution = config.visualization.slice_resolution as u128;
    let slice_count = if !config.visualization.generate_slices {
        0
    } else if config.visualization.slice_frequency_indices.is_empty() {
        // Match compute_slices: for short sweeps, step_by(max(n / 10, 1))
        // can retain more than ten frequency slices (up to nineteen).
        let step = (frequency_count / 10).max(1);
        frequency_count.saturating_add(step - 1) / step
    } else {
        config.visualization.slice_frequency_indices.len() as u128
    };
    // Two grids (horizontal/vertical), each retaining values and validity
    // flags, plus coordinate/output overhead.
    let slice_bytes = resolution
        .saturating_mul(resolution)
        .saturating_mul(slice_count)
        .saturating_mul(32);

    let ir_bytes = super::impulse_response::estimated_transform_bytes;
    let mut transform_bytes = 0u128;
    if config.visualization.generate_impulse_response {
        transform_bytes =
            transform_bytes.saturating_add(ir_bytes(&config.visualization.impulse_response));
    }
    if config.visualization.binaural.enabled {
        transform_bytes = transform_bytes
            .saturating_add(ir_bytes(&config.visualization.binaural.ir_config).saturating_mul(2));
    }

    frequency_bytes
        .saturating_add(slice_bytes)
        .saturating_add(transform_bytes)
        // Mesh generation holds subdivision input/output faces, midpoint
        // caches, vertices and merged room elements at once. Linear field
        // caps alone do not bound exponential icosphere refinement.
        .saturating_add(
            (super::scattering_objects::estimate_scattering_element_count(
                &config.scattering_objects,
            ) as u128)
                .saturating_mul(1024),
        )
}

fn validate_working_set_estimate(config: &SimulationConfig, errors: &mut Vec<String>) {
    let estimated = estimated_working_set_bytes(config);
    if estimated > MAX_ESTIMATED_WORKING_SET_BYTES {
        errors.push(format!(
            "estimated browser working set ({estimated} bytes) exceeds the {} MiB limit",
            MAX_ESTIMATED_WORKING_SET_BYTES / (1024 * 1024)
        ));
    }
}

fn validate_room(room: &RoomGeometryConfig, errors: &mut Vec<String>) {
    match room {
        RoomGeometryConfig::Rectangular {
            width,
            depth,
            height,
        } => {
            for (name, value) in [("width", *width), ("depth", *depth), ("height", *height)] {
                finite_positive(value, name, errors);
            }
        }
        RoomGeometryConfig::LShaped {
            width1,
            depth1,
            width2,
            depth2,
            height,
        } => {
            for (name, value) in [
                ("width1", *width1),
                ("depth1", *depth1),
                ("width2", *width2),
                ("depth2", *depth2),
                ("height", *height),
            ] {
                finite_positive(value, name, errors);
            }
            if width2 > width1 {
                errors.push("L-shaped room width2 must not exceed width1".into());
            }
        }
    }
}

fn point_is_inside(room: &RoomGeometryConfig, point: &Point3DConfig) -> bool {
    if ![point.x, point.y, point.z]
        .iter()
        .all(|value| value.is_finite())
    {
        return false;
    }
    match room {
        RoomGeometryConfig::Rectangular {
            width,
            depth,
            height,
        } => {
            (0.0..=*width).contains(&point.x)
                && (0.0..=*depth).contains(&point.y)
                && (0.0..=*height).contains(&point.z)
        }
        RoomGeometryConfig::LShaped {
            width1,
            depth1,
            width2,
            depth2,
            height,
        } => {
            let in_front =
                point.x >= 0.0 && point.x <= *width1 && point.y >= 0.0 && point.y <= *depth1;
            let in_back = point.x >= 0.0
                && point.x <= *width2
                && point.y >= *depth1
                && point.y <= *depth1 + *depth2;
            (in_front || in_back) && point.z >= 0.0 && point.z <= *height
        }
    }
}

fn validate_point(point: &Point3DConfig, label: &str, errors: &mut Vec<String>) {
    for (axis, value) in [("x", point.x), ("y", point.y), ("z", point.z)] {
        if !value.is_finite() {
            errors.push(format!("{label}.{axis} must be finite"));
        }
    }
}

fn finite(value: f64, label: &str, errors: &mut Vec<String>) {
    if !value.is_finite() {
        errors.push(format!("{label} must be finite"));
    }
}

fn finite_positive(value: f64, label: &str, errors: &mut Vec<String>) {
    if !value.is_finite() || value <= 0.0 {
        errors.push(format!("{label} must be finite and positive"));
    }
}

fn finite_in_range(value: f64, min: f64, max: f64, label: &str, errors: &mut Vec<String>) {
    if !value.is_finite() || !(min..=max).contains(&value) {
        errors.push(format!("{label} must be finite and in [{min}, {max}]"));
    }
}

fn validate_crossover(crossover: &CrossoverConfig, index: usize, errors: &mut Vec<String>) {
    let label = format!("source {index} crossover");
    match crossover {
        CrossoverConfig::FullRange => {}
        CrossoverConfig::Lowpass { cutoff_freq, order }
        | CrossoverConfig::Highpass { cutoff_freq, order } => {
            finite_positive(*cutoff_freq, &format!("{label} cutoff frequency"), errors);
            if *order == 0 {
                errors.push(format!("{label} order must be positive"));
            }
        }
        CrossoverConfig::Bandpass {
            low_cutoff,
            high_cutoff,
            order,
        } => {
            finite_positive(*low_cutoff, &format!("{label} low cutoff"), errors);
            finite_positive(*high_cutoff, &format!("{label} high cutoff"), errors);
            if high_cutoff <= low_cutoff {
                errors.push(format!("{label} high cutoff must exceed low cutoff"));
            }
            if *order == 0 {
                errors.push(format!("{label} order must be positive"));
            }
        }
    }
}

fn validate_directivity(directivity: &DirectivityConfig, index: usize, errors: &mut Vec<String>) {
    if let DirectivityConfig::Cardioid { front_back_ratio } = directivity {
        finite_positive(
            *front_back_ratio,
            &format!("source {index} front_back_ratio"),
            errors,
        );
    }
    if let DirectivityConfig::Spinorama {
        horizontal,
        vertical,
    } = directivity
    {
        for (name, curves) in [("horizontal", horizontal), ("vertical", vertical)] {
            if curves.len() > MAX_DIRECTIVITY_CURVES {
                errors.push(format!(
                    "source {index} {name} directivity has too many curves"
                ));
            }
            for curve in curves {
                finite(curve.angle, &format!("source {index} {name} angle"), errors);
                if curve.freq.len() != curve.spl.len() || curve.freq.is_empty() {
                    errors.push(format!(
                        "source {index} {name} directivity curves must have matching non-empty data"
                    ));
                }
                if curve.freq.len() > MAX_DIRECTIVITY_POINTS {
                    errors.push(format!(
                        "source {index} {name} directivity curve has too many points"
                    ));
                }
                for (&frequency, &spl) in curve.freq.iter().zip(&curve.spl) {
                    finite_positive(
                        frequency,
                        &format!("source {index} {name} frequency"),
                        errors,
                    );
                    finite(spl, &format!("source {index} {name} SPL"), errors);
                }
                if curve.freq.windows(2).any(|pair| pair[1] <= pair[0]) {
                    errors.push(format!(
                        "source {index} {name} directivity frequencies must be strictly increasing"
                    ));
                }
            }
        }
    }
}

fn validate_impulse_config(
    config: &super::ImpulseResponseConfig,
    label: &str,
    errors: &mut Vec<String>,
) {
    finite_positive(config.sample_rate, &format!("{label} sample rate"), errors);
    if let Some(duration) = config.duration {
        finite_positive(duration, &format!("{label} duration"), errors);
        let requested_samples = duration * config.sample_rate;
        if !requested_samples.is_finite()
            || requested_samples > super::MAX_IMPULSE_RESPONSE_SAMPLES as f64
        {
            errors.push(format!(
                "{label} duration and sample rate request too many samples (maximum {})",
                super::MAX_IMPULSE_RESPONSE_SAMPLES
            ));
        }
    }
    if !config.min_freq.is_finite() || config.min_freq < 0.0 {
        errors.push(format!(
            "{label} minimum frequency must be finite and non-negative"
        ));
    }
    if let Some(max) = config.max_freq
        && (!max.is_finite() || max <= config.min_freq)
    {
        errors.push(format!("{label} maximum frequency must exceed minimum"));
    }
    if let Some(size) = config.fft_size
        && (size < 2 || !size.is_power_of_two() || size > super::MAX_IMPULSE_RESPONSE_SAMPLES)
    {
        errors.push(format!(
            "{label} FFT size must be a power of two, at least 2, and no larger than {}",
            super::MAX_IMPULSE_RESPONSE_SAMPLES
        ));
    }
}

fn validate_materials(config: &SimulationConfig, errors: &mut Vec<String>) {
    let materials = [
        &config.wall_materials.left,
        &config.wall_materials.right,
        &config.wall_materials.front,
        &config.wall_materials.back,
        &config.wall_materials.floor,
        &config.wall_materials.ceiling,
    ];
    for (index, material) in materials.into_iter().enumerate() {
        validate_material_config(material, &format!("wall material {index}"), errors);
    }
}

fn validate_material_config(material: &WallMaterialConfig, label: &str, errors: &mut Vec<String>) {
    if let WallMaterialConfig::Preset { name } = material {
        let normalized = name.to_ascii_lowercase();
        if !matches!(
            normalized.as_str(),
            "concrete"
                | "brick"
                | "drywall"
                | "gypsum"
                | "plaster"
                | "glass"
                | "wood_thin"
                | "wood-thin"
                | "thin_wood"
                | "wood_thick"
                | "wood-thick"
                | "thick_wood"
                | "wood"
                | "carpet_thin"
                | "carpet-thin"
                | "thin_carpet"
                | "carpet_thick"
                | "carpet-thick"
                | "thick_carpet"
                | "carpet"
                | "acoustic_tile"
                | "acoustic-tile"
                | "ceiling_tile"
                | "curtains"
                | "drapes"
                | "acoustic_foam"
                | "acoustic-foam"
                | "foam"
                | "hardwood"
                | "wood_floor"
                | "wood-floor"
                | "concrete_floor"
                | "concrete-floor"
        ) {
            errors.push(format!("unknown {label} preset `{name}`"));
        }
    }
    if let WallMaterialConfig::Custom {
        absorption,
        reflection_phase,
        impedance,
        ..
    } = material
    {
        if absorption
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            errors.push(format!("{label} absorption must be finite and in [0, 1]"));
        }
        if reflection_phase.iter().any(|value| !value.is_finite()) {
            errors.push(format!(
                "{label} reflection_phase must contain only finite radians"
            ));
        }
        if let Some(impedance) = impedance {
            if impedance
                .real
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                errors.push(format!(
                    "{label} impedance real parts must be finite and non-negative"
                ));
            }
            if impedance.imag.iter().any(|value| !value.is_finite()) {
                errors.push(format!("{label} impedance imaginary parts must be finite"));
            }
        }
    }
}

fn validate_scattering_objects(
    config: &SimulationConfig,
    room_contains: &impl Fn(&Point3DConfig) -> bool,
    errors: &mut Vec<String>,
) {
    if config.scattering_objects.len() > MAX_SCATTERING_OBJECTS {
        errors.push(format!(
            "scattering object count must not exceed {MAX_SCATTERING_OBJECTS}"
        ));
    }
    for (index, object) in config.scattering_objects.iter().enumerate() {
        let label = format!("scattering object {index}");
        match object {
            ScatteringObjectConfig::Box(box_object) => {
                validate_point(&box_object.center, &format!("{label} center"), errors);
                if !room_contains(&box_object.center) {
                    errors.push(format!("{label} center is outside the room"));
                }
                if box_object
                    .dimensions
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    errors.push(format!("{label} dimensions must be finite and positive"));
                } else {
                    validate_box_extent(
                        &box_object.center,
                        [
                            box_object.dimensions[0] / 2.0,
                            box_object.dimensions[1] / 2.0,
                            box_object.dimensions[2] / 2.0,
                        ],
                        room_contains,
                        &label,
                        errors,
                    );
                }
                validate_material_config(&box_object.material, &label, errors);
            }
            ScatteringObjectConfig::Sphere(sphere) => {
                validate_point(&sphere.center, &format!("{label} center"), errors);
                if !room_contains(&sphere.center) {
                    errors.push(format!("{label} center is outside the room"));
                }
                finite_positive(sphere.radius, &format!("{label} radius"), errors);
                if sphere.subdivisions == 0 || sphere.subdivisions > MAX_OBJECT_SUBDIVISIONS {
                    errors.push(format!(
                        "{label} subdivisions must be between 1 and {MAX_OBJECT_SUBDIVISIONS}"
                    ));
                }
                if sphere.radius.is_finite() && sphere.radius > 0.0 {
                    validate_box_extent(
                        &sphere.center,
                        [sphere.radius; 3],
                        room_contains,
                        &label,
                        errors,
                    );
                }
                validate_material_config(&sphere.material, &label, errors);
            }
            ScatteringObjectConfig::Cylinder(cylinder) => {
                validate_point(
                    &cylinder.base_center,
                    &format!("{label} base center"),
                    errors,
                );
                if !room_contains(&cylinder.base_center) {
                    errors.push(format!("{label} base center is outside the room"));
                }
                finite_positive(cylinder.radius, &format!("{label} radius"), errors);
                finite_positive(cylinder.height, &format!("{label} height"), errors);
                if cylinder.radius.is_finite()
                    && cylinder.radius > 0.0
                    && cylinder.height.is_finite()
                    && cylinder.height > 0.0
                {
                    let center = Point3DConfig {
                        x: cylinder.base_center.x,
                        y: cylinder.base_center.y,
                        z: cylinder.base_center.z + cylinder.height / 2.0,
                    };
                    validate_box_extent(
                        &center,
                        [cylinder.radius, cylinder.radius, cylinder.height / 2.0],
                        room_contains,
                        &label,
                        errors,
                    );
                }
                if cylinder.radial_segments == 0
                    || cylinder.radial_segments > MAX_OBJECT_SUBDIVISIONS
                {
                    errors.push(format!(
                        "{label} radial segments must be between 1 and {MAX_OBJECT_SUBDIVISIONS}"
                    ));
                }
                validate_material_config(&cylinder.material, &label, errors);
            }
        }
    }
}

fn validate_box_extent(
    center: &Point3DConfig,
    half_extents: [f64; 3],
    room_contains: &impl Fn(&Point3DConfig) -> bool,
    label: &str,
    errors: &mut Vec<String>,
) {
    for mask in 0..8 {
        let corner = Point3DConfig {
            x: center.x
                + if mask & 1 == 0 {
                    -half_extents[0]
                } else {
                    half_extents[0]
                },
            y: center.y
                + if mask & 2 == 0 {
                    -half_extents[1]
                } else {
                    half_extents[1]
                },
            z: center.z
                + if mask & 4 == 0 {
                    -half_extents[2]
                } else {
                    half_extents[2]
                },
        };
        if !room_contains(&corner) {
            errors.push(format!("{label} extends outside the room"));
            return;
        }
    }
}

fn validate_bem_config(config: &SimulationConfig, errors: &mut Vec<String>) {
    // The room-acoustics BEM path uses its native default boundary model:
    // an explicit impedance table would be silently ignored, so reject it
    // loudly at validation instead of burying the gap in provenance.
    if matches!(config.solver.method.as_str(), "bem" | "hybrid-bem") {
        let materials = &config.wall_materials;
        for (label, material) in [
            ("left", &materials.left),
            ("right", &materials.right),
            ("front", &materials.front),
            ("back", &materials.back),
            ("floor", &materials.floor),
            ("ceiling", &materials.ceiling),
        ] {
            if material.has_impedance() {
                errors.push(format!(
                    "{label} wall impedance is not applied by the bem/hybrid-bem solver; \
                     remove the impedance table or use an image-source/modal method"
                ));
            }
        }
    }
    let bem = &config.solver.bem_config;
    finite_positive(bem.tolerance, "BEM tolerance", errors);
    if bem.max_iterations == 0
        || bem.restart == 0
        || bem.max_elements == 0
        || bem.min_elements_per_meter == 0
    {
        errors.push("BEM iteration, restart, element, and mesh limits must be positive".into());
    }
    finite_positive(
        bem.elements_per_wavelength,
        "BEM elements_per_wavelength",
        errors,
    );
    finite_positive(bem.beta_scale, "BEM beta_scale", errors);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SimulationConfig {
        serde_json::from_str(&crate::create_default_config()).expect("default config is valid JSON")
    }

    #[test]
    fn default_config_passes_runtime_validation() {
        validate_simulation_config(&default_config()).expect("default config should be runnable");
    }

    #[test]
    fn scattering_subdivision_growth_is_admitted_before_mesh_generation() {
        let mut config = default_config();
        config.solver.method = "bem".into();
        for subdivisions in [7, 32, MAX_OBJECT_SUBDIVISIONS, usize::MAX] {
            config.scattering_objects = vec![ScatteringObjectConfig::Sphere(crate::SphereObject {
                name: "budget-test".into(),
                center: Point3DConfig {
                    x: 2.0,
                    y: 2.0,
                    z: 1.0,
                },
                radius: 0.1,
                subdivisions,
                material: WallMaterialConfig::default(),
            })];
            let errors = validate_simulation_config(&config).unwrap_err();
            assert!(errors.iter().any(|error| error.contains("working set")));
        }
        if let ScatteringObjectConfig::Sphere(sphere) = &mut config.scattering_objects[0] {
            sphere.subdivisions = 2;
        }
        assert_eq!(
            super::super::scattering_objects::estimate_scattering_element_count(
                &config.scattering_objects
            ),
            320
        );
    }

    #[test]
    fn working_set_accounts_for_listener_product_and_automatic_fft() {
        let mut config = default_config();
        config.visualization.generate_slices = false;
        config.visualization.generate_impulse_response = false;
        config.visualization.binaural.enabled = false;
        config.frequencies.num_points = 10_000;
        config.listening_positions = vec![config.listening_positions[0]; 1024];
        let errors = validate_simulation_config(&config).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("working set")));

        let mut config = default_config();
        config.visualization.generate_slices = false;
        config.visualization.generate_impulse_response = true;
        config.visualization.impulse_response.fft_size = None;
        config.visualization.impulse_response.sample_rate = 48_000.0;
        config.visualization.impulse_response.duration = Some(60.0);
        let errors = validate_simulation_config(&config).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("working set")));
        config.frequencies.num_points = usize::MAX;
        assert!(estimated_working_set_bytes(&config) > MAX_ESTIMATED_WORKING_SET_BYTES);
    }

    #[test]
    fn max_independent_caps_product_is_rejected_gracefully() {
        // Every dimension below is individually admissible, but their product
        // (frequencies × sources × listeners plus full-resolution slices) far
        // exceeds the browser working set. Admission must reject the
        // combination with a working-set error, not admit it or panic.
        let mut config = default_config();
        config.frequencies.num_points = MAX_FREQUENCY_POINTS;
        config.sources = vec![config.sources[0].clone(); MAX_SOURCES];
        config.listening_positions = vec![config.listening_positions[0]; MAX_LISTENING_POSITIONS];
        config.visualization.generate_slices = true;
        config.visualization.slice_resolution = MAX_SLICE_RESOLUTION;
        config.visualization.slice_frequency_indices.clear();
        config.visualization.generate_impulse_response = false;
        config.visualization.binaural.enabled = false;

        let estimated = estimated_working_set_bytes(&config);
        assert!(
            estimated > MAX_ESTIMATED_WORKING_SET_BYTES,
            "max-caps product must exceed the working-set budget, got {estimated}"
        );
        let errors = validate_simulation_config(&config).unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains("working set")),
            "max-caps product must be rejected with a working-set error, got {errors:?}"
        );
    }

    #[test]
    fn working_set_counts_every_automatically_selected_slice() {
        let mut with_slices = default_config();
        with_slices.visualization.slice_resolution = 32;
        with_slices.visualization.generate_impulse_response = false;
        with_slices.visualization.binaural.enabled = false;
        with_slices.frequencies.num_points = 19;

        let mut without_slices = with_slices.clone();
        without_slices.visualization.generate_slices = false;

        let slice_bytes = estimated_working_set_bytes(&with_slices)
            - estimated_working_set_bytes(&without_slices);
        assert_eq!(slice_bytes, 19 * 32 * 32 * 32);
    }

    #[test]
    fn accepts_multiple_listeners_for_listener_axis_results() {
        let mut config = default_config();
        config
            .listening_positions
            .push(config.listening_positions[0].clone());
        validate_simulation_config(&config).expect("multiple listeners are supported");
    }

    #[test]
    fn legacy_configs_default_to_canonical_peak_conventions() {
        let mut legacy = serde_json::to_value(default_config()).expect("default config JSON");
        legacy
            .as_object_mut()
            .expect("configuration object")
            .remove("conventions");
        let config: SimulationConfig = serde_json::from_value(legacy).expect("legacy config");

        assert_eq!(
            config.conventions.harmonic_time,
            sonium_xem::HarmonicConvention::ExpMinusIwt
        );
        assert!(!config.conventions.rms_amplitudes);
        validate_simulation_config(&config).expect("legacy config should remain runnable");
    }

    #[test]
    fn rejects_noncanonical_or_incompatible_acoustic_conventions() {
        let mut plus_i = default_config();
        plus_i.conventions.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let errors = validate_simulation_config(&plus_i).expect_err("plus-i is unsupported");
        assert!(errors.iter().any(|error| error.contains("exp(-iωt)")));

        let mut rms = default_config();
        rms.conventions.rms_amplitudes = true;
        let errors = validate_simulation_config(&rms).expect_err("RMS amplitudes are unsupported");
        assert!(errors.iter().any(|error| error.contains("peak, not RMS")));

        let mut inward = default_config();
        inward.conventions.outward_normals = false;
        let errors =
            validate_simulation_config(&inward).expect_err("inward normals are unsupported");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("outward physical-domain normals"))
        );
    }

    #[test]
    fn l_shape_rejects_points_in_the_cutout() {
        let mut config = default_config();
        config.room = RoomGeometryConfig::LShaped {
            width1: 5.0,
            depth1: 4.0,
            width2: 3.0,
            depth2: 3.0,
            height: 2.5,
        };
        config.sources[0].position = Point3DConfig {
            x: 4.0,
            y: 5.0,
            z: 1.0,
        };
        let errors = validate_simulation_config(&config).expect_err("cutout is outside room");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("source 0 is outside"))
        );
    }

    #[test]
    fn rejects_unbounded_frequency_requests_and_unknown_methods() {
        let mut config = default_config();
        config.frequencies.num_points = MAX_FREQUENCY_POINTS + 1;
        config.solver.method = "unknown".into();
        let errors = validate_simulation_config(&config).expect_err("invalid config");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("frequency point count"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unsupported solver method"))
        );
    }

    #[test]
    fn rejects_combinations_with_an_excessive_working_set() {
        let mut config = default_config();
        config.frequencies.num_points = 100_000;
        config.visualization.slice_resolution = MAX_SLICE_RESOLUTION;
        let errors = validate_simulation_config(&config).expect_err("working set is too large");
        assert!(errors.iter().any(|error| error.contains("working set")));
    }

    #[test]
    fn rejects_impulse_requests_that_overflow_the_sample_budget() {
        let mut config = default_config();
        config.visualization.generate_impulse_response = true;
        config.visualization.impulse_response.duration = Some(1_000.0);
        let errors = validate_simulation_config(&config).expect_err("IR is too large");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("too many samples"))
        );
    }

    #[test]
    fn rejects_unknown_materials_and_unsorted_directivity_data() {
        let mut config = default_config();
        config.wall_materials.left = WallMaterialConfig::Preset {
            name: "mystery-wall".into(),
        };
        config.sources[0].directivity = DirectivityConfig::Spinorama {
            horizontal: vec![crate::SpinoramaCurve {
                angle: 0.0,
                freq: vec![1_000.0, 100.0],
                spl: vec![0.0, 0.0],
            }],
            vertical: Vec::new(),
        };
        let errors = validate_simulation_config(&config).expect_err("invalid material/directivity");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unknown wall material"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("frequencies must be strictly increasing"))
        );
    }

    #[test]
    fn rejects_nonfinite_complex_material_phase() {
        let mut config = default_config();
        config.wall_materials.left = WallMaterialConfig::Custom {
            name: "phased".into(),
            absorption: [0.2; 6],
            reflection_phase: [f64::NAN; 6],
            impedance: None,
        };
        let errors = validate_simulation_config(&config).expect_err("invalid phase");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("reflection_phase"))
        );
    }

    #[test]
    fn legacy_custom_material_json_defaults_to_zero_phase() {
        let material: WallMaterialConfig = serde_json::from_str(
            r#"{"type":"custom","name":"legacy","absorption":[0.1,0.1,0.2,0.2,0.3,0.3]}"#,
        )
        .expect("legacy material JSON");
        match material {
            WallMaterialConfig::Custom {
                reflection_phase, ..
            } => assert_eq!(reflection_phase, [0.0; 6]),
            WallMaterialConfig::Preset { .. } => panic!("expected custom material"),
        }
    }

    #[test]
    fn rejects_nonpassive_impedance_material() {
        let mut config = default_config();
        config.wall_materials.left = WallMaterialConfig::Custom {
            name: "active".into(),
            absorption: [0.2; 6],
            reflection_phase: [0.0; 6],
            impedance: Some(crate::ImpedanceBands {
                real: [-1.0; 6],
                imag: [0.0; 6],
            }),
        };
        let errors = validate_simulation_config(&config).expect_err("active impedance");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("impedance real parts"))
        );
    }

    #[test]
    fn rejects_nonpositive_ambient_density() {
        let mut config = default_config();
        config.solver.air_density = 0.0;
        let errors = validate_simulation_config(&config).expect_err("invalid density");
        assert!(errors.iter().any(|error| error.contains("air_density")));
    }

    #[test]
    fn rejects_zero_source_orientation() {
        let mut config = default_config();
        config.sources[0].orientation = Point3DConfig {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let errors = validate_simulation_config(&config).expect_err("invalid orientation");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("orientation must be finite and non-zero"))
        );
    }

    #[test]
    fn rejects_invalid_scattering_object_geometry_and_mesh_limits() {
        let mut config = default_config();
        config.solver.method = "bem".into();
        config.scattering_objects = vec![ScatteringObjectConfig::Sphere(crate::SphereObject {
            name: "invalid".into(),
            center: Point3DConfig {
                x: 100.0,
                y: 2.0,
                z: 1.0,
            },
            radius: 0.0,
            material: WallMaterialConfig::default(),
            subdivisions: MAX_OBJECT_SUBDIVISIONS + 1,
        })];
        let errors = validate_simulation_config(&config).expect_err("invalid scattering object");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("center is outside"))
        );
        assert!(errors.iter().any(|error| error.contains("radius")));
        assert!(errors.iter().any(|error| error.contains("subdivisions")));
    }

    #[test]
    fn rejects_unbounded_source_collections() {
        let mut config = default_config();
        config.sources = vec![config.sources[0].clone(); MAX_SOURCES + 1];
        let errors = validate_simulation_config(&config).expect_err("source collection too large");
        assert!(errors.iter().any(|error| error.contains("source count")));
    }

    #[test]
    fn model_hash_is_deterministic_and_changes_with_configuration() {
        let config = default_config();
        let first = crate::simulation_model_hash(&config);
        let second = crate::simulation_model_hash(&config.clone());
        assert_eq!(first, second);

        let mut changed = config;
        if let RoomGeometryConfig::Rectangular { width, .. } = &mut changed.room {
            *width += 0.01;
        }
        assert_ne!(first, crate::simulation_model_hash(&changed));

        let mut medium_changed = default_config();
        medium_changed.solver.air_density += 0.01;
        assert_ne!(first, crate::simulation_model_hash(&medium_changed));
    }
}
