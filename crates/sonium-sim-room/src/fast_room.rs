//! Fast, phase-preserving approximation for rectangular rooms.
//!
//! The low-frequency field is evaluated from the analytical rigid-wall
//! shoebox eigenfunctions with an Eyring-derived damping time.  Above the
//! Schroeder transition a coherent image-source sum takes over.  This is not
//! a replacement for FEM when detailed boundary geometry matters; it is a
//! deliberately cheap preview/data-generation backend.

use anyhow::{Result, bail};
use num_complex::Complex64;
use sonium_bem::materials::{AmbientMedium, PorousLayer, PorousModel};
use sonium_xem::{
    BoundaryConfig, Point3D, PorousModelConfig, RoomGeometry, RoomSimulation, SurfaceConfig,
};
use std::f64::consts::PI;

use crate::SimulationOutput;

const AIR_DENSITY: f64 = sonium_xem::constants::AIR_DENSITY_20C;

/// Accuracy/cost controls for the fast rectangular-room solver.
#[derive(Debug, Clone, Copy)]
pub struct FastRoomConfig {
    /// Lowest modal loss factor. Prevents singular poles in nearly rigid rooms.
    pub modal_damping_floor: f64,
    /// Compute modes up to this multiple of the blend centre.
    pub modal_frequency_margin: f64,
    /// Maximum total number of wall reflections in an image path.
    pub image_order: usize,
    /// Ignore image paths whose propagation time exceeds this value.
    pub max_path_time: f64,
    /// Minimum automatically selected modal/image transition frequency.
    pub min_transition_hz: f64,
    /// Maximum automatically selected modal/image transition frequency.
    pub max_transition_hz: f64,
}

impl Default for FastRoomConfig {
    fn default() -> Self {
        Self {
            // Numerical pole regularization only; boundary absorption owns
            // physical low-frequency decay.
            modal_damping_floor: 0.0005,
            modal_frequency_margin: 1.25,
            image_order: 12,
            max_path_time: 0.250,
            min_transition_hz: 150.0,
            max_transition_hz: 500.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Mode {
    nx: usize,
    ny: usize,
    nz: usize,
    omega: f64,
    normalization: f64,
}

#[derive(Debug, Clone, Copy)]
struct WallReflections {
    x0: Complex64,
    x1: Complex64,
    y0: Complex64,
    y1: Complex64,
    z0: Complex64,
    z1: Complex64,
}

/// Return the normalized rigid-walled shoebox zero-mode pressure contribution.
///
/// The reference assumes unit source strength and the Neumann-mode
/// normalization used by [`run_fast_hybrid_with_config`]. It is a modal
/// contribution, not a complete room response at arbitrary frequency.
pub fn rigid_shoebox_zero_mode_pressure_reference(
    dimensions: [f64; 3],
    frequency_hz: f64,
    speed_of_sound: f64,
) -> Result<Complex64> {
    validate_shoebox_reference_inputs(dimensions, speed_of_sound)?;
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        bail!("reference frequency must be positive and finite");
    }

    let volume = dimensions.iter().product::<f64>();
    let omega = 2.0 * PI * frequency_hz;
    Ok(Complex64::new(
        -speed_of_sound * speed_of_sound / (volume * omega * omega),
        0.0,
    ))
}

/// Return the analytical rigid-walled shoebox frequency for one mode.
///
/// This is intentionally kept separate from [`build_modes`] so callers and
/// tests can validate the production modal enumeration against the closed-form
/// Neumann spectrum without depending on its loop bounds or ordering.
pub fn rigid_shoebox_mode_frequency_reference(
    dimensions: [f64; 3],
    mode_indices: [usize; 3],
    speed_of_sound: f64,
) -> Result<f64> {
    validate_shoebox_reference_inputs(dimensions, speed_of_sound)?;
    let squared_wavenumber = mode_indices
        .into_iter()
        .zip(dimensions)
        .map(|(index, dimension)| (index as f64 / dimension).powi(2))
        .sum::<f64>();
    Ok(0.5 * speed_of_sound * squared_wavenumber.sqrt())
}

fn validate_shoebox_reference_inputs(dimensions: [f64; 3], speed_of_sound: f64) -> Result<()> {
    if dimensions
        .iter()
        .any(|&dimension| !dimension.is_finite() || dimension <= 0.0)
    {
        bail!("shoebox dimensions must be positive and finite");
    }
    if !speed_of_sound.is_finite() || speed_of_sound <= 0.0 {
        bail!("speed of sound must be positive and finite");
    }
    Ok(())
}

/// Run the fast modal/image-source approximation.
pub fn run_fast_hybrid(simulation: &RoomSimulation) -> Result<SimulationOutput> {
    run_fast_hybrid_with_config(simulation, &FastRoomConfig::default())
}

/// Sample the production fast-hybrid field at arbitrary probe positions.
///
/// This reduced-order field API reuses the exact modal/image-source evaluator
/// through a one-frequency temporary simulation; it never labels the result
/// as FEM or measured data.
pub fn sample_fast_hybrid_field(
    simulation: &RoomSimulation,
    source_index: usize,
    frequency_hz: f64,
    positions: &[Point3D],
) -> Result<Vec<Complex64>> {
    sample_fast_hybrid_field_with_config(
        simulation,
        &FastRoomConfig::default(),
        source_index,
        frequency_hz,
        positions,
    )
}

/// Configurable form of [`sample_fast_hybrid_field`].
pub fn sample_fast_hybrid_field_with_config(
    simulation: &RoomSimulation,
    config: &FastRoomConfig,
    source_index: usize,
    frequency_hz: f64,
    positions: &[Point3D],
) -> Result<Vec<Complex64>> {
    if source_index >= simulation.sources.len() {
        bail!("source index {source_index} is outside the simulation");
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        bail!("sample frequency must be finite and positive");
    }
    if positions.is_empty() {
        bail!("field sampling requires at least one position");
    }
    if positions
        .iter()
        .any(|point| ![point.x, point.y, point.z].into_iter().all(f64::is_finite))
    {
        bail!("field sample positions must be finite");
    }

    let mut sample_simulation = simulation.clone();
    sample_simulation.frequencies = vec![frequency_hz];
    sample_simulation.listening_positions = positions.to_vec();
    let output = run_fast_hybrid_with_config(&sample_simulation, config)?;
    Ok(output.pressures[source_index]
        .iter()
        .map(|listener| listener[0])
        .collect())
}

/// Evidence from a three-level modal truncation study.
///
/// The pressures are evaluated by the same production modal evaluator used by
/// [`run_fast_hybrid_with_config`]. The report deliberately keeps the cutoff
/// and mode-count axes beside the complex responses so a caller cannot claim
/// convergence without showing that the modal basis actually grew.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalConvergenceReport {
    /// Increasing modal-frequency cutoffs used for the three levels.
    pub cutoff_frequencies_hz: [f64; 3],
    /// Number of retained modes at each cutoff, including the zero mode.
    pub mode_counts: [usize; 3],
    /// Complex modal pressure at the selected source/listener/frequency.
    pub pressures: [Complex64; 3],
    /// Absolute complex-pressure changes between adjacent levels.
    pub absolute_errors: [f64; 2],
    /// Relative complex-pressure changes between adjacent levels.
    pub relative_errors: [f64; 2],
    /// Relative convergence budget used to produce this report.
    pub relative_tolerance: f64,
    /// Absolute convergence budget used to produce this report.
    pub absolute_tolerance: f64,
}

impl ModalConvergenceReport {
    /// Whether both adjacent modal truncation changes satisfy the stored
    /// absolute-plus-relative error budget.
    pub fn is_converged(&self) -> bool {
        self.require_converged().is_ok()
    }

    /// Require a valid, converged report before promoting it as evidence.
    pub fn require_converged(&self) -> Result<(), String> {
        if self
            .cutoff_frequencies_hz
            .iter()
            .any(|&cutoff| !cutoff.is_finite() || cutoff <= 0.0)
            || self
                .cutoff_frequencies_hz
                .windows(2)
                .any(|window| window[1] <= window[0])
        {
            return Err("modal-convergence cutoffs must be finite and strictly increasing".into());
        }
        if self.mode_counts.contains(&0) {
            return Err("modal-convergence mode counts must be positive".into());
        }
        if self
            .mode_counts
            .windows(2)
            .any(|window| window[1] < window[0])
        {
            return Err("modal-convergence mode counts must be monotonic".into());
        }
        if !self.relative_tolerance.is_finite()
            || self.relative_tolerance < 0.0
            || !self.absolute_tolerance.is_finite()
            || self.absolute_tolerance < 0.0
            || (self.relative_tolerance == 0.0 && self.absolute_tolerance == 0.0)
        {
            return Err(
                "modal-convergence tolerances must be finite, non-negative, and non-zero".into(),
            );
        }
        if self
            .pressures
            .iter()
            .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
            || self
                .absolute_errors
                .iter()
                .chain(self.relative_errors.iter())
                .any(|error| !error.is_finite() || *error < 0.0)
        {
            return Err("modal-convergence report contains non-finite values".into());
        }

        for (level, &absolute_error) in self.absolute_errors.iter().enumerate() {
            let scale = self.pressures[level]
                .norm()
                .max(self.pressures[level + 1].norm());
            let limit = self.absolute_tolerance + self.relative_tolerance * scale;
            if !limit.is_finite() || absolute_error > limit {
                return Err(format!(
                    "modal-convergence level {} to {} exceeds budget: absolute={}, relative={}, limit={}",
                    level,
                    level + 1,
                    absolute_error,
                    self.relative_errors[level],
                    limit
                ));
            }
        }
        Ok(())
    }
}

/// Run a three-level modal truncation study with the default fast-room
/// numerical controls.
pub fn run_modal_convergence(
    simulation: &RoomSimulation,
    source_index: usize,
    listener_index: usize,
    frequency_hz: f64,
    cutoff_frequencies_hz: [f64; 3],
    relative_tolerance: f64,
    absolute_tolerance: f64,
) -> Result<ModalConvergenceReport> {
    run_modal_convergence_with_config(
        simulation,
        &FastRoomConfig::default(),
        source_index,
        listener_index,
        frequency_hz,
        cutoff_frequencies_hz,
        relative_tolerance,
        absolute_tolerance,
    )
}

/// Configurable production modal truncation/convergence study.
#[allow(clippy::too_many_arguments)]
pub fn run_modal_convergence_with_config(
    simulation: &RoomSimulation,
    config: &FastRoomConfig,
    source_index: usize,
    listener_index: usize,
    frequency_hz: f64,
    cutoff_frequencies_hz: [f64; 3],
    relative_tolerance: f64,
    absolute_tolerance: f64,
) -> Result<ModalConvergenceReport> {
    let RoomGeometry::Rectangular(room) = &simulation.room else {
        bail!("modal-convergence currently supports rectangular rooms only");
    };
    let dimensions = [room.width, room.depth, room.height];
    if dimensions
        .iter()
        .any(|&dimension| !dimension.is_finite() || dimension <= 0.0)
    {
        bail!("room dimensions must be positive and finite");
    }
    if !simulation.speed_of_sound.is_finite() || simulation.speed_of_sound <= 0.0 {
        bail!("speed of sound must be positive and finite");
    }
    if source_index >= simulation.sources.len() {
        bail!("source index {source_index} is outside the simulation");
    }
    if listener_index >= simulation.listening_positions.len() {
        bail!("listener index {listener_index} is outside the simulation");
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        bail!("modal-convergence frequency must be finite and positive");
    }
    if cutoff_frequencies_hz
        .iter()
        .any(|&cutoff| !cutoff.is_finite() || cutoff <= 0.0)
        || cutoff_frequencies_hz
            .windows(2)
            .any(|window| window[1] <= window[0])
    {
        bail!("modal-convergence cutoffs must be finite, positive, and strictly increasing");
    }
    if !relative_tolerance.is_finite()
        || relative_tolerance < 0.0
        || !absolute_tolerance.is_finite()
        || absolute_tolerance < 0.0
        || (relative_tolerance == 0.0 && absolute_tolerance == 0.0)
    {
        bail!("modal-convergence tolerances must be finite, non-negative, and non-zero");
    }
    if !config.modal_damping_floor.is_finite() || config.modal_damping_floor <= 0.0 {
        bail!("modal-convergence damping floor must be finite and positive");
    }

    let volume = dimensions.iter().product::<f64>();
    let rt60 = eyring_rt60(
        &simulation.boundaries,
        dimensions,
        volume,
        simulation.speed_of_sound,
    )?;
    if !rt60.is_finite() || rt60 <= 0.0 {
        bail!("modal-convergence room decay estimate must be finite and positive");
    }
    let source = simulation.sources[source_index].position;
    let listener = simulation.listening_positions[listener_index];
    let source_gain = simulation.sources[source_index].amplitude_towards(&listener, frequency_hz);
    if !source_gain.is_finite() {
        bail!("modal-convergence source gain must be finite");
    }

    let mut mode_counts = [0; 3];
    let mut pressures = [Complex64::new(0.0, 0.0); 3];
    for (level, &cutoff) in cutoff_frequencies_hz.iter().enumerate() {
        let modes = build_modes(dimensions, simulation.speed_of_sound, cutoff);
        mode_counts[level] = modes.len();
        pressures[level] = source_gain
            * modal_pressure(
                &modes,
                dimensions,
                volume,
                source,
                listener,
                frequency_hz,
                simulation.speed_of_sound,
                rt60,
                config.modal_damping_floor,
            );
        if !pressures[level].re.is_finite() || !pressures[level].im.is_finite() {
            bail!("modal-convergence pressure is non-finite at level {level}");
        }
    }

    let mut absolute_errors = [0.0; 2];
    let mut relative_errors = [0.0; 2];
    for level in 0..2 {
        absolute_errors[level] = (pressures[level + 1] - pressures[level]).norm();
        let scale = pressures[level].norm().max(pressures[level + 1].norm());
        relative_errors[level] = if scale > 0.0 {
            absolute_errors[level] / scale
        } else {
            0.0
        };
    }

    Ok(ModalConvergenceReport {
        cutoff_frequencies_hz,
        mode_counts,
        pressures,
        absolute_errors,
        relative_errors,
        relative_tolerance,
        absolute_tolerance,
    })
}

/// Run the approximation with explicit cost/accuracy controls.
pub fn run_fast_hybrid_with_config(
    simulation: &RoomSimulation,
    config: &FastRoomConfig,
) -> Result<SimulationOutput> {
    let RoomGeometry::Rectangular(room) = &simulation.room else {
        bail!("fast-hybrid currently supports rectangular rooms only");
    };
    let dimensions = [room.width, room.depth, room.height];
    if dimensions
        .iter()
        .any(|&dimension| !dimension.is_finite() || dimension <= 0.0)
    {
        bail!("room dimensions must be positive and finite");
    }
    if !simulation.speed_of_sound.is_finite() || simulation.speed_of_sound <= 0.0 {
        bail!("speed of sound must be positive and finite");
    }
    if config.image_order == 0
        || !config.modal_damping_floor.is_finite()
        || config.modal_damping_floor <= 0.0
        || !config.modal_frequency_margin.is_finite()
        || config.modal_frequency_margin <= 0.0
        || !config.max_path_time.is_finite()
        || config.max_path_time <= 0.0
        || !config.min_transition_hz.is_finite()
        || config.min_transition_hz <= 0.0
        || !config.max_transition_hz.is_finite()
        || config.max_transition_hz < config.min_transition_hz
    {
        bail!("fast-hybrid accuracy controls must be positive and ordered");
    }
    if simulation
        .frequencies
        .iter()
        .any(|&frequency| !frequency.is_finite() || frequency <= 0.0)
    {
        bail!("frequencies must be positive and finite");
    }

    let volume = dimensions.iter().product::<f64>();
    let rt60 = eyring_rt60(
        &simulation.boundaries,
        dimensions,
        volume,
        simulation.speed_of_sound,
    )?;
    let transition =
        (2000.0 * (rt60 / volume).sqrt()).clamp(config.min_transition_hz, config.max_transition_hz);
    let modes = build_modes(
        dimensions,
        simulation.speed_of_sound,
        transition * config.modal_frequency_margin,
    );

    let mut pressures = vec![
        vec![
            vec![Complex64::new(0.0, 0.0); simulation.frequencies.len()];
            simulation.listening_positions.len()
        ];
        simulation.sources.len()
    ];

    for (source_idx, source) in simulation.sources.iter().enumerate() {
        for (listener_idx, listener) in simulation.listening_positions.iter().enumerate() {
            for (frequency_idx, &frequency) in simulation.frequencies.iter().enumerate() {
                let modal_weight = modal_blend_weight(frequency, transition);
                let source_gain = source.amplitude_towards(listener, frequency);
                let modal = if modal_weight > 0.0 {
                    modal_pressure(
                        &modes,
                        dimensions,
                        volume,
                        source.position,
                        *listener,
                        frequency,
                        simulation.speed_of_sound,
                        rt60,
                        config.modal_damping_floor,
                    )
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let images = if modal_weight < 1.0 {
                    image_pressure(
                        dimensions,
                        source.position,
                        *listener,
                        frequency,
                        simulation.speed_of_sound,
                        &simulation.boundaries,
                        config.image_order,
                        config.max_path_time,
                    )?
                } else {
                    Complex64::new(0.0, 0.0)
                };
                pressures[source_idx][listener_idx][frequency_idx] =
                    source_gain * (modal_weight * modal + (1.0 - modal_weight) * images);
            }
        }
    }

    Ok(SimulationOutput::with_uniform_provenance(
        simulation.frequencies.clone(),
        pressures,
        simulation.sources.iter().map(|s| s.name.clone()).collect(),
        crate::FrequencyProvenance::ReducedOrder,
    ))
}

fn build_modes(dimensions: [f64; 3], c: f64, maximum_frequency: f64) -> Vec<Mode> {
    let limits = dimensions.map(|length| (2.0 * maximum_frequency * length / c).ceil() as usize);
    let mut modes = Vec::new();
    for nx in 0..=limits[0] {
        for ny in 0..=limits[1] {
            for nz in 0..=limits[2] {
                let frequency = 0.5
                    * c
                    * ((nx as f64 / dimensions[0]).powi(2)
                        + (ny as f64 / dimensions[1]).powi(2)
                        + (nz as f64 / dimensions[2]).powi(2))
                    .sqrt();
                if frequency <= maximum_frequency {
                    let normalization = [nx, ny, nz]
                        .into_iter()
                        .map(|n| if n == 0 { 1.0 } else { 2.0 })
                        .product();
                    modes.push(Mode {
                        nx,
                        ny,
                        nz,
                        omega: 2.0 * PI * frequency,
                        normalization,
                    });
                }
            }
        }
    }
    modes
}

#[allow(clippy::too_many_arguments)]
fn modal_pressure(
    modes: &[Mode],
    dimensions: [f64; 3],
    volume: f64,
    source: Point3D,
    listener: Point3D,
    frequency: f64,
    c: f64,
    rt60: f64,
    damping_floor: f64,
) -> Complex64 {
    let omega = 2.0 * PI * frequency;
    modes.iter().fold(Complex64::new(0.0, 0.0), |sum, mode| {
        let source_shape = mode_shape(*mode, dimensions, source);
        let listener_shape = mode_shape(*mode, dimensions, listener);
        // exp(-iωt): the damped pole has a negative imaginary term.
        let damping_ratio = if mode.omega > 0.0 && rt60.is_finite() {
            (1000.0_f64.ln() / (mode.omega * rt60)).max(damping_floor)
        } else {
            damping_floor
        };
        let denominator = Complex64::new(
            mode.omega * mode.omega - omega * omega,
            -2.0 * damping_ratio * mode.omega * omega,
        );
        sum + mode.normalization * source_shape * listener_shape * c * c / (volume * denominator)
    })
}

fn mode_shape(mode: Mode, dimensions: [f64; 3], point: Point3D) -> f64 {
    (PI * mode.nx as f64 * point.x / dimensions[0]).cos()
        * (PI * mode.ny as f64 * point.y / dimensions[1]).cos()
        * (PI * mode.nz as f64 * point.z / dimensions[2]).cos()
}

#[allow(clippy::too_many_arguments)]
fn image_pressure(
    dimensions: [f64; 3],
    source: Point3D,
    listener: Point3D,
    frequency: f64,
    c: f64,
    boundaries: &BoundaryConfig,
    max_order: usize,
    max_path_time: f64,
) -> Result<Complex64> {
    let reflections = wall_reflections(boundaries, frequency, c)?;
    let axis_images = [
        axis_images(
            source.x,
            dimensions[0],
            reflections.x0,
            reflections.x1,
            max_order,
        ),
        axis_images(
            source.y,
            dimensions[1],
            reflections.y0,
            reflections.y1,
            max_order,
        ),
        axis_images(
            source.z,
            dimensions[2],
            reflections.z0,
            reflections.z1,
            max_order,
        ),
    ];
    let k = 2.0 * PI * frequency / c;
    let max_distance = c * max_path_time;
    let mut pressure = Complex64::new(0.0, 0.0);
    for x in &axis_images[0] {
        for y in &axis_images[1] {
            if x.2 + y.2 > max_order {
                continue;
            }
            for z in &axis_images[2] {
                if x.2 + y.2 + z.2 > max_order {
                    continue;
                }
                let dx = listener.x - x.0;
                let dy = listener.y - y.0;
                let dz = listener.z - z.0;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                if distance < 1e-8 || distance > max_distance {
                    continue;
                }
                let phase = Complex64::from_polar(1.0, k * distance);
                pressure += x.1 * y.1 * z.1 * phase / (4.0 * PI * distance);
            }
        }
    }
    Ok(pressure)
}

fn axis_images(
    source: f64,
    length: f64,
    lower: Complex64,
    upper: Complex64,
    max_order: usize,
) -> Vec<(f64, Complex64, usize)> {
    let max_n = max_order as isize;
    let mut result = Vec::with_capacity(4 * max_order + 2);
    for n in -max_n..=max_n {
        let paired_order = 2 * n.unsigned_abs();
        if paired_order <= max_order {
            result.push((
                2.0 * n as f64 * length + source,
                (lower * upper).powu(n.unsigned_abs() as u32),
                paired_order,
            ));
        }
        let lower_count = (n - 1).unsigned_abs();
        let upper_count = n.unsigned_abs();
        let order = lower_count + upper_count;
        if order <= max_order {
            result.push((
                2.0 * n as f64 * length - source,
                lower.powu(lower_count as u32) * upper.powu(upper_count as u32),
                order,
            ));
        }
    }
    result
}

fn modal_blend_weight(frequency: f64, transition: f64) -> f64 {
    let start = transition / 2.0_f64.sqrt();
    let end = transition * 2.0_f64.sqrt();
    if frequency <= start {
        1.0
    } else if frequency >= end {
        0.0
    } else {
        let t = (frequency / start).ln() / (end / start).ln();
        0.5 * (1.0 + (PI * t).cos())
    }
}

fn eyring_rt60(
    boundaries: &BoundaryConfig,
    dimensions: [f64; 3],
    volume: f64,
    speed_of_sound: f64,
) -> Result<f64> {
    let surfaces = [
        (&boundaries.floor, dimensions[0] * dimensions[1]),
        (&boundaries.ceiling, dimensions[0] * dimensions[1]),
        (
            boundaries.left_wall.as_ref().unwrap_or(&boundaries.walls),
            dimensions[1] * dimensions[2],
        ),
        (
            boundaries.right_wall.as_ref().unwrap_or(&boundaries.walls),
            dimensions[1] * dimensions[2],
        ),
        (
            boundaries.front_wall.as_ref().unwrap_or(&boundaries.walls),
            dimensions[0] * dimensions[2],
        ),
        (
            boundaries.back_wall.as_ref().unwrap_or(&boundaries.walls),
            dimensions[0] * dimensions[2],
        ),
    ];
    let total_area: f64 = surfaces.iter().map(|(_, area)| area).sum();
    let mut absorbed = 0.0;
    for (surface, area) in surfaces {
        absorbed +=
            area * (1.0 - reflection(surface, 500.0, speed_of_sound)?.norm_sqr()).clamp(0.0, 1.0);
    }
    let mean_alpha = (absorbed / total_area).clamp(1e-4, 0.999);
    Ok(0.161 * volume / (-total_area * (1.0 - mean_alpha).ln()))
}

fn wall_reflections(
    boundaries: &BoundaryConfig,
    frequency: f64,
    c: f64,
) -> Result<WallReflections> {
    Ok(WallReflections {
        x0: reflection(
            boundaries.left_wall.as_ref().unwrap_or(&boundaries.walls),
            frequency,
            c,
        )?,
        x1: reflection(
            boundaries.right_wall.as_ref().unwrap_or(&boundaries.walls),
            frequency,
            c,
        )?,
        y0: reflection(
            boundaries.front_wall.as_ref().unwrap_or(&boundaries.walls),
            frequency,
            c,
        )?,
        y1: reflection(
            boundaries.back_wall.as_ref().unwrap_or(&boundaries.walls),
            frequency,
            c,
        )?,
        z0: reflection(&boundaries.floor, frequency, c)?,
        z1: reflection(&boundaries.ceiling, frequency, c)?,
    })
}

fn reflection(surface: &SurfaceConfig, frequency: f64, c: f64) -> Result<Complex64> {
    if !frequency.is_finite() || frequency <= 0.0 || !c.is_finite() || c <= 0.0 {
        bail!("reflection frequency and sound speed must be finite and positive");
    }
    let z0 = AIR_DENSITY * c;
    let result = match surface {
        SurfaceConfig::Rigid => Complex64::new(1.0, 0.0),
        SurfaceConfig::Absorption { coefficient } => {
            if !coefficient.is_finite() || !(0.0..=1.0).contains(coefficient) {
                bail!("absorption coefficient must be finite and between zero and one");
            }
            Complex64::new((1.0 - coefficient).sqrt(), 0.0)
        }
        SurfaceConfig::Impedance { real, imag } => {
            let z = Complex64::new(*real, *imag);
            (z - z0) / (z + z0)
        }
        SurfaceConfig::Admittance { real, imag } => {
            let beta = Complex64::new(*real, *imag);
            (Complex64::new(1.0, 0.0) - beta) / (Complex64::new(1.0, 0.0) + beta)
        }
        SurfaceConfig::PorousLayer {
            model,
            thickness,
            flow_resistivity,
            air_gap,
            ambient,
        } => {
            let medium = AmbientMedium::try_new(ambient.density, ambient.speed_of_sound)?;
            let layer = PorousLayer {
                model: match model {
                    PorousModelConfig::DelanyBazley => PorousModel::DelanyBazley,
                    PorousModelConfig::Miki => PorousModel::Miki,
                },
                thickness: *thickness,
                flow_resistivity: *flow_resistivity,
                air_gap: air_gap.unwrap_or(0.0),
            };
            let beta = layer.try_normalized_admittance(frequency, medium)?;
            (Complex64::new(1.0, 0.0) - beta) / (Complex64::new(1.0, 0.0) + beta)
        }
    };
    if !result.re.is_finite() || !result.im.is_finite() {
        bail!("material reflection is singular or nonfinite");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_xem::{Directivity, DirectivityGrid, RectangularRoom, Source};

    #[test]
    fn invalid_materials_return_errors_instead_of_panicking() {
        let bad_porous = SurfaceConfig::PorousLayer {
            model: PorousModelConfig::Miki,
            thickness: -0.1,
            flow_resistivity: 10_000.0,
            air_gap: None,
            ambient: Default::default(),
        };
        for surface in [
            bad_porous,
            SurfaceConfig::Absorption {
                coefficient: f64::NAN,
            },
            SurfaceConfig::Admittance {
                real: -1.0,
                imag: 0.0,
            },
            SurfaceConfig::Impedance {
                real: f64::NAN,
                imag: 0.0,
            },
        ] {
            assert!(reflection(&surface, 500.0, 343.0).is_err());
            let mut simulation = simulation();
            simulation.boundaries.floor = surface;
            assert!(run_fast_hybrid(&simulation).is_err());
        }
    }

    fn simulation() -> RoomSimulation {
        let source = Source::new(
            Point3D::new(1.0, 1.2, 1.1),
            Directivity::Grid(DirectivityGrid::omnidirectional()),
            1.0,
        )
        .with_name("main".into());
        let mut simulation = RoomSimulation::new(
            RoomGeometry::Rectangular(RectangularRoom::new(5.0, 4.0, 2.8)),
            vec![source],
            vec![Point3D::new(3.4, 2.7, 1.2)],
        );
        simulation.frequencies = vec![40.0, 100.0, 300.0, 1000.0];
        simulation.boundaries.walls = SurfaceConfig::Absorption { coefficient: 0.2 };
        simulation
    }

    #[test]
    fn fast_hybrid_preserves_output_shape_and_finite_phase() {
        let output = run_fast_hybrid(&simulation()).unwrap();
        assert_eq!(output.source_names, ["main"]);
        assert_eq!(output.pressures[0][0].len(), 4);
        assert!(
            output.pressures[0][0]
                .iter()
                .all(|p| p.re.is_finite() && p.im.is_finite())
        );
        assert!(output.pressures[0][0].iter().any(|p| p.im.abs() > 1e-10));
    }

    #[test]
    fn sampled_field_matches_listener_evaluation_and_rejects_bad_requests() {
        let simulation = simulation();
        let output = run_fast_hybrid(&simulation).unwrap();
        let sampled = sample_fast_hybrid_field(
            &simulation,
            0,
            simulation.frequencies[0],
            &simulation.listening_positions,
        )
        .unwrap();
        assert_eq!(sampled, vec![output.pressures[0][0][0]]);
        assert!(
            sample_fast_hybrid_field(&simulation, 1, 100.0, &simulation.listening_positions)
                .is_err()
        );
        assert!(
            sample_fast_hybrid_field(&simulation, 0, 0.0, &simulation.listening_positions).is_err()
        );
        assert!(sample_fast_hybrid_field(&simulation, 0, 100.0, &[]).is_err());
        assert!(
            sample_fast_hybrid_field(&simulation, 0, 100.0, &[Point3D::new(f64::NAN, 0.0, 0.0)])
                .is_err()
        );
    }

    #[test]
    fn modal_convergence_reports_growing_basis_and_finite_adjacent_errors() {
        let report =
            run_modal_convergence(&simulation(), 0, 0, 40.0, [100.0, 200.0, 400.0], 1.0, 1.0)
                .expect("valid modal convergence request");

        assert!(report.mode_counts[0] > 0);
        assert!(
            report
                .mode_counts
                .windows(2)
                .all(|window| window[1] >= window[0])
        );
        assert!(
            report
                .pressures
                .iter()
                .all(|pressure| pressure.re.is_finite() && pressure.im.is_finite())
        );
        assert!(
            report
                .absolute_errors
                .iter()
                .chain(report.relative_errors.iter())
                .all(|error| error.is_finite() && *error >= 0.0)
        );
        assert!(report.is_converged());
    }

    #[test]
    fn modal_convergence_accepts_the_release_fixture_budget() {
        let source = Source::new(
            Point3D::new(0.37, 0.43, 0.51),
            Directivity::Grid(DirectivityGrid::omnidirectional()),
            1.0,
        )
        .with_name("main".into());
        let mut simulation = RoomSimulation::new(
            RoomGeometry::Rectangular(RectangularRoom::new(2.0, 2.0, 2.0)),
            vec![source],
            vec![Point3D::new(1.37, 1.43, 1.51)],
        );
        simulation.speed_of_sound = 343.0;
        simulation.boundaries.walls = SurfaceConfig::Rigid;
        simulation.boundaries.floor = SurfaceConfig::Rigid;
        simulation.boundaries.ceiling = SurfaceConfig::Rigid;

        let report =
            run_modal_convergence(&simulation, 0, 0, 30.0, [120.0, 240.0, 480.0], 0.15, 1.0e-6)
                .expect("release modal convergence fixture must evaluate");
        report
            .require_converged()
            .expect("release modal convergence fixture must meet its budget");
    }

    #[test]
    fn modal_convergence_rejects_malformed_requests_before_modal_work() {
        let simulation = simulation();
        let invalid_cutoffs =
            run_modal_convergence(&simulation, 0, 0, 40.0, [200.0, 100.0, 400.0], 0.1, 1.0e-12);
        assert!(invalid_cutoffs.is_err());

        let invalid_tolerance =
            run_modal_convergence(&simulation, 0, 0, 40.0, [100.0, 200.0, 400.0], 0.0, 0.0);
        assert!(invalid_tolerance.is_err());

        let mut invalid_config = FastRoomConfig::default();
        invalid_config.modal_damping_floor = f64::NAN;
        assert!(
            run_modal_convergence_with_config(
                &simulation,
                &invalid_config,
                0,
                0,
                40.0,
                [100.0, 200.0, 400.0],
                0.1,
                1.0e-12,
            )
            .is_err()
        );
    }

    #[test]
    fn modal_convergence_report_rejects_non_monotonic_metadata() {
        let mut report =
            run_modal_convergence(&simulation(), 0, 0, 40.0, [100.0, 200.0, 400.0], 1.0, 1.0)
                .unwrap();
        report.mode_counts[1] = 0;
        assert!(
            report
                .require_converged()
                .unwrap_err()
                .contains("mode counts")
        );

        report.mode_counts[1] = report.mode_counts[0];
        report.cutoff_frequencies_hz[2] = report.cutoff_frequencies_hz[1];
        assert!(report.require_converged().unwrap_err().contains("cutoffs"));
    }

    #[test]
    fn rejects_non_rectangular_rooms() {
        let mut simulation = simulation();
        simulation.room =
            RoomGeometry::LShaped(sonium_xem::LShapedRoom::new(4.0, 4.0, 2.0, 2.0, 2.8));
        assert!(run_fast_hybrid(&simulation).is_err());
    }

    #[test]
    fn blend_is_continuous_and_bounded() {
        let transition = 300.0;
        let values =
            [100.0, 212.132, 300.0, 424.264, 600.0].map(|f| modal_blend_weight(f, transition));
        assert!(values.windows(2).all(|w| w[0] >= w[1]));
        assert!(values.into_iter().all(|v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn cpu_modal_path_matches_compact_rigid_shoebox_zero_mode_reference() {
        let mut simulation = RoomSimulation::new(
            RoomGeometry::Rectangular(RectangularRoom::new(0.1, 0.1, 0.1)),
            vec![Source::omnidirectional(
                Point3D::new(0.031, 0.037, 0.043),
                1.0,
            )],
            vec![Point3D::new(0.067, 0.052, 0.061)],
        );
        simulation.boundaries.walls = SurfaceConfig::Rigid;
        simulation.boundaries.floor = SurfaceConfig::Rigid;
        simulation.boundaries.ceiling = SurfaceConfig::Rigid;
        simulation.frequencies = vec![40.0, 80.0];

        let output = run_fast_hybrid(&simulation).expect("compact CPU reference case must solve");
        for (index, &frequency) in simulation.frequencies.iter().enumerate() {
            let reference = rigid_shoebox_zero_mode_pressure_reference(
                [0.1, 0.1, 0.1],
                frequency,
                simulation.speed_of_sound,
            )
            .expect("valid zero-mode reference inputs");
            let source_gain = simulation.sources[0]
                .amplitude_towards(&simulation.listening_positions[0], frequency);
            let expected = source_gain * reference;
            let actual = output.pressures[0][0][index];
            let tolerance = expected.norm().max(1.0) * 1.0e-12;
            assert!(
                (actual - expected).norm() <= tolerance,
                "frequency {frequency} Hz: CPU modal pressure {actual:?} differs from zero-mode reference {expected:?}"
            );
        }
    }

    #[test]
    fn cpu_modal_basis_matches_independent_rigid_shoebox_reference() {
        let dimensions = [4.0, 5.0, 6.0];
        let speed_of_sound = 343.0;
        let maximum_frequency = 60.0;
        let modes = build_modes(dimensions, speed_of_sound, maximum_frequency);

        // This list is intentionally independent of build_modes' enumeration.
        // It covers every mode below 60 Hz for this asymmetric shoebox, including
        // the first axial, tangential, and second axial modes.
        let expected_indices = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
        ];
        assert_eq!(modes.len(), expected_indices.len());

        for (nx, ny, nz) in expected_indices {
            let mode = modes
                .iter()
                .find(|mode| (mode.nx, mode.ny, mode.nz) == (nx, ny, nz))
                .unwrap_or_else(|| panic!("missing mode ({nx}, {ny}, {nz})"));
            let reference_frequency =
                rigid_shoebox_mode_frequency_reference(dimensions, [nx, ny, nz], speed_of_sound)
                    .expect("valid shoebox reference inputs");
            let actual_frequency = mode.omega / std::f64::consts::TAU;
            let frequency_tolerance = 1.0e-12 * reference_frequency.max(1.0);
            assert!((actual_frequency - reference_frequency).abs() <= frequency_tolerance);

            let active_axes = usize::from(nx != 0) + usize::from(ny != 0) + usize::from(nz != 0);
            assert_eq!(mode.normalization, 2.0_f64.powi(active_axes as i32));
            assert!(actual_frequency <= maximum_frequency + 1.0e-12);
        }
    }

    #[test]
    fn zero_mode_reference_rejects_nonphysical_inputs() {
        assert!(rigid_shoebox_zero_mode_pressure_reference([0.0, 0.1, 0.1], 40.0, 343.0).is_err());
        assert!(rigid_shoebox_zero_mode_pressure_reference([0.1, 0.1, 0.1], 0.0, 343.0).is_err());
        assert!(rigid_shoebox_zero_mode_pressure_reference([0.1, 0.1, 0.1], 40.0, 0.0).is_err());
    }

    #[test]
    fn mode_frequency_reference_includes_zero_mode_and_rejects_nonfinite_inputs() {
        assert_eq!(
            rigid_shoebox_mode_frequency_reference([4.0, 5.0, 6.0], [0, 0, 0], 343.0).unwrap(),
            0.0
        );
        assert!(
            rigid_shoebox_mode_frequency_reference([f64::NAN, 5.0, 6.0], [1, 0, 0], 343.0).is_err()
        );
        assert!(
            rigid_shoebox_mode_frequency_reference([4.0, 5.0, 6.0], [1, 0, 0], f64::INFINITY)
                .is_err()
        );
    }

    #[test]
    fn fast_hybrid_rejects_nonfinite_inputs_before_numerical_setup() {
        let mut invalid_room = simulation();
        invalid_room.room = RoomGeometry::Rectangular(RectangularRoom::new(f64::NAN, 4.0, 2.8));
        assert!(run_fast_hybrid(&invalid_room).is_err());

        let mut invalid_speed = simulation();
        invalid_speed.speed_of_sound = f64::NAN;
        assert!(run_fast_hybrid(&invalid_speed).is_err());

        let mut invalid_frequency = simulation();
        invalid_frequency.frequencies = vec![f64::NAN];
        assert!(run_fast_hybrid(&invalid_frequency).is_err());

        let mut config = FastRoomConfig::default();
        config.min_transition_hz = 500.0;
        config.max_transition_hz = 150.0;
        assert!(run_fast_hybrid_with_config(&simulation(), &config).is_err());
    }
}
