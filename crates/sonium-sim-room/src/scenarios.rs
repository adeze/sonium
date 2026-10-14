//! Room scenario definitions for RoomEQ data generation
//!
//! Defines 20 scenarios covering small/medium/large rooms with
//! stereo, 2.1, multi-sub, multi-seat, and surround (5.0/5.1/5.1.4/5.2.4)
//! configurations.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sonium_xem::{
    BoundaryConfig, CrossoverFilter, FrequencySpacing, Point3D, RectangularRoom, RoomGeometry,
    RoomSimulation, Source, SurfaceConfig,
};
use std::collections::HashSet;

/// A named scenario for data generation
#[derive(Clone)]
pub struct Scenario {
    /// Unique name for file/directory naming
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// The simulation configuration
    pub simulation: RoomSimulation,
    /// Source names matching `simulation.sources` ordering
    pub source_names: Vec<String>,
}

impl Scenario {
    /// Validate the axes and physical admission boundary used by the data
    /// generator.
    ///
    /// Scenario definitions are configuration, not solver output. Rejecting
    /// malformed definitions here prevents a source/listener tensor from
    /// being generated with labels or coordinates that do not describe the
    /// actual simulation. This is independent of any FEM/BEM solve so the
    /// complete built-in catalog can be checked cheaply.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || !portable_label(&self.name) {
            return Err("scenario name must be a non-empty portable label".into());
        }
        if self.description.trim().is_empty() {
            return Err(format!(
                "scenario '{}' must have a non-empty description",
                self.name
            ));
        }
        if self.source_names.len() != self.simulation.sources.len() {
            return Err(format!(
                "scenario '{}' has {} source labels for {} sources",
                self.name,
                self.source_names.len(),
                self.simulation.sources.len()
            ));
        }
        if self.simulation.sources.is_empty() {
            return Err(format!(
                "scenario '{}' must contain at least one source",
                self.name
            ));
        }
        if self.simulation.listening_positions.is_empty() {
            return Err(format!(
                "scenario '{}' must contain at least one listening position",
                self.name
            ));
        }

        let mut source_labels = HashSet::with_capacity(self.source_names.len());
        for (index, (label, source)) in self
            .source_names
            .iter()
            .zip(&self.simulation.sources)
            .enumerate()
        {
            if !portable_label(label) {
                return Err(format!(
                    "scenario '{}' source label {index} is not portable",
                    self.name
                ));
            }
            if label != &source.name {
                return Err(format!(
                    "scenario '{}' source label {index} '{}' does not match source name '{}'",
                    self.name, label, source.name
                ));
            }
            if !source_labels.insert(label.to_ascii_lowercase()) {
                return Err(format!(
                    "scenario '{}' source labels are not unique case-insensitively",
                    self.name
                ));
            }
            if !point_is_finite(source.position)
                || !source.amplitude.is_finite()
                || source.amplitude < 0.0
            {
                return Err(format!(
                    "scenario '{}' source {index} has invalid position or amplitude",
                    self.name
                ));
            }
        }

        let (width, depth, height) = self.simulation.room.dimensions();
        if [width, depth, height, self.simulation.speed_of_sound]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(format!(
                "scenario '{}' has invalid room dimensions or sound speed",
                self.name
            ));
        }
        if self
            .simulation
            .sources
            .iter()
            .map(|source| source.position)
            .chain(self.simulation.listening_positions.iter().copied())
            .any(|point| !point_inside_room(point, width, depth, height))
        {
            return Err(format!(
                "scenario '{}' contains a source or listener outside its room",
                self.name
            ));
        }

        if self.simulation.frequencies.is_empty()
            || self
                .simulation
                .frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
            || self
                .simulation
                .frequencies
                .windows(2)
                .any(|window| window[1] <= window[0])
        {
            return Err(format!(
                "scenario '{}' must have a finite strictly increasing positive frequency grid",
                self.name
            ));
        }
        Ok(())
    }
}

fn portable_label(value: &str) -> bool {
    crate::is_portable_filename_stem(value)
}

fn point_is_finite(point: Point3D) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn point_inside_room(point: Point3D, width: f64, depth: f64, height: f64) -> bool {
    point_is_finite(point)
        && point.x >= 0.0
        && point.x <= width
        && point.y >= 0.0
        && point.y <= depth
        && point.z >= 0.0
        && point.z <= height
}

/// Default frequency configuration
const DEFAULT_MIN_FREQ: f64 = 20.0;
const DEFAULT_MAX_FREQ: f64 = 2000.0;
const DEFAULT_NUM_POINTS: usize = 200;

/// Runtime frequency configuration (overridable via CLI)
pub struct FrequencyConfig {
    pub min_freq: f64,
    pub max_freq: f64,
    pub num_points: usize,
    /// Grid spacing (log default; linear for modal/narrowband work).
    pub spacing: FrequencySpacing,
}

impl Default for FrequencyConfig {
    fn default() -> Self {
        Self {
            min_freq: DEFAULT_MIN_FREQ,
            max_freq: DEFAULT_MAX_FREQ,
            num_points: DEFAULT_NUM_POINTS,
            spacing: FrequencySpacing::default(),
        }
    }
}

/// Crossover frequency for sub/satellite splits
const CROSSOVER_FREQ: f64 = 80.0;
/// Butterworth order for crossover filters
const CROSSOVER_ORDER: u32 = 4;

/// Generate 5 listening positions: the main LP plus 4 positions offset by a
/// fixed linear distance (head-width scale) in the up/down/left/right
/// directions as seen from the speaker midpoint.
///
/// Heads don't grow with the room, so a fixed linear offset is the right
/// model — earlier versions scaled the offset by `dist · tan(angle)`, which
/// produced 18 cm offsets in a 2 m room and 31 cm in a 3.5 m room for the
/// same nominal "5°".
/// Minimum distance (m) from a listening position to any room boundary.
const LP_WALL_MARGIN: f64 = 0.1;

fn clamp_lp_to_room(p: Point3D, room: &RectangularRoom) -> Point3D {
    Point3D::new(
        p.x.clamp(LP_WALL_MARGIN, room.width - LP_WALL_MARGIN),
        p.y.clamp(LP_WALL_MARGIN, room.depth - LP_WALL_MARGIN),
        p.z.clamp(LP_WALL_MARGIN, room.height - LP_WALL_MARGIN),
    )
}

/// Generate 5 listening positions: the main LP plus 4 positions offset by a
/// fixed linear distance (head-width scale) in the up/down/left/right
/// directions as seen from the speaker midpoint.  All positions are clamped
/// to stay inside the room with a small wall margin so they never land
/// outside the mesh or inside a boundary layer.
fn multi_lp_from_speakers(
    lp: Point3D,
    speakers: &[Point3D],
    offset_m: f64,
    room: &RectangularRoom,
) -> Vec<Point3D> {
    // Compute speaker midpoint
    let n = speakers.len() as f64;
    let mid = Point3D::new(
        speakers.iter().map(|s| s.x).sum::<f64>() / n,
        speakers.iter().map(|s| s.y).sum::<f64>() / n,
        speakers.iter().map(|s| s.z).sum::<f64>() / n,
    );

    // Direction from speaker midpoint to LP
    let dx = lp.x - mid.x;
    let dy = lp.y - mid.y;
    let dz = lp.z - mid.z;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist < 1e-6 {
        return vec![clamp_lp_to_room(lp, room); 5];
    }

    // Normalized forward direction
    let fwd = (dx / dist, dy / dist, dz / dist);

    // Build a "right" vector perpendicular to forward and world-up (0,0,1):
    // right = forward × up
    let rx = fwd.1; // fy*1 - fz*0
    let ry = -fwd.0; // fz*0 - fx*1
    let rz = 0.0; // fx*0 - fy*0
    let rlen = (rx * rx + ry * ry + rz * rz).sqrt();
    let right = if rlen > 1e-6 {
        (rx / rlen, ry / rlen, rz / rlen)
    } else {
        // Forward is nearly vertical, use (1,0,0) as fallback right
        (1.0, 0.0, 0.0)
    };

    // Build "up" vector = right × forward
    let ux = right.1 * fwd.2 - right.2 * fwd.1;
    let uy = right.2 * fwd.0 - right.0 * fwd.2;
    let uz = right.0 * fwd.1 - right.1 * fwd.0;

    let offset = offset_m;

    vec![
        clamp_lp_to_room(lp, room), // center (primary)
        clamp_lp_to_room(
            Point3D::new(
                lp.x + right.0 * offset,
                lp.y + right.1 * offset,
                lp.z + right.2 * offset,
            ),
            room,
        ), // right
        clamp_lp_to_room(
            Point3D::new(
                lp.x - right.0 * offset,
                lp.y - right.1 * offset,
                lp.z - right.2 * offset,
            ),
            room,
        ), // left
        clamp_lp_to_room(
            Point3D::new(lp.x + ux * offset, lp.y + uy * offset, lp.z + uz * offset),
            room,
        ), // up
        clamp_lp_to_room(
            Point3D::new(lp.x - ux * offset, lp.y - uy * offset, lp.z - uz * offset),
            room,
        ), // down
    ]
}

/// Linear offset (m) for multi-measurement positions — chosen as one head-half-width
/// so the four off-center LPs roughly trace where a real head would shift during
/// listening.
const MULTI_LP_OFFSET_M: f64 = 0.10;

fn make_hp_source(position: Point3D, name: &str, aim: Point3D) -> Source {
    Source::classical(position, 60.0, 40.0, 1.0)
        .with_name(name.to_string())
        .with_crossover(CrossoverFilter::Highpass {
            cutoff_freq: CROSSOVER_FREQ,
            order: CROSSOVER_ORDER,
        })
        .aimed_at(aim)
}

fn make_lp_source(position: Point3D, name: &str) -> Source {
    // Subwoofers use the omnidirectional grid directivity, so orientation has
    // no acoustic effect; we leave it at its default.
    Source::omnidirectional(position, 1.0)
        .with_name(name.to_string())
        .with_crossover(CrossoverFilter::Lowpass {
            cutoff_freq: CROSSOVER_FREQ,
            order: CROSSOVER_ORDER,
        })
}

fn make_fullrange_source(position: Point3D, name: &str, aim: Point3D) -> Source {
    Source::classical(position, 60.0, 40.0, 1.0)
        .with_name(name.to_string())
        .aimed_at(aim)
}

fn default_boundaries() -> BoundaryConfig {
    BoundaryConfig {
        floor: SurfaceConfig::Absorption { coefficient: 0.3 },
        ceiling: SurfaceConfig::Absorption { coefficient: 0.05 },
        walls: SurfaceConfig::Absorption { coefficient: 0.1 },
        ..Default::default()
    }
}

fn concrete_basement_boundaries() -> BoundaryConfig {
    BoundaryConfig {
        floor: SurfaceConfig::Absorption { coefficient: 0.1 },
        ceiling: SurfaceConfig::Absorption { coefficient: 0.1 },
        walls: SurfaceConfig::Absorption { coefficient: 0.05 },
        ..Default::default()
    }
}

fn treated_front_boundaries() -> BoundaryConfig {
    BoundaryConfig {
        floor: SurfaceConfig::Absorption { coefficient: 0.3 },
        ceiling: SurfaceConfig::Absorption { coefficient: 0.1 },
        walls: SurfaceConfig::Absorption { coefficient: 0.08 },
        front_wall: Some(SurfaceConfig::Absorption { coefficient: 0.4 }),
        back_wall: Some(SurfaceConfig::Absorption { coefficient: 0.2 }),
        left_wall: None,
        right_wall: None,
    }
}

fn asymmetric_side_boundaries() -> BoundaryConfig {
    BoundaryConfig {
        floor: SurfaceConfig::Absorption { coefficient: 0.25 },
        ceiling: SurfaceConfig::Absorption { coefficient: 0.08 },
        walls: SurfaceConfig::Absorption { coefficient: 0.12 },
        front_wall: None,
        back_wall: None,
        left_wall: Some(SurfaceConfig::Absorption { coefficient: 0.25 }),
        right_wall: Some(SurfaceConfig::Absorption { coefficient: 0.05 }),
    }
}

fn make_simulation_with_boundaries(
    room: RectangularRoom,
    sources: Vec<Source>,
    listening_positions: Vec<Point3D>,
    boundaries: BoundaryConfig,
    freq_config: &FrequencyConfig,
) -> RoomSimulation {
    let mut sim = RoomSimulation::with_frequencies_spaced(
        RoomGeometry::Rectangular(room),
        sources,
        listening_positions,
        freq_config.min_freq,
        freq_config.max_freq,
        freq_config.num_points,
        freq_config.spacing,
    );
    sim.boundaries = boundaries;
    sim
}

fn make_simulation(
    room: RectangularRoom,
    sources: Vec<Source>,
    listening_positions: Vec<Point3D>,
    freq_config: &FrequencyConfig,
) -> RoomSimulation {
    make_simulation_with_boundaries(
        room,
        sources,
        listening_positions,
        default_boundaries(),
        freq_config,
    )
}

/// Generate all 20 scenarios with the given frequency configuration
pub fn all_scenarios_with_config(freq_config: &FrequencyConfig) -> Vec<Scenario> {
    vec![
        scenario_01_small_stereo_2_0(freq_config),
        scenario_02_small_stereo_2_1(freq_config),
        scenario_03_small_stereo_2_2_mso(freq_config),
        scenario_03_small_stereo_2_2_cardioid(freq_config),
        scenario_03_small_stereo_2_2_group(freq_config),
        scenario_04_medium_stereo(freq_config),
        scenario_05_medium_2_1(freq_config),
        scenario_06_medium_multi_sub_4(freq_config),
        scenario_07_medium_multi_seat(freq_config),
        scenario_08_large_stereo(freq_config),
        scenario_09_large_2_1(freq_config),
        scenario_10_large_multi_sub_4(freq_config),
        scenario_11_large_multi_seat_2_1(freq_config),
        scenario_12_medium_multi_sub_multi_seat(freq_config),
        scenario_13_medium_surround_5_0(freq_config),
        scenario_14_medium_surround_5_1(freq_config),
        scenario_15_medium_surround_5_1_4(freq_config),
        scenario_16_large_surround_5_1(freq_config),
        scenario_17_large_surround_5_1_4(freq_config),
        scenario_18_medium_surround_5_2_4_multi_seat(freq_config),
    ]
}

/// Generate all 20 scenarios with default frequency configuration
pub fn all_scenarios() -> Vec<Scenario> {
    all_scenarios_with_config(&FrequencyConfig::default())
}

/// Look up a scenario by name (using default frequency configuration)
pub fn scenario_by_name(name: &str) -> Option<Scenario> {
    all_scenarios().into_iter().find(|s| s.name == name)
}

/// Look up a scenario by name with a specific frequency configuration
pub fn scenario_by_name_with_config(name: &str, freq_config: &FrequencyConfig) -> Option<Scenario> {
    all_scenarios_with_config(freq_config)
        .into_iter()
        .find(|s| s.name == name)
}

// ============================================================================
// Small room: 3 x 3 x 2.4 m
// ============================================================================

fn small_room() -> RectangularRoom {
    RectangularRoom::new(3.0, 3.0, 2.4)
}

/// Scenario 1: Small room, stereo 2.0, fullrange, 5 LPs (center + 4 at ±5°)
fn scenario_01_small_stereo_2_0(freq_config: &FrequencyConfig) -> Scenario {
    let room = small_room();
    let left_pos = Point3D::new(0.8, 0.3, 1.1);
    let right_pos = Point3D::new(2.2, 0.3, 1.1);
    let lp = Point3D::new(1.5, 2.0, 1.1);
    let left = make_fullrange_source(left_pos, "left", lp);
    let right = make_fullrange_source(right_pos, "right", lp);
    let lps = multi_lp_from_speakers(lp, &[left_pos, right_pos], MULTI_LP_OFFSET_M, &room);

    Scenario {
        name: "small_stereo_2_0".to_string(),
        description: "Small 3x3x2.4m room, stereo 2.0, fullrange, 5 LPs".to_string(),
        simulation: make_simulation(room, vec![left, right], lps, freq_config),
        source_names: vec!["left".to_string(), "right".to_string()],
    }
}

/// Scenario 2: Small room, 2.1, HP@80Hz mains + LP@80Hz sub, 5 LPs (center + 4 at ±5°)
fn scenario_02_small_stereo_2_1(freq_config: &FrequencyConfig) -> Scenario {
    let room = small_room();
    let left_pos = Point3D::new(0.8, 0.3, 1.1);
    let right_pos = Point3D::new(2.2, 0.3, 1.1);
    let lp = Point3D::new(1.5, 2.0, 1.1);
    let left = make_hp_source(left_pos, "left", lp);
    let right = make_hp_source(right_pos, "right", lp);
    let sub = make_lp_source(Point3D::new(0.3, 0.3, 0.15), "subwoofer");
    let lps = multi_lp_from_speakers(lp, &[left_pos, right_pos], MULTI_LP_OFFSET_M, &room);

    Scenario {
        name: "small_stereo_2_1".to_string(),
        description: "Small 3x3x2.4m room, 2.1, sub at front-left corner, 5 LPs".to_string(),
        simulation: make_simulation(room, vec![left, right, sub], lps, freq_config),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 3a: Small room, 2 subs (corners) + HP mains, 5 LPs (MSO)
fn scenario_03_small_stereo_2_2_mso(freq_config: &FrequencyConfig) -> Scenario {
    let room = small_room();
    let left_pos = Point3D::new(0.8, 0.3, 1.1);
    let right_pos = Point3D::new(2.2, 0.3, 1.1);
    let lp = Point3D::new(1.5, 2.0, 1.1);
    let left = make_hp_source(left_pos, "left", lp);
    let right = make_hp_source(right_pos, "right", lp);
    let sub1 = make_lp_source(Point3D::new(0.15, 0.15, 0.15), "sub1");
    let sub2 = make_lp_source(Point3D::new(2.85, 0.15, 0.15), "sub2");
    let lps = multi_lp_from_speakers(lp, &[left_pos, right_pos], MULTI_LP_OFFSET_M, &room);

    Scenario {
        name: "small_stereo_2_2_mso".to_string(),
        description: "Small 3x3x2.4m room, 2 subs in front corners (MSO), 5 LPs".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, sub1, sub2],
            lps,
            concrete_basement_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "sub1".to_string(),
            "sub2".to_string(),
        ],
    }
}

/// Scenario 3b: Small room, 2 vertically-stacked subs + HP mains, 5 LPs.
/// Note: this is a stacked-sub configuration, not a true cardioid (which
/// would require front+rear units with delay/inversion). Name retained for
/// dataset stability.
fn scenario_03_small_stereo_2_2_cardioid(freq_config: &FrequencyConfig) -> Scenario {
    let room = small_room();
    let left_pos = Point3D::new(0.8, 0.3, 1.1);
    let right_pos = Point3D::new(2.2, 0.3, 1.1);
    let lp = Point3D::new(1.5, 2.0, 1.1);
    let left = make_hp_source(left_pos, "left", lp);
    let right = make_hp_source(right_pos, "right", lp);
    // Stacked subs at front-left: bottom at z=0.15, top at z=0.65 (0.5 m apart).
    let sub_bottom = make_lp_source(Point3D::new(0.3, 0.3, 0.15), "sub_bottom");
    let sub_top = make_lp_source(Point3D::new(0.3, 0.3, 0.65), "sub_top");
    let lps = multi_lp_from_speakers(lp, &[left_pos, right_pos], MULTI_LP_OFFSET_M, &room);

    Scenario {
        name: "small_stereo_2_2_cardioid".to_string(),
        description: "Small 3x3x2.4m room, stacked cardioid subs, 5 LPs".to_string(),
        simulation: make_simulation(
            room,
            vec![left, right, sub_bottom, sub_top],
            lps,
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "sub_bottom".to_string(),
            "sub_top".to_string(),
        ],
    }
}

/// Scenario 3c: Small room, 2 subs (grouped with mains) + HP mains, 5 LPs
/// Each sub is directly below its corresponding main speaker.
/// Naming convention: `{main}_sub` signals Group config to config gen.
fn scenario_03_small_stereo_2_2_group(freq_config: &FrequencyConfig) -> Scenario {
    let room = small_room();
    let left_pos = Point3D::new(0.8, 0.3, 1.1);
    let right_pos = Point3D::new(2.2, 0.3, 1.1);
    let lp = Point3D::new(1.5, 2.0, 1.1);
    let left = make_hp_source(left_pos, "left", lp);
    let right = make_hp_source(right_pos, "right", lp);
    // Subs directly below their respective mains, on the floor
    let left_sub = make_lp_source(Point3D::new(0.8, 0.3, 0.15), "left_sub");
    let right_sub = make_lp_source(Point3D::new(2.2, 0.3, 0.15), "right_sub");
    let lps = multi_lp_from_speakers(lp, &[left_pos, right_pos], MULTI_LP_OFFSET_M, &room);

    Scenario {
        name: "small_stereo_2_2_group".to_string(),
        description: "Small 3x3x2.4m room, grouped subs below mains, 5 LPs".to_string(),
        simulation: make_simulation(
            room,
            vec![left, right, left_sub, right_sub],
            lps,
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "left_sub".to_string(),
            "right_sub".to_string(),
        ],
    }
}

// ============================================================================
// Medium room: 5 x 4 x 2.5 m
// ============================================================================

fn medium_room() -> RectangularRoom {
    RectangularRoom::new(5.0, 4.0, 2.5)
}

/// Scenario 4: Medium room, stereo 2.0, fullrange, 1 LP
fn scenario_04_medium_stereo(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);
    let left = make_fullrange_source(Point3D::new(1.2, 0.4, 1.1), "left", lp);
    let right = make_fullrange_source(Point3D::new(3.8, 0.4, 1.1), "right", lp);

    Scenario {
        name: "medium_stereo_2_0".to_string(),
        description: "Medium 5x4x2.5m room, stereo 2.0, fullrange".to_string(),
        simulation: make_simulation(room, vec![left, right], vec![lp], freq_config),
        source_names: vec!["left".to_string(), "right".to_string()],
    }
}

/// Scenario 5: Medium room, 2.1, 1 LP
fn scenario_05_medium_2_1(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);
    let left = make_hp_source(Point3D::new(1.2, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(3.8, 0.4, 1.1), "right", lp);
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "medium_stereo_2_1".to_string(),
        description: "Medium 5x4x2.5m room, 2.1".to_string(),
        simulation: make_simulation(room, vec![left, right, sub], vec![lp], freq_config),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 6: Medium room, 4 subs (corners) + HP mains, 1 LP
fn scenario_06_medium_multi_sub_4(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);
    let left = make_hp_source(Point3D::new(1.2, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(3.8, 0.4, 1.1), "right", lp);
    let sub1 = make_lp_source(Point3D::new(0.15, 0.15, 0.15), "sub1");
    let sub2 = make_lp_source(Point3D::new(4.85, 0.15, 0.15), "sub2");
    let sub3 = make_lp_source(Point3D::new(0.15, 3.85, 0.15), "sub3");
    let sub4 = make_lp_source(Point3D::new(4.85, 3.85, 0.15), "sub4");

    Scenario {
        name: "medium_multi_sub_4".to_string(),
        description: "Medium 5x4x2.5m room, 4 corner subs".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, sub1, sub2, sub3, sub4],
            vec![lp],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "sub1".to_string(),
            "sub2".to_string(),
            "sub3".to_string(),
            "sub4".to_string(),
        ],
    }
}

/// Scenario 7: Medium room, stereo fullrange, 3 listening positions
fn scenario_07_medium_multi_seat(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp0 = Point3D::new(2.5, 2.5, 1.1); // center seat
    let lp1 = Point3D::new(1.5, 2.5, 1.1); // left seat
    let lp2 = Point3D::new(3.5, 2.5, 1.1); // right seat
    // Aim speakers at the center seat (canonical "sweet spot" for multi-seat).
    let left = make_fullrange_source(Point3D::new(1.2, 0.4, 1.1), "left", lp0);
    let right = make_fullrange_source(Point3D::new(3.8, 0.4, 1.1), "right", lp0);

    Scenario {
        name: "medium_multi_seat".to_string(),
        description: "Medium 5x4x2.5m room, stereo, 3 seats".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right],
            vec![lp0, lp1, lp2],
            asymmetric_side_boundaries(),
            freq_config,
        ),
        source_names: vec!["left".to_string(), "right".to_string()],
    }
}

// ============================================================================
// Large room: 7 x 5.5 x 2.6 m
// ============================================================================

fn large_room() -> RectangularRoom {
    RectangularRoom::new(7.0, 5.5, 2.6)
}

/// Scenario 8: Large room, stereo 2.0, fullrange, 1 LP
fn scenario_08_large_stereo(freq_config: &FrequencyConfig) -> Scenario {
    let room = large_room();
    let lp = Point3D::new(3.5, 3.5, 1.1);
    let left = make_fullrange_source(Point3D::new(1.5, 0.4, 1.1), "left", lp);
    let right = make_fullrange_source(Point3D::new(5.5, 0.4, 1.1), "right", lp);

    Scenario {
        name: "large_stereo_2_0".to_string(),
        description: "Large 7x5.5x2.6m room, stereo 2.0, fullrange".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right],
            vec![lp],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec!["left".to_string(), "right".to_string()],
    }
}

/// Scenario 9: Large room, 2.1, 1 LP
fn scenario_09_large_2_1(freq_config: &FrequencyConfig) -> Scenario {
    let room = large_room();
    let lp = Point3D::new(3.5, 3.5, 1.1);
    let left = make_hp_source(Point3D::new(1.5, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(5.5, 0.4, 1.1), "right", lp);
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "large_stereo_2_1".to_string(),
        description: "Large 7x5.5x2.6m room, 2.1".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, sub],
            vec![lp],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 10: Large room, 4 subs + HP mains, 1 LP
fn scenario_10_large_multi_sub_4(freq_config: &FrequencyConfig) -> Scenario {
    let room = large_room();
    let lp = Point3D::new(3.5, 3.5, 1.1);
    let left = make_hp_source(Point3D::new(1.5, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(5.5, 0.4, 1.1), "right", lp);
    let sub1 = make_lp_source(Point3D::new(0.15, 0.15, 0.15), "sub1");
    let sub2 = make_lp_source(Point3D::new(6.85, 0.15, 0.15), "sub2");
    let sub3 = make_lp_source(Point3D::new(0.15, 5.35, 0.15), "sub3");
    let sub4 = make_lp_source(Point3D::new(6.85, 5.35, 0.15), "sub4");

    Scenario {
        name: "large_multi_sub_4".to_string(),
        description: "Large 7x5.5x2.6m room, 4 corner subs".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, sub1, sub2, sub3, sub4],
            vec![lp],
            concrete_basement_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "sub1".to_string(),
            "sub2".to_string(),
            "sub3".to_string(),
            "sub4".to_string(),
        ],
    }
}

/// Scenario 11: Large room, 2.1, 3 listening positions
fn scenario_11_large_multi_seat_2_1(freq_config: &FrequencyConfig) -> Scenario {
    let room = large_room();
    let lp0 = Point3D::new(3.5, 3.5, 1.1);
    let lp1 = Point3D::new(2.3, 3.5, 1.1);
    let lp2 = Point3D::new(4.7, 3.5, 1.1);
    let left = make_hp_source(Point3D::new(1.5, 0.4, 1.1), "left", lp0);
    let right = make_hp_source(Point3D::new(5.5, 0.4, 1.1), "right", lp0);
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "large_multi_seat_2_1".to_string(),
        description: "Large 7x5.5x2.6m room, 2.1, 3 seats".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, sub],
            vec![lp0, lp1, lp2],
            asymmetric_side_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 12: Medium room, 2 subs + HP mains, 2 listening positions
fn scenario_12_medium_multi_sub_multi_seat(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp0 = Point3D::new(2.5, 2.5, 1.1);
    let lp1 = Point3D::new(2.5, 3.2, 1.1);
    let left = make_hp_source(Point3D::new(1.2, 0.4, 1.1), "left", lp0);
    let right = make_hp_source(Point3D::new(3.8, 0.4, 1.1), "right", lp0);
    let sub1 = make_lp_source(Point3D::new(0.15, 0.15, 0.15), "sub1");
    let sub2 = make_lp_source(Point3D::new(4.85, 0.15, 0.15), "sub2");

    Scenario {
        name: "medium_multi_sub_multi_seat".to_string(),
        description: "Medium 5x4x2.5m room, 2 subs, 2 seats".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, sub1, sub2],
            vec![lp0, lp1],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "sub1".to_string(),
            "sub2".to_string(),
        ],
    }
}

// ============================================================================
// Medium room surround: 5 x 4 x 2.5 m
// Speaker placement follows ITU-R BS.775 / ITU-R BS.2051 guidelines:
//   - L/R at ±30° from center, front wall
//   - Center at 0° (front wall center)
//   - SL/SR at ±110° from center, behind listener
//   - Sub at front-left corner, floor level
//   - Height channels near ceiling, at ±45° front / ±135° rear
// ============================================================================

/// Scenario 13: Medium room, 5.0 surround (L/R/C/SL/SR), fullrange, 1 LP
fn scenario_13_medium_surround_5_0(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);
    let left = make_fullrange_source(Point3D::new(1.2, 0.4, 1.1), "left", lp);
    let right = make_fullrange_source(Point3D::new(3.8, 0.4, 1.1), "right", lp);
    let center = make_fullrange_source(Point3D::new(2.5, 0.4, 1.1), "center", lp);
    let surr_left = make_fullrange_source(Point3D::new(0.4, 3.5, 1.1), "surround_left", lp);
    let surr_right = make_fullrange_source(Point3D::new(4.6, 3.5, 1.1), "surround_right", lp);

    Scenario {
        name: "medium_surround_5_0".to_string(),
        description: "Medium 5x4x2.5m room, 5.0 surround, fullrange".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, center, surr_left, surr_right],
            vec![lp],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "center".to_string(),
            "surround_left".to_string(),
            "surround_right".to_string(),
        ],
    }
}

/// Scenario 14: Medium room, 5.1 surround (L/R/C/SL/SR + LFE), HP mains, 1 LP
fn scenario_14_medium_surround_5_1(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);
    let left = make_hp_source(Point3D::new(1.2, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(3.8, 0.4, 1.1), "right", lp);
    let center = make_hp_source(Point3D::new(2.5, 0.4, 1.1), "center", lp);
    let surr_left = make_hp_source(Point3D::new(0.4, 3.5, 1.1), "surround_left", lp);
    let surr_right = make_hp_source(Point3D::new(4.6, 3.5, 1.1), "surround_right", lp);
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "medium_surround_5_1".to_string(),
        description: "Medium 5x4x2.5m room, 5.1 surround".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, center, surr_left, surr_right, sub],
            vec![lp],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "center".to_string(),
            "surround_left".to_string(),
            "surround_right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 15: Medium room, 5.1.4 immersive (5.1 + 4 height channels), HP all, 1 LP
fn scenario_15_medium_surround_5_1_4(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);
    // Ear-level speakers
    let left = make_hp_source(Point3D::new(1.2, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(3.8, 0.4, 1.1), "right", lp);
    let center = make_hp_source(Point3D::new(2.5, 0.4, 1.1), "center", lp);
    let surr_left = make_hp_source(Point3D::new(0.4, 3.5, 1.1), "surround_left", lp);
    let surr_right = make_hp_source(Point3D::new(4.6, 3.5, 1.1), "surround_right", lp);
    // Height speakers near ceiling — aimed downward at the LP, as Atmos
    // height channels are intended to fire toward the listener.
    let top_fl = make_hp_source(Point3D::new(1.2, 1.0, 2.3), "top_front_left", lp);
    let top_fr = make_hp_source(Point3D::new(3.8, 1.0, 2.3), "top_front_right", lp);
    let top_rl = make_hp_source(Point3D::new(1.2, 3.5, 2.3), "top_rear_left", lp);
    let top_rr = make_hp_source(Point3D::new(3.8, 3.5, 2.3), "top_rear_right", lp);
    // Subwoofer
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "medium_surround_5_1_4".to_string(),
        description: "Medium 5x4x2.5m room, 5.1.4 immersive audio".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![
                left, right, center, surr_left, surr_right, top_fl, top_fr, top_rl, top_rr, sub,
            ],
            vec![lp],
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "center".to_string(),
            "surround_left".to_string(),
            "surround_right".to_string(),
            "top_front_left".to_string(),
            "top_front_right".to_string(),
            "top_rear_left".to_string(),
            "top_rear_right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

// ============================================================================
// Large room surround: 7 x 5.5 x 2.6 m
// ============================================================================

/// Scenario 16: Large room, 5.1 surround, HP mains, 1 LP
fn scenario_16_large_surround_5_1(freq_config: &FrequencyConfig) -> Scenario {
    let room = large_room();
    let lp = Point3D::new(3.5, 3.5, 1.1);
    let left = make_hp_source(Point3D::new(1.5, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(5.5, 0.4, 1.1), "right", lp);
    let center = make_hp_source(Point3D::new(3.5, 0.4, 1.1), "center", lp);
    let surr_left = make_hp_source(Point3D::new(0.5, 4.5, 1.1), "surround_left", lp);
    let surr_right = make_hp_source(Point3D::new(6.5, 4.5, 1.1), "surround_right", lp);
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "large_surround_5_1".to_string(),
        description: "Large 7x5.5x2.6m room, 5.1 surround".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![left, right, center, surr_left, surr_right, sub],
            vec![lp],
            asymmetric_side_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "center".to_string(),
            "surround_left".to_string(),
            "surround_right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 17: Large room, 5.1.4 immersive, HP all, 1 LP
fn scenario_17_large_surround_5_1_4(freq_config: &FrequencyConfig) -> Scenario {
    let room = large_room();
    let lp = Point3D::new(3.5, 3.5, 1.1);
    // Ear-level speakers
    let left = make_hp_source(Point3D::new(1.5, 0.4, 1.1), "left", lp);
    let right = make_hp_source(Point3D::new(5.5, 0.4, 1.1), "right", lp);
    let center = make_hp_source(Point3D::new(3.5, 0.4, 1.1), "center", lp);
    let surr_left = make_hp_source(Point3D::new(0.5, 4.5, 1.1), "surround_left", lp);
    let surr_right = make_hp_source(Point3D::new(6.5, 4.5, 1.1), "surround_right", lp);
    // Height speakers near ceiling — aimed at LP.
    let top_fl = make_hp_source(Point3D::new(1.5, 1.0, 2.4), "top_front_left", lp);
    let top_fr = make_hp_source(Point3D::new(5.5, 1.0, 2.4), "top_front_right", lp);
    let top_rl = make_hp_source(Point3D::new(1.5, 4.5, 2.4), "top_rear_left", lp);
    let top_rr = make_hp_source(Point3D::new(5.5, 4.5, 2.4), "top_rear_right", lp);
    // Subwoofer
    let sub = make_lp_source(Point3D::new(0.5, 0.5, 0.15), "subwoofer");

    Scenario {
        name: "large_surround_5_1_4".to_string(),
        description: "Large 7x5.5x2.6m room, 5.1.4 immersive audio".to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            vec![
                left, right, center, surr_left, surr_right, top_fl, top_fr, top_rl, top_rr, sub,
            ],
            vec![lp],
            asymmetric_side_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "center".to_string(),
            "surround_left".to_string(),
            "surround_right".to_string(),
            "top_front_left".to_string(),
            "top_front_right".to_string(),
            "top_rear_left".to_string(),
            "top_rear_right".to_string(),
            "subwoofer".to_string(),
        ],
    }
}

/// Scenario 18: Medium room, 5.2.4 immersive, two physical subs and five LPs.
///
/// Every bed, height, and subwoofer source is evaluated at the same ordered
/// listening positions. The generated RoomEQ config can therefore exercise
/// all-channel spatial optimization and MSO without mixing seat identities.
fn scenario_18_medium_surround_5_2_4_multi_seat(freq_config: &FrequencyConfig) -> Scenario {
    let room = medium_room();
    let lp = Point3D::new(2.5, 2.5, 1.1);

    let left_pos = Point3D::new(1.2, 0.4, 1.1);
    let right_pos = Point3D::new(3.8, 0.4, 1.1);
    let center_pos = Point3D::new(2.5, 0.4, 1.1);
    let surround_left_pos = Point3D::new(0.4, 3.5, 1.1);
    let surround_right_pos = Point3D::new(4.6, 3.5, 1.1);
    let top_front_left_pos = Point3D::new(1.2, 1.0, 2.3);
    let top_front_right_pos = Point3D::new(3.8, 1.0, 2.3);
    let top_rear_left_pos = Point3D::new(1.2, 3.5, 2.3);
    let top_rear_right_pos = Point3D::new(3.8, 3.5, 2.3);

    let speaker_positions = [
        left_pos,
        right_pos,
        center_pos,
        surround_left_pos,
        surround_right_pos,
        top_front_left_pos,
        top_front_right_pos,
        top_rear_left_pos,
        top_rear_right_pos,
    ];
    let listening_positions =
        multi_lp_from_speakers(lp, &speaker_positions, MULTI_LP_OFFSET_M, &room);

    let sources = vec![
        make_hp_source(left_pos, "left", lp),
        make_hp_source(right_pos, "right", lp),
        make_hp_source(center_pos, "center", lp),
        make_hp_source(surround_left_pos, "surround_left", lp),
        make_hp_source(surround_right_pos, "surround_right", lp),
        make_hp_source(top_front_left_pos, "top_front_left", lp),
        make_hp_source(top_front_right_pos, "top_front_right", lp),
        make_hp_source(top_rear_left_pos, "top_rear_left", lp),
        make_hp_source(top_rear_right_pos, "top_rear_right", lp),
        make_lp_source(Point3D::new(0.5, 0.5, 0.15), "sub1"),
        make_lp_source(Point3D::new(4.5, 0.5, 0.15), "sub2"),
    ];

    Scenario {
        name: "medium_surround_5_2_4_multi_seat".to_string(),
        description: "Medium 5x4x2.5m room, 5.2.4 immersive audio, two subs, five seats"
            .to_string(),
        simulation: make_simulation_with_boundaries(
            room,
            sources,
            listening_positions,
            treated_front_boundaries(),
            freq_config,
        ),
        source_names: vec![
            "left".to_string(),
            "right".to_string(),
            "center".to_string(),
            "surround_left".to_string(),
            "surround_right".to_string(),
            "top_front_left".to_string(),
            "top_front_right".to_string(),
            "top_rear_left".to_string(),
            "top_rear_right".to_string(),
            "sub1".to_string(),
            "sub2".to_string(),
        ],
    }
}

fn jitter_absorption_coefficient(rng: &mut SmallRng, base: f64, amount: f64) -> f64 {
    let delta: f64 = rng.random_range(-amount..amount);
    let value = base + delta;
    value.clamp(0.02, 0.9)
}

fn jitter_surface(rng: &mut SmallRng, surface: &SurfaceConfig, amount: f64) -> SurfaceConfig {
    match surface {
        SurfaceConfig::Absorption { coefficient } => SurfaceConfig::Absorption {
            coefficient: jitter_absorption_coefficient(rng, *coefficient, amount),
        },
        other => other.clone(),
    }
}

fn jitter_boundaries(rng: &mut SmallRng, base: &BoundaryConfig, amount: f64) -> BoundaryConfig {
    BoundaryConfig {
        floor: jitter_surface(rng, &base.floor, amount),
        ceiling: jitter_surface(rng, &base.ceiling, amount),
        walls: jitter_surface(rng, &base.walls, amount),
        front_wall: base
            .front_wall
            .as_ref()
            .map(|s| jitter_surface(rng, s, amount)),
        back_wall: base
            .back_wall
            .as_ref()
            .map(|s| jitter_surface(rng, s, amount)),
        left_wall: base
            .left_wall
            .as_ref()
            .map(|s| jitter_surface(rng, s, amount)),
        right_wall: base
            .right_wall
            .as_ref()
            .map(|s| jitter_surface(rng, s, amount)),
    }
}

fn jitter_point_within_room(
    rng: &mut SmallRng,
    p: Point3D,
    max_dx: f64,
    max_dy: f64,
    max_dz: f64,
    room: &RectangularRoom,
) -> Point3D {
    let dx: f64 = rng.random_range(-max_dx..max_dx);
    let dy: f64 = rng.random_range(-max_dy..max_dy);
    let dz: f64 = rng.random_range(-max_dz..max_dz);

    let mut x = p.x + dx;
    let mut y = p.y + dy;
    let mut z = p.z + dz;

    if x < 0.1 {
        x = 0.1;
    }
    if x > room.width - 0.1 {
        x = room.width - 0.1;
    }
    if y < 0.1 {
        y = 0.1;
    }
    if y > room.depth - 0.1 {
        y = room.depth - 0.1;
    }
    if z < 0.1 {
        z = 0.1;
    }
    if z > room.height - 0.1 {
        z = room.height - 0.1;
    }

    Point3D::new(x, y, z)
}

fn randomized_scenario_variant(base: &Scenario, variant_index: usize, seed: u64) -> Scenario {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut simulation = base.simulation.clone();

    let room = match simulation.room {
        RoomGeometry::Rectangular(ref room) => room,
        _ => return base.clone(),
    };

    let max_dx = room.width * 0.03;
    let max_dy = room.depth * 0.03;
    let max_dz = room.height * 0.02;

    for source in &mut simulation.sources {
        source.position =
            jitter_point_within_room(&mut rng, source.position, max_dx, max_dy, max_dz, room);
    }

    let lp_max_dx = room.width * 0.04;
    let lp_max_dy = room.depth * 0.04;
    let lp_max_dz = room.height * 0.03;

    for lp in &mut simulation.listening_positions {
        *lp = jitter_point_within_room(&mut rng, *lp, lp_max_dx, lp_max_dy, lp_max_dz, room);
    }

    simulation.boundaries = jitter_boundaries(&mut rng, &simulation.boundaries, 0.05);

    Scenario {
        name: format!("{}_v{}", base.name, variant_index + 1),
        description: base.description.clone(),
        simulation,
        source_names: base.source_names.clone(),
    }
}

pub fn randomized_scenarios(seed: u64, variants_per_scenario: usize) -> Vec<Scenario> {
    let base = all_scenarios();
    let mut out = Vec::new();

    for (idx, scenario) in base.into_iter().enumerate() {
        let scenario_seed = seed ^ ((idx as u64) << 32);
        for v in 0..variants_per_scenario {
            let seed_v = scenario_seed ^ (v as u64);
            out.push(randomized_scenario_variant(&scenario, v, seed_v));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn randomized_scenario_variants_preserve_admission_invariants() {
        let variants = randomized_scenarios(0x5eed, 2);
        assert_eq!(variants.len(), 40);
        for scenario in variants {
            scenario
                .validate()
                .unwrap_or_else(|error| panic!("{}: {error}", scenario.name));
        }
    }

    #[test]
    fn every_builtin_scenario_passes_full_admission_validation() {
        let config = FrequencyConfig {
            min_freq: 20.0,
            max_freq: 2_000.0,
            num_points: 17,
            spacing: FrequencySpacing::default(),
        };
        let scenarios = all_scenarios_with_config(&config);
        assert_eq!(scenarios.len(), 20);
        for scenario in scenarios {
            scenario
                .validate()
                .unwrap_or_else(|error| panic!("{}: {error}", scenario.name));
        }
    }

    #[test]
    fn scenario_validation_rejects_axis_and_physical_metadata_drift() {
        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.source_names[0] = "wrong-source".into();
        let error = scenario.validate().unwrap_err();
        assert!(error.contains("does not match source name"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.listening_positions[0] = Point3D::new(4.0, 2.0, 1.0);
        let error = scenario.validate().unwrap_err();
        assert!(error.contains("outside its room"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.frequencies[1] = scenario.simulation.frequencies[0];
        let error = scenario.validate().unwrap_err();
        assert!(error.contains("strictly increasing"));
    }

    #[test]
    fn scenario_validation_rejects_duplicate_or_nonportable_labels() {
        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.source_names[1] = scenario.source_names[0].clone();
        scenario.simulation.sources[1].name = scenario.source_names[0].clone();
        let error = scenario.validate().unwrap_err();
        assert!(error.contains("not unique"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.name = "bad/name".into();
        let error = scenario.validate().unwrap_err();
        assert!(error.contains("portable label"));
    }

    #[test]
    fn scenario_validation_rejects_platform_specific_filename_labels() {
        for invalid_name in ["trailing-dot.", "trailing-space ", "CON", "lpt9"] {
            let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
            scenario.name = invalid_name.into();
            let error = scenario.validate().unwrap_err();
            assert!(
                error.contains("portable label"),
                "expected scenario name {invalid_name:?} to be rejected"
            );
        }

        for invalid_label in ["trailing-dot.", "trailing-space ", "CON", "lpt9"] {
            let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
            scenario.source_names[0] = invalid_label.into();
            scenario.simulation.sources[0].name = invalid_label.into();
            let error = scenario.validate().unwrap_err();
            assert!(
                error.contains("source label 0 is not portable"),
                "expected source label {invalid_label:?} to be rejected"
            );
        }
    }

    #[test]
    fn scenario_validation_rejects_empty_axes_and_non_finite_physics() {
        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.description = "  ".into();
        assert!(scenario.validate().unwrap_err().contains("description"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.source_names.pop();
        assert!(scenario.validate().unwrap_err().contains("source labels"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.source_names.clear();
        scenario.simulation.sources.clear();
        assert!(
            scenario
                .validate()
                .unwrap_err()
                .contains("at least one source")
        );

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.listening_positions.clear();
        assert!(
            scenario
                .validate()
                .unwrap_err()
                .contains("listening position")
        );

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.sources[0].amplitude = -1.0;
        assert!(scenario.validate().unwrap_err().contains("amplitude"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.sources[0].position = Point3D::new(f64::NAN, 1.0, 1.0);
        assert!(scenario.validate().unwrap_err().contains("position"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.room = RoomGeometry::Rectangular(RectangularRoom::new(0.0, 4.0, 2.5));
        assert!(scenario.validate().unwrap_err().contains("room dimensions"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.speed_of_sound = f64::NAN;
        assert!(scenario.validate().unwrap_err().contains("sound speed"));

        let mut scenario = scenario_by_name("small_stereo_2_0").unwrap();
        scenario.simulation.frequencies[0] = f64::NAN;
        assert!(scenario.validate().unwrap_err().contains("frequency grid"));
    }

    use super::*;

    #[test]
    fn test_all_scenarios_count() {
        let scenarios = all_scenarios();
        assert_eq!(scenarios.len(), 20);
    }

    #[test]
    fn test_scenario_names_unique() {
        let scenarios = all_scenarios();
        let names: Vec<&str> = scenarios.iter().map(|s| s.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len(), "Scenario names must be unique");
    }

    #[test]
    fn test_all_scenarios_have_valid_frequencies() {
        for scenario in all_scenarios() {
            assert_eq!(
                scenario.simulation.frequencies.len(),
                DEFAULT_NUM_POINTS,
                "Scenario {} should have {} frequency points",
                scenario.name,
                DEFAULT_NUM_POINTS,
            );
            assert!(scenario.simulation.frequencies[0] >= DEFAULT_MIN_FREQ - 1.0);
            assert!(*scenario.simulation.frequencies.last().unwrap() <= DEFAULT_MAX_FREQ + 1.0);
        }
    }

    #[test]
    fn test_source_names_match_sources() {
        for scenario in all_scenarios() {
            assert_eq!(
                scenario.source_names.len(),
                scenario.simulation.sources.len(),
                "Scenario {} source_names/sources mismatch",
                scenario.name,
            );
        }
    }

    #[test]
    fn test_scenario_lookup() {
        let s = scenario_by_name("medium_stereo_2_0");
        assert!(s.is_some());
        assert!(scenario_by_name("nonexistent").is_none());
    }
}
