//! Radioss and OpenRadioss comparison harness (contract C3).
//!
//! This module single-sources small solid-mechanics cases (mesh, SI units,
//! materials, boundary conditions, loads, probes) and renders them as
//! starter (`_0000.rad`) and engine (`_0001.rad`) decks, runs an available
//! engine as a subprocess, converts the binary time history through the
//! official `th_to_csv` tool, and compares probe channels against closed
//! forms. Card layouts follow the Altair Radioss Reference Guide block
//! format (fixed-width fields); engine verification is deferred until an
//! engine is available (see the worker report).
//!
//! Deck flow: starter reads `<case>_0000.rad` and writes `<case>_0000.rst`;
//! the engine reads `<case>_0001.rad` plus the restart and writes the `T01`
//! time history. Binary T01 wall channels are impulses, but `th_to_csv`
//! differentiates them like HyperGraph does, so the CSV already carries
//! forces (proven by energy balance on the hertz leg: `U = (2/5) F delta`);
//! engine decks request `/TH/TITLE` so `th_to_csv` emits full channel names.
//!
//! Phase B (ticket P1-B-W1-0) adds [`RadExtendedCase`]: shell elements with
//! `/PROP/SHELL` cards, quad/triangle shell meshing, plastic `/MAT/LAW2`
//! and `/MAT/LAW36` cards, `/GRAV` loading, and guide-shaped `/TH/SHEL`
//! plus `/TH/BRIC` element requests. The Phase-A [`RadCase`] writer is
//! frozen (other lanes assert byte-identical decks from it); every new
//! block follows the Altair Radioss Reference Guide block format with the
//! part identifier in the element-block header (`/SHELL/part_ID`,
//! `/BRICK/part_ID`) and variable cards on `/TH` requests.

// Rust guideline compliant 2026-02-21

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Failure of a Radioss harness operation.
#[derive(Debug)]
pub enum RadError {
    /// Filesystem or subprocess failure.
    Io(std::io::Error),
    /// A time-history file or probe table is malformed.
    Parse(String),
    /// The starter or engine subprocess reported failure.
    EngineFailed {
        /// Failing stage (`starter`, `engine`, or `th_to_csv`).
        stage: &'static str,
        /// Process exit code when available.
        code: Option<i32>,
        /// Tail of the combined output.
        output: String,
    },
    /// A requested time-history channel is absent.
    MissingChannel {
        /// Requested channel name.
        channel: String,
        /// Channels present in the file.
        available: Vec<String>,
    },
    /// A probe value misses its tolerance.
    ProbeMismatch {
        /// Probe label.
        label: String,
        /// Simulated value.
        actual: f64,
        /// Expected value.
        expected: f64,
        /// Absolute tolerance used.
        abs: f64,
        /// Relative tolerance used.
        rel: f64,
    },
}

impl fmt::Display for RadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "radioss harness I/O failure: {error}"),
            Self::Parse(message) => write!(f, "radioss harness parse failure: {message}"),
            Self::EngineFailed {
                stage,
                code,
                output,
            } => write!(f, "radioss {stage} failed (code {code:?}): {output}"),
            Self::MissingChannel { channel, available } => {
                write!(f, "time history lacks channel {channel}; have {available:?}")
            }
            Self::ProbeMismatch {
                label,
                actual,
                expected,
                abs,
                rel,
            } => write!(
                f,
                "probe {label} mismatch: actual {actual:.6e}, expected {expected:.6e} (abs {abs:.1e}, rel {rel:.1e})"
            ),
        }
    }
}

impl std::error::Error for RadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Elastic solid material rendered as `/MAT/LAW1`.
#[derive(Debug, Clone, Copy)]
pub struct RadElasticMaterial {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Mass density in kg/m^3.
    pub density_kg_per_m3: f64,
}

/// Brick mesh with 1-based element connectivity.
#[derive(Debug, Clone)]
pub struct RadMesh {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Eight-node brick connectivity, 1-based node ids.
    pub bricks: Vec<[usize; 8]>,
}

impl RadMesh {
    /// Count nodes of the mesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Append a free node (e.g. a rigid-wall driver) and return its 1-based id.
    pub fn add_node(&mut self, point_m: [f64; 3]) -> usize {
        self.nodes.push(point_m);
        self.nodes.len()
    }

    /// Count brick elements of the mesh.
    pub fn num_bricks(&self) -> usize {
        self.bricks.len()
    }

    /// Validate index ranges of every brick.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] when a brick references a missing node.
    pub fn validate(&self) -> Result<(), RadError> {
        for (index, brick) in self.bricks.iter().enumerate() {
            for node in brick {
                if *node == 0 || *node > self.nodes.len() {
                    return Err(RadError::Parse(format!(
                        "brick {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Named node set rendered as `/GRNOD`.
#[derive(Debug, Clone)]
pub struct RadNodeSet {
    /// Set name used in deck comments.
    pub name: String,
    /// Member node ids, 1-based.
    pub nodes: Vec<usize>,
}

/// Clamped translational and rotational flags rendered as `/BCS`.
#[derive(Debug, Clone)]
pub struct RadBoundaryCondition {
    /// Name used in deck comments.
    pub name: String,
    /// Index into [`RadCase::node_sets`].
    pub set: usize,
    /// Locked `[Tx, Ty, Tz, Rx, Ry, Rz]` flags.
    pub lock: [bool; 6],
}

/// Piecewise-linear load curve rendered as `/FUNCT`.
#[derive(Debug, Clone)]
pub struct RadFunction {
    /// Curve name used in deck comments.
    pub name: String,
    /// Abscissa/ordinate pairs (time in s, scale factor).
    pub points: Vec<(f64, f64)>,
}

/// Concentrated nodal load rendered as `/CLOAD`.
#[derive(Debug, Clone)]
pub struct RadConcentratedLoad {
    /// Load name used in deck comments.
    pub name: String,
    /// Index into [`RadCase::node_sets`].
    pub set: usize,
    /// Load direction: 1..3 force X/Y/Z, 4..6 moment X/Y/Z.
    pub direction: usize,
    /// Index into [`RadCase::functions`].
    pub function: usize,
    /// Force scale in N.
    pub force_scale_n: f64,
}

/// Infinite-plane rigid wall rendered as `/RWALL/PLANE`.
#[derive(Debug, Clone)]
pub struct RadWall {
    /// Wall name used in deck comments.
    pub name: String,
    /// Outward unit normal of the wall plane.
    pub normal: [f64; 3],
    /// A point on the wall plane in metres.
    pub point_m: [f64; 3],
    /// Coulomb friction coefficient (0 = frictionless, selects Slide 0/2).
    pub friction: f64,
    /// Imposed wall velocity in m/s (`None` = fixed wall).
    pub velocity_m_per_s: Option<[f64; 3]>,
    /// 1-based driver node id for moving walls (`None` = fixed wall,
    /// rendered as node_ID 0 with an explicit M line). The driver node must
    /// exist in the mesh at [`RadWall::point_m`]; moving walls are rendered
    /// with Mass 0, which the reference guide defines as a constant imposed
    /// velocity (`VX0/VY0/VZ0`), so no `/IMPVEL` card is needed.
    pub driver_node: Option<usize>,
    /// Index into the case node sets attached as the secondary group
    /// (`grnd_ID1`; `None` = no group). Must exclude driver nodes.
    pub secondary_set: Option<usize>,
    /// Auto-capture distance in metres (`Dsearch`, 0 = group only).
    pub search_distance_m: f64,
}

/// A probe channel compared against a closed form.
#[derive(Debug, Clone)]
pub struct RadProbe {
    /// Probe label used in reports and `probes.json`.
    pub label: String,
    /// Time-history channel name, e.g. `DZ/123` or `FNZ/RWALL/2`.
    pub channel: String,
    /// Expected value in SI units.
    pub expected: f64,
    /// Absolute tolerance in SI units.
    pub abs_tol: f64,
    /// Relative tolerance, dimensionless.
    pub rel_tol: f64,
}

/// A single-sourced Radioss comparison case.
#[derive(Debug, Clone)]
pub struct RadCase {
    /// Case stem: decks are `<name>_0000.rad` / `<name>_0001.rad`.
    pub name: String,
    /// Title line written into both decks.
    pub title: String,
    /// Brick mesh in SI metres.
    pub mesh: RadMesh,
    /// Elastic solid material.
    pub material: RadElasticMaterial,
    /// Node sets referenced by BCs, loads, and outputs.
    pub node_sets: Vec<RadNodeSet>,
    /// Boundary conditions.
    pub bcs: Vec<RadBoundaryCondition>,
    /// Load curves.
    pub functions: Vec<RadFunction>,
    /// Concentrated loads.
    pub cloads: Vec<RadConcentratedLoad>,
    /// Rigid walls.
    pub walls: Vec<RadWall>,
    /// Node ids recorded with `/TH/NODE`.
    pub th_nodes: Vec<usize>,
    /// Probes compared after the run.
    pub probes: Vec<RadProbe>,
    /// Engine stop time in seconds.
    pub stop_time_s: f64,
    /// Time-history output period in seconds.
    pub th_period_s: f64,
    /// Animation output period in seconds.
    pub anim_period_s: f64,
}

fn field_int(value: usize) -> String {
    format!("{value:>10}")
}

fn field_float(value: f64) -> String {
    format!("{value:>20.12E}")
}

/// Render one `/RWALL/PLANE` block (shared by the basic and extended writers).
///
/// Card order follows the reference guide: title, `node_ID/Slide/grnds`,
/// `Dsearch/fric`, then `M` (fixed wall) or `Mass/VX0/VY0/VZ0` (moving
/// wall), then the `M1 = M + normal` orientation point. Moving walls use
/// Mass 0, i.e. constant imposed velocity, so no `/IMPVEL` card is needed.
fn write_wall_block(deck: &mut String, index: usize, wall: &RadWall) {
    deck.push_str(&format!("/RWALL/PLANE/{}/1\n", index + 1));
    deck.push_str(&format!("{}\n", wall.name));
    let slide = if wall.friction > 0.0 { 2 } else { 0 };
    let grnd1 = wall.secondary_set.map_or(0, |set| set + 1);
    deck.push_str(&format!(
        "{}{}{}{}\n",
        field_int(wall.driver_node.unwrap_or(0)),
        field_int(slide),
        field_int(grnd1),
        field_int(0),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_float(wall.search_distance_m),
        field_float(wall.friction),
        field_float(0.0),
        field_float(0.0),
        field_int(0),
    ));
    if wall.driver_node.is_some() {
        let velocity = wall.velocity_m_per_s.unwrap_or([0.0, 0.0, 0.0]);
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_float(0.0),
            field_float(velocity[0]),
            field_float(velocity[1]),
            field_float(velocity[2]),
        ));
    } else {
        deck.push_str(&format!(
            "{}{}{}\n",
            field_float(wall.point_m[0]),
            field_float(wall.point_m[1]),
            field_float(wall.point_m[2]),
        ));
    }
    deck.push_str(&format!(
        "{}{}{}\n",
        field_float(wall.point_m[0] + wall.normal[0]),
        field_float(wall.point_m[1] + wall.normal[1]),
        field_float(wall.point_m[2] + wall.normal[2]),
    ));
}

/// Render the starter deck (`<name>_0000.rad`) of a case.
///
/// Emits SI units, nodes, bricks, one elastic part, node sets, boundary
/// conditions, load curves, concentrated loads, rigid walls, and the
/// starter-side `/TH` requests.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadCase, cantilever_case, write_starter};
/// let case = cantilever_case();
/// let deck = write_starter(&case);
/// assert!(deck.contains("#RADIOSS STARTER"));
/// assert!(deck.contains("/END"));
/// ```
pub fn write_starter(case: &RadCase) -> String {
    let mut deck = String::new();
    deck.push_str("#RADIOSS STARTER\n");
    deck.push_str(&format!(
        "# sonium_qa rad case {} (SI units: kg, m, s)\n",
        case.name
    ));
    deck.push_str("/BEGIN\n");
    deck.push_str(&format!("{}\n", case.title));
    // Invers + input/work unit systems (mandatory /BEGIN cards).
    deck.push_str("      2022\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/UNIT/1\nSI mass-length-time\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/NODE\n");
    for (index, node) in case.mesh.nodes.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(index + 1),
            field_float(node[0]),
            field_float(node[1]),
            field_float(node[2])
        ));
    }
    // Proven against the engine: part id lives in the header, not per row.
    deck.push_str("/BRICK/1\n");
    for (index, brick) in case.mesh.bricks.iter().enumerate() {
        let mut line = field_int(index + 1);
        for node in brick {
            line.push_str(&field_int(*node));
        }
        line.push('\n');
        deck.push_str(&line);
    }
    for (index, set) in case.node_sets.iter().enumerate() {
        deck.push_str(&format!("/GRNOD/NODE/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", set.name));
        for chunk in set.nodes.chunks(10) {
            let mut line = String::new();
            for node in chunk {
                line.push_str(&field_int(*node));
            }
            line.push('\n');
            deck.push_str(&line);
        }
    }
    deck.push_str("/MAT/LAW1/1/1\n");
    deck.push_str("linear elastic solid\n");
    // Proven against the engine: line 1 is density, line 2 is E and nu.
    deck.push_str(&format!(
        "{}\n",
        field_float(case.material.density_kg_per_m3)
    ));
    deck.push_str(&format!(
        "{}{}\n",
        field_float(case.material.youngs_modulus_pa),
        field_float(case.material.poisson_ratio)
    ));
    deck.push_str("/PROP/SOLID/1/1\nelastic brick property\n");
    // HEPH bricks + full geometric nonlinearities (best practice;
    // proven 1.6% vs EB on the coarse cantilever, Isolid=1 gave 41%).
    deck.push_str("        24         2\n");
    deck.push_str("/PART/1\nelastic part\n");
    deck.push_str("         1         1\n");
    for (index, function) in case.functions.iter().enumerate() {
        deck.push_str(&format!("/FUNCT/{}\n", index + 1));
        deck.push_str(&format!("{}\n", function.name));
        for (x, y) in &function.points {
            deck.push_str(&format!("{}{}\n", field_float(*x), field_float(*y)));
        }
    }
    for (index, bcs) in case.bcs.iter().enumerate() {
        deck.push_str(&format!("/BCS/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", bcs.name));
        // Proven against the engine + reference guide: one Trarot/Skew/grnd
        // line. Trarot packs the six lock flags as "TTT RRR" (1 = fixed),
        // right-justified in its field; skew 0 selects the global frame.
        let trarot = format!(
            "{}{}{} {}{}{}",
            i32::from(bcs.lock[0]),
            i32::from(bcs.lock[1]),
            i32::from(bcs.lock[2]),
            i32::from(bcs.lock[3]),
            i32::from(bcs.lock[4]),
            i32::from(bcs.lock[5]),
        );
        deck.push_str(&format!(
            "{:>10}{}{}\n",
            trarot,
            field_int(0),
            field_int(bcs.set + 1)
        ));
    }
    for (index, cload) in case.cloads.iter().enumerate() {
        deck.push_str(&format!("/CLOAD/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", cload.name));
        deck.push_str(&format!(
            "{}{:>10}{}{}{}{}{}{}\n",
            field_int(cload.function + 1),
            direction_code(cload.direction),
            field_int(0),
            field_int(0),
            field_int(cload.set + 1),
            field_int(0),
            field_float(1.0),
            field_float(cload.force_scale_n),
        ));
    }
    for (index, wall) in case.walls.iter().enumerate() {
        write_wall_block(&mut deck, index, wall);
    }
    // Proven against the engine: the variables line precedes the node ids.
    let mut th_group = 0;
    write_th_block(
        &mut deck,
        &mut th_group,
        "NODE",
        "time history nodes",
        &["DX".to_string(), "DY".to_string(), "DZ".to_string()],
        &case.th_nodes,
    );
    if !case.walls.is_empty() {
        // The FN group expands to FNX/FNY/FNZ; th_to_csv reports wall
        // forces directly (it differentiates the T01 impulses).
        let wall_ids: Vec<usize> = (1..=case.walls.len()).collect();
        write_th_block(
            &mut deck,
            &mut th_group,
            "RWALL",
            "time history walls",
            &["FN".to_string()],
            &wall_ids,
        );
    }
    deck.push_str("/END\n");
    deck
}

/// Render the engine deck (`<name>_0001.rad`) of a case.
///
/// Requests a quasi-static explicit run to `stop_time_s` with `/TFILE`
/// history output, `/TH/TITLE` full channel names for `th_to_csv`, and
/// animation states.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadCase, cantilever_case, write_engine};
/// let case = cantilever_case();
/// let deck = write_engine(&case);
/// assert!(deck.contains("#RADIOSS ENGINE"));
/// assert!(deck.contains("/TH/TITLE"));
/// ```
pub fn write_engine(case: &RadCase) -> String {
    // Proven against the engine: /RUN takes the bare stop time (no title,
    // no step count), /ANIM/DT takes start+period, and the deck ends after
    // the last card (no /STOP or /END terminator).
    let mut deck = String::new();
    deck.push_str("#RADIOSS ENGINE\n");
    deck.push_str(&format!("# engine controls for {}\n", case.name));
    deck.push_str(&format!("/RUN/{}/1\n", case.name));
    deck.push_str(&format!("{}\n", field_float(case.stop_time_s)));
    deck.push_str("/TFILE\n");
    deck.push_str(&format!("{}\n", field_float(case.th_period_s)));
    deck.push_str("/TH/TITLE\n");
    deck.push_str("/ANIM/DT\n");
    deck.push_str(&format!(
        "{}{}\n",
        field_float(0.0),
        field_float(case.anim_period_s)
    ));
    deck.push_str("/PRINT/-1000\n");
    deck
}

/// Build the quasi-static cantilever prove-out case.
///
/// A brick-meshed steel beam clamped at `x = 0` carries a ramped tip load;
/// the tip displacement probe replays the Euler-Bernoulli closed form.
pub fn cantilever_case() -> RadCase {
    let (mesh, root_set, tip_set, tip_node) = cantilever_brick_mesh(10, 2, 2, 1.0, 0.05, 0.05);
    let expected =
        super::beam_euler_cantilever_point(1.0, 1.0, 200.0e9, 0.05 * 0.05f64.powi(3) / 12.0, 100.0);
    // /CLOAD applies the scale to EACH node of the set.
    let tip_scale = 100.0 / tip_set.len() as f64;
    RadCase {
        name: "cantilever".to_string(),
        title: "sonium_qa rad cantilever quasi-static tip load".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "tip".to_string(),
                nodes: tip_set,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp root".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "tip load".to_string(),
            set: 1,
            direction: 3,
            function: 0,
            force_scale_n: tip_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![tip_node],
        probes: vec![RadProbe {
            label: "tip DZ".to_string(),
            channel: format!("DZ/{tip_node}"),
            expected,
            abs_tol: 1e-9,
            rel_tol: 0.05,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

/// Build the Hertz sphere prove-out case.
///
/// An elastic ball rests on a fixed frictionless rigid plane while a top
/// plane presses it down at constant velocity. The case probe asserts the
/// pole kinematics (the top pole rides the imposed-velocity wall); the
/// force leg lives in the QA binary as custom checks (wall-force balance
/// from Newton's 3rd law plus a contact-stiffness window), because the
/// Hertz patch `a = sqrt(R delta) ~ 0.5 mm` is unresolvable on 10 mm
/// bricks, so the Hertz force magnitude itself is out of reach here.
/// Quasi-static Hertz is duration-independent, and the 1 s run grows a
/// contact-mode instability past ~150k cycles, so the press stops at 0.1 s
/// (137k cycles) where the parasitic kinetic energy is still 1e-7 J.
pub fn hertz_sphere_case() -> RadCase {
    let radius = 0.05;
    let mut mesh = sphere_ball_mesh(10, radius);
    let press_speed = 1.0e-4;
    let stop_time = 0.1;
    // Total top-wall travel; the top pole rides the wall down by this.
    let travel = press_speed * stop_time;
    let top_pole = mesh
        .nodes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1[2].total_cmp(&b.1[2]))
        .map(|(index, _)| index + 1)
        .unwrap_or(1);
    // Split secondary groups (bottom/top halves): a node secondary to
    // both walls trips starter warning 312 (incompatible kinematic
    // conditions). Contact only touches the poles, so the split is exact.
    // The driver node is appended after so it stays out of both groups.
    let mut bottom_nodes = Vec::new();
    let mut top_nodes = Vec::new();
    for (index, node) in mesh.nodes.iter().enumerate() {
        if node[2] <= radius {
            bottom_nodes.push(index + 1);
        } else {
            top_nodes.push(index + 1);
        }
    }
    let top_point = [0.0, 0.0, 2.0 * radius];
    let driver = mesh.add_node(top_point);
    RadCase {
        name: "hertz_sphere".to_string(),
        title: "sonium_qa rad hertz sphere between rigid planes".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "ball bottom".to_string(),
                nodes: bottom_nodes,
            },
            RadNodeSet {
                name: "ball top".to_string(),
                nodes: top_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant press velocity".to_string(),
            points: vec![(0.0, 1.0), (stop_time, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![
            RadWall {
                name: "fixed bottom plane".to_string(),
                normal: [0.0, 0.0, 1.0],
                point_m: [0.0, 0.0, 0.0],
                friction: 0.0,
                velocity_m_per_s: None,
                driver_node: None,
                secondary_set: Some(0),
                search_distance_m: 0.0,
            },
            RadWall {
                name: "moving top plane".to_string(),
                normal: [0.0, 0.0, -1.0],
                point_m: top_point,
                friction: 0.0,
                velocity_m_per_s: Some([0.0, 0.0, -press_speed]),
                driver_node: Some(driver),
                secondary_set: Some(1),
                search_distance_m: 0.0,
            },
        ],
        th_nodes: vec![top_pole],
        probes: vec![RadProbe {
            label: "top pole DZ".to_string(),
            channel: format!("DZ/{top_pole}"),
            expected: -travel,
            abs_tol: 1e-9,
            rel_tol: 0.10,
        }],
        stop_time_s: stop_time,
        th_period_s: 0.001,
        anim_period_s: 0.025,
    }
}

/// Mesh a cantilever beam with 8-node bricks.
///
/// Returns the mesh, the root/tip node sets, and the tip-center node id.
/// Node ids are 1-based.
pub fn cantilever_brick_mesh(
    nx: usize,
    ny: usize,
    nz: usize,
    length_m: f64,
    width_m: f64,
    height_m: f64,
) -> (RadMesh, Vec<usize>, Vec<usize>, usize) {
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    let id = |i: usize, j: usize, k: usize| i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1;
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                nodes.push([
                    length_m * i as f64 / nx as f64,
                    width_m * (j as f64 / ny as f64 - 0.5),
                    height_m * (k as f64 / nz as f64 - 0.5),
                ]);
            }
        }
    }
    let mut bricks = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                bricks.push([
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1),
                ]);
            }
        }
    }
    let mut root_set = Vec::new();
    let mut tip_set = Vec::new();
    for j in 0..=ny {
        for k in 0..=nz {
            root_set.push(id(0, j, k));
            tip_set.push(id(nx, j, k));
        }
    }
    let tip_node = id(nx, ny / 2, nz / 2);
    (RadMesh { nodes, bricks }, root_set, tip_set, tip_node)
}

/// Mesh a solid ball with 8-node bricks.
///
/// Keeps the Cartesian cells fully inside the sphere and projects the
/// region boundary radially onto the surface, so every brick keeps a
/// positive volume. The ball is centered at `(0, 0, radius)`.
pub fn sphere_ball_mesh(cells_per_side: usize, radius_m: f64) -> RadMesh {
    let n = cells_per_side.max(2);
    let step = 2.0 * radius_m / n as f64;
    let corner = |i: usize| -radius_m + step * i as f64;
    let inside = |i: usize, j: usize, k: usize| {
        let (x, y, z) = (corner(i), corner(j), corner(k));
        x * x + y * y + z * z <= radius_m * radius_m
    };
    let mut kept = vec![false; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let all_inside = (0..=1)
                    .all(|di| (0..=1).all(|dj| (0..=1).all(|dk| inside(i + di, j + dj, k + dk))));
                kept[(i * n + j) * n + k] = all_inside;
            }
        }
    }
    // Boundary nodes touch at least one dropped cell; project them outward.
    let mut node_id = vec![0usize; (n + 1) * (n + 1) * (n + 1)];
    let mut nodes = Vec::new();
    let mut is_boundary = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                let mut used = false;
                let mut boundary = false;
                for di in 0..=1 {
                    for dj in 0..=1 {
                        for dk in 0..=1 {
                            let (ci, cj, ck) = (i + 1 - di, j + 1 - dj, k + 1 - dk);
                            if ci == 0 || cj == 0 || ck == 0 || ci > n || cj > n || ck > n {
                                continue;
                            }
                            if kept[((ci - 1) * n + (cj - 1)) * n + (ck - 1)] {
                                used = true;
                            } else {
                                boundary = true;
                            }
                        }
                    }
                }
                // Outer-grid nodes of kept cells are boundary by construction.
                if i == 0 || j == 0 || k == 0 || i == n || j == n || k == n {
                    boundary = true;
                }
                if !used {
                    continue;
                }
                let (mut x, mut y, mut z) = (corner(i), corner(j), corner(k));
                if boundary {
                    let r = (x * x + y * y + z * z).sqrt();
                    if r > 1e-12 {
                        let scale = radius_m / r;
                        x *= scale;
                        y *= scale;
                        z *= scale;
                    }
                }
                node_id[(i * (n + 1) + j) * (n + 1) + k] = nodes.len() + 1;
                nodes.push([x, y, z + radius_m]);
                is_boundary.push(boundary);
            }
        }
    }
    let _ = is_boundary;
    let mut bricks = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if !kept[(i * n + j) * n + k] {
                    continue;
                }
                let at = |di: usize, dj: usize, dk: usize| {
                    node_id[((i + di) * (n + 1) + (j + dj)) * (n + 1) + (k + dk)]
                };
                bricks.push([
                    at(0, 0, 0),
                    at(1, 0, 0),
                    at(1, 1, 0),
                    at(0, 1, 0),
                    at(0, 0, 1),
                    at(1, 0, 1),
                    at(1, 1, 1),
                    at(0, 1, 1),
                ]);
            }
        }
    }
    RadMesh { nodes, bricks }
}

/// Resolve the engine executable from the environment.
///
/// Returns `RADISS_EXE` when set, else `OPENCOURANT_EXE`, else `None` so
/// binaries can report the documented skip message and exit 0.
pub fn engine_exe() -> Option<PathBuf> {
    for name in ["RADISS_EXE", "OPENCOURANT_EXE"] {
        if let Ok(path) = std::env::var(name) {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

/// Names the engine flavor behind an executable path.
pub fn engine_flavor(exe: &Path) -> &'static str {
    let haystack = exe.to_string_lossy().to_lowercase();
    if haystack.contains("courant") {
        "OpenCourant"
    } else {
        "Radioss"
    }
}

/// Rank a starter/engine candidate filename (lower is better).
///
/// Prefers double-precision SMP builds (`*_linux64_gf` without `_sp` or
/// `ompi` suffixes); single-precision before MPI; anything else last.
fn candidate_rank(file_name: &str) -> u32 {
    let exact_dp = file_name == "starter_linux64_gf" || file_name == "engine_linux64_gf";
    if exact_dp {
        return 0;
    }
    let lower = file_name.to_lowercase();
    let is_sp = lower.contains("_sp");
    // MPI builds need mpirun; SMP/OpenMP builds run directly.
    let is_mpi = lower.contains("ompi") || lower.contains("_mpi");
    match (is_sp, is_mpi) {
        (false, false) => 1,
        (true, false) => 2,
        (false, true) => 3,
        (true, true) => 4,
    }
}

/// Best `prefix*` executable entry of `dir` by [`candidate_rank`].
fn best_sibling(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(u32, String, PathBuf)> = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(prefix) {
            continue;
        }
        let rank = candidate_rank(&name);
        let replace = best
            .as_ref()
            .map(|(best_rank, best_name, _)| {
                (rank, &name) < (*best_rank, best_name)
            })
            .unwrap_or(true);
        if replace {
            best = Some((rank, name, entry.path()));
        }
    }
    best.map(|(_, _, path)| path)
}

/// Resolve the starter/engine binary pair behind one configured path.
///
/// OpenCourant/Radioss ship separate `starter_*` and `engine_*` binaries,
/// so one configured path cannot serve both stages directly. Accepts:
/// - a directory (e.g. `.../exec`): picks the best `starter_*`/`engine_*`
///   pair, preferring double-precision SMP builds;
/// - a `starter_*` binary: pairs it with the same-suffix `engine_*` sibling;
/// - an `engine_*` binary: pairs it with the same-suffix `starter_*`
///   sibling;
/// - anything else: uses the path for both stages (legacy single-binary).
///
/// # Errors
///
/// Returns [`RadError::Parse`] when a directory holds no usable pair.
pub fn resolve_engine_pair(exe: &Path) -> Result<(PathBuf, PathBuf), RadError> {
    if exe.is_dir() {
        let starter = best_sibling(exe, "starter");
        let engine = best_sibling(exe, "engine");
        match (starter, engine) {
            (Some(starter), Some(engine)) => return Ok((starter, engine)),
            _ => {
                return Err(RadError::Parse(format!(
                    "engine directory {} holds no starter_*/engine_* pair",
                    exe.display()
                )));
            }
        }
    }
    let file_name = exe
        .file_name()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    let parent = exe.parent();
    if file_name.starts_with("starter") {
        let sibling = file_name.replacen("starter", "engine", 1);
        if let Some(parent) = parent {
            let candidate = parent.join(&sibling);
            if candidate.is_file() {
                return Ok((exe.to_path_buf(), candidate));
            }
            if let Some(engine) = best_sibling(parent, "engine") {
                return Ok((exe.to_path_buf(), engine));
            }
        }
        return Ok((exe.to_path_buf(), exe.to_path_buf()));
    }
    if file_name.starts_with("engine") {
        let sibling = file_name.replacen("engine", "starter", 1);
        if let Some(parent) = parent {
            let candidate = parent.join(&sibling);
            if candidate.is_file() {
                return Ok((candidate, exe.to_path_buf()));
            }
            if let Some(starter) = best_sibling(parent, "starter") {
                return Ok((starter, exe.to_path_buf()));
            }
        }
        return Ok((exe.to_path_buf(), exe.to_path_buf()));
    }
    Ok((exe.to_path_buf(), exe.to_path_buf()))
}

/// Output of a starter-plus-engine run.
#[derive(Debug)]
pub struct RunOutput {
    /// Engine flavor that ran.
    pub engine: String,
    /// Converted time-history table.
    pub history: TimeHistory,
    /// Path of the converted CSV (or native T01 when already CSV).
    pub history_path: PathBuf,
}

/// Run the starter then the engine for a checked-in deck pair.
///
/// Both stages run with the working directory set to `decks_dir`; after a
/// successful engine run the `T01` file is converted with `th_to_csv`
/// (resolved beside the engine or on `PATH`) and parsed. Native CSV
/// histories are parsed directly. The `case` argument single-sources the
/// deck stems; C3 contract note: the frozen text sketches
/// `run_case(decks_dir, exe)`, but the stem must come from somewhere, so
/// the shipped form takes the case explicitly (owner clarification).
///
/// # Errors
///
/// Returns [`RadError`] when a stage fails, when conversion is unavailable
/// for a binary history, or when parsing fails.
pub fn run_case(decks_dir: &Path, exe: &Path, case: &RadCase) -> Result<RunOutput, RadError> {
    let starter_deck = decks_dir.join(format!("{}_0000.rad", case.name));
    let engine_deck = decks_dir.join(format!("{}_0001.rad", case.name));
    let (starter_exe, engine_exe) = resolve_engine_pair(exe)?;
    run_stage(&starter_exe, "starter", &starter_deck, decks_dir)?;
    run_stage(&engine_exe, "engine", &engine_deck, decks_dir)?;
    let t01 = decks_dir.join(format!("{}T01", case.name.to_uppercase()));
    let t01 = if t01.exists() {
        t01
    } else {
        decks_dir.join(format!("{}T01", case.name))
    };
    let csv = t01.with_extension("csv");
    if !is_probably_csv(&t01)? {
        convert_with_th_to_csv(exe, &t01, decks_dir)?;
    }
    let history_path = if csv.exists() { csv } else { t01.clone() };
    let history = parse_t01(&history_path)?;
    Ok(RunOutput {
        engine: engine_flavor(exe).to_string(),
        history,
        history_path,
    })
}

fn run_stage(exe: &Path, stage: &'static str, deck: &Path, dir: &Path) -> Result<(), RadError> {
    let output = Command::new(exe)
        .arg("-i")
        .arg(deck)
        .current_dir(dir)
        // The engine blocks on stdin after abnormal ends; never inherit it.
        .stdin(std::process::Stdio::null())
        .output()?;
    if !output.status.success() {
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        let tail: String = combined
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(RadError::EngineFailed {
            stage,
            code: output.status.code(),
            output: tail,
        });
    }
    Ok(())
}

fn is_probably_csv(path: &Path) -> Result<bool, RadError> {
    let bytes = std::fs::read(path)?;
    let head = &bytes[..bytes.len().min(4096)];
    Ok(!head.contains(&0) && head.contains(&b','))
}

fn convert_with_th_to_csv(exe: &Path, t01: &Path, dir: &Path) -> Result<(), RadError> {
    // `exe` may be the exec directory itself or one engine binary; the
    // converter ships as `th_to_csv*` beside the engine binaries.
    let search_dir = if exe.is_dir() {
        Some(exe.to_path_buf())
    } else {
        exe.parent().map(Path::to_path_buf)
    };
    let mut candidates = Vec::new();
    if let Some(search_dir) = search_dir {
        candidates.push(search_dir.join("th_to_csv"));
        if let Some(suffixed) = best_sibling(&search_dir, "th_to_csv")
            && suffixed.file_name().map(|name| name != "th_to_csv") == Some(true)
        {
            candidates.push(suffixed);
        }
    }
    candidates.push(PathBuf::from("th_to_csv"));
    let mut last_error = None;
    for tool in candidates {
        match Command::new(&tool).arg(t01).current_dir(dir).output() {
            Ok(output) if output.status.success() => return Ok(()),
            Ok(output) => {
                last_error = Some(format!(
                    "{} exited {}",
                    tool.display(),
                    output.status.code().unwrap_or(-1)
                ));
            }
            Err(error) => last_error = Some(error.to_string()),
        }
    }
    Err(RadError::Parse(format!(
        "binary T01 needs th_to_csv conversion: {}",
        last_error.unwrap_or_else(|| "converter missing".to_string())
    )))
}

/// A parsed time history: time grid plus named channels.
#[derive(Debug, Clone, Default)]
pub struct TimeHistory {
    /// Time stamps in seconds.
    pub times: Vec<f64>,
    /// Channel name to samples, one per time stamp.
    pub channels: BTreeMap<String, Vec<f64>>,
}

impl TimeHistory {
    /// Look up a channel by name.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::MissingChannel`] when the name is absent.
    pub fn channel(&self, name: &str) -> Result<&[f64], RadError> {
        self.channels
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| RadError::MissingChannel {
                channel: name.to_string(),
                available: self.channels.keys().cloned().collect(),
            })
    }

    /// Final sample of a channel.
    ///
    /// # Errors
    ///
    /// Returns [`RadError`] when the channel is absent or empty.
    pub fn final_value(&self, name: &str) -> Result<f64, RadError> {
        let values = self.channel(name)?;
        values
            .last()
            .copied()
            .ok_or_else(|| RadError::Parse(format!("channel {name} has no samples")))
    }

    /// Linearly interpolate a channel at a time within the grid.
    ///
    /// # Errors
    ///
    /// Returns [`RadError`] when the channel is absent, empty, or the time
    /// falls outside the grid.
    pub fn value_at(&self, name: &str, time: f64) -> Result<f64, RadError> {
        let values = self.channel(name)?;
        if self.times.len() != values.len() || values.is_empty() {
            return Err(RadError::Parse(format!("channel {name} is malformed")));
        }
        if time < self.times[0] || time > self.times[self.times.len() - 1] {
            return Err(RadError::Parse(format!(
                "time {time} is outside [{}..{}]",
                self.times[0],
                self.times[self.times.len() - 1]
            )));
        }
        let upper = self
            .times
            .partition_point(|grid| *grid <= time)
            .min(values.len() - 1);
        if upper == 0 {
            return Ok(values[0]);
        }
        let (t0, t1) = (self.times[upper - 1], self.times[upper]);
        if t1 <= t0 {
            return Ok(values[upper]);
        }
        let weight = (time - t0) / (t1 - t0);
        Ok(values[upper - 1] * (1.0 - weight) + values[upper] * weight)
    }
}

/// Parse a `th_to_csv` time-history CSV file.
///
/// Expects a header row whose first column is time (any name containing
/// `time`, case-insensitive) followed by one column per channel. Binary
/// T01 input is rejected with guidance to convert first.
///
/// # Errors
///
/// Returns [`RadError::Parse`] for binary input, malformed rows, or
/// non-finite samples.
pub fn parse_t01(path: &Path) -> Result<TimeHistory, RadError> {
    let text = std::fs::read(path)?;
    if text.contains(&0) {
        return Err(RadError::Parse(format!(
            "binary T01 at {} needs th_to_csv conversion first",
            path.display()
        )));
    }
    let text = String::from_utf8(text)
        .map_err(|_| RadError::Parse(format!("{} is not valid UTF-8", path.display())))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| RadError::Parse(format!("{} is empty", path.display())))?;
    let names: Vec<String> = header
        .split(',')
        .map(|cell| cell.trim().to_string())
        .collect();
    if names.len() < 2 || !names[0].to_lowercase().contains("time") {
        return Err(RadError::Parse(format!(
            "{} header must start with a time column",
            path.display()
        )));
    }
    let mut times = Vec::new();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len() - 1];
    for (row, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != names.len() {
            return Err(RadError::Parse(format!(
                "{} row {} has {} cells, expected {}",
                path.display(),
                row + 2,
                cells.len(),
                names.len()
            )));
        }
        let time: f64 = cells[0].trim().parse().map_err(|_| {
            RadError::Parse(format!(
                "{} row {} has a bad time stamp",
                path.display(),
                row + 2
            ))
        })?;
        if !time.is_finite() {
            return Err(RadError::Parse(format!(
                "{} row {} time stamp is non-finite",
                path.display(),
                row + 2
            )));
        }
        times.push(time);
        for (column, cell) in cells.iter().skip(1).enumerate() {
            let value: f64 = cell.trim().parse().map_err(|_| {
                RadError::Parse(format!(
                    "{} row {} column {} is not a number",
                    path.display(),
                    row + 2,
                    names[column + 1]
                ))
            })?;
            if !value.is_finite() {
                return Err(RadError::Parse(format!(
                    "{} row {} column {} is non-finite",
                    path.display(),
                    row + 2,
                    names[column + 1]
                )));
            }
            columns[column].push(value);
        }
    }
    let mut channels = BTreeMap::new();
    for (index, name) in names.iter().skip(1).enumerate() {
        channels.insert(
            normalize_channel(name),
            std::mem::take(&mut columns[index]),
        );
    }
    Ok(TimeHistory { times, channels })
}

/// Normalize a `th_to_csv` column header to probe-channel form.
///
/// Element channels arrive as `"<title> <id> <VAR>"` (e.g.
/// `"time history nodes  95  DX"`); wall channels carry the wall title
/// between the id and the variable (`"time history walls  1  fixed
/// bottom plane  FNZ"`). Probes use `VAR/id` for nodes and `VAR/TYPE/id`
/// for shells (`SHEL`), bricks (`BRIC`), and walls (`RWALL`), with the
/// type recovered from the title keywords our writer emits.
/// Energy channels (`INTERNAL ENERGY`, ...) match no element pattern and
/// pass through with quotes stripped.
fn normalize_channel(raw: &str) -> String {
    let unquoted = raw.trim().trim_matches('"').trim().to_string();
    let tokens: Vec<&str> = unquoted.split_whitespace().collect();
    if tokens.len() >= 3 {
        let var = tokens[tokens.len() - 1];
        let var_ok =
            !var.is_empty() && var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if var_ok {
            let head = &tokens[..tokens.len() - 1];
            let title = head.join(" ").to_lowercase();
            if title.contains("wall") {
                // Wall titles carry no digits, so the first integer token
                // is the wall id.
                if let Some(id) = head
                    .iter()
                    .find(|token| token.chars().all(|c| c.is_ascii_digit()))
                {
                    return format!("{var}/RWALL/{id}");
                }
            }
            let id = tokens[tokens.len() - 2];
            let id_ok = !id.is_empty() && id.chars().all(|c| c.is_ascii_digit());
            if id_ok {
                if title.contains("shell") {
                    return format!("{var}/SHEL/{id}");
                }
                if title.contains("brick") {
                    return format!("{var}/BRIC/{id}");
                }
                if title.contains("node") {
                    return format!("{var}/{id}");
                }
            }
        }
    }
    unquoted
}

/// Differentiate an impulse channel into a force channel.
///
/// Raw binary T01 records running impulses `J(t) = int F dt`; centered
/// differences recover the force at interior stamps (one-sided at the
/// ends). Not used on `th_to_csv` output, which already carries forces.
pub fn impulse_to_force(times: &[f64], impulse: &[f64]) -> Vec<f64> {
    let n = times.len().min(impulse.len());
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    let mut force = vec![0.0; n];
    force[0] = slope(times[0], impulse[0], times[1], impulse[1]);
    for i in 1..n - 1 {
        force[i] = slope(times[i - 1], impulse[i - 1], times[i + 1], impulse[i + 1]);
    }
    force[n - 1] = slope(times[n - 2], impulse[n - 2], times[n - 1], impulse[n - 1]);
    force
}

fn slope(t0: f64, j0: f64, t1: f64, j1: f64) -> f64 {
    if t1 > t0 {
        (j1 - j0) / (t1 - t0)
    } else {
        0.0
    }
}

/// Compare a probe value against its expectation.
///
/// Passes when `|actual - expected| <= abs + rel * |expected|`.
///
/// # Errors
///
/// Returns [`RadError::ProbeMismatch`] on missing tolerance or
/// non-finite inputs.
pub fn assert_probe_close(
    actual: f64,
    expected: f64,
    abs: f64,
    rel: f64,
    label: &str,
) -> Result<(), RadError> {
    if !actual.is_finite() || !expected.is_finite() || !abs.is_finite() || !rel.is_finite() {
        return Err(RadError::ProbeMismatch {
            label: label.to_string(),
            actual,
            expected,
            abs,
            rel,
        });
    }
    if (actual - expected).abs() <= abs + rel * expected.abs() {
        Ok(())
    } else {
        Err(RadError::ProbeMismatch {
            label: label.to_string(),
            actual,
            expected,
            abs,
            rel,
        })
    }
}

/// Render the checked-in `probes.json` artifact of a case.
///
/// Records expected probe values, tolerances, and the producing engine
/// version (`"none (prove-out deferred)"` until an engine runs).
pub fn probes_json(case: &RadCase, engine_version: &str) -> String {
    let mut out = String::from("{\n");
    out.push_str(&format!("  \"case\": \"{}\",\n", case.name));
    out.push_str(&format!("  \"engine\": \"{engine_version}\",\n"));
    out.push_str("  \"units\": \"SI\",\n");
    out.push_str("  \"probes\": [\n");
    for (index, probe) in case.probes.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"label\": \"{}\",\n", probe.label));
        out.push_str(&format!("      \"channel\": \"{}\",\n", probe.channel));
        out.push_str(&format!("      \"expected\": {:.12e},\n", probe.expected));
        out.push_str(&format!("      \"abs_tol\": {:.3e},\n", probe.abs_tol));
        out.push_str(&format!("      \"rel_tol\": {:.3e}\n", probe.rel_tol));
        out.push_str(if index + 1 == case.probes.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

// =====================================================================
// Phase-B extensions (P1-B-W1-0). Additive: nothing above this line
// changes behavior or writer output.
// =====================================================================

/// Shell property set rendered as `/PROP/SHELL` (TYPE1).
///
/// Field names and defaults follow the Reference Guide `/PROP/TYPE1`
/// page: `Ishell` selects the 4-node formulation, `Ismstr` the
/// small-strain switch, `Ish3n` the 3-node formulation, `N` the
/// through-thickness integration count, and `Thick` the thickness.
#[derive(Debug, Clone)]
pub struct RadShellProperty {
    /// Property title written on the card.
    pub name: String,
    /// Four-node formulation flag (24 = QEPH, recommended general use).
    pub ishell: i32,
    /// Small-strain flag (2 = full geometric nonlinearities).
    pub ismstr: i32,
    /// Three-node formulation flag (2 = C0 large rotation).
    pub ish3n: i32,
    /// Drilling stiffness flag (2 = off).
    pub idrill: i32,
    /// Through-thickness integration points, 1..10 (5 recommended).
    pub n_thickness: i32,
    /// Shell thickness in metres.
    pub thickness_m: f64,
    /// Thickness-change flag (2 = constant thickness).
    pub ithick: i32,
    /// Plane-stress plasticity flag (2 = radial return).
    pub iplas: i32,
}

impl RadShellProperty {
    /// QEPH shell property with best-practice defaults.
    ///
    /// Uses Ishell 24 (QEPH), Ismstr 2, Ish3n 2, Idrill 2, N 5,
    /// constant thickness, and radial-return plasticity per the
    /// Reference Guide and the Radioss modeling best practices.
    pub fn qeph(name: &str, thickness_m: f64) -> Self {
        Self {
            name: name.to_string(),
            ishell: 24,
            ismstr: 2,
            ish3n: 2,
            idrill: 2,
            n_thickness: 5,
            thickness_m,
            ithick: 2,
            iplas: 2,
        }
    }

    /// Check ranges of the property card.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for non-positive thickness or an
    /// out-of-range through-thickness count.
    pub fn validate(&self) -> Result<(), RadError> {
        if !self.thickness_m.is_finite() || self.thickness_m <= 0.0 {
            return Err(RadError::Parse(format!(
                "shell property {} needs a positive thickness",
                self.name
            )));
        }
        if self.n_thickness < 1 || self.n_thickness > 9 {
            return Err(RadError::Parse(format!(
                "shell property {} needs 1..9 thickness points",
                self.name
            )));
        }
        Ok(())
    }
}

/// Shell element connectivity with 1-based node ids.
#[derive(Debug, Clone, Default)]
pub struct RadShellMesh {
    /// Four-node shell connectivity, 1-based node ids.
    pub quads: Vec<[usize; 4]>,
    /// Three-node shell connectivity, 1-based node ids.
    pub tris: Vec<[usize; 3]>,
}

impl RadShellMesh {
    /// Count four-node shells.
    pub fn num_quads(&self) -> usize {
        self.quads.len()
    }

    /// Count three-node shells.
    pub fn num_tris(&self) -> usize {
        self.tris.len()
    }

    /// Check whether the mesh holds no shells.
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.tris.is_empty()
    }

    /// Validate index ranges against a node count.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] when a shell references a missing node.
    pub fn validate(&self, num_nodes: usize) -> Result<(), RadError> {
        for (index, quad) in self.quads.iter().enumerate() {
            for node in quad {
                if *node == 0 || *node > num_nodes {
                    return Err(RadError::Parse(format!(
                        "shell {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        for (index, tri) in self.tris.iter().enumerate() {
            for node in tri {
                if *node == 0 || *node > num_nodes {
                    return Err(RadError::Parse(format!(
                        "tria {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Johnson-Cook plastic parameters rendered as `/MAT/LAW2`.
#[derive(Debug, Clone)]
pub struct RadJohnsonCook {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Mass density in kg/m^3.
    pub density_kg_per_m3: f64,
    /// Yield stress `a` in Pa.
    pub yield_pa: f64,
    /// Hardening modulus `b` in Pa (0 = perfectly plastic).
    pub hardening_pa: f64,
    /// Hardening exponent `n` (1 = linear).
    pub hardening_exp: f64,
    /// Rate coefficient `c` (0 = rate independent).
    pub rate_coeff: f64,
    /// Reference strain rate in 1/s.
    pub ref_strain_rate: f64,
    /// Failure plastic strain (1e20 disables failure).
    pub fail_plastic_strain: f64,
    /// Maximum stress cutoff in Pa (1e20 disables the cap).
    pub max_stress_pa: f64,
}

impl RadJohnsonCook {
    /// Rate-independent elastic-perfectly plastic metal.
    pub fn perfectly_plastic(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        density_kg_per_m3: f64,
        yield_pa: f64,
    ) -> Self {
        Self {
            youngs_modulus_pa,
            poisson_ratio,
            density_kg_per_m3,
            yield_pa,
            hardening_pa: 0.0,
            hardening_exp: 1.0,
            rate_coeff: 0.0,
            ref_strain_rate: 1.0,
            fail_plastic_strain: 1.0e20,
            max_stress_pa: 1.0e20,
        }
    }

    /// Rate-independent power-law hardening metal.
    pub fn power_hardening(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        density_kg_per_m3: f64,
        yield_pa: f64,
        hardening_pa: f64,
        hardening_exp: f64,
    ) -> Self {
        Self {
            youngs_modulus_pa,
            poisson_ratio,
            density_kg_per_m3,
            yield_pa,
            hardening_pa,
            hardening_exp,
            rate_coeff: 0.0,
            ref_strain_rate: 1.0,
            fail_plastic_strain: 1.0e20,
            max_stress_pa: 1.0e20,
        }
    }
}

/// Tabulated hardening rendered as `/MAT/LAW36` with one curve.
#[derive(Debug, Clone)]
pub struct RadTabulatedHardening {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Mass density in kg/m^3.
    pub density_kg_per_m3: f64,
    /// Yield stress vs plastic strain, at least two rising points.
    pub points: Vec<(f64, f64)>,
}

impl RadTabulatedHardening {
    /// Elastic-perfectly plastic response at one yield stress.
    pub fn perfectly_plastic(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        density_kg_per_m3: f64,
        yield_pa: f64,
    ) -> Self {
        Self {
            youngs_modulus_pa,
            poisson_ratio,
            density_kg_per_m3,
            points: vec![(0.0, yield_pa), (0.5, yield_pa)],
        }
    }

    /// Check the hardening curve shape.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for fewer than two points, negative
    /// strains, or non-increasing abscissae.
    pub fn validate(&self) -> Result<(), RadError> {
        if self.points.len() < 2 {
            return Err(RadError::Parse(
                "LAW36 hardening needs at least two points".to_string(),
            ));
        }
        for window in self.points.windows(2) {
            if !window[0].0.is_finite()
                || !window[1].0.is_finite()
                || !window[0].1.is_finite()
                || !window[1].1.is_finite()
                || window[0].0 < 0.0
                || window[1].0 <= window[0].0
                || window[0].1 <= 0.0
                || window[1].1 <= 0.0
            {
                return Err(RadError::Parse(
                    "LAW36 hardening needs rising strains and positive stresses".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Plastic material selected by an extended case.
#[derive(Debug, Clone)]
pub enum RadPlasticMaterial {
    /// Johnson-Cook law (`/MAT/LAW2`).
    JohnsonCook(RadJohnsonCook),
    /// Tabulated isotropic hardening (`/MAT/LAW36`, one curve).
    Tabulated(RadTabulatedHardening),
}

/// Material of an extended case.
#[derive(Debug, Clone)]
pub enum RadMaterial {
    /// Linear elastic solid (`/MAT/LAW1`).
    Elastic(RadElasticMaterial),
    /// Elasto-plastic solid (`/MAT/LAW2` or `/MAT/LAW36`).
    Plastic(RadPlasticMaterial),
}

/// Gravity load rendered as `/GRAV`.
#[derive(Debug, Clone)]
pub struct RadGravity {
    /// Load title written on the card.
    pub name: String,
    /// Global direction `X`, `Y`, or `Z` (right-justified per the guide).
    pub direction: char,
    /// Gravity value in m/s^2 (constant) or curve scale.
    pub value_m_per_s2: f64,
    /// Index into [`RadCase::functions`] (`None` = constant load).
    pub function: Option<usize>,
    /// Index into [`RadCase::node_sets`] (`None` = whole model).
    pub set: Option<usize>,
}

impl RadGravity {
    /// Check the direction flag.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for a direction outside `X`/`Y`/`Z`.
    pub fn validate(&self) -> Result<(), RadError> {
        if !matches!(self.direction, 'X' | 'Y' | 'Z') {
            return Err(RadError::Parse(format!(
                "gravity {} needs direction X, Y, or Z",
                self.name
            )));
        }
        if !self.value_m_per_s2.is_finite() {
            return Err(RadError::Parse(format!(
                "gravity {} needs a finite value",
                self.name
            )));
        }
        Ok(())
    }
}

/// Element time-history request with guide variable names.
#[derive(Debug, Clone, Default)]
pub struct RadThRequest {
    /// Variable names, e.g. `SX` or `M1` (8 characters max each).
    pub vars: Vec<String>,
    /// Element ids recorded (1-based).
    pub elems: Vec<usize>,
}

impl RadThRequest {
    /// Brick stress request (`SX SY SZ SXY SYZ SXZ PLAS`).
    pub fn brick_stress(elems: Vec<usize>) -> Self {
        Self {
            vars: ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ", "PLAS"]
                .iter()
                .map(ToString::to_string)
                .collect(),
            elems,
        }
    }

    /// Shell resultant request (`F1 F2 F12 M1 M2 M12 THIC`).
    pub fn shell_resultants(elems: Vec<usize>) -> Self {
        Self {
            vars: ["F1", "F2", "F12", "M1", "M2", "M12", "THIC"]
                .iter()
                .map(ToString::to_string)
                .collect(),
            elems,
        }
    }

    /// Check variable spellings and element ranges.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for over-long names or out-of-range ids.
    pub fn validate(&self, num_elems: usize, label: &str) -> Result<(), RadError> {
        for var in &self.vars {
            if var.is_empty() || var.len() > 8 {
                return Err(RadError::Parse(format!(
                    "{label} variable {var} must use 1..8 characters"
                )));
            }
        }
        for elem in &self.elems {
            if *elem == 0 || *elem > num_elems {
                return Err(RadError::Parse(format!(
                    "{label} references missing element {elem}"
                )));
            }
        }
        Ok(())
    }
}

/// An extended Radioss comparison case with shells or plasticity.
///
/// Wraps a base [`RadCase`] (nodes, bricks, sets, boundary conditions,
/// loads, walls, node probes, run controls) and adds shell elements, a
/// shell property, a possibly plastic material, gravity, and element
/// time-history requests. Brick-only plastic cases wrap with empty
/// shells; shell cases wrap with empty bricks.
#[derive(Debug, Clone)]
pub struct RadExtendedCase {
    /// Base case: mesh nodes, bricks, sets, BCs, loads, probes, controls.
    pub base: RadCase,
    /// Material card selected for every part.
    pub material: RadMaterial,
    /// Shell elements indexing `base.mesh.nodes`.
    pub shells: RadShellMesh,
    /// Shell property (`None` when the case has no shells).
    pub shell_prop: Option<RadShellProperty>,
    /// Gravity load (`None` disables `/GRAV`).
    pub gravity: Option<RadGravity>,
    /// Node time-history variables (default `DX DY DZ`).
    pub th_node_vars: Vec<String>,
    /// Four-node shell time-history request.
    pub th_shell: RadThRequest,
    /// Brick time-history request.
    pub th_brick: RadThRequest,
}

impl RadExtendedCase {
    /// Wrap a base case with its elastic material and node outputs.
    pub fn wrap_elastic(base: RadCase) -> Self {
        let material = RadMaterial::Elastic(base.material);
        Self {
            base,
            material,
            shells: RadShellMesh::default(),
            shell_prop: None,
            gravity: None,
            th_node_vars: ["DX", "DY", "DZ"].iter().map(ToString::to_string).collect(),
            th_shell: RadThRequest::default(),
            th_brick: RadThRequest::default(),
        }
    }

    /// Part id carrying shells (1 when shells exist).
    fn shell_part(&self) -> Option<usize> {
        if self.shells.is_empty() {
            None
        } else {
            Some(1)
        }
    }

    /// Part id carrying bricks (2 beside shells, else 1).
    fn solid_part(&self) -> Option<usize> {
        if self.base.mesh.bricks.is_empty() {
            None
        } else if self.shell_part().is_some() {
            Some(2)
        } else {
            Some(1)
        }
    }

    /// Check every index, id, and card range of the case.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for the first inconsistency found.
    pub fn validate(&self) -> Result<(), RadError> {
        self.base.mesh.validate()?;
        self.shells.validate(self.base.mesh.num_nodes())?;
        if !self.shells.is_empty() {
            match &self.shell_prop {
                Some(prop) => prop.validate()?,
                None => {
                    return Err(RadError::Parse(
                        "shell elements need a shell property".to_string(),
                    ));
                }
            }
        }
        if let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(table)) = &self.material {
            table.validate()?;
        }
        if let Some(gravity) = &self.gravity {
            gravity.validate()?;
            if let Some(function) = gravity.function
                && function >= self.base.functions.len()
            {
                return Err(RadError::Parse(format!(
                    "gravity {} references missing function {function}",
                    gravity.name
                )));
            }
            if let Some(set) = gravity.set
                && set >= self.base.node_sets.len()
            {
                return Err(RadError::Parse(format!(
                    "gravity {} references missing node set {set}",
                    gravity.name
                )));
            }
        }
        for node in &self.base.th_nodes {
            if *node == 0 || *node > self.base.mesh.num_nodes() {
                return Err(RadError::Parse(format!(
                    "TH/NODE references missing node {node}"
                )));
            }
        }
        self.th_shell
            .validate(self.shells.num_quads(), "/TH/SHEL")?;
        self.th_brick
            .validate(self.base.mesh.num_bricks(), "/TH/BRIC")?;
        Ok(())
    }
}

fn field_var(name: &str) -> String {
    format!("{name:<10}")
}

/// Concentrated-load direction code: 1..3 force X/Y/Z, 4..6 moment X/Y/Z.
///
/// Right-justified per the guide (`/CLOAD` rejects integer directions).
fn direction_code(direction: usize) -> &'static str {
    match direction {
        1 => "X",
        2 => "Y",
        3 => "Z",
        4 => "XX",
        5 => "YY",
        6 => "ZZ",
        other => panic!("RadConcentratedLoad direction {other} outside 1..=6"),
    }
}

fn field_blank() -> String {
    "          ".to_string()
}

/// Render the starter deck of an extended case.
///
/// Emits the guide-shaped blocks `/SHELL/part_ID`, `/SH3N/part_ID`,
/// `/BRICK/part_ID`, `/PROP/SHELL`, `/MAT/LAW2` or `/MAT/LAW36`,
/// `/GRAV`, and `/TH` requests with variable cards. The engine deck is
/// unchanged from [`write_engine`]; use [`write_extended_engine`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{shell_clamped_plate_case, write_extended_starter};
/// let case = shell_clamped_plate_case();
/// case.validate().unwrap();
/// let deck = write_extended_starter(&case);
/// assert!(deck.contains("/SHELL/1"));
/// assert!(deck.contains("/PROP/SHELL/1/1"));
/// assert!(deck.contains("/TH/SHEL/2"));
/// ```
pub fn write_extended_starter(case: &RadExtendedCase) -> String {
    let base = &case.base;
    let mut deck = String::new();
    deck.push_str("#RADIOSS STARTER\n");
    deck.push_str(&format!(
        "# sonium_qa rad extended case {} (SI units: kg, m, s)\n",
        base.name
    ));
    deck.push_str("/BEGIN\n");
    deck.push_str(&format!("{}\n", base.title));
    // Invers + input/work unit systems (mandatory /BEGIN cards).
    deck.push_str("      2022\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/UNIT/1\nSI mass-length-time\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/NODE\n");
    for (index, node) in base.mesh.nodes.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(index + 1),
            field_float(node[0]),
            field_float(node[1]),
            field_float(node[2])
        ));
    }
    if let Some(part) = case.solid_part() {
        deck.push_str(&format!("/BRICK/{part}\n"));
        for (index, brick) in base.mesh.bricks.iter().enumerate() {
            let mut line = field_int(index + 1);
            for node in brick {
                line.push_str(&field_int(*node));
            }
            line.push('\n');
            deck.push_str(&line);
        }
    }
    if let Some(part) = case.shell_part() {
        if !case.shells.quads.is_empty() {
            deck.push_str(&format!("/SHELL/{part}\n"));
            for (index, quad) in case.shells.quads.iter().enumerate() {
                let mut line = field_int(index + 1);
                for node in quad {
                    line.push_str(&field_int(*node));
                }
                line.push('\n');
                deck.push_str(&line);
            }
        }
        if !case.shells.tris.is_empty() {
            deck.push_str(&format!("/SH3N/{part}\n"));
            for (index, tri) in case.shells.tris.iter().enumerate() {
                let mut line = field_int(index + 1);
                for node in tri {
                    line.push_str(&field_int(*node));
                }
                line.push('\n');
                deck.push_str(&line);
            }
        }
    }
    for (index, set) in base.node_sets.iter().enumerate() {
        deck.push_str(&format!("/GRNOD/NODE/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", set.name));
        for chunk in set.nodes.chunks(10) {
            let mut line = String::new();
            for node in chunk {
                line.push_str(&field_int(*node));
            }
            line.push('\n');
            deck.push_str(&line);
        }
    }
    write_extended_material(&mut deck, case);
    if let Some(part) = case.shell_part() {
        let prop = case.shell_prop.as_ref().expect("validated shell property");
        deck.push_str(&format!("/PROP/SHELL/{part}/1\n"));
        deck.push_str(&format!("{}\n", prop.name));
        deck.push_str(&format!(
            "{:>10}{:>10}{:>10}{:>10}{}{}{}{}{}{}\n",
            prop.ishell,
            prop.ismstr,
            prop.ish3n,
            prop.idrill,
            field_int(0),
            field_int(0),
            field_float(0.0),
            field_int(0),
            field_int(0),
            field_int(0),
        ));
        // Membrane, bending, and rotational hourglass plus damping are
        // unused by QEPH/QBAT; the card stays blank (guide defaults).
        deck.push_str(&format!(
            "{}{}{}{}{}\n",
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
        ));
        deck.push_str(&format!(
            "{:>10}{}{}{}{}{:>10}{:>10}{}\n",
            prop.n_thickness,
            field_int(0),
            field_float(prop.thickness_m),
            field_float(0.0),
            field_int(0),
            prop.ithick,
            prop.iplas,
            field_int(0),
        ));
    }
    if let Some(part) = case.solid_part() {
        deck.push_str(&format!("/PROP/SOLID/{part}/1\nelastic brick property\n"));
        // HEPH bricks + full geometric nonlinearities (best practice;
        // proven 1.6% vs EB on the coarse cantilever, Isolid=1 gave 41%).
        deck.push_str("        24         2\n");
    }
    if let Some(part) = case.shell_part() {
        deck.push_str(&format!("/PART/{part}\nshell part\n"));
        deck.push_str(&format!("{}{}\n", field_int(part), field_int(1)));
    }
    if let Some(part) = case.solid_part() {
        deck.push_str(&format!("/PART/{part}\nsolid part\n"));
        deck.push_str(&format!("{}{}\n", field_int(part), field_int(1)));
    }
    for (index, function) in base.functions.iter().enumerate() {
        deck.push_str(&format!("/FUNCT/{}\n", index + 1));
        deck.push_str(&format!("{}\n", function.name));
        for (x, y) in &function.points {
            deck.push_str(&format!("{}{}\n", field_float(*x), field_float(*y)));
        }
    }
    if let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(table)) = &case.material {
        deck.push_str(&format!("/FUNCT/{}\n", base.functions.len() + 1));
        deck.push_str("LAW36 yield stress vs plastic strain\n");
        for (strain, stress) in &table.points {
            deck.push_str(&format!("{}{}\n", field_float(*strain), field_float(*stress)));
        }
    }
    for (index, bcs) in base.bcs.iter().enumerate() {
        deck.push_str(&format!("/BCS/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", bcs.name));
        // Proven against the engine + reference guide: one Trarot/Skew/grnd
        // line (Trarot packs the six lock flags as "TTT RRR", 1 = fixed).
        let trarot = format!(
            "{}{}{} {}{}{}",
            i32::from(bcs.lock[0]),
            i32::from(bcs.lock[1]),
            i32::from(bcs.lock[2]),
            i32::from(bcs.lock[3]),
            i32::from(bcs.lock[4]),
            i32::from(bcs.lock[5]),
        );
        deck.push_str(&format!(
            "{:>10}{}{}\n",
            trarot,
            field_int(0),
            field_int(bcs.set + 1)
        ));
    }
    for (index, cload) in base.cloads.iter().enumerate() {
        deck.push_str(&format!("/CLOAD/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", cload.name));
        deck.push_str(&format!(
            "{}{:>10}{}{}{}{}{}{}\n",
            field_int(cload.function + 1),
            direction_code(cload.direction),
            field_int(0),
            field_int(0),
            field_int(cload.set + 1),
            field_int(0),
            field_float(1.0),
            field_float(cload.force_scale_n),
        ));
    }
    if let Some(gravity) = &case.gravity {
        deck.push_str("/GRAV/1/1\n");
        deck.push_str(&format!("{}\n", gravity.name));
        deck.push_str(&format!(
            "{}{:>10}{}{}{}{}{}{}\n",
            field_int(gravity.function.map_or(0, |id| id + 1)),
            gravity.direction,
            field_int(0),
            field_int(0),
            field_int(gravity.set.map_or(0, |id| id + 1)),
            field_blank(),
            field_float(1.0),
            field_float(gravity.value_m_per_s2),
        ));
    }
    for (index, wall) in base.walls.iter().enumerate() {
        write_wall_block(&mut deck, index, wall);
    }
    let mut th_group = 0;
    write_th_block(
        &mut deck,
        &mut th_group,
        "NODE",
        "time history nodes",
        &case.th_node_vars,
        &base.th_nodes,
    );
    write_th_block(
        &mut deck,
        &mut th_group,
        "SHEL",
        "time history shells",
        &case.th_shell.vars,
        &case.th_shell.elems,
    );
    write_th_block(
        &mut deck,
        &mut th_group,
        "BRIC",
        "time history bricks",
        &case.th_brick.vars,
        &case.th_brick.elems,
    );
    if !base.walls.is_empty() {
        let wall_ids: Vec<usize> = (1..=base.walls.len()).collect();
        write_th_block(
            &mut deck,
            &mut th_group,
            "RWALL",
            "time history walls",
            &["FN".to_string()],
            &wall_ids,
        );
    }
    deck.push_str("/END\n");
    deck
}

fn write_extended_material(deck: &mut String, case: &RadExtendedCase) {
    match &case.material {
        RadMaterial::Elastic(material) => {
            deck.push_str("/MAT/LAW1/1/1\n");
            deck.push_str("linear elastic solid\n");
            // Proven against the engine: line 1 is density, line 2 is E/nu.
            deck.push_str(&format!(
                "{}\n",
                field_float(material.density_kg_per_m3)
            ));
            deck.push_str(&format!(
                "{}{}\n",
                field_float(material.youngs_modulus_pa),
                field_float(material.poisson_ratio)
            ));
        }
        RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(jc)) => {
            deck.push_str("/MAT/LAW2/1/1\n");
            deck.push_str("johnson-cook elasto-plastic metal\n");
            deck.push_str(&format!("{}\n", field_float(jc.density_kg_per_m3)));
            // Classic input (Iflag 0), total strain rate (VP 2), and the
            // guide default pressure cutoff.
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(jc.youngs_modulus_pa),
                field_float(jc.poisson_ratio),
                field_int(0),
                field_int(2),
                field_float(-1.0e20),
            ));
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(jc.yield_pa),
                field_float(jc.hardening_pa),
                field_float(jc.hardening_exp),
                field_float(jc.fail_plastic_strain),
                field_float(jc.max_stress_pa),
            ));
            // Rate block: ICC 1 and Fsmooth 1 are the guide defaults;
            // isotropic hardening (Chard 0); no thermal softening run.
            deck.push_str(&format!(
                "{}{}{}{}{}{}\n",
                field_float(jc.rate_coeff),
                field_float(jc.ref_strain_rate),
                field_int(1),
                field_int(1),
                field_float(1.0e20),
                field_float(0.0),
            ));
            // Isothermal steel thermal line (unused without /HEAT).
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(1.0),
                field_float(1793.0),
                field_float(jc.density_kg_per_m3 * 500.0),
                field_float(293.0),
                field_float(1793.0),
            ));
        }
        RadMaterial::Plastic(RadPlasticMaterial::Tabulated(table)) => {
            let curve = case.base.functions.len() + 1;
            deck.push_str("/MAT/LAW36/1/1\n");
            deck.push_str("tabulated isotropic hardening metal\n");
            deck.push_str(&format!(
                "{}\n",
                field_float(table.density_kg_per_m3)
            ));
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(table.youngs_modulus_pa),
                field_float(table.poisson_ratio),
                field_float(1.0e20),
                field_float(1.0e20),
                field_float(2.0e20),
            ));
            // One hardening curve, smoothed rates, full isotropic model,
            // guide-default filter and total-rate flag.
            deck.push_str(&format!(
                "{}{}{}{}{}{}{}\n",
                field_int(1),
                field_int(1),
                field_float(0.0),
                field_float(10_000.0),
                field_float(3.0e20),
                field_blank(),
                field_int(0),
            ));
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_int(0),
                field_float(1.0),
                field_int(0),
                field_float(0.0),
                field_float(0.0),
            ));
            deck.push_str(&format!("{}\n", field_int(curve)));
            deck.push_str(&format!("{}\n", field_float(1.0)));
            deck.push_str(&format!("{}\n", field_float(0.0)));
        }
    }
}

fn write_th_block(
    deck: &mut String,
    next_group: &mut usize,
    keyword: &str,
    title: &str,
    vars: &[String],
    ids: &[usize],
) {
    if vars.is_empty() || ids.is_empty() {
        return;
    }
    // Proven against the engine (error 79): TH group ids share one
    // namespace across all /TH/* types, so the counter is deck-global.
    for var_card in vars.chunks(10) {
        *next_group += 1;
        deck.push_str(&format!("/TH/{keyword}/{}\n", *next_group));
        deck.push_str(&format!("{title}\n"));
        let mut line = String::new();
        for var in var_card {
            line.push_str(&field_var(var));
        }
        line.push('\n');
        deck.push_str(&line);
        for id in ids {
            deck.push_str(&format!("{}{}\n", field_int(*id), field_int(0)));
        }
    }
}

/// Render the engine deck of an extended case.
///
/// Run controls live in the wrapped base case, so this forwards to
/// [`write_engine`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{shell_scordelis_lo_case, write_extended_engine};
/// let case = shell_scordelis_lo_case();
/// let deck = write_extended_engine(&case);
/// assert!(deck.contains("#RADIOSS ENGINE"));
/// ```
pub fn write_extended_engine(case: &RadExtendedCase) -> String {
    write_engine(&case.base)
}

/// Run the starter then the engine for an extended case.
///
/// Decks resolve from the wrapped base case name; see [`run_case`].
///
/// # Errors
///
/// Returns [`RadError`] when a stage fails or parsing fails.
pub fn run_extended_case(
    decks_dir: &Path,
    exe: &Path,
    case: &RadExtendedCase,
) -> Result<RunOutput, RadError> {
    run_case(decks_dir, exe, &case.base)
}

/// Render the checked-in `probes.json` artifact of an extended case.
///
/// Records expected probe values, tolerances, and the producing
/// engine version. See [`probes_json`].
pub fn extended_probes_json(case: &RadExtendedCase, engine_version: &str) -> String {
    probes_json(&case.base, engine_version)
}

/// Time-history channel of a shell variable, e.g. `M1/SHEL/12`.
///
/// Element channels carry the `/TH` group so shell and brick ids never
/// collide; node channels keep the established `DZ/95` spelling. The
/// spelling is confirmed only up to the `th_to_csv` naming, which no
/// host engine can verify yet.
pub fn shel_channel(var: &str, shell: usize) -> String {
    format!("{var}/SHEL/{shell}")
}

/// Time-history channel of a brick variable, e.g. `SX/BRIC/4`.
///
/// Brick stress channels from `/TH/BRIC` use this group-qualified
/// spelling (see [`shel_channel`]).
pub fn bric_channel(var: &str, brick: usize) -> String {
    format!("{var}/BRIC/{brick}")
}

/// Quad mesh of the full Scordelis-Lo roof.
#[derive(Debug, Clone)]
pub struct ScordelisShellMesh {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Quad connectivity, 1-based node ids, outward normals.
    pub quads: Vec<[usize; 4]>,
    /// Diaphragm ring at `z = 0`.
    pub end_lo: Vec<usize>,
    /// Diaphragm ring at `z = L`.
    pub end_hi: Vec<usize>,
    /// Free-edge midspan probe node (theta = +40 deg, z = L/2).
    pub probe_node: usize,
    /// Quad id nearest the apex midspan (theta = 0, z = L/2).
    pub probe_shell: usize,
}

/// Mesh the Scordelis-Lo roof with quads.
///
/// Covers theta in [-40, +40] degrees over `n_theta` divisions and the
/// length over `n_z` divisions. Both counts must be even so the probe
/// node and shell sit exactly on the free edge and apex midspan.
///
/// # Panics
///
/// Panics for odd or zero divisions.
pub fn scordelis_lo_shell_mesh(n_theta: usize, n_z: usize) -> ScordelisShellMesh {
    assert!(n_theta >= 2 && n_theta.is_multiple_of(2), "n_theta must be even");
    assert!(n_z >= 2 && n_z.is_multiple_of(2), "n_z must be even");
    let (radius, length) = (25.0, 50.0);
    let half_angle = 40.0_f64.to_radians();
    let id = |iz: usize, it: usize| iz * (n_theta + 1) + it + 1;
    let mut nodes = Vec::with_capacity((n_theta + 1) * (n_z + 1));
    for iz in 0..=n_z {
        for it in 0..=n_theta {
            let theta = -half_angle + 2.0 * half_angle * it as f64 / n_theta as f64;
            nodes.push([
                radius * theta.sin(),
                radius * theta.cos(),
                length * iz as f64 / n_z as f64,
            ]);
        }
    }
    let mut quads = Vec::with_capacity(n_theta * n_z);
    for iz in 0..n_z {
        for it in 0..n_theta {
            // Winding (iz,it) -> (iz+1,it) -> (iz+1,it+1) -> (iz,it+1)
            // keeps the outward normal (dz cross dtheta points out).
            quads.push([
                id(iz, it),
                id(iz + 1, it),
                id(iz + 1, it + 1),
                id(iz, it + 1),
            ]);
        }
    }
    let mut end_lo = Vec::with_capacity(n_theta + 1);
    let mut end_hi = Vec::with_capacity(n_theta + 1);
    for it in 0..=n_theta {
        end_lo.push(id(0, it));
        end_hi.push(id(n_z, it));
    }
    ScordelisShellMesh {
        nodes,
        quads,
        end_lo,
        end_hi,
        probe_node: id(n_z / 2, n_theta),
        probe_shell: (n_z / 2 - 1) * n_theta + (n_theta / 2 - 1) + 1,
    }
}

/// Polar shell mesh of a clamped circular plate.
#[derive(Debug, Clone)]
pub struct ClampedPlateShellMesh {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Quad connectivity between rings, 1-based node ids.
    pub quads: Vec<[usize; 4]>,
    /// Center fan triangles, 1-based node ids.
    pub tris: Vec<[usize; 3]>,
    /// Clamped rim nodes.
    pub rim: Vec<usize>,
    /// Center probe node.
    pub center_node: usize,
    /// Ring-1 probe quad id for shell resultants.
    pub probe_shell: usize,
    /// Centroid radius of the probe quad in metres.
    pub probe_radius_m: f64,
    /// Tributary area per node in m^2 (sums to the disk area).
    pub nodal_area_m2: Vec<f64>,
}

/// Mesh a disk with a center fan plus quad rings.
///
/// Ring `i` sits at `radius * i / n_rings` with `n_sectors` nodes; the
/// center fan covers the hub with triangles and every outer band with
/// quads. Tributary areas follow the half-station ring bands.
///
/// # Panics
///
/// Panics for fewer than two rings or three sectors.
pub fn clamped_plate_shell_mesh(
    radius_m: f64,
    n_rings: usize,
    n_sectors: usize,
) -> ClampedPlateShellMesh {
    assert!(n_rings >= 2, "plate mesh needs at least two rings");
    assert!(n_sectors >= 3, "plate mesh needs at least three sectors");
    let ring_radius = |i: usize| radius_m * i as f64 / n_rings as f64;
    // Node 1 is the center; ring i sector j follows in row-major order.
    let id = |i: usize, j: usize| 1 + (i - 1) * n_sectors + j + 1;
    let mut nodes = vec![[0.0, 0.0, 0.0]];
    for i in 1..=n_rings {
        for j in 0..n_sectors {
            let theta = 2.0 * std::f64::consts::PI * j as f64 / n_sectors as f64;
            nodes.push([
                ring_radius(i) * theta.cos(),
                ring_radius(i) * theta.sin(),
                0.0,
            ]);
        }
    }
    let mut tris = Vec::with_capacity(n_sectors);
    for j in 0..n_sectors {
        tris.push([1, id(1, j), id(1, (j + 1) % n_sectors)]);
    }
    let mut quads = Vec::with_capacity((n_rings - 1) * n_sectors);
    for i in 1..n_rings {
        for j in 0..n_sectors {
            quads.push([
                id(i, j),
                id(i, (j + 1) % n_sectors),
                id(i + 1, (j + 1) % n_sectors),
                id(i + 1, j),
            ]);
        }
    }
    let mut rim = Vec::with_capacity(n_sectors);
    for j in 0..n_sectors {
        rim.push(id(n_rings, j));
    }
    // Half-station bands: the center owns the hub disk, each ring owns
    // its band split over sectors, and the rim band closes at `radius`.
    let station = |i: usize| {
        // Outer half-station of ring i (i = 0 is the center hub edge).
        if i == 0 {
            ring_radius(1) / 2.0
        } else if i >= n_rings {
            radius_m
        } else {
            (ring_radius(i) + ring_radius(i + 1)) / 2.0
        }
    };
    let mut nodal_area_m2 = vec![0.0; nodes.len()];
    nodal_area_m2[0] = std::f64::consts::PI * station(0).powi(2);
    let mut inner = station(0);
    for i in 1..=n_rings {
        let outer = station(i);
        let band = std::f64::consts::PI * (outer.powi(2) - inner.powi(2)) / n_sectors as f64;
        for j in 0..n_sectors {
            nodal_area_m2[id(i, j) - 1] = band;
        }
        inner = outer;
    }
    ClampedPlateShellMesh {
        nodes,
        quads,
        tris,
        rim,
        center_node: 1,
        probe_shell: 1,
        probe_radius_m: (ring_radius(1) + ring_radius(2)) / 2.0,
        nodal_area_m2,
    }
}

/// Quasi-static ramp-hold curve for explicit gravity and pressure.
///
/// Ramps 0 to 1 over the first 80 percent of the stop time, then holds
/// full load so final-value probes read a settled state.
fn ramp_hold_function() -> RadFunction {
    RadFunction {
        name: "ramp 0 to 1 then hold".to_string(),
        points: vec![(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)],
    }
}

/// Build the Scordelis-Lo shell comparison case.
///
/// A full-roof QEPH model (R = 25, L = 50, t = 0.25) with diaphragm
/// ends carries self weight through `/GRAV`. The free-edge midspan
/// probe is a REGRESSION baseline (-0.2574), not the MacNeal target
/// (-0.3024): the settled engine answer sits 15% stiff across every
/// formulation/mesh/material variant tried, while in-process MITC4
/// converges to the target. See the probe comment for the evidence.
pub fn shell_scordelis_lo_case() -> RadExtendedCase {
    // 32x32 with a slow ramp; the hold average is the settled static.
    let mesh = scordelis_lo_shell_mesh(32, 32);
    // Density carries the 90 Pa self weight through rho * g * t.
    let density = 90.0 / (9.81 * 0.25);
    let base = RadCase {
        name: "shell_scordelis_lo".to_string(),
        title: "sonium_qa rad scordelis-lo roof quasi-static gravity".to_string(),
        mesh: RadMesh {
            nodes: mesh.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: 4.32e8,
            poisson_ratio: 0.0,
            density_kg_per_m3: density,
        },
        node_sets: vec![
            RadNodeSet {
                name: "diaphragm z=0".to_string(),
                nodes: mesh.end_lo,
            },
            RadNodeSet {
                name: "diaphragm z=L".to_string(),
                nodes: mesh.end_hi,
            },
        ],
        bcs: vec![
            RadBoundaryCondition {
                name: "diaphragm z=0".to_string(),
                set: 0,
                lock: [true, true, false, false, false, false],
            },
            RadBoundaryCondition {
                name: "diaphragm z=L".to_string(),
                set: 1,
                lock: [true, true, false, false, false, false],
            },
        ],
        functions: vec![RadFunction {
            // First mode period is 1.19 s with higher-mode hash; the 5 s
            // ramp (Tr/T1 > 4) tracks quasi-statically and the 3 s hold
            // averages the residual corner transient to its center.
            name: "slow ramp 0 to 1 then hold".to_string(),
            points: vec![(0.0, 0.0), (5.0, 1.0), (8.0, 1.0)],
        }],
        cloads: Vec::new(),
        walls: Vec::new(),
        th_nodes: vec![mesh.probe_node],
        probes: vec![
            RadProbe {
                // REGRESSION (not validation): the settled engine answer is
                // -0.2574 across QEPH/BATOZ, meshes 16-48, Ismstr 1/2, and
                // LAW1/LAW2, while MITC4 and the MacNeal target agree at
                // -0.3024. The engine's flat-facet quads lock on this
                // membrane-dominated curved benchmark; sonium-fem is
                // authoritative here and the engine leg tracks drift only.
                label: "free edge mid DY".to_string(),
                channel: format!("DY/{}", mesh.probe_node),
                expected: -0.2574,
                abs_tol: 5e-3,
                rel_tol: 0.02,
            },
            RadProbe {
                label: "apex shell THIC".to_string(),
                channel: shel_channel("THIC", mesh.probe_shell),
                expected: 0.25,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
        ],
        stop_time_s: 8.0,
        th_period_s: 0.02,
        anim_period_s: 0.8,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: mesh.quads,
        tris: Vec::new(),
    };
    case.shell_prop = Some(RadShellProperty::qeph("roof section", 0.25));
    case.gravity = Some(RadGravity {
        name: "self weight".to_string(),
        direction: 'Y',
        value_m_per_s2: -9.81,
        function: Some(0),
        set: None,
    });
    case.th_shell = RadThRequest::shell_resultants(vec![mesh.probe_shell]);
    case
}

/// Clamped circular plate center deflection (Kirchhoff closed form).
fn clamped_plate_center_deflection(
    pressure_pa: f64,
    radius_m: f64,
    young_pa: f64,
    poisson: f64,
    thick_m: f64,
) -> f64 {
    let rigidity = young_pa * thick_m.powi(3) / (12.0 * (1.0 - poisson.powi(2)));
    -pressure_pa * radius_m.powi(4) / (64.0 * rigidity)
}

/// Clamped circular plate circumferential moment at a radius.
///
/// `M_t(r) = q/16 [(1+nu) a^2 - (1+3nu) r^2]` (Timoshenko plates,
/// axisymmetric bending). The probe quad local axis 1 runs along its
/// N1-N2 edge (circumferential), so `M1` reads this component.
fn clamped_plate_hoop_moment(
    pressure_pa: f64,
    radius_m: f64,
    poisson: f64,
    at_radius_m: f64,
) -> f64 {
    pressure_pa / 16.0
        * ((1.0 + poisson) * radius_m.powi(2) - (1.0 + 3.0 * poisson) * at_radius_m.powi(2))
}

/// Build the clamped circular plate shell comparison case.
///
/// A polar QEPH disk (a = 1, t = 0.02) with a clamped rim carries
/// uniform pressure as ring-wise nodal `/CLOAD`s; the center probe
/// replays the Kirchhoff closed form and the probe shell checks the
/// hoop moment plus constant thickness.
pub fn shell_clamped_plate_case() -> RadExtendedCase {
    let (radius, thick, young, poisson, pressure) = (1.0, 0.02, 200.0e9, 0.3, 1000.0);
    let (n_rings, n_sectors) = (8, 32);
    let mesh = clamped_plate_shell_mesh(radius, n_rings, n_sectors);
    // One node set per ring: tributary areas are uniform within a ring.
    let mut node_sets = vec![RadNodeSet {
        name: "rim".to_string(),
        nodes: mesh.rim.clone(),
    }];
    let mut cloads = Vec::new();
    let ring_nodes = |i: usize| {
        if i == 0 {
            vec![1]
        } else {
            (0..n_sectors).map(|j| 1 + (i - 1) * n_sectors + j + 1).collect()
        }
    };
    for i in 0..=n_rings {
        let nodes = ring_nodes(i);
        let area = mesh.nodal_area_m2[nodes[0] - 1];
        node_sets.push(RadNodeSet {
            name: format!("pressure ring {i}"),
            nodes: nodes.clone(),
        });
        cloads.push(RadConcentratedLoad {
            name: format!("pressure ring {i}"),
            set: node_sets.len() - 1,
            direction: 3,
            function: 0,
            force_scale_n: -pressure * area,
        });
    }
    let center = clamped_plate_center_deflection(pressure, radius, young, poisson, thick);
    let hoop = clamped_plate_hoop_moment(pressure, radius, poisson, mesh.probe_radius_m);
    let base = RadCase {
        name: "shell_clamped_plate".to_string(),
        title: "sonium_qa rad clamped plate quasi-static pressure".to_string(),
        mesh: RadMesh {
            nodes: mesh.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: young,
            poisson_ratio: poisson,
            density_kg_per_m3: 7800.0,
        },
        node_sets,
        bcs: vec![RadBoundaryCondition {
            name: "clamp rim".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![ramp_hold_function()],
        cloads,
        walls: Vec::new(),
        th_nodes: vec![mesh.center_node],
        probes: vec![
            RadProbe {
                label: "center DZ".to_string(),
                channel: format!("DZ/{}", mesh.center_node),
                expected: center,
                abs_tol: 1e-9,
                rel_tol: 0.05,
            },
            RadProbe {
                label: "probe shell THIC".to_string(),
                channel: shel_channel("THIC", mesh.probe_shell),
                expected: thick,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
            RadProbe {
                label: "probe shell M1".to_string(),
                channel: shel_channel("M1", mesh.probe_shell),
                // M1 carries moment per unit length per thickness squared.
                expected: hoop / thick.powi(2),
                abs_tol: 1e3,
                rel_tol: 0.15,
            },
        ],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: mesh.quads,
        tris: mesh.tris,
    };
    case.shell_prop = Some(RadShellProperty::qeph("plate section", thick));
    case.th_shell = RadThRequest::shell_resultants(vec![mesh.probe_shell]);
    case
}

/// Elastic-perfectly plastic steel as a LAW36 material.
///
/// Matches the thick-cylinder autofrettage case inputs (E, nu, rho,
/// yield) so the plasticity lane can adopt it with a wrap call.
pub fn perfectly_plastic_steel_law36(
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
    density_kg_per_m3: f64,
    yield_pa: f64,
) -> RadPlasticMaterial {
    RadPlasticMaterial::Tabulated(RadTabulatedHardening::perfectly_plastic(
        youngs_modulus_pa,
        poisson_ratio,
        density_kg_per_m3,
        yield_pa,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brick_volume(mesh: &RadMesh, brick: &[usize; 8]) -> f64 {
        // Decompose the hex into five tets sharing the main diagonal.
        let p = |i: usize| mesh.nodes[i - 1];
        let tet = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| {
            let (ab, ac, ad) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                [d[0] - a[0], d[1] - a[1], d[2] - a[2]],
            );
            ((ab[1] * ac[2] - ab[2] * ac[1]) * ad[0]
                + (ab[2] * ac[0] - ab[0] * ac[2]) * ad[1]
                + (ab[0] * ac[1] - ab[1] * ac[0]) * ad[2])
                / 6.0
        };
        let n: Vec<[f64; 3]> = brick.iter().map(|id| p(*id)).collect();
        tet(n[0], n[1], n[3], n[4]).abs()
            + tet(n[1], n[2], n[3], n[6]).abs()
            + tet(n[1], n[3], n[4], n[6]).abs()
            + tet(n[3], n[4], n[6], n[7]).abs()
            + tet(n[1], n[4], n[5], n[6]).abs()
    }

    #[test]
    fn cantilever_mesh_is_valid_and_indexed() {
        let (mesh, root, tip, tip_node) = cantilever_brick_mesh(4, 1, 1, 1.0, 0.1, 0.1);
        assert_eq!(mesh.num_nodes(), 5 * 2 * 2);
        assert_eq!(mesh.num_bricks(), 4);
        mesh.validate().unwrap();
        assert_eq!(root.len(), 4);
        assert_eq!(tip.len(), 4);
        assert!(tip.contains(&tip_node));
        for brick in &mesh.bricks {
            assert!(brick_volume(&mesh, brick) > 0.0);
        }
    }

    #[test]
    fn sphere_mesh_is_all_positive_volume() {
        let mesh = sphere_ball_mesh(8, 0.05);
        mesh.validate().unwrap();
        assert!(!mesh.bricks.is_empty());
        for brick in &mesh.bricks {
            assert!(
                brick_volume(&mesh, brick) > 1e-15,
                "inverted brick: {brick:?}"
            );
        }
        // Surface nodes sit on the sphere within projection tolerance.
        let mut surface = 0;
        for node in &mesh.nodes {
            let r = (node[0] * node[0] + node[1] * node[1] + (node[2] - 0.05).powi(2)).sqrt();
            assert!(r <= 0.05 + 1e-9, "node escapes the ball: {node:?}");
            if r > 0.049 {
                surface += 1;
            }
        }
        assert!(surface > 0, "projection must place nodes on the surface");
    }

    #[test]
    fn starter_and_engine_decks_carry_required_cards() {
        for case in [cantilever_case(), hertz_sphere_case()] {
            case.mesh.validate().unwrap();
            let starter = write_starter(&case);
            for card in [
                "#RADIOSS STARTER",
                "/BEGIN",
                "/UNIT/1",
                "/NODE",
                "/BRICK",
                "/MAT/LAW1/1/1",
                "/PROP/SOLID/1/1",
                "/PART/1",
                "/END",
            ] {
                assert!(starter.contains(card), "{} lacks {card}", case.name);
            }
            let engine = write_engine(&case);
            for card in [
                "#RADIOSS ENGINE",
                "/RUN/",
                "/TFILE",
                "/TH/TITLE",
                "/ANIM/DT",
            ] {
                assert!(engine.contains(card), "{} lacks {card}", case.name);
            }
            // Proven against the engine: no /STOP or /END terminator cards.
            assert!(!engine.contains("/STOP"), "{} carries /STOP", case.name);
            assert!(!engine.contains("/END"), "{} carries /END", case.name);
        }
        let starter = write_starter(&cantilever_case());
        assert!(starter.contains("/BCS/1"));
        assert!(starter.contains("/CLOAD/1/1"));
        let starter = write_starter(&hertz_sphere_case());
        assert!(starter.contains("/RWALL/PLANE/1/1"));
        assert!(starter.contains("/RWALL/PLANE/2/1"));
    }

    #[test]
    fn rwall_plane_card_order_matches_reference() {
        let starter = write_starter(&hertz_sphere_case());
        let lines: Vec<&str> = starter.lines().collect();
        for wall_id in [1, 2] {
            let header = format!("/RWALL/PLANE/{wall_id}/1");
            let at = lines
                .iter()
                .position(|line| *line == header)
                .unwrap_or_else(|| panic!("{starter} lacks {header}"));
            // Title, node_ID/Slide/grnds, Dsearch/fric, then M (fixed) or
            // Mass/V (moving), then M1. Six lines per wall block.
            assert_eq!(lines.len() > at + 5, true, "truncated {header}");
            let node_slide: Vec<&str> = lines[at + 2].split_whitespace().collect();
            let mass_or_m: Vec<&str> = lines[at + 4].split_whitespace().collect();
            if wall_id == 1 {
                assert_eq!(node_slide[0], "0", "fixed wall needs node_ID 0");
                assert_eq!(mass_or_m.len(), 3, "fixed wall needs an M line");
            } else {
                assert_ne!(node_slide[0], "0", "moving wall needs a driver node");
                assert_eq!(mass_or_m.len(), 4, "moving wall needs Mass/VX0/VY0/VZ0");
                assert!(
                    mass_or_m[0].starts_with("0."),
                    "moving wall needs Mass 0 (imposed velocity)"
                );
            }
        }
    }

    #[test]
    fn wall_channels_normalize_past_wall_titles() {
        assert_eq!(
            normalize_channel(
                "\"time history walls                       1 fixed bottom plane                        FNZ\""
            ),
            "FNZ/RWALL/1"
        );
        assert_eq!(
            normalize_channel(
                "\"time history walls                       2 moving top plane                          FNX\""
            ),
            "FNX/RWALL/2"
        );
        assert_eq!(
            normalize_channel("\"time history nodes                       247                                           DZ\""),
            "DZ/247"
        );
    }

    #[test]
    fn time_history_csv_parses_and_interpolates() {
        let dir = std::env::temp_dir().join("sonium_qa_rad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sampleT01.csv");
        std::fs::write(
            &path,
            "Time,DZ/5,FNZ/RWALL/1\n0.0,0.0,0.0\n1.0,0.002,40.0\n2.0,0.004,80.0\n",
        )
        .unwrap();
        let history = parse_t01(&path).unwrap();
        assert_eq!(history.times, vec![0.0, 1.0, 2.0]);
        assert!((history.final_value("DZ/5").unwrap() - 0.004).abs() < 1e-15);
        assert!((history.value_at("DZ/5", 0.5).unwrap() - 0.001).abs() < 1e-15);
        assert!(history.channel("missing").is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn impulse_channels_differentiate_to_force() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let impulse = vec![0.0, 10.0, 40.0, 90.0];
        let force = impulse_to_force(&times, &impulse);
        assert!((force[1] - 20.0).abs() < 1e-12);
        assert!((force[2] - 40.0).abs() < 1e-12);
    }

    #[test]
    fn probe_comparator_accepts_and_rejects() {
        assert!(assert_probe_close(1.0, 1.0 + 1e-9, 1e-6, 1e-6, "close").is_ok());
        assert!(assert_probe_close(2.0, 1.0, 1e-6, 1e-6, "far").is_err());
        assert!(assert_probe_close(f64::NAN, 1.0, 1e-6, 1e-6, "nan").is_err());
    }

    fn quad_area(nodes: &[[f64; 3]], quad: &[usize; 4]) -> f64 {
        let (a, b, c, d) = (
            nodes[quad[0] - 1],
            nodes[quad[1] - 1],
            nodes[quad[2] - 1],
            nodes[quad[3] - 1],
        );
        let cross = |u: [f64; 3], v: [f64; 3]| {
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        };
        let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
        let norm = |u: [f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        0.5 * (norm(cross(sub(b, a), sub(c, a))) + norm(cross(sub(c, a), sub(d, a))))
    }

    #[test]
    fn scordelis_mesh_has_outward_quads_and_probes() {
        let mesh = scordelis_lo_shell_mesh(8, 8);
        assert_eq!(mesh.nodes.len(), 81);
        assert_eq!(mesh.quads.len(), 64);
        // Free-edge probe at theta = +40 deg, z = 25.
        let probe = mesh.nodes[mesh.probe_node - 1];
        assert!((probe[0] - 25.0 * 40.0_f64.to_radians().sin()).abs() < 1e-9);
        assert!((probe[2] - 25.0).abs() < 1e-12);
        // Every quad has a positive area and an outward normal.
        for quad in &mesh.quads {
            assert!(quad_area(&mesh.nodes, quad) > 1.0);
            let (a, b, d) = (
                mesh.nodes[quad[0] - 1],
                mesh.nodes[quad[1] - 1],
                mesh.nodes[quad[3] - 1],
            );
            let edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let edge2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            let radial = [a[0], a[1], 0.0];
            let dot = normal[0] * radial[0] + normal[1] * radial[1];
            assert!(dot > 0.0, "quad {quad:?} normal points inward");
        }
        // Probe shell straddles the apex midspan.
        assert!((1..=mesh.quads.len()).contains(&mesh.probe_shell));
        let quad = mesh.quads[mesh.probe_shell - 1];
        let cx: f64 = quad.iter().map(|n| mesh.nodes[n - 1][0]).sum::<f64>() / 4.0;
        let cz: f64 = quad.iter().map(|n| mesh.nodes[n - 1][2]).sum::<f64>() / 4.0;
        assert!(cx.abs() < 25.0 * 10.0_f64.to_radians());
        assert!((cz - 25.0).abs() < 50.0 / 8.0);
    }

    #[test]
    fn clamped_plate_mesh_tiles_the_disk() {
        let mesh = clamped_plate_shell_mesh(1.0, 8, 32);
        assert_eq!(mesh.nodes.len(), 1 + 8 * 32);
        assert_eq!(mesh.quads.len(), 7 * 32);
        assert_eq!(mesh.tris.len(), 32);
        let total: f64 = mesh.nodal_area_m2.iter().sum();
        assert!(
            (total - std::f64::consts::PI).abs() < 1e-9,
            "tributary areas sum to {total}"
        );
        for quad in &mesh.quads {
            assert!(quad_area(&mesh.nodes, quad) > 1e-8);
        }
        assert_eq!(mesh.center_node, 1);
        assert!((mesh.probe_radius_m - 0.1875).abs() < 1e-12);
    }

    #[test]
    fn shell_cases_validate_and_match_closed_forms() {
        for case in [shell_scordelis_lo_case(), shell_clamped_plate_case()] {
            case.validate().unwrap();
            assert!(!case.shells.is_empty());
            assert!(case.shell_prop.is_some());
        }
        let plate = shell_clamped_plate_case();
        let center = plate.base.probes[0].expected;
        let rigidity = 200.0e9 * 0.02_f64.powi(3) / (12.0 * (1.0 - 0.3_f64.powi(2)));
        let exact = -1000.0 / (64.0 * rigidity);
        assert!((center - exact).abs() < 1e-15);
        assert!((center + 1.066_406e-4).abs() < 1e-10);
        // Ring CLOADs integrate to the pressure resultant.
        let mut resultant = 0.0;
        for load in &plate.base.cloads {
            let set = &plate.base.node_sets[load.set];
            resultant += load.force_scale_n * set.nodes.len() as f64;
        }
        assert!(
            (resultant + 1000.0 * std::f64::consts::PI).abs() < 1e-6,
            "pressure resultant is {resultant}"
        );
        let roof = shell_scordelis_lo_case();
        // REGRESSION, not the closed form: OpenCourant converges to the
        // flat-facet limit (-0.2574) across QEPH/BATOZ/16-48 mesh/Ismstr/LAW1/LAW2.
        // The -0.3024 closed form stays authoritative for sonium-fem
        // (fem_shell_scordelis_lo golden); this probe only guards engine drift.
        assert!((roof.base.probes[0].expected + 0.2574).abs() < 1e-15);
        assert!((roof.base.probes[0].rel_tol - 0.02).abs() < 1e-12);
        assert!(roof.gravity.is_some());
    }

    #[test]
    fn extended_starter_carries_guide_blocks() {
        let roof = shell_scordelis_lo_case();
        let starter = write_extended_starter(&roof);
        for card in [
            "#RADIOSS STARTER",
            "/NODE",
            "/SHELL/1",
            "/GRNOD/NODE/1/1",
            "/MAT/LAW1/1/1",
            "/PROP/SHELL/1/1",
            "/PART/1",
            "/FUNCT/1",
            "/BCS/1",
            "/GRAV/1/1",
            "/TH/NODE/1",
            "/TH/SHEL/2",
            "/END",
        ] {
            assert!(starter.contains(card), "roof starter lacks {card}");
        }
        assert!(!starter.contains("/BRICK"));
        assert!(!starter.contains("/TH/BRIC"));
        let plate = shell_clamped_plate_case();
        let starter = write_extended_starter(&plate);
        for card in ["/SHELL/1", "/SH3N/1", "/CLOAD/1/1", "/TH/SHEL/2"] {
            assert!(starter.contains(card), "plate starter lacks {card}");
        }
        // TH variable cards carry the requested names.
        assert!(starter.contains("M1"));
        assert!(starter.contains("THIC"));
        assert!(starter.contains("DX"));
    }

    #[test]
    fn plastic_materials_render_law_cards() {
        let base = cantilever_case();
        let mut law2 = RadExtendedCase::wrap_elastic(base.clone());
        law2.material = RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(
            RadJohnsonCook::perfectly_plastic(200.0e9, 0.3, 7800.0, 250.0e6),
        ));
        law2.th_brick = RadThRequest::brick_stress(vec![1]);
        law2.validate().unwrap();
        let starter = write_extended_starter(&law2);
        for card in ["/MAT/LAW2/1/1", "/BRICK/1", "/TH/BRIC/2", "/PART/1"] {
            assert!(starter.contains(card), "LAW2 starter lacks {card}");
        }
        assert!(starter.contains("SX"));
        assert!(!starter.contains("/PROP/SHELL"));

        let mut law36 = RadExtendedCase::wrap_elastic(base);
        law36.material = RadMaterial::Plastic(perfectly_plastic_steel_law36(
            200.0e9, 0.3, 7800.0, 250.0e6,
        ));
        law36.validate().unwrap();
        let starter = write_extended_starter(&law36);
        assert!(starter.contains("/MAT/LAW36/1/1"));
        // The hardening curve follows the user functions.
        assert!(starter.contains("/FUNCT/2"));
        assert!(starter.contains("LAW36 yield stress vs plastic strain"));
    }

    #[test]
    fn extended_case_validation_catches_gaps() {
        let mut case = shell_scordelis_lo_case();
        case.shell_prop = None;
        assert!(case.validate().is_err());
        let mut case = shell_scordelis_lo_case();
        case.th_shell.elems = vec![10_000];
        assert!(case.validate().is_err());
        let mut case = shell_scordelis_lo_case();
        case.gravity.as_mut().unwrap().direction = 'Q';
        assert!(case.validate().is_err());
        let cantilever = RadExtendedCase::wrap_elastic(cantilever_case());
        cantilever.validate().unwrap();
    }

    #[test]
    fn element_channels_are_group_qualified() {
        assert_eq!(shel_channel("M1", 12), "M1/SHEL/12");
        assert_eq!(bric_channel("SX", 4), "SX/BRIC/4");
    }

    #[test]
    fn channel_headers_normalize_to_probe_form() {
        // Observed `th_to_csv` spellings (engine prove-out, Oct 2026).
        assert_eq!(
            normalize_channel("\"time history nodes                       95                                           DX         \""),
            "DX/95"
        );
        assert_eq!(
            normalize_channel("\"time history shells                       1                                           THIC         \""),
            "THIC/SHEL/1"
        );
        assert_eq!(
            normalize_channel("\"time history bricks                       25                                           SX         \""),
            "SX/BRIC/25"
        );
        // Energy channels pass through with quotes stripped.
        assert_eq!(normalize_channel("\"INTERNAL ENERGY\""), "INTERNAL ENERGY");
        assert_eq!(normalize_channel("\"TIME STEP\""), "TIME STEP");
        assert_eq!(
            normalize_channel("\"PERCENTAGE ADDED MASS\""),
            "PERCENTAGE ADDED MASS"
        );
    }

    fn fake_exec_dir(tag: &str, files: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sonium_qa_rad_pair_{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(file), "fake").unwrap();
        }
        dir
    }

    #[test]
    fn engine_pair_prefers_double_precision_smp() {
        let dir = fake_exec_dir(
            "rank",
            &[
                "engine_linux64_gf_ompi_sp",
                "engine_linux64_gf_sp",
                "engine_linux64_gf",
                "starter_linux64_gf_sp",
                "starter_linux64_gf",
            ],
        );
        let (starter, engine) = resolve_engine_pair(&dir).unwrap();
        assert_eq!(
            starter.file_name().unwrap(),
            "starter_linux64_gf"
        );
        assert_eq!(engine.file_name().unwrap(), "engine_linux64_gf");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_pair_matches_file_suffix() {
        let dir = fake_exec_dir("suffix", &["starter_linux64_gf_sp", "engine_linux64_gf_sp"]);
        let (starter, engine) =
            resolve_engine_pair(&dir.join("starter_linux64_gf_sp")).unwrap();
        assert_eq!(
            engine.file_name().unwrap(),
            "engine_linux64_gf_sp"
        );
        assert_eq!(
            starter.file_name().unwrap(),
            "starter_linux64_gf_sp"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_pair_rejects_empty_directory() {
        let dir = fake_exec_dir("empty", &["unrelated.txt"]);
        assert!(resolve_engine_pair(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_flavor_sees_install_path() {
        assert_eq!(
            engine_flavor(Path::new("/opt/OpenCourant/exec/engine_linux64_gf")),
            "OpenCourant"
        );
        assert_eq!(
            engine_flavor(Path::new("/opt/radioss/exec/engine_linux64_gf")),
            "Radioss"
        );
    }
}
