//! Bolted-flange 3D contact assembly.
//!
//! Two tetrahedral blocks (flange + base) clamped by a single central bolt
//! (discrete axial spring with prescribed pretension shortening) across an
//! augmented-Lagrangian contact interface with regularized Coulomb friction.
//! A study-level Newton-Uzawa loop with sparse-direct inner solves drives
//! preload staging followed by external tensile/shear load steps, recording
//! the joint diagram (bolt force vs external load) per step.

use serde::{Deserialize, Serialize};
use sonium_fem::assembly::generic::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, box_mesh_tetrahedra};
use sonium_fem::physics::solid::contact::{ContactConfig, ContactManager};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{
    MatrixClass, Ordering, SparseDirectConfig, SparseDirectMethod, TripletMatrix, factorize_sparse,
};

use super::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};

// Rust guideline compliant 2026-02-21

/// DOFs per node: `[ux, uy, uz]`.
const DOFS_PER_NODE: usize = 3;

/// Initial interface overclosure in m.
///
/// Augmented-Lagrangian pairs activate on `lambda + eps * gap < 0`, so an
/// exactly touching interface (`gap = 0`, `lambda = 0`) starts inactive and
/// leaves the flange rigidly unconstrained (singular first Newton system).
/// A 0.1 nm overclosure pre-activates every interface pair; the resulting
/// preload offset (`k_stack * delta ~` newtons on a 20 kN preload) is far
/// below every validation gate.
const INITIAL_OVERCLOSURE_M: f64 = 1.0e-10;

/// Flange/base block dimensions and mesh density.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoltedFlangeGeometry {
    /// In-plane length in m.
    pub length_m: f64,
    /// In-plane width in m.
    pub width_m: f64,
    /// Flange (upper block) thickness in m.
    pub flange_thickness_m: f64,
    /// Base (lower block) thickness in m.
    pub base_thickness_m: f64,
    /// Elements along x per block.
    pub nx: usize,
    /// Elements along y per block.
    pub ny: usize,
    /// Elements through the flange thickness.
    pub nz_flange: usize,
    /// Elements through the base thickness.
    pub nz_base: usize,
}

/// Isotropic metal shared by flange, base, and bolt.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoltedFlangeMaterial {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
    /// Bolt axial stiffness in N/m (`E_b A_b / L_b`).
    pub bolt_stiffness_n_per_m: f64,
}

impl Default for BoltedFlangeMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            bolt_stiffness_n_per_m: 1.0e8,
        }
    }
}

/// Axial stiffness of a bolt shank (`E A / L`).
pub fn bolt_axial_stiffness_n_per_m(
    youngs_modulus_pa: f64,
    diameter_m: f64,
    grip_length_m: f64,
) -> f64 {
    youngs_modulus_pa * std::f64::consts::PI * diameter_m * diameter_m / 4.0 / grip_length_m
}

/// Clamping stiffness of a single-bolt stack by the Rotscher pressure cone.
///
/// Single cone frustum through the total clamped thickness `thickness_m`
/// with hole diameter `hole_diameter_m` and cone half-angle 30 degrees
/// (`tan = 1/sqrt(3)`, the steel default): `k_c = pi E d_h tan(a) /
/// (2 ln(((l tan(a) + d_w + d_h)(d_w - d_h)) / ((l tan(a) + d_w - d_h)(d_w + d_h))))`
/// with bearing diameter `d_w` from `bearing_diameter_m`.
pub fn rotscher_cone_stiffness_n_per_m(
    youngs_modulus_pa: f64,
    thickness_m: f64,
    hole_diameter_m: f64,
    bearing_diameter_m: f64,
) -> f64 {
    let tan_cone = 1.0 / 3.0f64.sqrt();
    let numerator = (thickness_m * tan_cone + bearing_diameter_m + hole_diameter_m)
        * (bearing_diameter_m - hole_diameter_m);
    let denominator = (thickness_m * tan_cone + bearing_diameter_m - hole_diameter_m)
        * (bearing_diameter_m + hole_diameter_m);
    std::f64::consts::PI * youngs_modulus_pa * hole_diameter_m * tan_cone
        / (2.0 * (numerator / denominator).ln())
}

/// Bolted-flange assembly study: preload staging then tensile/shear loading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoltedFlangeStudy {
    /// Study name for reports.
    pub name: String,
    /// Block dimensions and mesh density.
    pub geometry: BoltedFlangeGeometry,
    /// Isotropic metal plus bolt stiffness.
    pub material: BoltedFlangeMaterial,
    /// Coulomb friction coefficient at the flange interface (0 = frictionless).
    pub friction_coefficient: f64,
    /// Bolt pretension force in N (converted to spring shortening).
    pub preload_force_n: f64,
    /// External tensile (separating) force in N on the flange top.
    pub external_tensile_n: f64,
    /// Transverse shear force in N on the flange top.
    pub transverse_shear_n: f64,
    /// Load steps ramping the pretension shortening.
    pub preload_steps: usize,
    /// Load steps ramping the external tensile/shear loads.
    pub load_steps: usize,
    /// Cap on inner Newton iterations per Uzawa sweep.
    pub max_newton_iterations: usize,
    /// Cap on Uzawa multiplier sweeps per load step.
    pub max_uzawa_iterations: usize,
    /// Newton residual tolerance, relative to the applied force scale.
    pub newton_residual_tolerance: f64,
    /// Uzawa convergence tolerance on the multiplier update norm (Pa).
    pub uzawa_tolerance_pa: f64,
    /// Augmented-Lagrangian penalty parameter (Pa/m).
    pub contact_epsilon: f64,
    /// BVH contact search radius in m.
    pub contact_search_radius_m: f64,
}

impl BoltedFlangeStudy {
    /// Small introductory flange pair (2 x 1 x 1 blocks, single bolt).
    pub fn introductory() -> Self {
        Self {
            name: "introductory-bolted-flange".to_string(),
            geometry: BoltedFlangeGeometry {
                length_m: 0.20,
                width_m: 0.10,
                flange_thickness_m: 0.02,
                base_thickness_m: 0.03,
                nx: 20,
                ny: 10,
                nz_flange: 4,
                nz_base: 6,
            },
            material: BoltedFlangeMaterial::default(),
            friction_coefficient: 0.3,
            preload_force_n: 20.0e3,
            external_tensile_n: 0.0,
            transverse_shear_n: 0.0,
            preload_steps: 4,
            load_steps: 6,
            max_newton_iterations: 30,
            max_uzawa_iterations: 25,
            newton_residual_tolerance: 1.0e-6,
            uzawa_tolerance_pa: 1.0e3,
            contact_epsilon: 1.0e13,
            contact_search_radius_m: 0.01,
        }
    }
}

/// One converged preload/load step of the flange assembly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoltedFlangeStep {
    /// 1-based load step index.
    pub step: usize,
    /// True while ramping pretension, false while ramping external loads.
    pub preload_phase: bool,
    /// Applied load fraction within the current phase.
    pub load_factor: f64,
    /// Bolt spring shortening in m.
    pub shortening_m: f64,
    /// Applied external tensile force in N.
    pub external_tensile_n: f64,
    /// Bolt tension in N.
    pub bolt_force_n: f64,
    /// Interface clamp force in N (z-equilibrium: bolt minus external).
    pub clamp_force_n: f64,
    /// Measured interface clamp force in N: contact z-resultant over the
    /// flange-side nodes, independent of the bolt/tensile bookkeeping.
    /// Must equal `clamp_force_n` when the step is equilibrated.
    pub contact_clamp_n: f64,
    /// Measured interface shear resultants in N (x, y): physical contact
    /// resultants over the flange-side nodes. In stick, x must equal
    /// minus the applied shear (friction opposes the load; weak-spring
    /// slack aside); y must be ~0.
    pub contact_shear_x_n: f64,
    pub contact_shear_y_n: f64,
    /// Flange-top-center transverse drift in m (slip indicator).
    pub flange_ux_m: f64,
    /// Coulomb utilization `Q / (mu N)` from global resultants.
    pub coulomb_utilization: f64,
    /// Active contact pairs at convergence.
    pub active_pairs: usize,
    /// Max penetration at convergence in m.
    pub max_penetration_m: f64,
    /// Inner Newton iterations used.
    pub newton_iterations: usize,
    /// Uzawa sweeps used.
    pub uzawa_iterations: usize,
    /// Max-abs free residual at convergence in N.
    pub residual_n: f64,
}

/// Bolted-flange assembly response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoltedFlangeResult {
    /// Echoed study.
    pub study: BoltedFlangeStudy,
    /// Node count over both blocks.
    pub nodes: usize,
    /// Total displacement DOFs.
    pub dofs: usize,
    /// Interface contact triangles per side.
    pub contact_faces_per_side: usize,
    /// Per-step converged states (preload phase first).
    pub steps: Vec<BoltedFlangeStep>,
    /// Whether every load step converged.
    pub converged: bool,
    /// Model assumption tag.
    pub model_assumption: StructuralModelAssumption,
    /// Execution provenance.
    pub execution: StructuralExecutionProvenance,
}

fn validate_flange_inputs(study: &BoltedFlangeStudy) -> Result<(), StructuralMechanicsError> {
    let geometry = &study.geometry;
    for (field, value) in [
        ("bolted_flange.length_m", geometry.length_m),
        ("bolted_flange.width_m", geometry.width_m),
        (
            "bolted_flange.flange_thickness_m",
            geometry.flange_thickness_m,
        ),
        ("bolted_flange.base_thickness_m", geometry.base_thickness_m),
        (
            "bolted_flange.youngs_modulus_pa",
            study.material.youngs_modulus_pa,
        ),
        (
            "bolted_flange.bolt_stiffness_n_per_m",
            study.material.bolt_stiffness_n_per_m,
        ),
        ("bolted_flange.preload_force_n", study.preload_force_n),
        (
            "bolted_flange.newton_residual_tolerance",
            study.newton_residual_tolerance,
        ),
        ("bolted_flange.uzawa_tolerance_pa", study.uzawa_tolerance_pa),
        ("bolted_flange.contact_epsilon", study.contact_epsilon),
        (
            "bolted_flange.contact_search_radius_m",
            study.contact_search_radius_m,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if !study.material.poisson_ratio.is_finite()
        || !(0.0..0.5).contains(&study.material.poisson_ratio)
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bolted_flange.poisson_ratio",
            value: study.material.poisson_ratio.to_string(),
        });
    }
    for (field, value) in [
        (
            "bolted_flange.friction_coefficient",
            study.friction_coefficient,
        ),
        ("bolted_flange.external_tensile_n", study.external_tensile_n),
        ("bolted_flange.transverse_shear_n", study.transverse_shear_n),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if geometry.nx == 0 || geometry.ny == 0 || geometry.nz_flange == 0 || geometry.nz_base == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bolted_flange.mesh_resolution",
            value: format!(
                "({}, {}, {}, {})",
                geometry.nx, geometry.ny, geometry.nz_flange, geometry.nz_base
            ),
        });
    }
    if study.preload_steps == 0
        || study.load_steps == 0
        || study.max_newton_iterations == 0
        || study.max_uzawa_iterations == 0
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "bolted_flange.iterations",
            value: format!(
                "preload_steps={}, load_steps={}, newton={}, uzawa={}",
                study.preload_steps,
                study.load_steps,
                study.max_newton_iterations,
                study.max_uzawa_iterations
            ),
        });
    }
    Ok(())
}

/// Stack the flange block on the base block (touching at the interface).
///
/// Returns the combined mesh plus the base-block node count (node ownership
/// split for interface classification).
fn flange_assembly_mesh(geometry: &BoltedFlangeGeometry) -> (Mesh, usize) {
    let base_top = geometry.base_thickness_m;
    let flange_top = base_top + geometry.flange_thickness_m;
    let mut mesh = Mesh::new(3);
    let blocks = [
        (0.0, base_top, geometry.nx, geometry.ny, geometry.nz_base),
        (
            base_top - INITIAL_OVERCLOSURE_M,
            flange_top,
            geometry.nx,
            geometry.ny,
            geometry.nz_flange,
        ),
    ];
    let mut base_nodes = 0usize;
    for (block, (z_min, z_max, nx, ny, nz)) in blocks.into_iter().enumerate() {
        let block_mesh = box_mesh_tetrahedra(
            0.0,
            geometry.length_m,
            0.0,
            geometry.width_m,
            z_min,
            z_max,
            nx,
            ny,
            nz,
        );
        let node_offset = mesh.nodes.len();
        for point in block_mesh.nodes {
            mesh.add_node(point);
        }
        for element in block_mesh.elements {
            let nodes = element
                .nodes
                .into_iter()
                .map(|node| node + node_offset)
                .collect();
            mesh.add_element(element.element_type, nodes);
        }
        if block == 0 {
            base_nodes = mesh.nodes.len();
        }
    }
    mesh.detect_boundaries();
    (mesh, base_nodes)
}

/// Split the interface triangles by block ownership (base top vs flange bottom).
///
/// Ownership (not normal sign) classifies the faces: tet-boundary winding is
/// not consistently outward, so the normal test mislabels sides.
fn interface_contact_faces(
    mesh: &Mesh,
    base_nodes: usize,
    interface_z: f64,
    tolerance: f64,
) -> Result<(Vec<[usize; 3]>, Vec<[usize; 3]>), StructuralMechanicsError> {
    let mut base_top = Vec::new();
    let mut flange_bottom = Vec::new();
    for boundary in &mesh.boundaries {
        if boundary.nodes.len() != 3 {
            continue;
        }
        let nodes = [boundary.nodes[0], boundary.nodes[1], boundary.nodes[2]];
        if nodes.iter().any(|node| *node >= mesh.nodes.len()) {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "flange contact boundary references an unknown node".to_string(),
            });
        }
        let on_interface = nodes
            .iter()
            .all(|node| (mesh.nodes[*node].z - interface_z).abs() <= tolerance);
        if !on_interface {
            continue;
        }
        if nodes.iter().all(|node| *node < base_nodes) {
            base_top.push(orient_up(mesh, nodes));
        } else if nodes.iter().all(|node| *node >= base_nodes) {
            flange_bottom.push(orient_down(mesh, nodes));
        } else {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: "flange interface face straddles both blocks".to_string(),
            });
        }
    }
    if base_top.is_empty() || flange_bottom.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!(
                "flange interface selected no opposing faces (base={}, flange={})",
                base_top.len(),
                flange_bottom.len()
            ),
        });
    }
    Ok((base_top, flange_bottom))
}

/// Orient a triangle so its z-normal is positive.
fn orient_up(mesh: &Mesh, nodes: [usize; 3]) -> [usize; 3] {
    if face_normal_z(mesh, nodes) < 0.0 {
        [nodes[0], nodes[2], nodes[1]]
    } else {
        nodes
    }
}

/// Orient a triangle so its z-normal is negative.
fn orient_down(mesh: &Mesh, nodes: [usize; 3]) -> [usize; 3] {
    if face_normal_z(mesh, nodes) > 0.0 {
        [nodes[0], nodes[2], nodes[1]]
    } else {
        nodes
    }
}

fn face_normal_z(mesh: &Mesh, nodes: [usize; 3]) -> f64 {
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
}

fn mesh_coordinate_tolerance(mesh: &Mesh) -> f64 {
    let mut extent = 1.0f64;
    for point in &mesh.nodes {
        extent = extent
            .max(point.x.abs())
            .max(point.y.abs())
            .max(point.z.abs());
    }
    1.0e-9 * extent.max(1.0)
}

fn nearest_node(mesh: &Mesh, target: [f64; 3]) -> usize {
    let mut best = 0usize;
    let mut best_distance = f64::INFINITY;
    for (node, point) in mesh.nodes.iter().enumerate() {
        let distance = (point.x - target[0])
            .hypot(point.y - target[1])
            .hypot(point.z - target[2]);
        if distance < best_distance {
            best_distance = distance;
            best = node;
        }
    }
    best
}

fn triplet_matvec(matrix: &TripletMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.num_rows];
    for index in 0..matrix.nnz() {
        result[matrix.rows[index]] += matrix.values[index] * vector[matrix.cols[index]];
    }
    result
}

/// Solve the bolted-flange preload + external-load assembly.
pub fn run_bolted_flange(
    study: &BoltedFlangeStudy,
) -> Result<BoltedFlangeResult, StructuralMechanicsError> {
    validate_flange_inputs(study)?;
    let geometry = &study.geometry;
    let (mesh, base_nodes) = flange_assembly_mesh(geometry);
    mesh.validate_connectivity()
        .map_err(|errors| StructuralMechanicsError::InvalidMesh {
            reason: format!("flange mesh connectivity: {errors:?}"),
        })?;
    let tolerance = mesh_coordinate_tolerance(&mesh);
    let interface_z = geometry.base_thickness_m;
    let flange_top_z = interface_z + geometry.flange_thickness_m;
    let (base_faces, flange_faces) =
        interface_contact_faces(&mesh, base_nodes, interface_z, tolerance)?;
    let reference_coords = mesh
        .nodes
        .iter()
        .flat_map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    // One-sided pairing (base = master, flange = slave): each interface
    // point pairs once, so multipliers read as true pressure and the
    // Coulomb cap is exactly mu*N (two-sided pairing double-counts).
    let mut contact = ContactManager::new_two_body(
        &base_faces,
        &flange_faces,
        reference_coords,
        ContactConfig {
            epsilon: study.contact_epsilon,
            uzawa_tolerance: study.uzawa_tolerance_pa,
            max_uzawa_iterations: study.max_uzawa_iterations,
            search_radius: study.contact_search_radius_m,
            bvh_margin: study.contact_search_radius_m.max(1.0e-9),
            friction_coefficient: study.friction_coefficient,
            two_body: true,
        },
    );

    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    let material = IsotropicMaterial::new(
        study.material.youngs_modulus_pa,
        study.material.poisson_ratio,
    );
    let form = ElasticityForm::new_3d(material);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2);

    // Base bottom fully fixed; bolt anchor is the fixed base-bottom center.
    // Bolt spring: flange-top center to the fixed base-bottom center.
    let center = [geometry.length_m / 2.0, geometry.width_m / 2.0];
    let bolt_top = nearest_node(&mesh, [center[0], center[1], flange_top_z]);
    let bolt_anchor = nearest_node(&mesh, [center[0], center[1], 0.0]);
    let off_center = nearest_node(
        &mesh,
        [
            geometry.length_m / 2.0,
            geometry.width_m / 4.0,
            flange_top_z,
        ],
    );
    let tilt_node = nearest_node(
        &mesh,
        [
            geometry.length_m / 4.0,
            geometry.width_m / 2.0,
            flange_top_z,
        ],
    );
    let base_fixed: Vec<bool> = (0..dofs)
        .map(|dof| mesh.nodes[dof / DOFS_PER_NODE].z.abs() <= tolerance)
        .collect();

    let bolt_dof_top = bolt_top * DOFS_PER_NODE + 2;
    let bolt_dof_anchor = bolt_anchor * DOFS_PER_NODE + 2;
    let bolt_stiffness = study.material.bolt_stiffness_n_per_m;
    let full_shortening_m = study.preload_force_n / bolt_stiffness;

    // Flange-top nodes carry the distributed external loads.
    let top_nodes: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| (point.z - flange_top_z).abs() <= tolerance)
        .map(|(node, _)| node)
        .collect();
    if top_nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "flange top selected no loaded nodes".to_string(),
        });
    }

    let direct_config = SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SupernodalLu,
        ordering: Ordering::Auto,
        equilibrate: false,
        ..SparseDirectConfig::default()
    };

    let total_steps = study.preload_steps + study.load_steps;
    let mut solution = vec![0.0; dofs];
    let mut steps = Vec::with_capacity(total_steps);
    for step in 1..=total_steps {
        let preload_phase = step <= study.preload_steps;
        let (load_factor, shortening_m, tensile_n, shear_n) = if preload_phase {
            let factor = step as f64 / study.preload_steps as f64;
            (factor, full_shortening_m * factor, 0.0, 0.0)
        } else {
            let factor = (step - study.preload_steps) as f64 / study.load_steps as f64;
            (
                factor,
                full_shortening_m,
                study.external_tensile_n * factor,
                study.transverse_shear_n * factor,
            )
        };
        let mut external = vec![0.0; dofs];
        for &node in &top_nodes {
            external[node * DOFS_PER_NODE] += shear_n / top_nodes.len() as f64;
            external[node * DOFS_PER_NODE + 2] += tensile_n / top_nodes.len() as f64;
        }
        let force_scale = external
            .iter()
            .map(|value| value.abs())
            .fold(
                bolt_stiffness * shortening_m / top_nodes.len() as f64,
                f64::max,
            )
            .max(1.0e-300);
        let residual_gate = study.newton_residual_tolerance * force_scale;
        // Accepted chatter-plateau bar: active-set/friction-boundary
        // flip-flop stagnates Newton at ~3e-3 of the nodal force scale
        // (measured floors ~1.3 N on a ~440 N scale); the bar sits ~2.4x
        // above. Acceptance still requires a genuine plateau (5 flat
        // iterations), so floors cannot chase the bar upward.
        let plateau_bar = 7.0e-3 * force_scale;

        let mut newton_iterations = 0usize;
        let mut uzawa_iterations = 0usize;
        let mut multiplier_update = f64::INFINITY;
        let mut residual = f64::INFINITY;
        let mut converged = false;
        for uzawa in 1..=study.max_uzawa_iterations {
            uzawa_iterations = uzawa;
            // Friction homotopy: only the very first sweep of the study
            // runs frictionless, so the normal multipliers establish a
            // sane Coulomb cap before friction engages (starting friction
            // at lambda = 0 caps tractions at ~0 with arbitrary slip
            // directions, which Newton cannot resolve). Later steps warm-
            // start from established multipliers, so they keep friction
            // from sweep 1: a frictionless sweep under shear load would
            // drift to the mm-scale frictionless equilibrium (weak-spring
            // balanced) and hand the frictional sweeps a state with 1e6x
            // the stick slip to undo.
            let effective_friction = if uzawa == 1 && step == 1 {
                0.0
            } else {
                study.friction_coefficient
            };
            contact.set_friction_coefficient(effective_friction);
            // Every sweep shares one stabilization: weak ground springs on
            // five flange-top DOFs (Tx, Ty via bolt_top x/y; Rz via
            // off_center x; Rx via off_center z; Ry via tilt_node z; Tz
            // via the bolt). Hard pins in frictionless sweeps were tried
            // and removed: releasing them when friction engages injects a
            // pin-reaction transient that chatters grazing contact at
            // liftoff. The springs stay on in every sweep, so no sweep
            // boundary carries a restraint discontinuity.
            // Lateral springs run at 1e-2 of the bolt stiffness: sub-N at
            // stick-regime slip, and they contain gross-slip drift to mm
            // scale so post-onset steps still converge. The z springs stay
            // at 1e-6 so liftoff steps keep exact bolt/tensile equilibrium.
            // Slip onset is read from the drift knee / Coulomb utilization,
            // not from regularization-limited drift magnitudes.
            let is_fixed = base_fixed.clone();
            let weak_springs: Vec<(usize, f64)> = vec![
                (bolt_top * DOFS_PER_NODE, bolt_stiffness * 1.0e-2),
                (bolt_top * DOFS_PER_NODE + 1, bolt_stiffness * 1.0e-2),
                (off_center * DOFS_PER_NODE, bolt_stiffness * 1.0e-2),
                (off_center * DOFS_PER_NODE + 2, bolt_stiffness * 1.0e-6),
                (tilt_node * DOFS_PER_NODE + 2, bolt_stiffness * 1.0e-6),
            ];
            let free: Vec<usize> = (0..dofs).filter(|dof| !is_fixed[*dof]).collect();
            let mut global_to_free = vec![usize::MAX; dofs];
            for (position, &dof) in free.iter().enumerate() {
                global_to_free[dof] = position;
            }
            contact.detect_contact(&solution);
            // Newton stall acceptance: active-set chatter plateaus the
            // residual above the tight gate; accept the plateau once it
            // stops improving and let Uzawa advance the multipliers.
            let mut best_residual = f64::INFINITY;
            let mut since_improvement = 0usize;
            for iteration in 1..=study.max_newton_iterations {
                newton_iterations = iteration;
                let mut internal = triplet_matvec(&stiffness, &solution);
                // Bolt spring with prescribed shortening (tension positive).
                // Virtual-work sign convention: the internal force is
                // +k*e on the top node (physical pull is downward), so the
                // tangent stays the positive [[k,-k],[-k,k]] block.
                let extension = shortening_m + solution[bolt_dof_top] - solution[bolt_dof_anchor];
                let bolt_force = bolt_stiffness * extension;
                internal[bolt_dof_top] += bolt_force;
                internal[bolt_dof_anchor] -= bolt_force;
                contact.contact_residual(&solution, &mut internal);
                for &(dof, stiffness) in &weak_springs {
                    internal[dof] += stiffness * solution[dof];
                }
                residual = free
                    .iter()
                    .map(|&dof| (internal[dof] - external[dof]).abs())
                    .fold(0.0, f64::max);
                // Smooth 2-norm merit for the line search: the max-norm
                // gate is nonsmooth, and Newton steps that reduce the
                // global error can raise a single concentrated DOF,
                // which a max-norm search then rejects forever (spurious
                // stall with a static worst DOF). Gates stay max-norm
                // (no DOF left behind); only the search uses the 2-norm.
                let merit: f64 = free
                    .iter()
                    .map(|&dof| {
                        let value = internal[dof] - external[dof];
                        value * value
                    })
                    .sum::<f64>()
                    .sqrt();
                if residual <= residual_gate {
                    break;
                }
                if residual < 0.5 * best_residual {
                    best_residual = residual;
                    since_improvement = 0;
                } else {
                    since_improvement += 1;
                }
                if since_improvement >= 5 && residual <= plateau_bar {
                    break;
                }
                // Inexact-Uzawa inner cap: with lambda far from converged
                // the nm-scale active-set flip threshold makes full Newton
                // convergence impossible; hand the plateau back to Uzawa
                // rather than churning (or drifting) in place.
                if since_improvement >= 8 {
                    break;
                }
                let mut tangent = TripletMatrix::<f64>::new(dofs, dofs);
                for index in 0..stiffness.nnz() {
                    tangent.push(
                        stiffness.rows[index],
                        stiffness.cols[index],
                        stiffness.values[index],
                    );
                }
                tangent.push(bolt_dof_top, bolt_dof_top, bolt_stiffness);
                tangent.push(bolt_dof_top, bolt_dof_anchor, -bolt_stiffness);
                tangent.push(bolt_dof_anchor, bolt_dof_top, -bolt_stiffness);
                tangent.push(bolt_dof_anchor, bolt_dof_anchor, bolt_stiffness);
                for (row, column, value) in contact.contact_tangent(&solution) {
                    tangent.push(row, column, value);
                }
                for &(dof, stiffness) in &weak_springs {
                    tangent.push(dof, dof, stiffness);
                }
                let mut reduced = TripletMatrix::<f64>::new(free.len(), free.len());
                for ((&row, &column), &value) in
                    tangent.rows.iter().zip(&tangent.cols).zip(&tangent.values)
                {
                    let free_row = global_to_free[row];
                    let free_column = global_to_free[column];
                    if free_row != usize::MAX && free_column != usize::MAX {
                        reduced.push(free_row, free_column, value);
                    }
                }
                let mut rhs = Vec::with_capacity(free.len());
                for &dof in &free {
                    rhs.push(-(internal[dof] - external[dof]));
                }
                let reduced_csr = reduced.to_csr();
                let mut correction = factorize_sparse(&reduced_csr, &direct_config, None)
                    .and_then(|factor| factor.solve(&rhs))
                    .map_err(|_error| StructuralMechanicsError::NonlinearNonConvergence {
                        model: StructuralModelAssumption::BoltedFlangeContact,
                        load_step: step,
                        iterations: iteration,
                        residual,
                    })?;
                // Trust region: a churned active set makes the tangent
                // intermittently garbage; cap the correction max-norm at
                // 0.1 mm (above single-step elastic motion, far below the
                // 25 mm element size) so one bad tangent cannot catapult
                // the state out of the contact search radius. Inactive at
                // convergence, where corrections vanish.
                let trust_radius_m = 1.0e-4;
                let correction_max = correction
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f64::max);
                if correction_max > trust_radius_m {
                    let scale = trust_radius_m / correction_max;
                    for value in &mut correction {
                        *value *= scale;
                    }
                }
                // Backtracking line search on the smooth 2-norm merit.
                // Full Newton steps overshoot across the friction blend
                // band (and active-set boundaries), where the tangent goes
                // stale; halving until the merit decreases restores
                // convergence. Trials only re-assemble the residual (no LU).
                let mut step_length = 1.0f64;
                let mut trial = solution.clone();
                let mut best_trial = solution.clone();
                let mut best_trial_merit = f64::INFINITY;
                for _ in 0..10 {
                    for (position, &dof) in free.iter().enumerate() {
                        trial[dof] = solution[dof] + step_length * correction[position];
                    }
                    let mut trial_internal = triplet_matvec(&stiffness, &trial);
                    let trial_extension =
                        shortening_m + trial[bolt_dof_top] - trial[bolt_dof_anchor];
                    let trial_bolt = bolt_stiffness * trial_extension;
                    trial_internal[bolt_dof_top] += trial_bolt;
                    trial_internal[bolt_dof_anchor] -= trial_bolt;
                    contact.contact_residual(&trial, &mut trial_internal);
                    for &(dof, stiffness) in &weak_springs {
                        trial_internal[dof] += stiffness * trial[dof];
                    }
                    let trial_merit: f64 = free
                        .iter()
                        .map(|&dof| {
                            let value = trial_internal[dof] - external[dof];
                            value * value
                        })
                        .sum::<f64>()
                        .sqrt();
                    if trial_merit.is_finite() && trial_merit < best_trial_merit {
                        best_trial_merit = trial_merit;
                        best_trial.clone_from(&trial);
                    }
                    if trial_merit < merit {
                        break;
                    }
                    step_length *= 0.5;
                }
                if !best_trial_merit.is_finite() {
                    return Err(StructuralMechanicsError::NonlinearNonConvergence {
                        model: StructuralModelAssumption::BoltedFlangeContact,
                        load_step: step,
                        iterations: iteration,
                        residual,
                    });
                }
                solution = best_trial;
                if solution.iter().any(|value| !value.is_finite()) {
                    return Err(StructuralMechanicsError::ValidationFailure {
                        check: "bolted_flange.newton_state_finite",
                        value: f64::NAN,
                    });
                }
            }
            multiplier_update = contact.update_multipliers(&solution);
            // A step converges only with BOTH the multipliers settled and
            // Newton at least at its accepted plateau: the multiplier gate
            // alone cannot see lateral disequilibrium (friction chatter
            // leaves gaps biased while lambda looks settled).
            if multiplier_update <= study.uzawa_tolerance_pa && residual <= plateau_bar {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::BoltedFlangeContact,
                load_step: step,
                iterations: newton_iterations.max(uzawa_iterations),
                residual: residual.max(multiplier_update),
            });
        }
        let extension = shortening_m + solution[bolt_dof_top] - solution[bolt_dof_anchor];
        let bolt_force = bolt_stiffness * extension;
        let clamp_force = bolt_force - tensile_n;
        let utilization = if study.friction_coefficient > 0.0 && clamp_force > 0.0 {
            shear_n / (study.friction_coefficient * clamp_force)
        } else {
            0.0
        };
        // Independent clamp measurement: contact z-resultant over the
        // flange-side nodes. `contact_residual` uses the internal sign
        // convention (matching the bolt internal force), so the physical
        // upward clamp on the flange is the negated sum.
        let mut contact_forces = vec![0.0; dofs];
        contact.contact_residual(&solution, &mut contact_forces);
        let mut contact_clamp = 0.0f64;
        let mut contact_shear_x = 0.0f64;
        let mut contact_shear_y = 0.0f64;
        for node in base_nodes..mesh.nodes.len() {
            contact_clamp -= contact_forces[node * DOFS_PER_NODE + 2];
            contact_shear_x -= contact_forces[node * DOFS_PER_NODE];
            contact_shear_y -= contact_forces[node * DOFS_PER_NODE + 1];
        }
        steps.push(BoltedFlangeStep {
            step,
            preload_phase,
            load_factor,
            shortening_m,
            external_tensile_n: tensile_n,
            bolt_force_n: bolt_force,
            clamp_force_n: clamp_force,
            contact_clamp_n: contact_clamp,
            contact_shear_x_n: contact_shear_x,
            contact_shear_y_n: contact_shear_y,
            flange_ux_m: solution[bolt_top * DOFS_PER_NODE],
            coulomb_utilization: utilization,
            active_pairs: contact.count_active_pairs(&solution),
            max_penetration_m: contact.max_penetration(&solution),
            newton_iterations,
            uzawa_iterations,
            residual_n: residual,
        });
    }

    Ok(BoltedFlangeResult {
        study: study.clone(),
        nodes: mesh.nodes.len(),
        dofs,
        contact_faces_per_side: base_faces.len(),
        steps,
        converged: true,
        model_assumption: StructuralModelAssumption::BoltedFlangeContact,
        execution: StructuralExecutionProvenance::cpu_bolted_flange_reference(),
    })
}
