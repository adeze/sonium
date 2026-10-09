//! D5 (review `acoustics-20250730.md` §3.5 promotion step 7): body-fitted,
//! mesh-converged reanalysis of projected motor topology designs.
//!
//! After density-based topology optimization with Heaviside projection, the
//! projected steel layout is a BINARY element set on the structured `(r, z)`
//! topology mesh. Trusting its BL requires removing the two approximations the
//! optimizer lives with: the density-interpolated permeability (clamped gray
//! cells) and the coarse-mesh field resolution. This module:
//!
//! 1. extracts the binary steel set from the (filtered, projected) physical
//!    densities ([`extract_binary_design`]) and the steel boundary as closed
//!    polygon loops ([`extract_steel_boundary`]),
//! 2. rebuilds a body-fitted mesh of the same motor domain
//!    ([`build_body_fitted_mesh`]), and
//! 3. re-solves the axisymmetric magnetostatic bias with the TRUE steel model
//!    (constant `μ_r = 1000`, or the D3 differential permeability re-extracted
//!    on the new mesh) over a refinement ladder, reporting BL per level and a
//!    mesh-convergence flag ([`reanalyze_projected_design`]).
//!
//! ## Remesh choice: integer-factor structured refinement
//!
//! The projected design IS the staircase union of coarse steel elements —
//! there is no underlying smooth curve to recover, and any smoothing (node
//! snapping, path (a); per-region constrained triangulation, path (b)) would
//! evaluate a DIFFERENT geometry than the one the optimizer produced, with an
//! arbitrary area/shape drift. Because the topology mesh is a uniform
//! rectilinear grid split along one consistent diagonal, every steel/air
//! interface edge of the projected design is a coarse mesh edge. Refining the
//! same generator by an integer factor `K` (same diagonal split direction)
//! embeds every coarse edge — including the diagonals, which are unions of `K`
//! refined diagonal edges — in the refined mesh. Region assignment by
//! containment of refined centroids in coarse elements therefore reproduces
//! the projected steel region EXACTLY (steel area preserved to roundoff, every
//! interface segment present as a mesh edge: the body-fitted property), while
//! the integer-`K` ladder gives a clean mesh-convergence study. This is the
//! fallback path (c) of the D5 plan, but on this structured mesh it delivers
//! the path (a)/(b) body-fitted guarantee; a general 2D constrained Delaunay
//! triangulation is deliberately not introduced.
//!
//! Non-design regions (magnet, voice coil, air gap, magnetic air) inherit the
//! coarse element's marker verbatim, so their geometry is kept EXACTLY as in
//! the topology/golden mesh — only the steel/air classification of the design
//! cells changes. Note the motor layout has no FIXED steel regions: both
//! `STEEL_POLE` and `STEEL_YOKE` are design regions, so the steel boundary
//! consists of the movable steel/air-design interface plus fixed edges against
//! the non-design regions and the outer domain boundary.

use crate::axisymmetric_linear::{
    AIR_GAP, AxisymmetricLinearConfig, MAGNETIC_AIR, MAGNETIC_AXIS, MAGNETIC_LOWER_END,
    MAGNETIC_OUTER_RADIUS, MAGNETIC_UPPER_END, PERMANENT_MAGNET, STEEL_POLE, STEEL_YOKE,
    VOICE_COIL, element_centroid, magnetic_bias_from_potential, register_boundary_groups,
    register_groups, remap_boundary_markers,
};
use crate::motor_topology::{
    LINEAR_STEEL_MU_R, MotorFactorization, MotorTopologyConfig, MotorTopologyError,
    MotorTopologyResult, MotorTopologyWorkspace, SteelPermeability, final_continuation_beta,
    optimize_motor_topology, physical_densities,
};
use crate::nonlinear_bias::{NonlinearSpeakerBiasError, steel_permeability_map_from_bias};
use sonium_fem::assembly::{
    RegionMap, assemble_bilinear_by_region, assemble_bilinear_where, assemble_linear_where,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::{ElementType, Mesh, rectangular_mesh_triangles};
use sonium_fem::optimization::{DensityFilter, HeavisideProjection, InterpolationScheme};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricAThetaForm, AzimuthalCurrentDensityForm, MU_0,
};
use sonium_fem::spaces::H1Space;
use std::collections::HashMap;

/// Coordinate quantization step (m) used when matching nodes across the
/// coarse and refined structured meshes. Refined grid coordinates are exact
/// dyadic refinements of the coarse ones; 1 pm is far below any mesh length
/// and far above roundoff.
const COORDINATE_QUANTUM_M: f64 = 1.0e-12;

/// Reanalysis settings for a projected motor topology design.
#[derive(Debug, Clone)]
pub struct ProjectedReanalysisConfig {
    /// Steel threshold over the physical densities. `None` uses the topology
    /// projection threshold `eta` when the run projected, else 0.5.
    pub threshold: Option<f64>,
    /// Integer refinement factors of the mesh-convergence ladder, applied to
    /// the structured topology grid (1 = topology resolution with the true
    /// binary material). Sorted ascending; duplicates removed. The default
    /// `[1, 2, 4, 8, 16]` reaches the 1 % convergence target on the
    /// pole-tip corner singularity; callers can use a shorter survey ladder
    /// when they only need a coarse discrepancy estimate.
    pub refinement_factors: Vec<usize>,
    /// Relative BL change between the two finest levels below which the
    /// body-fitted BL is declared mesh-converged.
    pub convergence_tolerance: f64,
    /// Steel permeability model of the reanalysis. With
    /// [`SteelPermeability::DifferentialFromBias`] the map is re-extracted
    /// from a nonlinear bias solve of the BINARY design on each refined mesh
    /// (steel cells carry weight 1); its `refresh_interval` field is
    /// meaningless here and ignored.
    pub steel_permeability: SteelPermeability,
}

impl Default for ProjectedReanalysisConfig {
    fn default() -> Self {
        Self {
            threshold: None,
            refinement_factors: vec![1, 2, 4, 8, 16],
            convergence_tolerance: 1.0e-2,
            steel_permeability: SteelPermeability::Constant,
        }
    }
}

/// One closed polygon loop of the steel boundary, traced with steel on the
/// left (outer loops wind counterclockwise, holes clockwise).
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// Ordered `(r, z)` vertices; the loop closes back to the first vertex.
    pub vertices_rz_m: Vec<[f64; 2]>,
    /// Shoelace signed area (m^2): positive for outer loops, negative for
    /// holes.
    pub signed_area_m2: f64,
    /// Whether this loop encloses an air pocket inside steel (negative signed
    /// area).
    pub is_hole: bool,
}

/// BL of one mesh-convergence level.
#[derive(Debug, Clone)]
pub struct ReanalysisLevel {
    /// Integer refinement factor applied to the topology grid.
    pub refinement_factor: usize,
    /// Refined mesh size.
    pub num_elements: usize,
    /// Refined node count.
    pub num_nodes: usize,
    /// Body-fitted true-material BL at this level (T·m).
    pub bl_tm: f64,
    /// `|BL_k - BL_{k-1}| / |BL_{k-1}|` against the previous level (`None` for
    /// the coarsest level).
    pub relative_change_from_previous: Option<f64>,
}

/// Result of a body-fitted reanalysis of a projected design.
#[derive(Debug, Clone)]
pub struct ProjectedReanalysisReport {
    /// Steel threshold applied to the physical densities.
    pub threshold: f64,
    /// Number of steel design elements in the binary design (topology mesh).
    pub steel_element_count: usize,
    /// Closed polygon loops of the steel boundary on the topology mesh
    /// (outer loops and holes; empty for an all-air design).
    pub boundary_loops: Vec<BoundaryLoop>,
    /// BL of the topology forward model at the physical (filtered, projected)
    /// densities with its density-interpolated permeability — the number the
    /// optimizer itself would report for the projected design.
    pub bl_projected_interpolated_tm: f64,
    /// Mesh-convergence ladder of body-fitted true-material solves.
    pub levels: Vec<ReanalysisLevel>,
    /// BL at the finest level (T·m): the trustworthy BL of the projected
    /// design.
    pub bl_body_fitted_tm: f64,
    /// `|bl_body_fitted - bl_projected_interpolated| / |bl_projected_interpolated|`.
    pub relative_discrepancy: f64,
    /// True when the ladder has at least two levels and the relative change
    /// between the two finest is at most the configured tolerance.
    pub mesh_converged: bool,
}

/// Recoverable projected-reanalysis error.
#[derive(Debug, thiserror::Error)]
pub enum ProjectedReanalysisError {
    #[error(transparent)]
    Topology(#[from] MotorTopologyError),
    #[error(transparent)]
    NonlinearBias(#[from] NonlinearSpeakerBiasError),
    #[error("projected reanalysis FEM failure: {0}")]
    Fem(String),
    #[error("invalid projected reanalysis configuration: {0}")]
    InvalidConfiguration(&'static str),
}

// ---------------------------------------------------------------------------
// CP1: binary design extraction and steel boundary polygons
// ---------------------------------------------------------------------------

/// Threshold a density vector over the design elements into the binary steel
/// element set (mesh element indices, in design order).
pub fn extract_binary_design(
    densities: &[f64],
    design_elements: &[usize],
    threshold: f64,
) -> Result<Vec<usize>, ProjectedReanalysisError> {
    if densities.len() != design_elements.len() {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "density vector must cover every design element",
        ));
    }
    if !threshold.is_finite() || !(0.0..1.0).contains(&threshold) {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "steel threshold must lie in (0, 1)",
        ));
    }
    Ok(design_elements
        .iter()
        .zip(densities)
        .filter_map(|(&element, &density)| (density >= threshold).then_some(element))
        .collect())
}

/// Extract the boundary of the union of `steel_elements` as closed polygon
/// loops on a triangle mesh.
///
/// Every steel element edge whose across-edge neighbor is absent (domain
/// boundary) or non-steel contributes one directed half-edge with steel on the
/// left. Half-edges chain into loops by the sharpest-right-turn rule at each
/// vertex, which resolves point-contact (checkerboard) vertices into tight
/// per-component loops. Outer loops wind counterclockwise (positive shoelace
/// area), holes clockwise.
pub fn extract_steel_boundary(mesh: &Mesh, steel_elements: &[usize]) -> Vec<BoundaryLoop> {
    let mut steel = vec![false; mesh.elements.len()];
    for &element in steel_elements {
        steel[element] = true;
    }

    // Undirected edge -> owning elements.
    let mut edge_owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != ElementType::Triangle {
            continue;
        }
        let v = element.vertices();
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            edge_owners
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(index);
        }
    }

    // Directed boundary half-edges of the steel set, steel on the left.
    let mut half_edges: Vec<(usize, usize)> = Vec::new();
    for (index, element) in mesh.elements.iter().enumerate() {
        if !steel[index] || element.element_type != ElementType::Triangle {
            continue;
        }
        let v = element.vertices();
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let across_non_steel = edge_owners[&(a.min(b), a.max(b))]
                .iter()
                .all(|&other| other == index || !steel[other]);
            if across_non_steel {
                half_edges.push((a, b));
            }
        }
    }

    let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
    for (edge, &(from, _)) in half_edges.iter().enumerate() {
        outgoing.entry(from).or_default().push(edge);
    }
    let point = |node: usize| [mesh.nodes[node].x, mesh.nodes[node].y];

    // Pairing: at the end vertex of each half-edge, continue with the outgoing
    // half-edge making the sharpest right turn (smallest counterclockwise turn
    // angle). On a valid partition this is a fixed-point-free permutation, so
    // loops are its cycles.
    let mut next = vec![usize::MAX; half_edges.len()];
    for (edge, &(from, to)) in half_edges.iter().enumerate() {
        let direction_in = {
            let (a, b) = (point(from), point(to));
            [b[0] - a[0], b[1] - a[1]]
        };
        let mut best: Option<(f64, usize)> = None;
        for &candidate in &outgoing[&to] {
            if candidate == edge {
                continue;
            }
            let (_, candidate_to) = half_edges[candidate];
            let p = point(candidate_to);
            let q = point(to);
            let direction_out = [p[0] - q[0], p[1] - q[1]];
            let cross = direction_in[0] * direction_out[1] - direction_in[1] * direction_out[0];
            let dot = direction_in[0] * direction_out[0] + direction_in[1] * direction_out[1];
            let theta = cross.atan2(dot);
            let ccw = if theta < 0.0 {
                theta + 2.0 * std::f64::consts::TAU
            } else {
                theta
            };
            if best.is_none_or(|(angle, _)| ccw < angle) {
                best = Some((ccw, candidate));
            }
        }
        if let Some((_, candidate)) = best {
            next[edge] = candidate;
        }
    }

    // Cycle decomposition of the pairing permutation.
    let mut visited = vec![false; half_edges.len()];
    let mut loops = Vec::new();
    for start in 0..half_edges.len() {
        if visited[start] || next[start] == usize::MAX {
            visited[start] = true;
            continue;
        }
        let mut vertices = Vec::new();
        let mut edge = start;
        while !visited[edge] {
            visited[edge] = true;
            vertices.push(point(half_edges[edge].0));
            edge = next[edge];
            if edge == usize::MAX {
                break;
            }
        }
        // Only cycles of the permutation are closed loops.
        if edge != start || vertices.len() < 3 {
            continue;
        }
        let signed_area = 0.5
            * vertices
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let b = vertices[(i + 1) % vertices.len()];
                    a[0] * b[1] - b[0] * a[1]
                })
                .sum::<f64>();
        loops.push(BoundaryLoop {
            vertices_rz_m: vertices,
            is_hole: signed_area < 0.0,
            signed_area_m2: signed_area,
        });
    }
    loops
}

// ---------------------------------------------------------------------------
// CP2: body-fitted structured remesh
// ---------------------------------------------------------------------------

/// Structured-grid description of the coarse topology mesh, recovered from
/// its node coordinates (the mesh comes from `rectangular_mesh_triangles`).
struct StructuredGrid {
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    radial_cells: usize,
    axial_cells: usize,
}

fn structured_grid(mesh: &Mesh) -> Result<StructuredGrid, ProjectedReanalysisError> {
    let quantize = |value: f64| (value / COORDINATE_QUANTUM_M).round() as i64;
    let mut xs: Vec<i64> = mesh.nodes.iter().map(|node| quantize(node.x)).collect();
    let mut ys: Vec<i64> = mesh.nodes.iter().map(|node| quantize(node.y)).collect();
    xs.sort_unstable();
    xs.dedup();
    ys.sort_unstable();
    ys.dedup();
    if xs.len() < 2 || ys.len() < 2 {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "topology mesh is not a structured rectilinear grid",
        ));
    }
    let radial_cells = xs.len() - 1;
    let axial_cells = ys.len() - 1;
    if (radial_cells + 1) * (axial_cells + 1) != mesh.num_nodes()
        || 2 * radial_cells * axial_cells != mesh.elements.len()
    {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "topology mesh is not a structured rectilinear grid",
        ));
    }
    let x_of = |q: i64| q as f64 * COORDINATE_QUANTUM_M;
    Ok(StructuredGrid {
        x_min: x_of(*xs.first().unwrap()),
        x_max: x_of(*xs.last().unwrap()),
        y_min: x_of(*ys.first().unwrap()),
        y_max: x_of(*ys.last().unwrap()),
        radial_cells,
        axial_cells,
    })
}

/// Minimum barycentric coordinate of point `(r, z)` in triangle `element`
/// (negative outside; the three coordinates always sum to 1).
fn min_barycentric(mesh: &Mesh, element: usize, r: f64, z: f64) -> f64 {
    let v = mesh.elements[element].vertices();
    let (a, b, c) = (&mesh.nodes[v[0]], &mesh.nodes[v[1]], &mesh.nodes[v[2]]);
    let denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if denominator.abs() < 1.0e-30 {
        return f64::NEG_INFINITY;
    }
    let lambda_a = ((b.y - c.y) * (r - c.x) + (c.x - b.x) * (z - c.y)) / denominator;
    let lambda_b = ((c.y - a.y) * (r - c.x) + (a.x - c.x) * (z - c.y)) / denominator;
    let lambda_c = 1.0 - lambda_a - lambda_b;
    lambda_a.min(lambda_b).min(lambda_c)
}

/// Coarse element containing the refined triangle `vertices` (of `refined`),
/// resolved among the two triangles of the arithmetic grid quad holding the
/// refined centroid.
///
/// Containment is decided over ALL THREE refined vertices, not the centroid:
/// refined centroids of diagonal sub-quads lie exactly ON the coarse diagonal
/// (the refined and coarse diagonal splits are consistent), where a
/// centroid-side test would systematically assign both sub-triangles to the
/// same side and corrupt the steel area along diagonal interfaces. Since
/// refined triangles never cross a coarse grid line or diagonal, exactly one
/// coarse triangle contains all three vertices (vertices on the shared
/// diagonal are on the boundary of both).
fn locate_coarse_element(
    coarse: &Mesh,
    grid: &StructuredGrid,
    refined: &Mesh,
    vertices: &[usize],
) -> Result<usize, ProjectedReanalysisError> {
    let dx = (grid.x_max - grid.x_min) / grid.radial_cells as f64;
    let dz = (grid.y_max - grid.y_min) / grid.axial_cells as f64;
    let (r, z) = element_centroid(refined, vertices);
    let i = ((r - grid.x_min) / dx)
        .floor()
        .clamp(0.0, (grid.radial_cells - 1) as f64) as usize;
    let j = ((z - grid.y_min) / dz)
        .floor()
        .clamp(0.0, (grid.axial_cells - 1) as f64) as usize;
    let first = 2 * (j * grid.radial_cells + i);
    let second = first + 1;
    let contains = |element: usize| {
        vertices.iter().all(|&node| {
            min_barycentric(
                coarse,
                element,
                refined.nodes[node].x,
                refined.nodes[node].y,
            ) >= -1.0e-9
        })
    };
    match (contains(first), contains(second)) {
        (true, _) => Ok(first),
        (false, true) => Ok(second),
        (false, false) => Err(ProjectedReanalysisError::Fem(
            "refined triangle is not contained in any coarse element".to_string(),
        )),
    }
}

/// Build a body-fitted mesh of the motor domain at integer refinement
/// `factor` over the coarse structured topology mesh.
///
/// `steel` flags the coarse elements belonging to the binary steel design
/// (length must equal `coarse.elements.len()`). Refined elements inherit the
/// containing coarse element's marker, except design cells (`STEEL_POLE` /
/// `STEEL_YOKE`) flagged air, which become `MAGNETIC_AIR`. Boundary markers
/// and physical groups replicate the golden magnetic mesh contract.
pub fn build_body_fitted_mesh(
    coarse: &Mesh,
    steel: &[bool],
    refinement: usize,
) -> Result<Mesh, ProjectedReanalysisError> {
    if steel.len() != coarse.elements.len() {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "steel flags must cover every coarse element",
        ));
    }
    if refinement == 0 {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "refinement factor must be at least one",
        ));
    }
    let grid = structured_grid(coarse)?;
    let mut mesh = rectangular_mesh_triangles(
        grid.x_min,
        grid.x_max,
        grid.y_min,
        grid.y_max,
        grid.radial_cells * refinement,
        grid.axial_cells * refinement,
    );
    for index in 0..mesh.elements.len() {
        let vertices = mesh.elements[index].vertices().to_vec();
        let element = locate_coarse_element(coarse, &grid, &mesh, &vertices)?;
        let marker = coarse.elements[element].marker;
        mesh.elements[index].marker =
            if matches!(marker, STEEL_POLE | STEEL_YOKE) && !steel[element] {
                MAGNETIC_AIR
            } else {
                marker
            };
    }
    register_groups(
        &mut mesh,
        &[
            (MAGNETIC_AIR, "magnetic air"),
            (PERMANENT_MAGNET, "permanent magnet"),
            (STEEL_POLE, "steel pole"),
            (VOICE_COIL, "voice coil"),
            (AIR_GAP, "air gap"),
            (STEEL_YOKE, "steel yoke"),
        ],
    )
    .map_err(|error| ProjectedReanalysisError::Fem(error.to_string()))?;
    remap_boundary_markers(
        &mut mesh,
        [
            MAGNETIC_AXIS,
            MAGNETIC_OUTER_RADIUS,
            MAGNETIC_LOWER_END,
            MAGNETIC_UPPER_END,
        ],
    );
    register_boundary_groups(
        &mut mesh,
        &[
            (MAGNETIC_AXIS, "magnetic symmetry axis"),
            (MAGNETIC_OUTER_RADIUS, "magnetic exterior"),
            (MAGNETIC_LOWER_END, "magnetic lower exterior"),
            (MAGNETIC_UPPER_END, "magnetic upper exterior"),
        ],
    )
    .map_err(|error| ProjectedReanalysisError::Fem(error.to_string()))?;
    Ok(mesh)
}

/// Verify the body-fitted property: every segment of every boundary loop is
/// covered by a chain of collinear `mesh` edges (a coarse interface segment is
/// subdivided into `K` refined edges, so coverage — not direct presence — is
/// the right check). Endpoint distances above `1e-9 m` off the segment line
/// disqualify an edge.
pub fn loops_are_body_fitted(mesh: &Mesh, loops: &[BoundaryLoop]) -> bool {
    let point = |node: usize| [mesh.nodes[node].x, mesh.nodes[node].y];
    let mut edges = Vec::new();
    for element in &mesh.elements {
        let v = element.vertices();
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            edges.push((point(a), point(b)));
        }
    }
    loops.iter().all(|loop_| {
        let n = loop_.vertices_rz_m.len();
        (0..n).all(|i| {
            let a = loop_.vertices_rz_m[i];
            let b = loop_.vertices_rz_m[(i + 1) % n];
            let d = [b[0] - a[0], b[1] - a[1]];
            let length_sq = d[0] * d[0] + d[1] * d[1];
            if length_sq <= 0.0 {
                return false;
            }
            // Intervals of mesh edges lying on the segment line.
            let mut intervals: Vec<(f64, f64)> = Vec::new();
            for (p, q) in &edges {
                let covered = |point: [f64; 2]| {
                    let w = [point[0] - a[0], point[1] - a[1]];
                    let cross = w[0] * d[1] - w[1] * d[0];
                    if cross.abs() > 1.0e-9 * length_sq.sqrt() {
                        return None;
                    }
                    let t = (w[0] * d[0] + w[1] * d[1]) / length_sq;
                    (-1.0e-9..=1.0 + 1.0e-9).contains(&t).then_some(t)
                };
                if let (Some(t0), Some(t1)) = (covered(*p), covered(*q)) {
                    intervals.push((t0.min(t1), t0.max(t1)));
                }
            }
            intervals.sort_by(|a, b| f64::total_cmp(&a.0, &b.0));
            // Merge intervals: the merged chain must reach from 0 to 1.
            let mut reached = 0.0;
            for (start, end) in intervals {
                if start > reached + 1.0e-9 {
                    break;
                }
                reached = reached.max(end);
            }
            reached >= 1.0 - 1.0e-9
        })
    })
}

/// Planar `(r, z)` area of the elements flagged `marker` (sum over markers).
#[cfg(test)]
fn marker_area(mesh: &Mesh, markers: [i32; 2]) -> f64 {
    mesh.elements
        .iter()
        .filter(|element| markers.contains(&element.marker))
        .map(|element| crate::axisymmetric_linear::triangle_area(mesh, element.vertices()))
        .sum()
}

// ---------------------------------------------------------------------------
// CP3: true-material forward solve and mesh convergence
// ---------------------------------------------------------------------------

/// Steel element indices (pole or yoke markers) of a (re)mesh, in element
/// order.
fn steel_elements_of(mesh: &Mesh) -> Vec<usize> {
    mesh.elements
        .iter()
        .enumerate()
        .filter_map(|(index, element)| {
            matches!(element.marker, STEEL_POLE | STEEL_YOKE).then_some(index)
        })
        .collect()
}

/// Solve the axisymmetric magnetostatic bias on `mesh` with the true steel
/// model and return BL (T·m).
///
/// The base assembly mirrors the golden forward model (regions, magnet
/// equivalent-current RHS, homogeneous Dirichlet outer boundaries, steel at
/// `μ_r = 1000`). With `steel_mu_r = Some(map)` (one entry per entry of
/// `steel_elements`) each steel element is corrected to its own relative
/// permeability via the linear-in-`1/μ` unit-stiffness splitting of the
/// topology workspace — no density interpolation anywhere.
fn solve_true_material_bl(
    driver: &AxisymmetricLinearConfig,
    mesh: &Mesh,
    steel_elements: &[usize],
    steel_mu_r: Option<&[f64]>,
) -> Result<f64, ProjectedReanalysisError> {
    let space = H1Space::new(PolynomialDegree::P1);
    let mut forms = RegionMap::new();
    forms.insert(MAGNETIC_AIR, AxisymmetricAThetaForm::new(MU_0));
    forms.insert(PERMANENT_MAGNET, AxisymmetricAThetaForm::new(1.05 * MU_0));
    forms.insert(
        STEEL_POLE,
        AxisymmetricAThetaForm::new(LINEAR_STEEL_MU_R * MU_0),
    );
    forms.insert(
        STEEL_YOKE,
        AxisymmetricAThetaForm::new(LINEAR_STEEL_MU_R * MU_0),
    );
    forms.insert(VOICE_COIL, AxisymmetricAThetaForm::new(MU_0));
    forms.insert(AIR_GAP, AxisymmetricAThetaForm::new(MU_0));
    let matrix = assemble_bilinear_by_region(&forms, &space, &space, mesh, 2)
        .map_err(|error| ProjectedReanalysisError::Fem(error.to_string()))?;
    let equivalent_current_density = driver.remanence_t / (MU_0 * 20.0e-3);
    let source = AzimuthalCurrentDensityForm::new(move |_, _| equivalent_current_density);
    let rhs = assemble_linear_where(&source, &space, mesh, 2, |_, element| {
        element.marker == PERMANENT_MAGNET
    });
    let mut system = FemSystem::new(matrix, rhs);
    for marker in [
        MAGNETIC_AXIS,
        MAGNETIC_OUTER_RADIUS,
        MAGNETIC_LOWER_END,
        MAGNETIC_UPPER_END,
    ] {
        apply_homogeneous_dirichlet(&mut system, mesh, &space, marker);
    }

    let mut matrix = system.matrix;
    if let Some(steel_mu_r) = steel_mu_r {
        if steel_mu_r.len() != steel_elements.len() {
            return Err(ProjectedReanalysisError::InvalidConfiguration(
                "steel permeability map must cover every steel element",
            ));
        }
        let dirichlet_dofs = system.dirichlet_dofs.clone();
        let unit_form = AxisymmetricAThetaForm {
            inv_mu: 1.0,
            axis_epsilon: 1.0e-12,
        };
        let steel_inv_mu = 1.0 / (LINEAR_STEEL_MU_R * MU_0);
        for (&element, &mu_r) in steel_elements.iter().zip(steel_mu_r) {
            let correction = 1.0 / (MU_0 * mu_r.max(1.0)) - steel_inv_mu;
            if correction == 0.0 {
                continue;
            }
            let triplet = assemble_bilinear_where(&unit_form, &space, &space, mesh, 2, {
                |index, _| index == element
            });
            for k in 0..triplet.nnz() {
                let (row, col) = (triplet.rows[k], triplet.cols[k]);
                if !dirichlet_dofs.contains(&row) && !dirichlet_dofs.contains(&col) {
                    matrix.push(row, col, correction * triplet.values[k]);
                }
            }
        }
    }

    let factorization = MotorFactorization::factorize(&matrix)?;
    let potential = factorization.solve(&system.rhs)?;
    let bias = magnetic_bias_from_potential(driver, mesh, potential)
        .map_err(|error| ProjectedReanalysisError::Fem(error.to_string()))?;
    Ok(bias.distributed_bl_tm)
}

/// Differential permeability map of the binary design on `mesh`, extracted
/// from a nonlinear bias solve with steel cells at weight 1 (true steel).
fn extract_binary_permeability_map(
    driver: &AxisymmetricLinearConfig,
    mesh: &Mesh,
    steel_elements: &[usize],
    config: &ProjectedReanalysisConfig,
) -> Result<Option<Vec<f64>>, ProjectedReanalysisError> {
    let SteelPermeability::DifferentialFromBias {
        bh_curve, solve, ..
    } = &config.steel_permeability
    else {
        return Ok(None);
    };
    if steel_elements.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let weights = vec![1.0; steel_elements.len()];
    let map = steel_permeability_map_from_bias(
        mesh,
        driver.remanence_t,
        steel_elements,
        &weights,
        bh_curve,
        solve,
    )?;
    Ok(Some(map.differential_relative_permeability))
}

/// Core reanalysis: binarize `physical` (already filtered/projected densities
/// over the design elements), remesh over the refinement ladder, and re-solve
/// with the true steel model. `interpolation` is only used for the comparison
/// BL of the topology forward model at `physical`.
fn reanalyze_physical(
    driver: &AxisymmetricLinearConfig,
    physical: &[f64],
    interpolation: InterpolationScheme,
    threshold: f64,
    config: &ProjectedReanalysisConfig,
) -> Result<ProjectedReanalysisReport, ProjectedReanalysisError> {
    if !threshold.is_finite() || !(0.0..1.0).contains(&threshold) {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "steel threshold must lie in (0, 1)",
        ));
    }
    if !config.convergence_tolerance.is_finite() || config.convergence_tolerance <= 0.0 {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "convergence tolerance must be positive and finite",
        ));
    }
    let mut factors = config.refinement_factors.clone();
    factors.sort_unstable();
    factors.dedup();
    if factors.is_empty() || factors.contains(&0) {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "refinement factors must be non-empty and at least one",
        ));
    }

    let workspace = MotorTopologyWorkspace::build(driver)?;
    let steel_set = extract_binary_design(physical, &workspace.design_elements, threshold)?;
    let boundary_loops = extract_steel_boundary(&workspace.mesh, &steel_set);
    let (_, bl_projected_interpolated_tm) = workspace.solve_forward(physical, interpolation)?;

    // Per-coarse-element steel flags for the remesher.
    let mut steel = vec![false; workspace.mesh.elements.len()];
    for &element in &steel_set {
        steel[element] = true;
    }

    let mut levels = Vec::with_capacity(factors.len());
    for &factor in &factors {
        let mesh = build_body_fitted_mesh(&workspace.mesh, &steel, factor)?;
        let steel_elements = steel_elements_of(&mesh);
        let steel_mu_r = extract_binary_permeability_map(driver, &mesh, &steel_elements, config)?;
        let bl_tm = solve_true_material_bl(driver, &mesh, &steel_elements, steel_mu_r.as_deref())?;
        let relative_change_from_previous = levels.last().map(|previous: &ReanalysisLevel| {
            (bl_tm - previous.bl_tm).abs() / previous.bl_tm.abs().max(1.0e-30)
        });
        levels.push(ReanalysisLevel {
            refinement_factor: factor,
            num_elements: mesh.elements.len(),
            num_nodes: mesh.num_nodes(),
            bl_tm,
            relative_change_from_previous,
        });
    }

    let bl_body_fitted_tm = levels.last().expect("non-empty ladder").bl_tm;
    let relative_discrepancy = (bl_body_fitted_tm - bl_projected_interpolated_tm).abs()
        / bl_projected_interpolated_tm.abs().max(1.0e-30);
    let mesh_converged = levels.len() >= 2
        && levels
            .last()
            .and_then(|level| level.relative_change_from_previous)
            .is_some_and(|change| change <= config.convergence_tolerance);

    Ok(ProjectedReanalysisReport {
        threshold,
        steel_element_count: steel_set.len(),
        boundary_loops,
        bl_projected_interpolated_tm,
        levels,
        bl_body_fitted_tm,
        relative_discrepancy,
        mesh_converged,
    })
}

/// Design element indices (pole/yoke cells) of the motor topology mesh for
/// `driver` — the ordering every density vector of this driver follows.
pub fn motor_design_elements(
    driver: &AxisymmetricLinearConfig,
) -> Result<Vec<usize>, ProjectedReanalysisError> {
    Ok(MotorTopologyWorkspace::build(driver)?.design_elements)
}

/// Reanalyze explicit physical densities (already filtered/projected) with an
/// explicit threshold. This is the entry point for designs that did not come
/// from [`optimize_motor_topology`] (e.g. the all-steel original geometry).
pub fn reanalyze_physical_design(
    driver: &AxisymmetricLinearConfig,
    physical_densities: &[f64],
    interpolation: InterpolationScheme,
    threshold: f64,
    config: &ProjectedReanalysisConfig,
) -> Result<ProjectedReanalysisReport, ProjectedReanalysisError> {
    reanalyze_physical(driver, physical_densities, interpolation, threshold, config)
}

/// Reanalyze the projected design of a topology run: replays the run's
/// density filter and Heaviside projection (at the final continuation beta)
/// over `result.densities`, binarizes at the projection threshold (or 0.5),
/// and runs the body-fitted mesh-convergence study.
pub fn reanalyze_projected_design(
    driver: &AxisymmetricLinearConfig,
    topology_config: &MotorTopologyConfig,
    result: &MotorTopologyResult,
    config: &ProjectedReanalysisConfig,
) -> Result<ProjectedReanalysisReport, ProjectedReanalysisError> {
    let workspace = MotorTopologyWorkspace::build(driver)?;
    if result.densities.len() != workspace.design_elements.len()
        || result.design_elements != workspace.design_elements
    {
        return Err(ProjectedReanalysisError::InvalidConfiguration(
            "topology result does not match the driver's design elements",
        ));
    }
    let filter = topology_config.density_filter_radius_m.map(|radius| {
        DensityFilter::build_from_elements(&workspace.mesh, radius, &workspace.design_elements)
    });
    let projection = final_continuation_beta(topology_config, result.iterations).map(|beta| {
        HeavisideProjection::new(
            beta,
            topology_config
                .projection
                .expect("beta implies projection")
                .eta,
        )
    });
    let (physical, _) = physical_densities(&result.densities, filter.as_ref(), projection.as_ref());
    let threshold = config
        .threshold
        .or(topology_config.projection.map(|projection| projection.eta))
        .unwrap_or(0.5);
    reanalyze_physical(
        driver,
        &physical,
        topology_config.interpolation,
        threshold,
        config,
    )
}

/// One-call driver: optimize the motor topology, then reanalyze the projected
/// design on body-fitted, mesh-converged geometry.
pub fn optimize_and_reanalyze(
    driver: &AxisymmetricLinearConfig,
    topology_config: &MotorTopologyConfig,
    reanalysis_config: &ProjectedReanalysisConfig,
) -> Result<(MotorTopologyResult, ProjectedReanalysisReport), ProjectedReanalysisError> {
    let result = optimize_motor_topology(driver, topology_config)?;
    let report = reanalyze_projected_design(driver, topology_config, &result, reanalysis_config)?;
    Ok((result, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axisymmetric_linear::{AxisymmetricLinearModel, element_centroid, triangle_area};
    use crate::motor_topology::{ProjectionContinuationConfig, optimize_motor_topology};
    use crate::nonlinear_bias::default_low_carbon_steel_bh_curve;
    use sonium_fem::physics::electromagnetics::magnetostatics::{
        BhCurve, NonlinearBiasSolveConfig,
    };
    use std::collections::HashSet;

    fn fixture_driver() -> AxisymmetricLinearConfig {
        AxisymmetricLinearConfig {
            magnetic_radial_cells: 8,
            magnetic_axial_cells: 8,
            structural_radial_cells: 6,
            structural_axial_cells: 1,
            acoustic_axial_cells: 4,
            ..AxisymmetricLinearConfig::default()
        }
    }

    fn steel_flags(workspace: &MotorTopologyWorkspace, steel_set: &[usize]) -> Vec<bool> {
        let mut flags = vec![false; workspace.mesh.elements.len()];
        for &element in steel_set {
            flags[element] = true;
        }
        flags
    }

    fn steel_area(mesh: &Mesh) -> f64 {
        marker_area(mesh, [STEEL_POLE, STEEL_YOKE])
    }

    fn loop_area_sum(loops: &[BoundaryLoop]) -> f64 {
        loops.iter().map(|loop_| loop_.signed_area_m2).sum()
    }

    // ---- CP1 ----

    #[test]
    fn binary_extraction_thresholds_densities() {
        let steel = extract_binary_design(&[0.2, 0.8, 0.5], &[3, 5, 7], 0.5).unwrap();
        assert_eq!(steel, vec![5, 7]);
        assert!(extract_binary_design(&[0.2], &[3, 5], 0.5).is_err());
        assert!(extract_binary_design(&[0.2], &[3], 1.0).is_err());
    }

    /// Full-steel design: the boundary is the design-region outline; loop
    /// areas sum exactly to the steel element area, and no loop segment
    /// separates two design elements (the movable interface is empty).
    #[test]
    fn full_steel_boundary_is_design_region_outline() {
        let workspace = MotorTopologyWorkspace::build(&fixture_driver()).unwrap();
        let steel_set = workspace.design_elements.clone();
        let loops = extract_steel_boundary(&workspace.mesh, &steel_set);
        assert!(!loops.is_empty());
        let steel_area = steel_area(&workspace.mesh);
        let relative = (loop_area_sum(&loops) - steel_area).abs() / steel_area;
        assert!(
            relative < 1.0e-12,
            "loop area sum deviates from steel area by {relative:e}"
        );
        for loop_ in &loops {
            assert!(loop_.vertices_rz_m.len() >= 3);
            assert!(!loop_.is_hole, "full-steel design has no air pockets");
        }
    }

    /// Checkerboard over fully-design quads: every steel quad is an island
    /// touching other steel quads only at vertices, so the pinch-resolving
    /// chaining must produce exactly one 4-segment loop per steel quad.
    #[test]
    fn checkerboard_boundary_loops_resolve_point_contacts() {
        let workspace = MotorTopologyWorkspace::build(&fixture_driver()).unwrap();
        let grid = structured_grid(&workspace.mesh).unwrap();
        let design: HashSet<usize> = workspace.design_elements.iter().copied().collect();
        let quad_of = |element: usize| element / 2;
        let is_full_design_quad =
            |quad: usize| design.contains(&(2 * quad)) && design.contains(&(2 * quad + 1));
        let mut steel_set = Vec::new();
        let mut expected_loops = 0;
        for &element in &workspace.design_elements {
            let quad = quad_of(element);
            let (i, j) = (quad % grid.radial_cells, quad / grid.radial_cells);
            if is_full_design_quad(quad) && (i + j) % 2 == 0 {
                steel_set.push(element);
                if element % 2 == 0 {
                    expected_loops += 1;
                }
            }
        }
        assert!(
            expected_loops >= 4,
            "fixture should have several full quads"
        );
        let loops = extract_steel_boundary(&workspace.mesh, &steel_set);
        assert_eq!(
            loops.len(),
            expected_loops,
            "each isolated checkerboard quad should give exactly one loop"
        );
        let quad_area = {
            let dx = (grid.x_max - grid.x_min) / grid.radial_cells as f64;
            let dz = (grid.y_max - grid.y_min) / grid.axial_cells as f64;
            dx * dz
        };
        for loop_ in &loops {
            assert_eq!(
                loop_.vertices_rz_m.len(),
                4,
                "quad loop should have 4 vertices"
            );
            assert!((loop_.signed_area_m2 - quad_area).abs() / quad_area < 1.0e-12);
        }
        let steel_area: f64 = steel_set
            .iter()
            .map(|&element| {
                triangle_area(&workspace.mesh, workspace.mesh.elements[element].vertices())
            })
            .sum();
        assert!((loop_area_sum(&loops) - steel_area).abs() / steel_area < 1.0e-12);
    }

    /// A single connected blob gives exactly one outer loop whose shoelace
    /// area equals the steel element area sum.
    #[test]
    fn single_blob_gives_single_loop_with_matching_area() {
        let workspace = MotorTopologyWorkspace::build(&fixture_driver()).unwrap();
        // Blob: design cells within 2.5 cell widths of the pole centre.
        let centre = [5.0e-3, 0.0];
        let radius = 2.5 * 40.0e-3 / 8.0;
        let steel_set: Vec<usize> = workspace
            .design_elements
            .iter()
            .copied()
            .filter(|&element| {
                let (r, z) =
                    element_centroid(&workspace.mesh, workspace.mesh.elements[element].vertices());
                let (dr, dz) = (r - centre[0], z - centre[1]);
                (dr * dr + dz * dz).sqrt() <= radius
            })
            .collect();
        assert!(steel_set.len() >= 4);
        let loops = extract_steel_boundary(&workspace.mesh, &steel_set);
        let outer: Vec<_> = loops.iter().filter(|loop_| !loop_.is_hole).collect();
        assert_eq!(
            outer.len(),
            1,
            "blob should give one outer loop, got {loops:?}"
        );
        let steel_area: f64 = steel_set
            .iter()
            .map(|&element| {
                triangle_area(&workspace.mesh, workspace.mesh.elements[element].vertices())
            })
            .sum();
        assert!((loop_area_sum(&loops) - steel_area).abs() / steel_area < 1.0e-12);
    }

    // ---- CP2 ----

    /// The refined mesh preserves the steel area exactly, keeps the non-design
    /// region geometry (magnet, coil, gap) exactly, has positively oriented
    /// triangles, and contains every steel-boundary segment as a mesh edge.
    #[test]
    fn remesh_is_body_fitted_and_region_preserving() {
        let driver = fixture_driver();
        let workspace = MotorTopologyWorkspace::build(&driver).unwrap();
        // Non-trivial binary design: pole fully steel, yoke air.
        let steel_set: Vec<usize> = workspace
            .design_elements
            .iter()
            .copied()
            .filter(|&element| workspace.mesh.elements[element].marker == STEEL_POLE)
            .collect();
        let flags = steel_flags(&workspace, &steel_set);
        let loops = extract_steel_boundary(&workspace.mesh, &steel_set);

        let refined = build_body_fitted_mesh(&workspace.mesh, &flags, 2).unwrap();
        for element in &refined.elements {
            let v = element.vertices();
            let (a, b, c) = (
                &refined.nodes[v[0]],
                &refined.nodes[v[1]],
                &refined.nodes[v[2]],
            );
            let twice_signed = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
            assert!(
                twice_signed > 0.0,
                "refined triangle must be positively oriented"
            );
        }

        let binary_steel_area: f64 = steel_set
            .iter()
            .map(|&element| {
                triangle_area(&workspace.mesh, workspace.mesh.elements[element].vertices())
            })
            .sum();
        let refined_steel_area = refined
            .elements
            .iter()
            .filter(|element| matches!(element.marker, STEEL_POLE | STEEL_YOKE))
            .map(|element| triangle_area(&refined, element.vertices()))
            .sum::<f64>();
        assert!(
            (refined_steel_area - binary_steel_area).abs() / binary_steel_area < 1.0e-12,
            "refined steel area {refined_steel_area} vs binary design {binary_steel_area}"
        );

        for marker in [PERMANENT_MAGNET, VOICE_COIL, AIR_GAP] {
            let coarse_area = marker_area(&workspace.mesh, [marker, marker]);
            let refined_area = marker_area(&refined, [marker, marker]);
            assert!(
                (refined_area - coarse_area).abs() / coarse_area < 1.0e-12,
                "non-design region {marker} area changed: {coarse_area} vs {refined_area}"
            );
        }

        assert!(
            loops_are_body_fitted(&refined, &loops),
            "every steel-boundary segment must be a refined mesh edge"
        );
    }

    /// At factor 1 the remeshed markers reproduce the binary classification
    /// exactly (steel cells keep their marker, air design cells become air).
    #[test]
    fn remesh_factor_one_reproduces_binary_classification() {
        let workspace = MotorTopologyWorkspace::build(&fixture_driver()).unwrap();
        let steel_set: Vec<usize> = workspace
            .design_elements
            .iter()
            .copied()
            .filter(|&element| workspace.mesh.elements[element].marker == STEEL_POLE)
            .collect();
        let flags = steel_flags(&workspace, &steel_set);
        let remeshed = build_body_fitted_mesh(&workspace.mesh, &flags, 1).unwrap();
        assert_eq!(remeshed.elements.len(), workspace.mesh.elements.len());
        for (index, element) in remeshed.elements.iter().enumerate() {
            let coarse_marker = workspace.mesh.elements[index].marker;
            let expected = if matches!(coarse_marker, STEEL_POLE | STEEL_YOKE) && !flags[index] {
                MAGNETIC_AIR
            } else {
                coarse_marker
            };
            assert_eq!(element.marker, expected, "element {index} marker mismatch");
        }
    }

    // ---- CP3 ----

    /// Validation 1: the original full-steel geometry through the whole
    /// pipeline reproduces the golden model BL. At refinement 1 the remeshed
    /// geometry and mesh are identical to the golden magnetic mesh, so only
    /// the linear solver differs (sparse Cholesky vs the golden complex direct
    /// solve) — both are direct methods, so agreement is at solver roundoff;
    /// the 1e-9 bound is two orders of magnitude above the observed ~1e-12.
    #[test]
    fn full_steel_pipeline_reproduces_golden_bl() {
        let driver = AxisymmetricLinearConfig::default();
        let golden = AxisymmetricLinearModel::build(driver.clone())
            .unwrap()
            .magnetic_bias
            .distributed_bl_tm;
        let design_elements = motor_design_elements(&driver).unwrap();
        let physical = vec![1.0; design_elements.len()];
        let report = reanalyze_physical_design(
            &driver,
            &physical,
            InterpolationScheme::Ramp { penalty: 3.0 },
            0.5,
            &ProjectedReanalysisConfig::default(),
        )
        .unwrap();
        assert_eq!(report.steel_element_count, design_elements.len());
        let level1 = &report.levels[0];
        assert_eq!(level1.refinement_factor, 1);
        let relative = (level1.bl_tm - golden).abs() / golden;
        eprintln!(
            "golden reproduction: golden={golden:.12}, K=1={:.12} (rel {relative:e}), K=2={:.12}",
            level1.bl_tm, report.bl_body_fitted_tm
        );
        assert!(
            relative < 1.0e-9,
            "K=1 body-fitted BL deviates from golden BL by {relative:e}"
        );
        // The interpolated comparison BL at all-ones densities is the golden
        // topology model itself.
        let interp_relative = (report.bl_projected_interpolated_tm - golden).abs() / golden;
        assert!(interp_relative < 1.0e-9);
    }

    /// Validation 2: short projected optimization on the 8x8 fixture, then
    /// reanalysis — reports the staircase-vs-body-fitted discrepancy and the
    /// convergence table, and asserts mesh convergence at the default 1 %
    /// tolerance.
    ///
    /// Ladder note: the pole-tip reentrant corner (permeability jump 1000:1)
    /// gives a measured BL convergence ratio of ~0.25–0.35 per grid doubling
    /// on this fixture (18.8 % then 5.2 % then 1.3 % for K = 1→2→4→8), so the
    /// 1 % flag needs the ladder out to K = 16 (~1.5 s solve time). The other
    /// tests stay at <= 2 levels.
    #[test]
    fn optimized_design_reanalysis_reports_converged_body_fitted_bl() {
        let driver = fixture_driver();
        let topology = MotorTopologyConfig {
            max_iterations: 12,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.65,
            projection: Some(ProjectionContinuationConfig {
                beta_initial: 1.0,
                beta_max: 16.0,
                eta: 0.5,
                continuation_interval: 4,
            }),
            ..MotorTopologyConfig::default()
        };
        let result = optimize_motor_topology(&driver, &topology).unwrap();
        assert!(
            result.gray_fraction < 0.2,
            "fixture should be near-binary, gray fraction {}",
            result.gray_fraction
        );
        let report = reanalyze_projected_design(
            &driver,
            &topology,
            &result,
            &ProjectedReanalysisConfig {
                refinement_factors: vec![1, 2, 4, 8, 16],
                ..ProjectedReanalysisConfig::default()
            },
        )
        .unwrap();
        eprintln!(
            "optimized design: steel cells {}/{}, loops {} ({} holes)",
            report.steel_element_count,
            result.design_elements.len(),
            report.boundary_loops.len(),
            report.boundary_loops.iter().filter(|l| l.is_hole).count()
        );
        eprintln!(
            "BL: interpolated(projected)={:.6} T·m, body-fitted={:.6} T·m, discrepancy={:.3e}",
            report.bl_projected_interpolated_tm,
            report.bl_body_fitted_tm,
            report.relative_discrepancy
        );
        for level in &report.levels {
            eprintln!(
                "  K={}: {} elements, BL={:.6} T·m, change={:?}",
                level.refinement_factor,
                level.num_elements,
                level.bl_tm,
                level.relative_change_from_previous
            );
        }
        assert!(report.bl_projected_interpolated_tm.is_finite());
        assert!(report.bl_body_fitted_tm.is_finite() && report.bl_body_fitted_tm > 0.0);
        assert!(
            report.mesh_converged,
            "body-fitted BL should converge within 1% over the K=1..16 ladder: {report:?}"
        );
    }

    #[test]
    fn default_reanalysis_ladder_reaches_convergence_resolution() {
        assert_eq!(
            ProjectedReanalysisConfig::default().refinement_factors,
            vec![1, 2, 4, 8, 16]
        );
    }

    /// Validation 3: degenerate designs — all-air and all-steel design
    /// regions — reanalyze without panics and give sensible BL ordering.
    #[test]
    fn degenerate_designs_reanalyze_without_panics() {
        let driver = fixture_driver();
        let design_elements = motor_design_elements(&driver).unwrap();
        let config = ProjectedReanalysisConfig {
            refinement_factors: vec![1],
            ..ProjectedReanalysisConfig::default()
        };
        let all_air = reanalyze_physical_design(
            &driver,
            &vec![0.0; design_elements.len()],
            InterpolationScheme::Ramp { penalty: 3.0 },
            0.5,
            &config,
        )
        .unwrap();
        assert_eq!(all_air.steel_element_count, 0);
        assert!(all_air.boundary_loops.is_empty());
        assert!(all_air.bl_body_fitted_tm.is_finite() && all_air.bl_body_fitted_tm >= 0.0);

        let all_steel = reanalyze_physical_design(
            &driver,
            &vec![1.0; design_elements.len()],
            InterpolationScheme::Ramp { penalty: 3.0 },
            0.5,
            &config,
        )
        .unwrap();
        assert_eq!(all_steel.steel_element_count, design_elements.len());
        assert!(
            all_steel.bl_body_fitted_tm > all_air.bl_body_fitted_tm,
            "steel path should raise BL: air={} steel={}",
            all_air.bl_body_fitted_tm,
            all_steel.bl_body_fitted_tm
        );
    }

    /// D3 linear limit on the reanalysis side: with a straight-line B-H curve
    /// at `mu_r = 1000`, the `DifferentialFromBias` reanalysis reproduces the
    /// `Constant` reanalysis.
    #[test]
    fn differential_permeability_linear_limit_matches_constant() {
        let driver = fixture_driver();
        let design_elements = motor_design_elements(&driver).unwrap();
        let physical = vec![1.0; design_elements.len()];
        let interpolation = InterpolationScheme::Ramp { penalty: 3.0 };
        let config = ProjectedReanalysisConfig {
            refinement_factors: vec![1],
            ..ProjectedReanalysisConfig::default()
        };
        let constant =
            reanalyze_physical_design(&driver, &physical, interpolation, 0.5, &config).unwrap();

        let permeability = LINEAR_STEEL_MU_R * MU_0;
        let linear_curve = BhCurve::new(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
                .into_iter()
                .map(|b| (b, b / permeability))
                .collect(),
        )
        .unwrap();
        let differential = reanalyze_physical_design(
            &driver,
            &physical,
            interpolation,
            0.5,
            &ProjectedReanalysisConfig {
                steel_permeability: SteelPermeability::DifferentialFromBias {
                    bh_curve: linear_curve,
                    solve: NonlinearBiasSolveConfig {
                        continuation_steps: 2,
                        ..NonlinearBiasSolveConfig::default()
                    },
                    refresh_interval: None,
                },
                ..config
            },
        )
        .unwrap();
        let relative = (differential.bl_body_fitted_tm - constant.bl_body_fitted_tm).abs()
            / constant.bl_body_fitted_tm;
        eprintln!(
            "linear-limit differential reanalysis: constant={:.9}, differential={:.9}, rel={relative:e}",
            constant.bl_body_fitted_tm, differential.bl_body_fitted_tm
        );
        assert!(
            relative < 1.0e-6,
            "linear B-H curve reanalysis should match the constant model, rel {relative:e}"
        );
    }

    /// Sanity: the default low-carbon steel B-H curve gives a differential
    /// reanalysis below the linear `mu_r = 1000` model (saturation can only
    /// lower the effective permeability).
    #[test]
    fn saturated_steel_lowers_body_fitted_bl() {
        let driver = fixture_driver();
        let design_elements = motor_design_elements(&driver).unwrap();
        let physical = vec![1.0; design_elements.len()];
        let interpolation = InterpolationScheme::Ramp { penalty: 3.0 };
        let config = ProjectedReanalysisConfig {
            refinement_factors: vec![1],
            ..ProjectedReanalysisConfig::default()
        };
        let constant =
            reanalyze_physical_design(&driver, &physical, interpolation, 0.5, &config).unwrap();
        let saturated = reanalyze_physical_design(
            &driver,
            &physical,
            interpolation,
            0.5,
            &ProjectedReanalysisConfig {
                steel_permeability: SteelPermeability::DifferentialFromBias {
                    bh_curve: default_low_carbon_steel_bh_curve(),
                    solve: NonlinearBiasSolveConfig {
                        continuation_steps: 4,
                        ..NonlinearBiasSolveConfig::default()
                    },
                    refresh_interval: None,
                },
                ..config
            },
        )
        .unwrap();
        eprintln!(
            "saturation: linear={:.6}, differential={:.6} T·m",
            constant.bl_body_fitted_tm, saturated.bl_body_fitted_tm
        );
        assert!(saturated.bl_body_fitted_tm.is_finite());
        assert!(
            saturated.bl_body_fitted_tm <= constant.bl_body_fitted_tm * 1.000_001,
            "saturated steel should not exceed the linear BL"
        );
    }

    // ---- CP4 ----

    #[test]
    fn optimize_and_reanalyze_roundtrip() {
        let driver = fixture_driver();
        let topology = MotorTopologyConfig {
            max_iterations: 4,
            tolerance: 1.0e-12,
            steel_volume_fraction: 0.7,
            projection: Some(ProjectionContinuationConfig {
                beta_initial: 1.0,
                beta_max: 4.0,
                eta: 0.5,
                continuation_interval: 2,
            }),
            ..MotorTopologyConfig::default()
        };
        let (result, report) =
            optimize_and_reanalyze(&driver, &topology, &Default::default()).unwrap();
        assert_eq!(report.threshold, 0.5);
        assert_eq!(report.levels.len(), 5);
        assert!(report.bl_body_fitted_tm.is_finite());
        assert!(!result.bl_history_tm.is_empty());
    }
}
