//! Bounded in-process evaluation of canonical projected geometry.
//!
//! Native GEO/Python emitters still reject projection operations because Gmsh
//! has no portable field-evaluation contract. This module provides the
//! backend-neutral analytic consumer used by previews, optimization probes,
//! and QA: signed distance, iso-surface admission, and closest-point queries
//! for the canonical primitive/boolean subset. Unsupported operations fail
//! explicitly rather than silently producing backend-specific geometry.

use meshlang_ast::{
    CanonicalGeometryIr, EntityId, GeometryNode, GeometryOperation, IrArgumentName,
    IrBinaryOperator, IrValue, PrimitiveKind, ProjectionMethod,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

const MAX_EVALUATION_DEPTH: usize = 128;
const MAX_CLOSEST_POINT_ITERATIONS: usize = 32;
/// Floor applied to closest-point tolerances: an exact-zero query would
/// otherwise divide by zero or loop forever in the projected iteration.
/// Callers passing `tolerance = 0.0` get `1e-10`, surfaced per-evaluation
/// in [`ProjectionEvaluation::effective_tolerance`].
pub const MIN_CLOSEST_POINT_TOLERANCE: f64 = 1.0e-10;

/// Grid nodes per thread above which iso-surface SDF sampling parallelizes.
/// A fixed node count cannot be right on every machine: the point where pool
/// dispatch pays off moves with thread count and per-node SDF cost
/// (recursion depth varies per node), so the gate scales with
/// [`std::thread::available_parallelism`]. Starting point, not a calibrated
/// value: tune with `compiler_bench` on the target machine.
const GRID_NODES_PER_THREAD: usize = 256;

/// Entity lookup index hoisting the per-query `BTreeMap` build. Batch
/// callers (previews, probes, optimization sweeps) should build one index
/// and use the `_with_index` entry points instead of paying the rebuild
/// per point.
pub type EntityNodeIndex<'a> = BTreeMap<EntityId, &'a GeometryNode>;

/// Build the entity lookup index for [`evaluate_projection_with_index`]
/// and [`mesh_projection_iso_surface_with_index`].
pub fn entity_node_index(ir: &CanonicalGeometryIr) -> EntityNodeIndex<'_> {
    ir.entities.iter().map(|node| (node.id, node)).collect()
}

/// Machine-adaptive parallel gate for iso-surface grid sampling:
/// `nodes >= GRID_NODES_PER_THREAD * threads`. `SONIUM_MESHLANG_PAR_THRESHOLD`
/// overrides with an absolute node count for tuning and forced-path tests;
/// `SONIUM_SERIAL=1` forces the serial path. Single-threaded processes
/// always run serially.
fn use_projection_parallel(grid_nodes: usize) -> bool {
    if std::env::var("SONIUM_SERIAL")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
    {
        return false;
    }
    if let Some(threshold) = std::env::var("SONIUM_MESHLANG_PAR_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        return grid_nodes >= threshold;
    }
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    if threads < 2 {
        return false;
    }
    grid_nodes >= GRID_NODES_PER_THREAD.saturating_mul(threads).max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectionEvaluation {
    pub method: ProjectionMethod,
    pub signed_distance: f64,
    pub closest_point: Option<[f64; 3]>,
    pub on_iso_surface: bool,
    /// Tolerance actually used: `requested.max(MIN_CLOSEST_POINT_TOLERANCE)`.
    pub effective_tolerance: f64,
}

/// A bounded triangle mesh sampled from a canonical iso-surface projection.
///
/// Vertices are shared across grid edges and triangles are emitted in a
/// deterministic order. The mesh is an in-process consumer for previews and
/// optimization probes; native GEO/Python backends still require an explicit
/// Gmsh-side field/meshing contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionMesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub bounds: [[f64; 3]; 2],
    pub cells: [usize; 3],
}

/// Versioned, backend-neutral projection mesh artifact.
///
/// The artifact is deliberately separate from native GEO/Python emission: it
/// gives previews, optimization probes, and WASM clients a stable hand-off
/// without implying that Gmsh can evaluate an arbitrary canonical field.
pub const PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const MAX_PROJECTION_MESH_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionMeshArtifact {
    pub schema_version: u32,
    pub mesh: ProjectionMesh,
}

#[derive(Debug, Error)]
pub enum ProjectionArtifactError {
    #[error("projection mesh artifact JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "projection mesh artifact schema version {found} is unsupported (expected {expected})"
    )]
    UnsupportedSchema { found: u32, expected: u32 },
    #[error("invalid projection mesh artifact: {0}")]
    InvalidMesh(String),
    #[error(
        "projection mesh artifact has {bytes} serialized bytes, exceeding the {maximum}-byte limit"
    )]
    ArtifactTooLarge { bytes: usize, maximum: usize },
}

impl ProjectionMesh {
    /// Validate the bounded mesh contract before handing it to another
    /// process or serializing it.
    pub fn validate(&self) -> Result<(), ProjectionArtifactError> {
        if self.bounds.iter().flatten().any(|value| !value.is_finite())
            || self.bounds[0]
                .iter()
                .zip(self.bounds[1].iter())
                .any(|(lower, upper)| upper <= lower)
        {
            return Err(ProjectionArtifactError::InvalidMesh(
                "bounds must be finite with positive extents".into(),
            ));
        }

        if self.cells.contains(&0) {
            return Err(ProjectionArtifactError::InvalidMesh(
                "grid dimensions must be non-zero".into(),
            ));
        }
        let total_cells = self
            .cells
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .ok_or_else(|| ProjectionArtifactError::InvalidMesh("grid size overflows".into()))?;
        if self
            .cells
            .iter()
            .any(|count| *count > MAX_ISO_SURFACE_CELLS_PER_AXIS)
            || total_cells > MAX_ISO_SURFACE_CELLS
        {
            return Err(ProjectionArtifactError::InvalidMesh(format!(
                "grid has {total_cells} cells, exceeding the resource limit"
            )));
        }

        if let Some((index, _)) = self
            .vertices
            .iter()
            .enumerate()
            .find(|(_, point)| point.iter().any(|value| !value.is_finite()))
        {
            return Err(ProjectionArtifactError::InvalidMesh(format!(
                "vertex {index} is non-finite"
            )));
        }

        for (triangle_index, triangle) in self.triangles.iter().enumerate() {
            if triangle
                .iter()
                .any(|index| (*index as usize) >= self.vertices.len())
            {
                return Err(ProjectionArtifactError::InvalidMesh(format!(
                    "triangle {triangle_index} references a missing vertex"
                )));
            }
            if triangle[0] == triangle[1]
                || triangle[1] == triangle[2]
                || triangle[0] == triangle[2]
            {
                return Err(ProjectionArtifactError::InvalidMesh(format!(
                    "triangle {triangle_index} is degenerate"
                )));
            }
        }

        Ok(())
    }
}

impl ProjectionMeshArtifact {
    pub fn from_mesh(mesh: ProjectionMesh) -> Result<Self, ProjectionArtifactError> {
        mesh.validate()?;
        Ok(Self {
            schema_version: PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
            mesh,
        })
    }

    pub fn validate(&self) -> Result<(), ProjectionArtifactError> {
        if self.schema_version != PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION {
            return Err(ProjectionArtifactError::UnsupportedSchema {
                found: self.schema_version,
                expected: PROJECTION_MESH_ARTIFACT_SCHEMA_VERSION,
            });
        }
        self.mesh.validate()
    }

    pub fn to_json(&self) -> Result<String, ProjectionArtifactError> {
        self.validate()?;
        let json = serde_json::to_string(self)?;
        if json.len() > MAX_PROJECTION_MESH_ARTIFACT_BYTES {
            return Err(ProjectionArtifactError::ArtifactTooLarge {
                bytes: json.len(),
                maximum: MAX_PROJECTION_MESH_ARTIFACT_BYTES,
            });
        }
        Ok(json)
    }

    pub fn from_json(json: &str) -> Result<Self, ProjectionArtifactError> {
        if json.len() > MAX_PROJECTION_MESH_ARTIFACT_BYTES {
            return Err(ProjectionArtifactError::ArtifactTooLarge {
                bytes: json.len(),
                maximum: MAX_PROJECTION_MESH_ARTIFACT_BYTES,
            });
        }
        let artifact: Self = serde_json::from_str(json)?;
        artifact.validate()?;
        Ok(artifact)
    }
}

const MAX_ISO_SURFACE_CELLS: usize = 262_144;
const MAX_ISO_SURFACE_CELLS_PER_AXIS: usize = 64;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ProjectionError {
    #[error("projection point must contain only finite coordinates")]
    NonFinitePoint,
    #[error("projection tolerance must be finite and non-negative")]
    InvalidTolerance,
    #[error("projection entity {0:?} does not exist")]
    UnknownEntity(EntityId),
    #[error("entity {0:?} is not a projection operation")]
    NotProjection(EntityId),
    #[error("entity {entity:?} has no projection input")]
    MissingInput { entity: EntityId },
    #[error("cyclic geometry dependency at entity {0:?}")]
    Cycle(EntityId),
    #[error("projection evaluation exceeded the maximum geometry depth")]
    DepthLimit,
    #[error("entity {entity:?} uses unsupported geometry operation {operation}")]
    UnsupportedOperation { entity: EntityId, operation: String },
    #[error("entity {entity:?} has invalid {argument:?} argument: {reason}")]
    InvalidArgument {
        entity: EntityId,
        argument: IrArgumentName,
        reason: String,
    },
    #[error("entity {entity:?} references a dynamic value in {argument:?}")]
    DynamicArgument {
        entity: EntityId,
        argument: IrArgumentName,
    },
    #[error("closest-point iteration did not converge (residual {residual})")]
    ClosestPointNotConverged { residual: f64 },
    #[error("projection evaluation produced a non-finite result")]
    NonFiniteResult,
    #[error("projection entity {entity:?} uses {method:?}, not iso-surface sampling")]
    NotIsoSurface {
        entity: EntityId,
        method: ProjectionMethod,
    },
    #[error("iso-surface grid dimensions must be non-zero")]
    InvalidGrid,
    #[error("iso-surface grid has {cells} cells, exceeding the {maximum}-cell resource limit")]
    GridTooLarge { cells: usize, maximum: usize },
    #[error("iso-surface bounds must be finite and have positive extents")]
    InvalidBounds,
}

/// Evaluate a canonical `GeometryOperation::Project` node at one point.
pub fn evaluate_projection(
    ir: &CanonicalGeometryIr,
    projection: EntityId,
    point: [f64; 3],
    tolerance: f64,
) -> Result<ProjectionEvaluation, ProjectionError> {
    let nodes = entity_node_index(ir);
    evaluate_projection_with_index(ir, &nodes, projection, point, tolerance)
}

/// [`evaluate_projection`] with a caller-provided [`entity_node_index`].
/// Batch callers evaluating many points against one IR build the index once
/// and reuse it here instead of rebuilding the entity map per query.
pub fn evaluate_projection_with_index(
    ir: &CanonicalGeometryIr,
    nodes: &EntityNodeIndex<'_>,
    projection: EntityId,
    point: [f64; 3],
    tolerance: f64,
) -> Result<ProjectionEvaluation, ProjectionError> {
    if point.iter().any(|value| !value.is_finite()) {
        return Err(ProjectionError::NonFinitePoint);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ProjectionError::InvalidTolerance);
    }
    let node = nodes
        .get(&projection)
        .ok_or(ProjectionError::UnknownEntity(projection))?;
    let method = match node.operation {
        GeometryOperation::Project { method } => method,
        _ => return Err(ProjectionError::NotProjection(projection)),
    };
    let input = node
        .inputs
        .first()
        .copied()
        .ok_or(ProjectionError::MissingInput { entity: projection })?;
    let mut stack = BTreeSet::new();
    let signed_distance = evaluate_sdf(ir, nodes, input, point, &mut stack, 0)?;
    if !signed_distance.is_finite() {
        return Err(ProjectionError::NonFiniteResult);
    }
    let effective_tolerance = tolerance.max(MIN_CLOSEST_POINT_TOLERANCE);
    let closest_point = if method == ProjectionMethod::ClosestPoint {
        Some(closest_point(ir, nodes, input, point, effective_tolerance)?)
    } else {
        None
    };
    Ok(ProjectionEvaluation {
        method,
        signed_distance,
        closest_point,
        on_iso_surface: signed_distance.abs() <= tolerance,
        effective_tolerance,
    })
}

/// Sample a canonical `GeometryOperation::Project { method: IsoSurface }`
/// node with a deterministic marching-tetrahedra grid.
///
/// The operation is deliberately bounded by both per-axis and total-cell
/// limits. Each cell is split into six tetrahedra, avoiding a large lookup
/// table while retaining a watertight shared-edge vertex map.
pub fn mesh_projection_iso_surface(
    ir: &CanonicalGeometryIr,
    projection: EntityId,
    bounds: [[f64; 3]; 2],
    cells: [usize; 3],
    tolerance: f64,
) -> Result<ProjectionMesh, ProjectionError> {
    let nodes = entity_node_index(ir);
    mesh_projection_iso_surface_with_index(ir, &nodes, projection, bounds, cells, tolerance)
}

/// [`mesh_projection_iso_surface`] with a caller-provided
/// [`entity_node_index`]. Batch callers meshing many projections against one
/// IR build the index once and reuse it here.
pub fn mesh_projection_iso_surface_with_index(
    ir: &CanonicalGeometryIr,
    nodes: &EntityNodeIndex<'_>,
    projection: EntityId,
    bounds: [[f64; 3]; 2],
    cells: [usize; 3],
    tolerance: f64,
) -> Result<ProjectionMesh, ProjectionError> {
    if bounds.iter().flatten().any(|value| !value.is_finite())
        || bounds[0]
            .iter()
            .zip(bounds[1].iter())
            .any(|(lower, upper)| upper <= lower)
    {
        return Err(ProjectionError::InvalidBounds);
    }
    if cells.contains(&0) {
        return Err(ProjectionError::InvalidGrid);
    }
    let total_cells = cells
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(ProjectionError::GridTooLarge {
            cells: usize::MAX,
            maximum: MAX_ISO_SURFACE_CELLS,
        })?;
    if cells
        .iter()
        .any(|count| *count > MAX_ISO_SURFACE_CELLS_PER_AXIS)
        || total_cells > MAX_ISO_SURFACE_CELLS
    {
        return Err(ProjectionError::GridTooLarge {
            cells: total_cells,
            maximum: MAX_ISO_SURFACE_CELLS,
        });
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ProjectionError::InvalidTolerance);
    }

    let projection_node = nodes
        .get(&projection)
        .ok_or(ProjectionError::UnknownEntity(projection))?;
    let method = match projection_node.operation {
        GeometryOperation::Project { method } => method,
        _ => return Err(ProjectionError::NotProjection(projection)),
    };
    if method != ProjectionMethod::IsoSurface {
        return Err(ProjectionError::NotIsoSurface {
            entity: projection,
            method,
        });
    }
    let input = projection_node
        .inputs
        .first()
        .copied()
        .ok_or(ProjectionError::MissingInput { entity: projection })?;

    let grid_shape = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
    let grid_nodes = grid_shape
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(ProjectionError::GridTooLarge {
            cells: total_cells,
            maximum: MAX_ISO_SURFACE_CELLS,
        })?;
    if grid_nodes > MAX_ISO_SURFACE_CELLS + 3 * MAX_ISO_SURFACE_CELLS_PER_AXIS.pow(2) {
        return Err(ProjectionError::GridTooLarge {
            cells: total_cells,
            maximum: MAX_ISO_SURFACE_CELLS,
        });
    }

    let step = [
        (bounds[1][0] - bounds[0][0]) / cells[0] as f64,
        (bounds[1][1] - bounds[0][1]) / cells[1] as f64,
        (bounds[1][2] - bounds[0][2]) / cells[2] as f64,
    ];
    // Grid coordinates in the same z-outer/x-inner order the serial loop
    // used, so `values[i]` keeps its marching-tetrahedra meaning on both
    // paths. Each node evaluation is independent (fresh cycle stack), and
    // indexed collect preserves order — the parallel path is value-identical.
    // Note: unlike the serial loop, the parallel path samples every node
    // before reporting the first non-finite value.
    let coords: Vec<(usize, usize, usize)> = (0..=cells[2])
        .flat_map(|z| (0..=cells[1]).flat_map(move |y| (0..=cells[0]).map(move |x| (x, y, z))))
        .collect();
    debug_assert_eq!(coords.len(), grid_nodes);
    let sample_node = |(x, y, z): (usize, usize, usize)| {
        let point = grid_point(bounds, step, x, y, z);
        evaluate_sdf(ir, nodes, input, point, &mut BTreeSet::new(), 0)
    };
    let values: Vec<f64> = if use_projection_parallel(coords.len()) {
        use rayon::prelude::*;
        coords
            .into_par_iter()
            .map(&sample_node)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        coords
            .into_iter()
            .map(&sample_node)
            .collect::<Result<Vec<_>, _>>()?
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ProjectionError::NonFiniteResult);
    }

    let mut mesh = ProjectionMesh {
        vertices: Vec::new(),
        triangles: Vec::new(),
        bounds,
        cells,
    };
    let mut edge_vertices = BTreeMap::<(usize, usize), u32>::new();
    let mut triangle_keys = BTreeSet::<[u32; 3]>::new();
    const TETRAHEDRA: [[usize; 4]; 6] = [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ];
    for z in 0..cells[2] {
        for y in 0..cells[1] {
            for x in 0..cells[0] {
                let corners = cube_corner_indices(cells, x, y, z);
                let positions = corners.map(|corner| {
                    let (gx, gy, gz) = grid_coordinates(corner, grid_shape);
                    grid_point(bounds, step, gx, gy, gz)
                });
                let corner_values = corners.map(|corner| values[corner]);
                for tetrahedron in TETRAHEDRA {
                    emit_tetrahedron(
                        &tetrahedron,
                        &corners,
                        &positions,
                        &corner_values,
                        &mut edge_vertices,
                        &mut triangle_keys,
                        &mut mesh,
                    );
                }
            }
        }
    }
    orient_mesh_triangles(&mut mesh);
    Ok(mesh)
}

fn grid_point(bounds: [[f64; 3]; 2], step: [f64; 3], x: usize, y: usize, z: usize) -> [f64; 3] {
    [
        bounds[0][0] + step[0] * x as f64,
        bounds[0][1] + step[1] * y as f64,
        bounds[0][2] + step[2] * z as f64,
    ]
}

fn grid_coordinates(index: usize, shape: [usize; 3]) -> (usize, usize, usize) {
    let x = index % shape[0];
    let plane = shape[0] * shape[1];
    let z = index / plane;
    let y = (index % plane) / shape[0];
    (x, y, z)
}

fn cube_corner_indices(cells: [usize; 3], x: usize, y: usize, z: usize) -> [usize; 8] {
    let shape = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
    let index = |x: usize, y: usize, z: usize| (z * shape[1] + y) * shape[0] + x;
    [
        index(x, y, z),
        index(x + 1, y, z),
        index(x + 1, y + 1, z),
        index(x, y + 1, z),
        index(x, y, z + 1),
        index(x + 1, y, z + 1),
        index(x + 1, y + 1, z + 1),
        index(x, y + 1, z + 1),
    ]
}

#[allow(clippy::too_many_arguments)]
fn emit_tetrahedron(
    tetrahedron: &[usize; 4],
    corners: &[usize; 8],
    positions: &[[f64; 3]; 8],
    values: &[f64; 8],
    edge_vertices: &mut BTreeMap<(usize, usize), u32>,
    triangle_keys: &mut BTreeSet<[u32; 3]>,
    mesh: &mut ProjectionMesh,
) {
    let inside = tetrahedron.map(|index| values[index] <= 0.0);
    let inside_count = inside.iter().filter(|value| **value).count();
    if inside_count == 0 || inside_count == 4 {
        return;
    }
    let (sum, count) = tetrahedron
        .iter()
        .zip(inside.iter())
        .filter_map(|(index, is_inside)| is_inside.then_some(positions[*index]))
        .fold(([0.0; 3], 0usize), |(sum, count), point| {
            (
                [sum[0] + point[0], sum[1] + point[1], sum[2] + point[2]],
                count + 1,
            )
        });
    let inside_center = [
        sum[0] / count as f64,
        sum[1] / count as f64,
        sum[2] / count as f64,
    ];
    let mut intersections = Vec::<(u32, [f64; 3])>::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            if inside[left] == inside[right] {
                continue;
            }
            let left_index = tetrahedron[left];
            let right_index = tetrahedron[right];
            let key = if corners[left_index] < corners[right_index] {
                (corners[left_index], corners[right_index])
            } else {
                (corners[right_index], corners[left_index])
            };
            let vertex = *edge_vertices.entry(key).or_insert_with(|| {
                let left_value = values[left_index];
                let right_value = values[right_index];
                let denominator = left_value - right_value;
                let t = if denominator == 0.0 {
                    0.5
                } else {
                    (left_value / denominator).clamp(0.0, 1.0)
                };
                let a = positions[left_index];
                let b = positions[right_index];
                let point = [
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ];
                let index = mesh.vertices.len() as u32;
                mesh.vertices.push(point);
                index
            });
            intersections.push((vertex, mesh.vertices[vertex as usize]));
        }
    }
    intersections.sort_by_key(|(index, _)| *index);
    intersections.dedup_by_key(|(index, _)| *index);
    if intersections.len() < 3 {
        return;
    }
    push_oriented_triangle(
        [intersections[0], intersections[1], intersections[2]],
        inside_center,
        triangle_keys,
        mesh,
    );
    if intersections.len() == 4 {
        push_oriented_triangle(
            [intersections[0], intersections[2], intersections[3]],
            inside_center,
            triangle_keys,
            mesh,
        );
    }
}

fn push_oriented_triangle(
    triangle: [(u32, [f64; 3]); 3],
    inside_center: [f64; 3],
    triangle_keys: &mut BTreeSet<[u32; 3]>,
    mesh: &mut ProjectionMesh,
) {
    let a = triangle[0].1;
    let b = triangle[1].1;
    let c = triangle[2].1;
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let center = [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ];
    let outward = [
        center[0] - inside_center[0],
        center[1] - inside_center[1],
        center[2] - inside_center[2],
    ];
    let mut indices = [triangle[0].0, triangle[1].0, triangle[2].0];
    if normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2] < 0.0 {
        indices.swap(1, 2);
    }
    if indices[0] == indices[1] || indices[1] == indices[2] || indices[0] == indices[2] {
        return;
    }
    let mut key = indices;
    key.sort_unstable();
    if triangle_keys.insert(key) {
        mesh.triangles.push(indices);
    }
}

/// Propagate a consistent winding across each manifold triangle component.
///
/// Marching tetrahedra orients each cut triangle against its local tetrahedron
/// interior. At shared grid edges, that local choice can disagree at a cell
/// boundary (especially for two-inside/two-outside cases). Native volume
/// consumers require opposite directed edge uses, so repair the winding using
/// deterministic breadth-first propagation while leaving non-manifold edges
/// available for the preview/native-surface adapter to reject explicitly.
fn orient_mesh_triangles(mesh: &mut ProjectionMesh) {
    let mut edges = BTreeMap::<(u32, u32), Vec<(usize, bool)>>::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        for edge_index in 0..3 {
            let from = triangle[edge_index];
            let to = triangle[(edge_index + 1) % 3];
            edges
                .entry((from.min(to), from.max(to)))
                .or_default()
                .push((triangle_index, from < to));
        }
    }

    let mut flips = vec![None; mesh.triangles.len()];
    for seed in 0..mesh.triangles.len() {
        if flips[seed].is_some() {
            continue;
        }
        flips[seed] = Some(false);
        let mut queue = VecDeque::from([seed]);
        while let Some(triangle_index) = queue.pop_front() {
            let triangle = mesh.triangles[triangle_index];
            let triangle_flipped = flips[triangle_index].unwrap_or(false);
            for edge_index in 0..3 {
                let from = triangle[edge_index];
                let to = triangle[(edge_index + 1) % 3];
                let key = (from.min(to), from.max(to));
                let Some(uses) = edges.get(&key) else {
                    continue;
                };
                if uses.len() != 2 {
                    continue;
                }
                let current_forward = (from < to) ^ triangle_flipped;
                for &(neighbor, neighbor_forward) in uses {
                    if neighbor == triangle_index {
                        continue;
                    }
                    let expected_flip = neighbor_forward == current_forward;
                    match flips[neighbor] {
                        Some(existing) if existing != expected_flip => {
                            // A non-orientable/non-manifold fragment is left
                            // untouched; strict native consumers reject it.
                        }
                        Some(_) => {}
                        None => {
                            flips[neighbor] = Some(expected_flip);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
    }

    for (triangle, flip) in mesh.triangles.iter_mut().zip(flips) {
        if flip.unwrap_or(false) {
            triangle.swap(1, 2);
        }
    }
}

fn evaluate_sdf(
    ir: &CanonicalGeometryIr,
    nodes: &BTreeMap<EntityId, &GeometryNode>,
    entity: EntityId,
    point: [f64; 3],
    stack: &mut BTreeSet<EntityId>,
    depth: usize,
) -> Result<f64, ProjectionError> {
    if depth > MAX_EVALUATION_DEPTH {
        return Err(ProjectionError::DepthLimit);
    }
    if !stack.insert(entity) {
        return Err(ProjectionError::Cycle(entity));
    }
    let node = nodes
        .get(&entity)
        .ok_or(ProjectionError::UnknownEntity(entity))?;
    let result = match &node.operation {
        GeometryOperation::Primitive { kind } => primitive_sdf(ir, node, *kind, point),
        GeometryOperation::Boolean { kind } => {
            if node.inputs.is_empty() {
                Err(ProjectionError::MissingInput { entity })
            } else {
                let mut distances = node
                    .inputs
                    .iter()
                    .map(|input| evaluate_sdf(ir, nodes, *input, point, stack, depth + 1));
                let first = distances.next().expect("non-empty inputs")?;
                distances.try_fold(first, |acc, distance| {
                    let value = distance?;
                    Ok(match kind {
                        meshlang_ast::BooleanOperation::Union => acc.min(value),
                        meshlang_ast::BooleanOperation::Intersection => acc.max(value),
                        meshlang_ast::BooleanOperation::Difference => acc.max(-value),
                    })
                })
            }
        }
        GeometryOperation::Array => {
            if node.inputs.is_empty() {
                Err(ProjectionError::MissingInput { entity })
            } else {
                node.inputs
                    .iter()
                    .map(|input| evaluate_sdf(ir, nodes, *input, point, stack, depth + 1))
                    .try_fold(f64::INFINITY, |acc, distance| Ok(acc.min(distance?)))
            }
        }
        GeometryOperation::Transform { kind } => {
            let input = node
                .inputs
                .first()
                .copied()
                .ok_or(ProjectionError::MissingInput { entity })?;
            let (local_point, scale) = transform_point(node, *kind, point)?;
            Ok(evaluate_sdf(ir, nodes, input, local_point, stack, depth + 1)? * scale)
        }
        GeometryOperation::Project { .. } => {
            let input = node
                .inputs
                .first()
                .copied()
                .ok_or(ProjectionError::MissingInput { entity })?;
            evaluate_sdf(ir, nodes, input, point, stack, depth + 1)
        }
        operation => Err(ProjectionError::UnsupportedOperation {
            entity,
            operation: format!("{operation:?}"),
        }),
    };
    stack.remove(&entity);
    result
}

fn primitive_sdf(
    ir: &CanonicalGeometryIr,
    node: &GeometryNode,
    kind: PrimitiveKind,
    point: [f64; 3],
) -> Result<f64, ProjectionError> {
    match kind {
        PrimitiveKind::Point => {
            let origin = vector_argument(ir, node, IrArgumentName::Origin, [0.0, 0.0, 0.0])?;
            Ok(distance(point, origin))
        }
        PrimitiveKind::Sphere => {
            let center = vector_argument(ir, node, IrArgumentName::Center, [0.0, 0.0, 0.0])?;
            let radius = scalar_argument(ir, node, IrArgumentName::Radius)?;
            positive_radius(node, IrArgumentName::Radius, radius)?;
            Ok(distance(point, center) - radius)
        }
        PrimitiveKind::Box => {
            let origin = vector_argument(ir, node, IrArgumentName::Origin, [0.0, 0.0, 0.0])?;
            let size = size_argument(ir, node)?;
            if size.iter().any(|value| *value <= 0.0) {
                return Err(ProjectionError::InvalidArgument {
                    entity: node.id,
                    argument: IrArgumentName::Size,
                    reason: "box dimensions must be positive".into(),
                });
            }
            let center = [
                origin[0] + size[0] * 0.5,
                origin[1] + size[1] * 0.5,
                origin[2] + size[2] * 0.5,
            ];
            let half = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
            let q = [
                (point[0] - center[0]).abs() - half[0],
                (point[1] - center[1]).abs() - half[1],
                (point[2] - center[2]).abs() - half[2],
            ];
            let outside = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
            let outside_distance = norm(outside);
            let inside_distance = q[0].max(q[1]).max(q[2]).min(0.0);
            Ok(outside_distance + inside_distance)
        }
        PrimitiveKind::Cylinder => {
            let center = vector_argument(ir, node, IrArgumentName::Origin, [0.0, 0.0, 0.0])?;
            let radius = scalar_argument(ir, node, IrArgumentName::Radius)?;
            let height = scalar_argument(ir, node, IrArgumentName::Height)?;
            positive_radius(node, IrArgumentName::Radius, radius)?;
            positive_radius(node, IrArgumentName::Height, height)?;
            let z = point[2] - (center[2] + height * 0.5);
            let radial = ((point[0] - center[0]).powi(2) + (point[1] - center[1]).powi(2)).sqrt();
            let q = [radial - radius, z.abs() - height * 0.5];
            let outside = [q[0].max(0.0), q[1].max(0.0)];
            Ok((outside[0].powi(2) + outside[1].powi(2)).sqrt() + q[0].max(q[1]).min(0.0))
        }
        _ => Err(ProjectionError::UnsupportedOperation {
            entity: node.id,
            operation: format!("primitive {kind:?}"),
        }),
    }
}

fn transform_point(
    node: &GeometryNode,
    kind: meshlang_ast::TransformOperation,
    point: [f64; 3],
) -> Result<([f64; 3], f64), ProjectionError> {
    match kind {
        meshlang_ast::TransformOperation::Translate => {
            let offset = vector_argument_from_node(node, IrArgumentName::Direction)?;
            Ok((
                [
                    point[0] - offset[0],
                    point[1] - offset[1],
                    point[2] - offset[2],
                ],
                1.0,
            ))
        }
        meshlang_ast::TransformOperation::Scale => {
            let scale = scalar_argument_from_node(node, IrArgumentName::Scale)?;
            positive_radius(node, IrArgumentName::Scale, scale)?;
            Ok((
                [point[0] / scale, point[1] / scale, point[2] / scale],
                scale,
            ))
        }
        _ => Err(ProjectionError::UnsupportedOperation {
            entity: node.id,
            operation: format!("transform {kind:?}"),
        }),
    }
}

fn closest_point(
    ir: &CanonicalGeometryIr,
    nodes: &BTreeMap<EntityId, &GeometryNode>,
    entity: EntityId,
    point: [f64; 3],
    tolerance: f64,
) -> Result<[f64; 3], ProjectionError> {
    let mut current = point;
    for _ in 0..MAX_CLOSEST_POINT_ITERATIONS {
        let distance = evaluate_sdf(ir, nodes, entity, current, &mut BTreeSet::new(), 0)?;
        if distance.abs() <= tolerance {
            return Ok(current);
        }
        let epsilon = 1.0e-6 * (1.0 + norm(current));
        let mut gradient = [0.0; 3];
        for axis in 0..3 {
            let mut plus = current;
            let mut minus = current;
            plus[axis] += epsilon;
            minus[axis] -= epsilon;
            gradient[axis] = (evaluate_sdf(ir, nodes, entity, plus, &mut BTreeSet::new(), 0)?
                - evaluate_sdf(ir, nodes, entity, minus, &mut BTreeSet::new(), 0)?)
                / (2.0 * epsilon);
        }
        let gradient_norm_sq = gradient.iter().map(|value| value * value).sum::<f64>();
        if !gradient_norm_sq.is_finite() || gradient_norm_sq <= 1.0e-20 {
            return Err(ProjectionError::NonFiniteResult);
        }
        let step = distance / gradient_norm_sq;
        for axis in 0..3 {
            current[axis] -= step * gradient[axis];
        }
        if current.iter().any(|value| !value.is_finite()) {
            return Err(ProjectionError::NonFiniteResult);
        }
    }
    let residual = evaluate_sdf(ir, nodes, entity, current, &mut BTreeSet::new(), 0)?.abs();
    Err(ProjectionError::ClosestPointNotConverged { residual })
}

fn scalar_argument(
    ir: &CanonicalGeometryIr,
    node: &GeometryNode,
    argument: IrArgumentName,
) -> Result<f64, ProjectionError> {
    let value = node
        .arguments
        .iter()
        .find(|entry| entry.name == argument)
        .map(|entry| &entry.value)
        .ok_or(ProjectionError::InvalidArgument {
            entity: node.id,
            argument,
            reason: "missing scalar".into(),
        })?;
    scalar_value(ir, node.id, argument, value)
}

fn scalar_argument_from_node(
    node: &GeometryNode,
    argument: IrArgumentName,
) -> Result<f64, ProjectionError> {
    let value = node
        .arguments
        .iter()
        .find(|entry| entry.name == argument)
        .map(|entry| &entry.value)
        .ok_or(ProjectionError::InvalidArgument {
            entity: node.id,
            argument,
            reason: "missing scalar".into(),
        })?;
    match value {
        IrValue::Integer(value) => Ok(*value as f64),
        IrValue::Scalar { value, .. } if value.is_finite() => Ok(*value),
        _ => Err(ProjectionError::DynamicArgument {
            entity: node.id,
            argument,
        }),
    }
}

fn scalar_value(
    ir: &CanonicalGeometryIr,
    entity: EntityId,
    argument: IrArgumentName,
    value: &IrValue,
) -> Result<f64, ProjectionError> {
    let scalar = match value {
        IrValue::Integer(value) => *value as f64,
        IrValue::Scalar { value, .. } => *value,
        IrValue::Parameter(parameter) => {
            let definition = ir
                .parameters
                .iter()
                .find(|entry| entry.id == *parameter)
                .ok_or(ProjectionError::DynamicArgument { entity, argument })?;
            return scalar_value(ir, entity, argument, &definition.value);
        }
        IrValue::Unary {
            op: meshlang_ast::IrUnaryOperator::Negate,
            operand,
        } => -scalar_value(ir, entity, argument, operand)?,
        IrValue::Binary { op, left, right } => {
            let left = scalar_value(ir, entity, argument, left)?;
            let right = scalar_value(ir, entity, argument, right)?;
            match op {
                IrBinaryOperator::Add => left + right,
                IrBinaryOperator::Subtract => left - right,
                IrBinaryOperator::Multiply => left * right,
                IrBinaryOperator::Divide if right != 0.0 => left / right,
                IrBinaryOperator::Power => left.powf(right),
                _ => {
                    return Err(ProjectionError::DynamicArgument { entity, argument });
                }
            }
        }
        _ => {
            return Err(ProjectionError::DynamicArgument { entity, argument });
        }
    };
    if scalar.is_finite() {
        Ok(scalar)
    } else {
        Err(ProjectionError::InvalidArgument {
            entity,
            argument,
            reason: "value is non-finite".into(),
        })
    }
}

fn vector_argument(
    ir: &CanonicalGeometryIr,
    node: &GeometryNode,
    argument: IrArgumentName,
    default: [f64; 3],
) -> Result<[f64; 3], ProjectionError> {
    let Some(value) = node
        .arguments
        .iter()
        .find(|entry| entry.name == argument)
        .map(|entry| &entry.value)
    else {
        return Ok(default);
    };
    vector_value(ir, node.id, argument, value)
}

fn vector_argument_from_node(
    node: &GeometryNode,
    argument: IrArgumentName,
) -> Result<[f64; 3], ProjectionError> {
    let value = node
        .arguments
        .iter()
        .find(|entry| entry.name == argument)
        .map(|entry| &entry.value)
        .ok_or(ProjectionError::InvalidArgument {
            entity: node.id,
            argument,
            reason: "missing vector".into(),
        })?;
    vector_value_from_node(node.id, argument, value)
}

fn vector_value(
    ir: &CanonicalGeometryIr,
    entity: EntityId,
    argument: IrArgumentName,
    value: &IrValue,
) -> Result<[f64; 3], ProjectionError> {
    match value {
        IrValue::Vector(values) if values.len() == 3 => Ok([
            scalar_value(ir, entity, argument, &values[0])?,
            scalar_value(ir, entity, argument, &values[1])?,
            scalar_value(ir, entity, argument, &values[2])?,
        ]),
        IrValue::Parameter(parameter) => {
            let definition = ir
                .parameters
                .iter()
                .find(|entry| entry.id == *parameter)
                .ok_or(ProjectionError::DynamicArgument { entity, argument })?;
            vector_value(ir, entity, argument, &definition.value)
        }
        _ => Err(ProjectionError::DynamicArgument { entity, argument }),
    }
}

fn vector_value_from_node(
    entity: EntityId,
    argument: IrArgumentName,
    value: &IrValue,
) -> Result<[f64; 3], ProjectionError> {
    match value {
        IrValue::Vector(values) if values.len() == 3 => values
            .iter()
            .map(|value| match value {
                IrValue::Integer(value) => Ok(*value as f64),
                IrValue::Scalar { value, .. } if value.is_finite() => Ok(*value),
                _ => Err(ProjectionError::DynamicArgument { entity, argument }),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|values| [values[0], values[1], values[2]]),
        _ => Err(ProjectionError::DynamicArgument { entity, argument }),
    }
}

fn size_argument(
    ir: &CanonicalGeometryIr,
    node: &GeometryNode,
) -> Result<[f64; 3], ProjectionError> {
    let entries = node
        .arguments
        .iter()
        .filter(|entry| entry.name == IrArgumentName::Size)
        .collect::<Vec<_>>();
    if entries.len() == 1 {
        if let IrValue::Vector(_) = &entries[0].value {
            return vector_value(ir, node.id, IrArgumentName::Size, &entries[0].value);
        }
    }
    let values = entries
        .iter()
        .map(|entry| scalar_value(ir, node.id, IrArgumentName::Size, &entry.value))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() == 3 {
        Ok([values[0], values[1], values[2]])
    } else {
        Err(ProjectionError::InvalidArgument {
            entity: node.id,
            argument: IrArgumentName::Size,
            reason: "expected three scalar dimensions or one 3-vector".into(),
        })
    }
}

fn positive_radius(
    node: &GeometryNode,
    argument: IrArgumentName,
    value: f64,
) -> Result<(), ProjectionError> {
    if value > 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(ProjectionError::InvalidArgument {
            entity: node.id,
            argument,
            reason: "value must be finite and positive".into(),
        })
    }
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    norm([left[0] - right[0], left[1] - right[1], left[2] - right[2]])
}

fn norm(value: [f64; 3]) -> f64 {
    value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}
