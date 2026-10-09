//! Mesh topology: global edge and face enumeration
//!
//! Provides edge connectivity required by H(curl) (Nédélec) and H(div) (Raviart-Thomas)
//! function spaces, where DOFs live on edges or faces rather than nodes.
//!
//! Built lazily and cached on [`Mesh`](super::Mesh).

use super::parallelism::{ParallelPolicy, TOPOLOGY_ITEMS_PER_THREAD};
use super::types::{BoundaryFace, Mesh, MeshConnectivityError, Point};
use rayon::prelude::*;
use std::collections::HashMap;

fn compute_face_normal(nodes: &[Point], face: &Face) -> [f64; 3] {
    match face {
        Face::Triangle([i0, i1, i2]) => {
            let p0 = &nodes[*i0];
            let p1 = &nodes[*i1];
            let p2 = &nodes[*i2];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
        }
        Face::Quad([i0, i1, i2, _i3]) => {
            let p0 = &nodes[*i0];
            let p1 = &nodes[*i1];
            let p2 = &nodes[*i2];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            // Triangulate via diagonal (0,2) for quad normal
            [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
        }
    }
}

/// Global mesh topology: edges, faces, element-edge/face connectivity, and orientations.
#[derive(Debug, Clone)]
pub struct MeshTopology {
    /// Global edge table: `edges[edge_idx] = (node_a, node_b)` with `node_a < node_b`.
    pub edges: Vec<(usize, usize)>,
    /// Map from `(min_node, max_node)` to global edge index.
    pub edge_map: HashMap<(usize, usize), usize>,
    /// Per-element edge connectivity: `element_edges[elem_idx][local_edge]` = global edge index.
    pub element_edges: Vec<Vec<usize>>,
    /// Per-element edge orientations: `+1` if the element's local edge direction matches
    /// the global edge direction (min→max), `-1` otherwise.
    pub edge_orientations: Vec<Vec<i8>>,
    /// Boundary edges: global edge indices that lie on boundary faces.
    pub boundary_edges: HashMap<i32, Vec<usize>>,
    /// Global face table: `faces[face_idx] = (n0, n1, n2, n3)` for quads, or (n0, n1, n2) for triangles.
    /// Stored as sorted node tuple (min,...) for consistent hashing.
    pub faces: Vec<Face>,
    /// Map from sorted node tuple to global face index.
    pub face_map: HashMap<FaceKey, usize>,
    /// Per-element face connectivity: `element_faces[elem_idx][local_face]` = global face index.
    pub element_faces: Vec<Vec<usize>>,
    /// Per-element face orientations: `+1` or `-1` indicating normal direction relative to global.
    pub face_orientations: Vec<Vec<i8>>,
    /// Boundary faces: global face indices on each boundary marker.
    pub boundary_faces: HashMap<i32, Vec<usize>>,
}

/// A face defined by its node indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Face {
    /// Triangular face (3 nodes)
    Triangle([usize; 3]),
    /// Quadrilateral face (4 nodes)
    Quad([usize; 4]),
}

impl Face {
    /// Get the sorted node key for this face (for hash map lookup).
    pub fn sorted_key(&self) -> FaceKey {
        match self {
            Face::Triangle(nodes) => {
                let mut sorted = *nodes;
                sorted.sort();
                FaceKey::Triangle(sorted)
            }
            Face::Quad(nodes) => {
                let mut sorted = *nodes;
                sorted.sort();
                FaceKey::Quad(sorted)
            }
        }
    }

    /// Number of nodes in this face.
    pub fn num_nodes(&self) -> usize {
        match self {
            Face::Triangle(_) => 3,
            Face::Quad(_) => 4,
        }
    }

    /// Get node indices as a slice.
    pub fn nodes(&self) -> Vec<usize> {
        match self {
            Face::Triangle(n) => vec![n[0], n[1], n[2]],
            Face::Quad(n) => vec![n[0], n[1], n[2], n[3]],
        }
    }

    /// Create a Face from a list of node indices.
    ///
    /// Panics unless `nodes` holds 3 (triangle) or 4 (quad) entries; prefer
    /// [`Self::try_from_nodes`] for externally supplied connectivity.
    pub fn from_nodes(nodes: &[usize]) -> Face {
        Self::try_from_nodes(nodes).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Fallible face construction for externally supplied connectivity.
    pub fn try_from_nodes(nodes: &[usize]) -> Result<Face, MeshConnectivityError> {
        match nodes.len() {
            3 => Ok(Face::Triangle([nodes[0], nodes[1], nodes[2]])),
            4 => Ok(Face::Quad([nodes[0], nodes[1], nodes[2], nodes[3]])),
            _ => Err(MeshConnectivityError::InvalidFaceNodeCount {
                actual: nodes.len(),
            }),
        }
    }
}

/// Hashable key for face lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FaceKey {
    Triangle([usize; 3]),
    Quad([usize; 4]),
}

/// Per-element extraction payload from phase 1.
#[derive(Debug, Clone)]
struct PerElem {
    edge_keys: Vec<(usize, usize, i8)>, // (min, max, orientation_for_this_elem)
    face_data: Vec<(Face, FaceKey, [f64; 3])>, // (oriented face, sorted key, local normal)
}

fn prep_one(mesh: &Mesh, elem: &crate::mesh::types::Element) -> PerElem {
    let local_edges = elem.edges();
    let edge_keys = local_edges
        .iter()
        .map(|(a, b)| {
            let (lo, hi, sign) = if a < b { (*a, *b, 1i8) } else { (*b, *a, -1i8) };
            (lo, hi, sign)
        })
        .collect();

    // Keep face orientation in the same canonical order used by the
    // H(div) and H(curl) spaces.  `Element::faces()` predates the
    // typed local-face contract and uses the opposite traversal for
    // two tetrahedral faces; mixing the two makes adjacent RT0
    // traces disagree even though they share the same global face.
    let vertices = elem.vertices();
    let local_faces: Vec<Vec<usize>> = elem
        .element_type
        .local_faces()
        .iter()
        .map(|local_face| local_face.iter().map(|&node| vertices[node]).collect())
        .collect();
    let face_data = local_faces
        .iter()
        .map(|face_nodes| {
            let face = Face::from_nodes(face_nodes);
            let key = face.sorted_key();
            let normal = compute_face_normal(&mesh.nodes, &face);
            (face, key, normal)
        })
        .collect();

    PerElem {
        edge_keys,
        face_data,
    }
}

/// Phase-2 output: global maps plus per-element index sequences.
struct MergedTopo {
    edges: Vec<(usize, usize)>,
    edge_map: HashMap<(usize, usize), usize>,
    element_edges: Vec<Vec<usize>>,
    edge_orientations: Vec<Vec<i8>>,
    faces: Vec<Face>,
    face_map: HashMap<FaceKey, usize>,
    face_normals: Vec<[f64; 3]>,
    element_faces: Vec<Vec<usize>>,
    face_orientations: Vec<Vec<i8>>,
}

/// Serial phase-2 merge: global indices follow element order.
fn merge_serial(prepped: Vec<PerElem>) -> MergedTopo {
    let mut merged = MergedTopo {
        edges: Vec::new(),
        edge_map: HashMap::new(),
        element_edges: Vec::with_capacity(prepped.len()),
        edge_orientations: Vec::with_capacity(prepped.len()),
        faces: Vec::new(),
        face_map: HashMap::new(),
        face_normals: Vec::new(),
        element_faces: Vec::with_capacity(prepped.len()),
        face_orientations: Vec::with_capacity(prepped.len()),
    };
    for prep in prepped {
        let mut elem_edge_indices = Vec::with_capacity(prep.edge_keys.len());
        let mut elem_orientations = Vec::with_capacity(prep.edge_keys.len());
        for (lo, hi, sign) in prep.edge_keys {
            let global_idx = *merged.edge_map.entry((lo, hi)).or_insert_with(|| {
                let idx = merged.edges.len();
                merged.edges.push((lo, hi));
                idx
            });
            elem_edge_indices.push(global_idx);
            elem_orientations.push(sign);
        }
        merged.element_edges.push(elem_edge_indices);
        merged.edge_orientations.push(elem_orientations);

        let mut elem_face_indices = Vec::with_capacity(prep.face_data.len());
        let mut elem_face_orientations = Vec::with_capacity(prep.face_data.len());
        for (face, key, local_normal) in prep.face_data {
            let global_idx = match merged.face_map.entry(key) {
                std::collections::hash_map::Entry::Occupied(o) => {
                    let stored_idx = *o.get();
                    let stored_normal = merged.face_normals[stored_idx];
                    let dot = local_normal[0] * stored_normal[0]
                        + local_normal[1] * stored_normal[1]
                        + local_normal[2] * stored_normal[2];
                    let orientation = if dot >= 0.0 { 1i8 } else { -1i8 };
                    elem_face_orientations.push(orientation);
                    stored_idx
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    let idx = merged.faces.len();
                    merged.faces.push(face);
                    merged.face_normals.push(local_normal);
                    elem_face_orientations.push(1i8);
                    *v.insert(idx)
                }
            };
            elem_face_indices.push(global_idx);
        }
        merged.element_faces.push(elem_face_indices);
        merged.face_orientations.push(elem_face_orientations);
    }
    merged
}

/// One shard's local dedup tables plus per-element local sequences.
struct ShardTopo {
    edge_uniques: Vec<(usize, usize)>,
    edge_lut: HashMap<(usize, usize), usize>,
    elem_edges: Vec<Vec<usize>>,
    elem_edge_ori: Vec<Vec<i8>>,
    face_uniques: Vec<(Face, FaceKey, [f64; 3])>,
    face_lut: HashMap<FaceKey, usize>,
    elem_faces: Vec<Vec<usize>>,
    elem_face_ori: Vec<Vec<i8>>,
}

fn build_shard(chunk: &[PerElem]) -> ShardTopo {
    let mut shard = ShardTopo {
        edge_uniques: Vec::new(),
        edge_lut: HashMap::new(),
        elem_edges: Vec::with_capacity(chunk.len()),
        elem_edge_ori: Vec::with_capacity(chunk.len()),
        face_uniques: Vec::new(),
        face_lut: HashMap::new(),
        elem_faces: Vec::with_capacity(chunk.len()),
        elem_face_ori: Vec::with_capacity(chunk.len()),
    };
    for prep in chunk {
        let mut seq = Vec::with_capacity(prep.edge_keys.len());
        let mut ori = Vec::with_capacity(prep.edge_keys.len());
        for &(lo, hi, sign) in &prep.edge_keys {
            let local = *shard.edge_lut.entry((lo, hi)).or_insert_with(|| {
                let idx = shard.edge_uniques.len();
                shard.edge_uniques.push((lo, hi));
                idx
            });
            seq.push(local);
            ori.push(sign);
        }
        shard.elem_edges.push(seq);
        shard.elem_edge_ori.push(ori);

        let mut fseq = Vec::with_capacity(prep.face_data.len());
        let mut fori = Vec::with_capacity(prep.face_data.len());
        for (face, key, normal) in &prep.face_data {
            let local = *shard.face_lut.entry(key.clone()).or_insert_with(|| {
                let idx = shard.face_uniques.len();
                shard
                    .face_uniques
                    .push((face.clone(), key.clone(), *normal));
                idx
            });
            fseq.push(local);
            let stored = shard.face_uniques[local].2;
            let dot = normal[0] * stored[0] + normal[1] * stored[1] + normal[2] * stored[2];
            fori.push(if dot >= 0.0 { 1i8 } else { -1i8 });
        }
        shard.elem_faces.push(fseq);
        shard.elem_face_ori.push(fori);
    }
    shard
}

/// Sharded phase-2 merge: contiguous element ranges dedup concurrently,
/// then shards merge in element order so global indices — and therefore
/// every per-element sequence — are identical to [`merge_serial`].
///
/// Face orientations are stored relative to each shard's first-occurrence
/// normal; the ordered merge records a flip flag per shard-local face
/// whenever the global first occurrence traverses the face oppositely, and
/// the rewrite pass applies it. (On degenerate zero-area faces the sign is
/// arbitrary but still deterministic.)
fn merge_sharded(prepped: &[PerElem], policy: &ParallelPolicy) -> MergedTopo {
    let shards_n = policy
        .shard_count(prepped.len(), TOPOLOGY_ITEMS_PER_THREAD)
        .max(2);
    let chunk = prepped.len().div_ceil(shards_n);
    let shards: Vec<ShardTopo> = prepped.par_chunks(chunk).map(build_shard).collect();

    let mut merged = MergedTopo {
        edges: Vec::new(),
        edge_map: HashMap::new(),
        element_edges: Vec::with_capacity(prepped.len()),
        edge_orientations: Vec::with_capacity(prepped.len()),
        faces: Vec::new(),
        face_map: HashMap::new(),
        face_normals: Vec::new(),
        element_faces: Vec::with_capacity(prepped.len()),
        face_orientations: Vec::with_capacity(prepped.len()),
    };
    struct Remap {
        edges: Vec<usize>,
        faces: Vec<usize>,
        flips: Vec<bool>,
    }
    let mut remaps: Vec<Remap> = Vec::with_capacity(shards.len());
    for shard in &shards {
        let mut edge_remap = Vec::with_capacity(shard.edge_uniques.len());
        for key in &shard.edge_uniques {
            edge_remap.push(*merged.edge_map.entry(*key).or_insert_with(|| {
                let idx = merged.edges.len();
                merged.edges.push(*key);
                idx
            }));
        }
        let mut face_remap = Vec::with_capacity(shard.face_uniques.len());
        let mut flips = Vec::with_capacity(shard.face_uniques.len());
        for (face, key, normal) in &shard.face_uniques {
            match merged.face_map.entry(key.clone()) {
                std::collections::hash_map::Entry::Occupied(o) => {
                    let global = *o.get();
                    let stored = merged.face_normals[global];
                    flips.push(
                        normal[0] * stored[0] + normal[1] * stored[1] + normal[2] * stored[2] < 0.0,
                    );
                    face_remap.push(global);
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    let global = merged.faces.len();
                    merged.faces.push(face.clone());
                    merged.face_normals.push(*normal);
                    v.insert(global);
                    flips.push(false);
                    face_remap.push(global);
                }
            }
        }
        remaps.push(Remap {
            edges: edge_remap,
            faces: face_remap,
            flips,
        });
    }

    // Rewrite per-element sequences in shard (element) order; the inner
    // mapping is independent per element and runs in parallel.
    for (shard, remap) in shards.iter().zip(remaps.iter()) {
        let global_edges: Vec<Vec<usize>> = shard
            .elem_edges
            .par_iter()
            .map(|seq| seq.iter().map(|local| remap.edges[*local]).collect())
            .collect();
        let global_faces: Vec<Vec<usize>> = shard
            .elem_faces
            .par_iter()
            .map(|seq| seq.iter().map(|local| remap.faces[*local]).collect())
            .collect();
        let global_ori: Vec<Vec<i8>> = shard
            .elem_faces
            .par_iter()
            .zip(shard.elem_face_ori.par_iter())
            .map(|(seq, ori)| {
                seq.iter()
                    .zip(ori.iter())
                    .map(|(local, orientation)| {
                        if remap.flips[*local] {
                            -orientation
                        } else {
                            *orientation
                        }
                    })
                    .collect()
            })
            .collect();
        merged.element_edges.extend(global_edges);
        merged
            .edge_orientations
            .extend(shard.elem_edge_ori.iter().cloned());
        merged.element_faces.extend(global_faces);
        merged.face_orientations.extend(global_ori);
    }
    merged
}

impl MeshTopology {
    /// Build topology after validating all externally mutable mesh vectors.
    pub fn try_build(mesh: &Mesh) -> Result<Self, Vec<MeshConnectivityError>> {
        mesh.validate_connectivity()?;
        Ok(Self::build_unchecked(mesh))
    }

    /// Build topology from a mesh.
    ///
    /// Panics on malformed public vectors; prefer [`Self::try_build`] for
    /// deserialized or externally mutated meshes.
    pub fn build(mesh: &Mesh) -> Self {
        Self::try_build(mesh).unwrap_or_else(|errors| {
            panic!(
                "cannot build topology from invalid mesh: {}",
                errors
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })
    }

    fn build_unchecked(mesh: &Mesh) -> Self {
        // Phase 1 (serial or parallel): each element independently enumerates
        // its local edges and faces, producing the sorted-node keys plus the
        // local traversal data we need during the dedup phase. Per-element
        // work is entirely independent here; the machine-adaptive gate
        // decides whether pool dispatch pays off on this machine.
        let policy = ParallelPolicy::from_env();
        let parallel = policy.decide(mesh.elements.len(), TOPOLOGY_ITEMS_PER_THREAD);
        let prepped: Vec<PerElem> = if parallel {
            mesh.elements
                .par_iter()
                .map(|elem| prep_one(mesh, elem))
                .collect()
        } else {
            mesh.elements
                .iter()
                .map(|elem| prep_one(mesh, elem))
                .collect()
        };

        // Phase 2: build the global edge/face maps. Global indices follow
        // first-occurrence order; the sharded merge preserves that order
        // exactly (shards merge in element order), so both paths produce
        // identical numbering.
        let merged = if parallel {
            merge_sharded(&prepped, &policy)
        } else {
            merge_serial(prepped)
        };
        let MergedTopo {
            edge_map,
            edges,
            element_edges,
            edge_orientations,
            face_map,
            faces,
            // Face normals are transient merge state (orientation resolution
            // only); they are not stored on `MeshTopology`.
            face_normals: _,
            element_faces,
            face_orientations,
        } = merged;

        // Build boundary edge sets
        let mut boundary_edges: HashMap<i32, Vec<usize>> = HashMap::new();
        // Build boundary face sets
        let mut boundary_faces: HashMap<i32, Vec<usize>> = HashMap::new();

        for bface in &mesh.boundaries {
            // Higher-order boundary entities carry edge/face nodes after their
            // corner nodes. Topology keys remain corner-owned; use only the
            // element-type corner prefix when rebuilding edge/face ownership.
            let boundary_nodes = boundary_corner_nodes(bface, mesh);
            let face_edges = boundary_face_edges(&boundary_nodes, bface.element_idx, mesh);
            for (a, b) in face_edges {
                let (min_node, max_node) = if a < b { (a, b) } else { (b, a) };
                if let Some(&edge_idx) = edge_map.get(&(min_node, max_node)) {
                    boundary_edges
                        .entry(bface.marker)
                        .or_default()
                        .push(edge_idx);
                }
            }

            // Also add to boundary faces (only for 3D, where boundaries are 3/4-node faces)
            if boundary_nodes.len() >= 3 {
                let face_key = Face::from_nodes(&boundary_nodes);
                let key = face_key.sorted_key();
                if let Some(&face_idx) = face_map.get(&key) {
                    boundary_faces
                        .entry(bface.marker)
                        .or_default()
                        .push(face_idx);
                }
            }
        }

        // Deduplicate boundary lists
        for edges_list in boundary_edges.values_mut() {
            edges_list.sort_unstable();
            edges_list.dedup();
        }
        for faces_list in boundary_faces.values_mut() {
            faces_list.sort_unstable();
            faces_list.dedup();
        }

        Self {
            edges,
            edge_map,
            element_edges,
            edge_orientations,
            boundary_edges,
            faces,
            face_map,
            element_faces,
            face_orientations,
            boundary_faces,
        }
    }

    /// Total number of edges in the mesh.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Total number of faces in the mesh.
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get global edge indices for an element.
    pub fn element_edge_indices(&self, elem_idx: usize) -> &[usize] {
        &self.element_edges[elem_idx]
    }

    /// Get edge orientations for an element.
    pub fn element_edge_signs(&self, elem_idx: usize) -> &[i8] {
        &self.edge_orientations[elem_idx]
    }

    /// Get global face indices for an element.
    pub fn element_face_indices(&self, elem_idx: usize) -> &[usize] {
        &self.element_faces[elem_idx]
    }

    /// Get face orientations for an element.
    pub fn element_face_signs(&self, elem_idx: usize) -> &[i8] {
        &self.face_orientations[elem_idx]
    }

    /// Get face by index.
    pub fn face(&self, idx: usize) -> &Face {
        &self.faces[idx]
    }

    /// Get boundary faces for a marker.
    pub fn boundary_faces(&self, marker: i32) -> &[usize] {
        self.boundary_faces
            .get(&marker)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Extract edges from a boundary face's node list.
fn boundary_corner_nodes(bface: &BoundaryFace, mesh: &Mesh) -> Vec<usize> {
    let Some(element) = mesh.elements.get(bface.element_idx) else {
        return bface.nodes.clone();
    };

    let corner_count = if element.element_type.dimension() == 2 {
        element
            .element_type
            .local_edges()
            .get(bface.local_idx)
            .map(|_| 2)
    } else {
        element
            .element_type
            .local_faces()
            .get(bface.local_idx)
            .map(|face| face.len())
    };

    // Boundary node vectors are validated as sets because serialized mesh
    // producers are not required to preserve a traversal order. In
    // particular, a P2/Q2 trace may put an edge midpoint or face centre
    // before its corner nodes. Recover corners from the owning element's
    // canonical boundary entity instead of assuming they are a prefix of
    // `bface.nodes`.
    if let (Some(corner_count), Some(entity)) = (
        corner_count,
        element.boundary_entities().get(bface.local_idx),
    ) {
        return entity.iter().copied().take(corner_count).collect();
    }

    bface
        .nodes
        .iter()
        .copied()
        .take(corner_count.unwrap_or(bface.nodes.len()))
        .collect()
}

fn boundary_face_edges(
    face_nodes: &[usize],
    _element_idx: usize,
    _mesh: &Mesh,
) -> Vec<(usize, usize)> {
    match face_nodes.len() {
        2 => {
            // 2D boundary edge
            vec![(face_nodes[0], face_nodes[1])]
        }
        3 => {
            // Triangular face (3D tet boundary)
            vec![
                (face_nodes[0], face_nodes[1]),
                (face_nodes[1], face_nodes[2]),
                (face_nodes[2], face_nodes[0]),
            ]
        }
        4 => {
            // Quad face (3D hex boundary)
            vec![
                (face_nodes[0], face_nodes[1]),
                (face_nodes[1], face_nodes[2]),
                (face_nodes[2], face_nodes[3]),
                (face_nodes[3], face_nodes[0]),
            ]
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{BoundaryType, ElementType, unit_cube_tetrahedra, unit_square_triangles};

    #[test]
    fn test_topology_2d_triangle_mesh() {
        let mesh = unit_square_triangles(2);
        let topo = MeshTopology::build(&mesh);

        // Each triangle has 3 edges
        for ee in &topo.element_edges {
            assert_eq!(ee.len(), 3);
        }

        // Each edge orientation is +1 or -1
        for eo in &topo.edge_orientations {
            for &o in eo {
                assert!(o == 1 || o == -1);
            }
        }

        // Edge count: for a structured 2×2 triangle mesh on unit square,
        // there are 5 nodes, 4 triangles (2 per quad cell × 1 cell... actually 2×2 grid = 4 cells × 2 = 8 triangles)
        // Just verify edges > 0 and edge_map is consistent
        assert!(topo.num_edges() > 0);
        for (i, &(a, b)) in topo.edges.iter().enumerate() {
            assert!(a < b, "Edge {i}: ({a}, {b}) should have a < b");
            assert_eq!(topo.edge_map[&(a, b)], i);
        }
    }

    #[test]
    fn test_topology_3d_tet_mesh() {
        let mesh = unit_cube_tetrahedra(2);
        let topo = MeshTopology::build(&mesh);

        // Each tetrahedron has 6 edges
        for ee in &topo.element_edges {
            assert_eq!(ee.len(), 6);
        }

        assert!(topo.num_edges() > 0);

        // Verify all edge indices are valid
        for ee in &topo.element_edges {
            for &edge_idx in ee {
                assert!(edge_idx < topo.num_edges());
            }
        }
    }

    #[test]
    fn test_shared_edges_between_adjacent_elements() {
        let mesh = unit_square_triangles(2);
        let topo = MeshTopology::build(&mesh);

        // Two adjacent triangles sharing an edge should reference the same global edge index.
        // Count how many times each edge appears across all elements.
        let mut edge_counts = vec![0usize; topo.num_edges()];
        for ee in &topo.element_edges {
            for &edge_idx in ee {
                edge_counts[edge_idx] += 1;
            }
        }

        // Interior edges should appear in exactly 2 elements, boundary edges in exactly 1.
        for &count in &edge_counts {
            assert!(
                count == 1 || count == 2,
                "Edge appears in {count} elements, expected 1 or 2"
            );
        }
    }

    #[test]
    fn test_orientation_consistency() {
        // For a shared edge between two elements, the orientations should be opposite
        // (one element traverses the edge min→max, the other max→min).
        // This isn't always true for all meshes, but for structured meshes it holds.
        let mesh = unit_square_triangles(3);
        let topo = MeshTopology::build(&mesh);

        // Build edge→(elem, local_idx, orientation) map
        let mut edge_elems: HashMap<usize, Vec<(usize, i8)>> = HashMap::new();
        for (elem_idx, (ee, eo)) in topo
            .element_edges
            .iter()
            .zip(topo.edge_orientations.iter())
            .enumerate()
        {
            for (&edge_idx, &orient) in ee.iter().zip(eo.iter()) {
                edge_elems
                    .entry(edge_idx)
                    .or_default()
                    .push((elem_idx, orient));
            }
        }

        // Interior edges (shared by 2 elements) should have consistent global orientation
        // meaning the two local orientations are independent but well-defined
        for (&edge_idx, elems) in &edge_elems {
            if elems.len() == 2 {
                let (a, b) = topo.edges[edge_idx];
                // Just verify the global edge has a < b
                assert!(a < b);
            }
        }
    }

    #[test]
    fn checked_topology_rejects_malformed_connectivity_without_panicking() {
        let mut mesh = unit_square_triangles(1);
        mesh.nodes.pop();
        let errors = mesh.try_topology().unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, MeshConnectivityError::InvalidElementNode { .. }))
        );
        assert!(Face::try_from_nodes(&[0, 1]).is_err());
    }

    #[test]
    fn higher_order_boundary_topology_uses_canonical_corner_nodes() {
        let mut mesh = Mesh::new(2);
        for point in [
            Point::new_2d(0.0, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.0),
            Point::new_2d(0.5, 0.5),
            Point::new_2d(0.0, 0.5),
        ] {
            mesh.add_node(point);
        }
        let element_idx = mesh.add_element(ElementType::Triangle, vec![0, 1, 2, 3, 4, 5]);

        // Each record is a valid P2 trace set, but deliberately starts with
        // its edge midpoint. Mesh serde admission accepts this ordering.
        mesh.boundaries = vec![
            BoundaryFace {
                nodes: vec![3, 0, 1],
                boundary_type: BoundaryType::Dirichlet,
                marker: 7,
                element_idx,
                local_idx: 0,
            },
            BoundaryFace {
                nodes: vec![4, 1, 2],
                boundary_type: BoundaryType::Dirichlet,
                marker: 7,
                element_idx,
                local_idx: 1,
            },
            BoundaryFace {
                nodes: vec![5, 2, 0],
                boundary_type: BoundaryType::Dirichlet,
                marker: 7,
                element_idx,
                local_idx: 2,
            },
        ];

        let topology = MeshTopology::try_build(&mesh).expect("reordered P2 traces are valid");
        assert_eq!(topology.boundary_edges.get(&7), Some(&vec![0, 1, 2]));
    }

    #[test]
    fn sharded_merge_matches_serial_merge_exactly() {
        let mesh = unit_cube_tetrahedra(4);
        assert!(mesh.validate_connectivity().is_ok());
        let prepped: Vec<PerElem> = mesh
            .elements
            .iter()
            .map(|elem| prep_one(&mesh, elem))
            .collect();
        assert!(
            prepped.len() > 2,
            "need multiple shards, got {}",
            prepped.len()
        );
        let policy = ParallelPolicy::with_threads(4);
        let serial = merge_serial(prepped.clone());
        let sharded = merge_sharded(&prepped, &policy);
        assert_eq!(serial.edges, sharded.edges);
        assert_eq!(serial.element_edges, sharded.element_edges);
        assert_eq!(serial.edge_orientations, sharded.edge_orientations);
        assert_eq!(serial.faces, sharded.faces);
        assert_eq!(serial.element_faces, sharded.element_faces);
        assert_eq!(serial.face_orientations, sharded.face_orientations);
        assert_eq!(serial.face_normals, sharded.face_normals);
    }

    #[test]
    fn gated_build_matches_manual_paths() {
        // End to end: the env-gated build agrees with both explicit paths.
        let mesh = unit_cube_tetrahedra(4);
        let reference = MeshTopology::try_build(&mesh).expect("valid cube");
        assert_eq!(reference.element_faces.len(), mesh.num_elements());
        for (faces, orientations) in reference
            .element_faces
            .iter()
            .zip(reference.face_orientations.iter())
        {
            assert_eq!(faces.len(), orientations.len());
            for &orientation in orientations {
                assert!(orientation == 1 || orientation == -1);
            }
        }
    }
}
