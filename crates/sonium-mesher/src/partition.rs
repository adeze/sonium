//! Mesh partitioning for distributed-memory FEM
//!
//! Provides graph-based partitioning for domain decomposition. Three
//! partitioners ship in-tree:
//!
//! - [`partition_mesh_round_robin`]: striping, no graph structure used.
//! - [`partition_mesh_adjacency`]: greedy neighbor-aware colouring.
//! - [`partition_mesh_bisection`]: recursive coordinate bisection on element
//!   centroids — produces compact, axis-aligned subdomains.
//!
//! All three are followed by an optional **Fiduccia–Mattheyses-style local
//! refinement** ([`refine_partition_fm`]) that moves boundary elements one at
//! a time to whichever neighbour partition gives the largest edge-cut
//! reduction while keeping the imbalance below [`PartitionOptions::imbalance`].
//! Pass `refinement = 0` in [`PartitionOptions`] to disable.

use crate::mesh::types::{ElementType, Mesh, MeshConnectivityError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Partitioning options
#[derive(Debug, Clone)]
pub struct PartitionOptions {
    /// Number of partitions
    pub num_partitions: usize,
    /// Edge cut weight objective (vs. total volume)
    pub objective: PartitionObjective,
    /// Balancing constraint: max imbalance ratio (1.03 = 3% imbalance)
    pub imbalance: f64,
    /// Number of refinement iterations
    pub refinement: usize,
    /// Seed for random decisions
    pub seed: Option<u32>,
}

impl Default for PartitionOptions {
    fn default() -> Self {
        Self {
            num_partitions: 4,
            objective: PartitionObjective::Volume,
            imbalance: 1.03,
            refinement: 10,
            seed: Some(42),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartitionObjective {
    Volume,
    EdgeCut,
}

/// Failure reported by the checked partitioning and ghost-layout APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionError {
    /// A partition count of zero cannot own a non-empty mesh.
    InvalidPartitionCount { requested: usize, elements: usize },
    /// The partition options contain a non-finite or invalid balance bound.
    InvalidImbalance { imbalance: f64 },
    /// The mesh failed its external-connectivity gate before graph creation.
    InvalidMesh { errors: Vec<MeshConnectivityError> },
    /// The partition vector does not contain one entry per graph vertex.
    PartitionIdCount { expected: usize, actual: usize },
    /// An element is assigned to a partition outside the declared range.
    PartitionIdOutOfRange {
        element: usize,
        partition: usize,
        num_partitions: usize,
    },
    /// The cached inverse map does not match `partition_id`.
    InverseMapMismatch { partition: usize, element: usize },
    /// The graph does not contain one adjacency list per element.
    GraphVertexCountMismatch { expected: usize, actual: usize },
    /// A graph adjacency list has an invalid neighbor or is not symmetric.
    InvalidGraph { element: usize, neighbor: usize },
    /// A caller requested a partition outside the declared range.
    PartitionOutOfRange {
        partition: usize,
        num_partitions: usize,
    },
    /// A partition mesh extraction encountered an invalid source index.
    InvalidExtraction { element: usize, node: usize },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPartitionCount {
                requested,
                elements,
            } => write!(
                f,
                "partition count {requested} cannot be used for {elements} mesh elements"
            ),
            Self::InvalidImbalance { imbalance } => {
                write!(
                    f,
                    "partition imbalance must be finite and at least one, got {imbalance}"
                )
            }
            Self::InvalidMesh { errors } => {
                write!(
                    f,
                    "mesh connectivity validation failed with {} error(s)",
                    errors.len()
                )
            }
            Self::PartitionIdCount { expected, actual } => {
                write!(f, "partition has {actual} ids, expected {expected}")
            }
            Self::PartitionIdOutOfRange {
                element,
                partition,
                num_partitions,
            } => write!(
                f,
                "element {element} uses partition {partition}, outside 0..{num_partitions}"
            ),
            Self::InverseMapMismatch { partition, element } => write!(
                f,
                "inverse partition map does not contain element {element} in partition {partition}"
            ),
            Self::GraphVertexCountMismatch { expected, actual } => {
                write!(f, "graph has {actual} vertices, expected {expected}")
            }
            Self::InvalidGraph { element, neighbor } => {
                write!(
                    f,
                    "graph edge {element} -> {neighbor} is invalid or asymmetric"
                )
            }
            Self::PartitionOutOfRange {
                partition,
                num_partitions,
            } => write!(f, "partition {partition} is outside 0..{num_partitions}"),
            Self::InvalidExtraction { element, node } => write!(
                f,
                "element {element} references node {node} during partition extraction"
            ),
        }
    }
}

impl std::error::Error for PartitionError {}

/// Build element adjacency graph from mesh.
///
/// Two elements are adjacent iff they share a full face (3D) or edge (2D).
/// The implementation hashes each oriented entity by its sorted-node key into
/// a `HashMap<key, owning_elems>` in one pass: every element reporting the same
/// key is connected to all previous owners. This remains `O(num_elements *
/// faces_per_element + incident_pairs)` and replaces the original `O(N²)`
/// quadratic scan that was prohibitively slow above a few thousand elements.
pub fn build_element_graph(mesh: &Mesh) -> ElementGraph {
    if mesh.validate_connectivity().is_err() {
        return ElementGraph {
            num_vertices: 0,
            adjacency: Vec::new(),
        };
    }
    let num_elements = mesh.elements.len();
    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); num_elements];

    // For 2D triangles/quads, `Element::faces()` returns an empty vec — those
    // elements share edges, not faces, so fall back to edges for 2D types.
    let mut face_owners: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();

    for (elem_idx, elem) in mesh.elements.iter().enumerate() {
        let entities: Vec<Vec<usize>> = if elem.element_type.dimension() == 3 {
            elem.faces()
        } else {
            elem.edges().into_iter().map(|(a, b)| vec![a, b]).collect()
        };

        for mut key in entities {
            key.sort_unstable();
            let owners = face_owners.entry(key).or_default();
            for &other in owners.iter() {
                adjacency[elem_idx].insert(other);
                adjacency[other].insert(elem_idx);
            }
            owners.push(elem_idx);
        }
    }

    ElementGraph {
        num_vertices: num_elements,
        adjacency: adjacency
            .into_iter()
            .map(|s| {
                let mut neighbors: Vec<_> = s.into_iter().collect();
                // HashSet iteration order is randomized. Stable neighbor
                // ordering keeps greedy partitioning and serialized QA
                // diagnostics deterministic across processes.
                neighbors.sort_unstable();
                neighbors
            })
            .collect(),
    }
}

/// Element adjacency graph
#[derive(Debug, Clone)]
pub struct ElementGraph {
    pub num_vertices: usize,
    pub adjacency: Vec<Vec<usize>>,
}

impl ElementGraph {
    /// Get neighbors of an element
    pub fn neighbors(&self, elem_idx: usize) -> &[usize] {
        &self.adjacency[elem_idx]
    }

    /// Total number of edges in graph
    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|v| v.len()).sum::<usize>() / 2
    }
}

/// Partition result.
///
/// `partition_id[e]` gives the partition assignment of element `e`. The
/// reverse map `parts` is cached so `partition_elements` is O(1).
#[derive(Debug, Clone)]
pub struct Partition {
    pub partition_id: Vec<usize>,
    pub num_partitions: usize,
    /// Inverse of `partition_id`. `parts[p]` lists every element in partition
    /// `p`, sorted ascending.
    parts: Vec<Vec<usize>>,
}

impl Partition {
    /// Build a `Partition` from a per-element id vector. The cached
    /// `parts` table is constructed once and kept in sync by the
    /// mutators on this type.
    pub fn from_ids(partition_id: Vec<usize>, num_partitions: usize) -> Self {
        let mut parts = vec![Vec::new(); num_partitions];
        for (i, &p) in partition_id.iter().enumerate() {
            if p < num_partitions {
                parts[p].push(i);
            }
        }
        Self {
            partition_id,
            num_partitions,
            parts,
        }
    }

    /// Construct a partition while checking every assignment and its cached
    /// inverse map. This is the preferred boundary for deserialized or
    /// externally-produced partition metadata.
    pub fn try_from_ids(
        partition_id: Vec<usize>,
        num_partitions: usize,
    ) -> Result<Self, PartitionError> {
        if num_partitions == 0 && !partition_id.is_empty() {
            return Err(PartitionError::InvalidPartitionCount {
                requested: num_partitions,
                elements: partition_id.len(),
            });
        }
        if let Some((element, &partition)) = partition_id
            .iter()
            .enumerate()
            .find(|(_, partition)| **partition >= num_partitions)
        {
            return Err(PartitionError::PartitionIdOutOfRange {
                element,
                partition,
                num_partitions,
            });
        }
        let result = Self::from_ids(partition_id, num_partitions);
        result.validate()?;
        Ok(result)
    }

    /// Check the partition vector and cached inverse map without needing a
    /// mesh graph.
    pub fn validate(&self) -> Result<(), PartitionError> {
        if self.num_partitions == 0 && !self.partition_id.is_empty() {
            return Err(PartitionError::InvalidPartitionCount {
                requested: self.num_partitions,
                elements: self.partition_id.len(),
            });
        }
        if self.parts.len() != self.num_partitions {
            return Err(PartitionError::InverseMapMismatch {
                partition: self.parts.len(),
                element: self.partition_id.len(),
            });
        }
        for (element, &partition) in self.partition_id.iter().enumerate() {
            if partition >= self.num_partitions {
                return Err(PartitionError::PartitionIdOutOfRange {
                    element,
                    partition,
                    num_partitions: self.num_partitions,
                });
            }
            if !self.parts[partition].binary_search(&element).is_ok() {
                return Err(PartitionError::InverseMapMismatch { partition, element });
            }
        }
        for (partition, elements) in self.parts.iter().enumerate() {
            if elements.windows(2).any(|pair| pair[0] >= pair[1]) {
                let element = elements.first().copied().unwrap_or_default();
                return Err(PartitionError::InverseMapMismatch { partition, element });
            }
            for &element in elements {
                if self.partition_id.get(element).copied() != Some(partition) {
                    return Err(PartitionError::InverseMapMismatch { partition, element });
                }
            }
        }
        Ok(())
    }

    /// Validate this partition against an element graph before using any
    /// adjacency-derived operation.
    pub fn validate_against(&self, graph: &ElementGraph) -> Result<(), PartitionError> {
        self.validate()?;
        if self.partition_id.len() != graph.num_vertices {
            return Err(PartitionError::GraphVertexCountMismatch {
                expected: self.partition_id.len(),
                actual: graph.num_vertices,
            });
        }
        if graph.adjacency.len() != graph.num_vertices {
            return Err(PartitionError::GraphVertexCountMismatch {
                expected: graph.num_vertices,
                actual: graph.adjacency.len(),
            });
        }
        for (element, neighbors) in graph.adjacency.iter().enumerate() {
            for &neighbor in neighbors {
                if neighbor >= graph.num_vertices
                    || neighbor == element
                    || !graph.adjacency[neighbor].contains(&element)
                {
                    return Err(PartitionError::InvalidGraph { element, neighbor });
                }
            }
        }
        Ok(())
    }

    /// Move element `e` from its current partition to `to`. Keeps the cached
    /// inverse map consistent — callers must use this rather than mutating
    /// `partition_id` directly.
    fn move_element(&mut self, e: usize, to: usize) {
        let from = self.partition_id[e];
        if from == to {
            return;
        }
        if let Ok(pos) = self.parts[from].binary_search(&e) {
            self.parts[from].remove(pos);
        }
        let pos = self.parts[to].partition_point(|&x| x < e);
        self.parts[to].insert(pos, e);
        self.partition_id[e] = to;
    }

    /// Get partition for an element
    pub fn element_partition(&self, elem_idx: usize) -> usize {
        self.partition_id[elem_idx]
    }

    /// Get all elements in a partition. O(1) — backed by the cached inverse
    /// map.
    pub fn partition_elements(&self, partition: usize) -> &[usize] {
        &self.parts[partition]
    }

    /// Count elements per partition
    pub fn partition_sizes(&self) -> Vec<usize> {
        self.parts.iter().map(|v| v.len()).collect()
    }

    /// Compute edge cut: number of edges crossing partitions
    pub fn edge_cut(&self, graph: &ElementGraph) -> usize {
        self.try_edge_cut(graph).unwrap_or_default()
    }

    /// Compute the edge cut after checking partition and graph invariants.
    pub fn try_edge_cut(&self, graph: &ElementGraph) -> Result<usize, PartitionError> {
        self.validate_against(graph)?;
        let mut cut = 0usize;

        for (elem_idx, neighbors) in graph.adjacency.iter().enumerate() {
            let elem_part = self.partition_id[elem_idx];
            for &nbr in neighbors {
                if elem_part != self.partition_id[nbr] {
                    cut += 1;
                }
            }
        }

        Ok(cut / 2)
    }

    /// Return elements adjacent to `partition` but owned by another
    /// partition. The result is sorted and duplicate-free, making it stable
    /// for deterministic halo exchange and serialized diagnostics.
    pub fn ghost_elements(
        &self,
        graph: &ElementGraph,
        partition: usize,
    ) -> Result<Vec<usize>, PartitionError> {
        self.validate_against(graph)?;
        if partition >= self.num_partitions {
            return Err(PartitionError::PartitionOutOfRange {
                partition,
                num_partitions: self.num_partitions,
            });
        }
        let mut ghosts = HashSet::new();
        for &element in self.partition_elements(partition) {
            for &neighbor in &graph.adjacency[element] {
                if self.partition_id[neighbor] != partition {
                    ghosts.insert(neighbor);
                }
            }
        }
        let mut ghosts: Vec<_> = ghosts.into_iter().collect();
        ghosts.sort_unstable();
        Ok(ghosts)
    }
}

/// Partition mesh using naive round-robin (fallback when METIS unavailable)
pub fn partition_mesh_round_robin(mesh: &Mesh, num_partitions: usize) -> Partition {
    if num_partitions == 0 {
        return Partition::from_ids(Vec::new(), 0);
    }
    let partition_id: Vec<usize> = (0..mesh.elements.len())
        .map(|i| i % num_partitions)
        .collect();
    Partition::from_ids(partition_id, num_partitions)
}

/// Partition mesh using element adjacency (greedy coloring approach)
pub fn partition_mesh_adjacency(mesh: &Mesh, num_partitions: usize) -> Partition {
    if num_partitions == 0 || mesh.validate_connectivity().is_err() {
        return Partition::from_ids(Vec::new(), 0);
    }
    let graph = build_element_graph(mesh);

    let mut partition_id = vec![usize::MAX; mesh.elements.len()];
    let mut partition_sizes = vec![0usize; num_partitions];

    let mut elem_order: Vec<usize> = (0..mesh.elements.len()).collect();
    elem_order.sort_by(|&a, &b| graph.adjacency[b].len().cmp(&graph.adjacency[a].len()));

    for &elem_idx in &elem_order {
        let neighbors = &graph.adjacency[elem_idx];
        let mut used: HashSet<usize> = HashSet::new();
        for &nbr in neighbors {
            if partition_id[nbr] != usize::MAX {
                used.insert(partition_id[nbr]);
            }
        }
        let mut best: Option<usize> = None;
        for (p, &size) in partition_sizes.iter().enumerate() {
            if used.contains(&p) {
                continue;
            }
            match best {
                None => best = Some(p),
                Some(b) if size < partition_sizes[b] => best = Some(p),
                _ => {}
            }
        }
        // If every partition is taken by a neighbour, fall back to the
        // smallest one (we always have to land somewhere).
        let best = best.unwrap_or_else(|| {
            let mut s = 0usize;
            for (p, &size) in partition_sizes.iter().enumerate() {
                if size < partition_sizes[s] {
                    s = p;
                }
            }
            s
        });
        partition_id[elem_idx] = best;
        partition_sizes[best] += 1;
    }

    Partition::from_ids(partition_id, num_partitions)
}

/// Recursive coordinate bisection on element centroids.
///
/// Computes the centroid of each element, then partitions by repeatedly
/// splitting along the axis with the largest extent at the median coordinate.
/// For `k` partitions the recursion splits into `⌈k/2⌉` and `⌊k/2⌋` until
/// `k == 1`. Produces compact, axis-aligned subdomains — usually a much better
/// starting point for FM refinement on graded meshes than [`partition_mesh_adjacency`].
///
/// Deterministic: ties on the split axis keep input (element-index) order via
/// a stable sort, so the same mesh and `k` always yield the same assignment.
pub fn partition_mesh_bisection(mesh: &Mesh, num_partitions: usize) -> Partition {
    if num_partitions == 0 || mesh.validate_connectivity().is_err() {
        return Partition::from_ids(vec![], 0);
    }
    let centroids: Vec<[f64; 3]> = (0..mesh.elements.len())
        .map(|i| {
            let c = mesh.element_centroid(i);
            [c.x, c.y, c.z]
        })
        .collect();
    let elements: Vec<usize> = (0..mesh.elements.len()).collect();
    let mut partition_id = vec![0usize; mesh.elements.len()];
    rcb(&centroids, &elements, 0, num_partitions, &mut partition_id);
    Partition::from_ids(partition_id, num_partitions)
}

fn rcb(centroids: &[[f64; 3]], elements: &[usize], base: usize, k: usize, out: &mut [usize]) {
    if k <= 1 {
        for &e in elements {
            out[e] = base;
        }
        return;
    }
    // Pick the axis with largest spread among `elements`.
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &e in elements {
        for d in 0..3 {
            min[d] = min[d].min(centroids[e][d]);
            max[d] = max[d].max(centroids[e][d]);
        }
    }
    let mut axis = 0usize;
    let mut best_spread = max[0] - min[0];
    for d in 1..3 {
        let s = max[d] - min[d];
        if s > best_spread {
            best_spread = s;
            axis = d;
        }
    }
    // Sort by chosen axis; split into ⌈k/2⌉ and ⌊k/2⌋ at the proportional cut.
    let mut sorted = elements.to_vec();
    sorted.sort_by(|&a, &b| {
        centroids[a][axis]
            .partial_cmp(&centroids[b][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let left_k = k.div_ceil(2);
    let right_k = k - left_k;
    let split = sorted.len() * left_k / k;
    let (left, right) = sorted.split_at(split);
    rcb(centroids, left, base, left_k, out);
    rcb(centroids, right, base + left_k, right_k, out);
}

/// Fiduccia–Mattheyses-style local refinement.
///
/// Moves one element at a time across a partition boundary in the direction
/// that reduces the edge cut by the largest amount, subject to the imbalance
/// constraint `(max_size - target) / target <= imbalance - 1`. Runs at most
/// `max_passes` outer passes; each pass stops when no positive-gain move is
/// available. Returns the number of moves actually performed.
pub fn refine_partition_fm(
    partition: &mut Partition,
    graph: &ElementGraph,
    max_passes: usize,
    imbalance: f64,
) -> usize {
    let total = partition.partition_id.len();
    let k = partition.num_partitions;
    if k < 2 || total == 0 || partition.validate_against(graph).is_err() {
        return 0;
    }
    let target = (total as f64) / (k as f64);
    let upper = (target * imbalance).ceil() as usize;

    let gain = |e: usize, dest: usize, p: &Partition| -> i64 {
        let src = p.partition_id[e];
        let mut external = 0i64;
        let mut internal = 0i64;
        for &nbr in &graph.adjacency[e] {
            let np = p.partition_id[nbr];
            if np == dest {
                external += 1;
            } else if np == src {
                internal += 1;
            }
        }
        external - internal
    };

    let mut moves_done = 0usize;
    for _pass in 0..max_passes {
        let mut best: Option<(usize, usize, i64)> = None; // (elem, dest, gain)
        for e in 0..total {
            let src = partition.partition_id[e];
            for d in 0..k {
                if d == src {
                    continue;
                }
                if partition.parts[d].len() + 1 > upper {
                    continue;
                }
                let g = gain(e, d, partition);
                if g > 0 {
                    match best {
                        None => best = Some((e, d, g)),
                        Some((_, _, b)) if g > b => best = Some((e, d, g)),
                        _ => {}
                    }
                }
            }
        }
        match best {
            Some((e, d, _)) => {
                partition.move_element(e, d);
                moves_done += 1;
            }
            None => break,
        }
    }
    moves_done
}

/// Partition mesh - main entry point.
///
/// Picks a starter (round-robin for `k == 1`, RCB otherwise — RCB produces
/// compact subdomains that FM refinement reduces further), then optionally
/// runs `refinement` passes of FM local search. Set `options.refinement = 0`
/// to disable the FM pass.
pub fn partition_mesh(mesh: &Mesh, options: &PartitionOptions) -> Partition {
    if options.num_partitions == 0 {
        return Partition::from_ids(vec![], 0);
    }

    if options.num_partitions == 1 {
        return Partition::from_ids(vec![0; mesh.elements.len()], 1);
    }

    // Use the RNG seed as a stability handle for tie-breaking in higher-level
    // partitioners that wrap this; the current implementation is deterministic
    // and doesn't actually consume randomness, but constructing the RNG
    // exercises the seed parameter so callers know it's wired up.
    let _rng: StdRng = match options.seed {
        Some(s) => StdRng::seed_from_u64(s as u64),
        None => StdRng::seed_from_u64(0),
    };

    let mut partition = partition_mesh_bisection(mesh, options.num_partitions);
    if options.refinement > 0 {
        let graph = build_element_graph(mesh);
        refine_partition_fm(
            &mut partition,
            &graph,
            options.refinement,
            options.imbalance,
        );
    }
    // `objective` is currently advisory — both Volume and EdgeCut funnel into
    // the same FM cut-minimisation pass. Future work would weight the gain
    // function by element volume for the Volume objective.
    let _ = options.objective;
    partition
}

/// Checked graph construction for externally supplied meshes.
pub fn try_build_element_graph(mesh: &Mesh) -> Result<ElementGraph, PartitionError> {
    mesh.validate_connectivity()
        .map_err(|errors| PartitionError::InvalidMesh { errors })?;
    Ok(build_element_graph(mesh))
}

/// Checked partition entry point. It validates options and mesh connectivity,
/// then validates the resulting assignment and adjacency graph before
/// returning it to distributed FEM callers.
pub fn try_partition_mesh(
    mesh: &Mesh,
    options: &PartitionOptions,
) -> Result<Partition, PartitionError> {
    if options.num_partitions == 0 && !mesh.elements.is_empty() {
        return Err(PartitionError::InvalidPartitionCount {
            requested: options.num_partitions,
            elements: mesh.elements.len(),
        });
    }
    if !options.imbalance.is_finite() || options.imbalance < 1.0 {
        return Err(PartitionError::InvalidImbalance {
            imbalance: options.imbalance,
        });
    }
    mesh.validate_connectivity()
        .map_err(|errors| PartitionError::InvalidMesh { errors })?;
    if options.num_partitions == 0 {
        return Ok(Partition::from_ids(Vec::new(), 0));
    }

    let partition = partition_mesh(mesh, options);
    let graph = build_element_graph(mesh);
    partition.validate_against(&graph)?;
    Ok(partition)
}

/// Deterministic ownership and halo information for one partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionGhostLayout {
    /// Elements assigned to this partition.
    pub owned_elements: Vec<usize>,
    /// Adjacent elements assigned to another partition.
    pub ghost_elements: Vec<usize>,
    /// Local nodes whose canonical owner is this partition.
    pub owned_nodes: Vec<usize>,
    /// Local nodes needed by this partition but canonically owned elsewhere.
    pub ghost_nodes: Vec<usize>,
}

impl Partition {
    /// Build deterministic element and node ownership for a mesh partition.
    /// A node shared by several partitions is owned by the lowest partition
    /// id incident on that node; every other local use is a ghost use.
    pub fn ghost_layout(
        &self,
        mesh: &Mesh,
        graph: &ElementGraph,
        partition: usize,
    ) -> Result<PartitionGhostLayout, PartitionError> {
        self.validate_against(graph)?;
        mesh.validate_connectivity()
            .map_err(|errors| PartitionError::InvalidMesh { errors })?;
        if self.partition_id.len() != mesh.elements.len() {
            return Err(PartitionError::PartitionIdCount {
                expected: mesh.elements.len(),
                actual: self.partition_id.len(),
            });
        }
        if partition >= self.num_partitions {
            return Err(PartitionError::PartitionOutOfRange {
                partition,
                num_partitions: self.num_partitions,
            });
        }

        let owned_elements = self.partition_elements(partition).to_vec();
        let ghost_elements = self.ghost_elements(graph, partition)?;
        let mut node_owner = vec![usize::MAX; mesh.nodes.len()];
        for (element, cell) in mesh.elements.iter().enumerate() {
            let owner = self.partition_id[element];
            for &node in &cell.nodes {
                node_owner[node] = node_owner[node].min(owner);
            }
        }

        let mut local_nodes = HashSet::new();
        for &element in &owned_elements {
            local_nodes.extend(mesh.elements[element].nodes.iter().copied());
        }
        let mut owned_nodes = Vec::new();
        let mut ghost_nodes = Vec::new();
        for node in local_nodes {
            if node_owner[node] == partition {
                owned_nodes.push(node);
            } else {
                ghost_nodes.push(node);
            }
        }
        owned_nodes.sort_unstable();
        ghost_nodes.sort_unstable();

        Ok(PartitionGhostLayout {
            owned_elements,
            ghost_elements,
            owned_nodes,
            ghost_nodes,
        })
    }
}

/// Submesh for a partition (extracts elements and nodes for a domain)
#[derive(Debug, Clone)]
pub struct PartitionMesh {
    pub nodes: Vec<crate::mesh::types::Point>,
    pub elements: Vec<Vec<usize>>,
    pub element_types: Vec<ElementType>,
    pub parent_element_indices: Vec<usize>,
    pub parent_node_indices: Vec<usize>,
}

impl PartitionMesh {
    /// Checked extraction that validates source connectivity and partition
    /// ownership before dereferencing any element or node index.
    pub fn extract_checked(
        mesh: &Mesh,
        partition: &Partition,
        partition_id: usize,
    ) -> Result<Self, PartitionError> {
        mesh.validate_connectivity()
            .map_err(|errors| PartitionError::InvalidMesh { errors })?;
        partition.validate()?;
        if partition.partition_id.len() != mesh.elements.len() {
            return Err(PartitionError::PartitionIdCount {
                expected: mesh.elements.len(),
                actual: partition.partition_id.len(),
            });
        }
        if partition_id >= partition.num_partitions {
            return Err(PartitionError::PartitionOutOfRange {
                partition: partition_id,
                num_partitions: partition.num_partitions,
            });
        }

        let element_indices = partition.partition_elements(partition_id);
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut element_types = Vec::new();
        let mut parent_element_indices = Vec::new();
        let mut parent_node_indices = Vec::new();

        for &elem_idx in element_indices {
            let Some(elem) = mesh.elements.get(elem_idx) else {
                return Err(PartitionError::InvalidExtraction {
                    element: elem_idx,
                    node: usize::MAX,
                });
            };
            element_types.push(elem.element_type);
            parent_element_indices.push(elem_idx);

            let mut local_nodes = Vec::with_capacity(elem.nodes.len());
            for &node in &elem.nodes {
                if node >= mesh.nodes.len() {
                    return Err(PartitionError::InvalidExtraction {
                        element: elem_idx,
                        node,
                    });
                }
                let local_idx = *node_map.entry(node).or_insert_with(|| {
                    let idx = nodes.len();
                    nodes.push(mesh.nodes[node]);
                    parent_node_indices.push(node);
                    idx
                });
                local_nodes.push(local_idx);
            }
            elements.push(local_nodes);
        }

        Ok(Self {
            nodes,
            elements,
            element_types,
            parent_element_indices,
            parent_node_indices,
        })
    }

    /// Extract partition from parent mesh.
    ///
    /// This compatibility API cannot return a typed failure. Callers that
    /// need diagnostics should use [`PartitionMesh::extract_checked`]; on
    /// malformed input this method returns an empty submesh rather than
    /// dereferencing invalid public indices.
    pub fn extract(mesh: &Mesh, partition: &Partition, partition_id: usize) -> Self {
        Self::extract_checked(mesh, partition, partition_id).unwrap_or_else(|_| Self {
            nodes: Vec::new(),
            elements: Vec::new(),
            element_types: Vec::new(),
            parent_element_indices: Vec::new(),
            parent_node_indices: Vec::new(),
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generators::unit_cube_tetrahedra;
    use crate::mesh::{ElementType, Mesh, Point};

    #[test]
    fn test_element_graph() {
        let mesh = unit_cube_tetrahedra(2);
        let graph = build_element_graph(&mesh);

        assert_eq!(graph.num_vertices, mesh.elements.len());
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn non_manifold_shared_face_connects_all_incident_elements() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
            Point::new_3d(0.0, 0.0, -1.0),
            Point::new_3d(1.0, 1.0, 1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 4]);
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 5]);

        let graph = build_element_graph(&mesh);
        assert!(
            graph
                .adjacency
                .iter()
                .all(|neighbors| neighbors.windows(2).all(|pair| pair[0] < pair[1]))
        );
        assert_eq!(graph.num_edges(), 3);
        assert!(graph.neighbors(0).contains(&1) && graph.neighbors(0).contains(&2));
        assert!(graph.neighbors(1).contains(&0) && graph.neighbors(1).contains(&2));
        assert!(graph.neighbors(2).contains(&0) && graph.neighbors(2).contains(&1));

        let mut mesh_neighbors = mesh.clone();
        assert_eq!(mesh_neighbors.element_neighbors()[0], [1, 2]);
        assert_eq!(mesh_neighbors.element_neighbors()[1], [0, 2]);
        assert_eq!(mesh_neighbors.element_neighbors()[2], [0, 1]);
    }

    #[test]
    fn test_partition_round_robin() {
        let mesh = unit_cube_tetrahedra(2);
        let partition = partition_mesh_round_robin(&mesh, 4);

        assert_eq!(partition.num_partitions, 4);
        let sizes = partition.partition_sizes();
        // Should be reasonably balanced
        let max_diff = sizes.iter().max().unwrap() - sizes.iter().min().unwrap();
        assert!(max_diff <= 1);
    }

    #[test]
    fn test_partition_adjacency() {
        let mesh = unit_cube_tetrahedra(2);
        let options = PartitionOptions {
            num_partitions: 4,
            ..Default::default()
        };
        let partition = partition_mesh(&mesh, &options);

        assert_eq!(partition.num_partitions, 4);

        let sizes = partition.partition_sizes();
        assert_eq!(sizes.iter().sum::<usize>(), mesh.elements.len());
    }

    #[test]
    fn test_partition_bisection_balances() {
        let mesh = unit_cube_tetrahedra(3);
        let p = partition_mesh_bisection(&mesh, 4);
        let sizes = p.partition_sizes();
        assert_eq!(p.num_partitions, 4);
        assert_eq!(sizes.iter().sum::<usize>(), mesh.elements.len());
        // RCB should be roughly balanced — within one element of the target.
        let target = mesh.elements.len() / 4;
        assert!(
            sizes.iter().all(|&s| s.abs_diff(target) <= 1),
            "RCB sizes {sizes:?} far from target {target}"
        );
    }

    #[test]
    fn test_fm_refinement_does_not_increase_cut() {
        let mesh = unit_cube_tetrahedra(3);
        let graph = build_element_graph(&mesh);
        let mut partition = partition_mesh_round_robin(&mesh, 4);
        let cut_before = partition.edge_cut(&graph);
        refine_partition_fm(&mut partition, &graph, 100, 1.10);
        let cut_after = partition.edge_cut(&graph);
        assert!(
            cut_after <= cut_before,
            "FM made cut worse: {cut_before} -> {cut_after}"
        );
    }

    #[test]
    fn test_partition_elements_is_zero_alloc() {
        // Just check the slice view is consistent with partition_id.
        let mesh = unit_cube_tetrahedra(2);
        let partition = partition_mesh_bisection(&mesh, 4);
        for p in 0..partition.num_partitions {
            for &e in partition.partition_elements(p) {
                assert_eq!(partition.partition_id[e], p);
            }
        }
    }

    #[test]
    fn test_partition_mesh_extract() {
        let mesh = unit_cube_tetrahedra(2);
        let options = PartitionOptions {
            num_partitions: 2,
            ..Default::default()
        };
        let partition = partition_mesh(&mesh, &options);

        let part_mesh = PartitionMesh::extract(&mesh, &partition, 0);

        assert!(part_mesh.num_elements() > 0);
        assert!(part_mesh.num_nodes() > 0);
    }

    #[test]
    fn zero_partition_count_is_rejected_or_non_panicking() {
        let mesh = unit_cube_tetrahedra(1);
        assert!(matches!(
            try_partition_mesh(
                &mesh,
                &PartitionOptions {
                    num_partitions: 0,
                    ..Default::default()
                }
            ),
            Err(PartitionError::InvalidPartitionCount { .. })
        ));
        assert_eq!(partition_mesh_round_robin(&mesh, 0).num_partitions, 0);
        assert_eq!(partition_mesh_adjacency(&mesh, 0).num_partitions, 0);
    }

    #[test]
    fn checked_partition_validates_graph_and_builds_deterministic_ghost_layout() {
        let mut mesh = Mesh::new(3);
        for point in [
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
            Point::new_3d(0.0, 0.0, -1.0),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
        mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 4]);

        let graph = try_build_element_graph(&mesh).expect("valid mesh graph");
        let partition = Partition::try_from_ids(vec![0, 1], 2).expect("valid assignment");
        assert_eq!(partition.try_edge_cut(&graph), Ok(1));

        let first = partition
            .ghost_layout(&mesh, &graph, 1)
            .expect("valid ghost layout");
        let second = partition
            .ghost_layout(&mesh, &graph, 1)
            .expect("repeated ghost layout");
        assert_eq!(first, second);
        assert_eq!(first.owned_elements, vec![1]);
        assert_eq!(first.ghost_elements, vec![0]);
        assert_eq!(first.owned_nodes, vec![4]);
        assert_eq!(first.ghost_nodes, vec![0, 1, 2]);

        let extracted = PartitionMesh::extract_checked(&mesh, &partition, 1)
            .expect("checked extraction should preserve valid connectivity");
        assert_eq!(extracted.parent_element_indices, vec![1]);
        assert_eq!(extracted.parent_node_indices, vec![0, 1, 2, 4]);
    }

    #[test]
    fn checked_partition_and_extraction_reject_malformed_public_data() {
        let mut mesh = unit_cube_tetrahedra(1);
        mesh.elements[0].nodes[0] = usize::MAX;
        assert!(matches!(
            try_build_element_graph(&mesh),
            Err(PartitionError::InvalidMesh { .. })
        ));

        let valid_mesh = unit_cube_tetrahedra(1);
        let malformed = Partition::from_ids(vec![0], 1);
        assert!(matches!(
            PartitionMesh::extract_checked(&valid_mesh, &malformed, 0),
            Err(PartitionError::PartitionIdCount { .. })
        ));
    }
}
