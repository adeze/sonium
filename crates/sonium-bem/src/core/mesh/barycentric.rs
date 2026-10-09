//! Barycentric refinement of a surface triangulation.
//!
//! For each input triangle `T = (v0, v1, v2)`, the barycentric refinement
//! introduces three edge midpoints `(e01, e12, e20)` and one centroid
//! `c`, then splits `T` into six sub-triangles
//! `(v0, e01, c)`, `(e01, v1, c)`, `(v1, e12, c)`, `(e12, v2, c)`,
//! `(v2, e20, c)`, `(e20, v0, c)`. Edge midpoints are *deduplicated*
//! across triangles that share an edge so the refined mesh stays
//! conforming (manifold).
//!
//! Layout of `refined_nodes`:
//!   ```text
//!   [ 0 .. n_orig_nodes )            — original vertices, unchanged
//!   [ n_orig_nodes .. + n_edge )     — edge midpoints, deduplicated
//!   [ above .. + n_orig_tri )        — per-triangle centroids
//!   ```
//!
//! This is the core construction the Stage-1 dual-mass preconditioner
//! (see `assembly::surface_mass`) is built on.

// Array2 use removed (Mesh::nodes migrated to Vec<[f64; 3]>)
use crate::core::types::{Element, ElementType, Mesh};
use std::collections::HashMap;

/// Refined-mesh handles indexing back into the input mesh.
pub struct BarycentricRefinement {
    /// All refined-mesh node coordinates, in the layout above.
    pub refined_nodes: Vec<[f64; 3]>,
    /// Refined sub-triangles as global node-index triples (CCW with
    /// inherited normal orientation from the parent triangle).
    pub refined_triangles: Vec<[usize; 3]>,
    /// Number of original nodes (offsets the start of edge midpoints).
    pub n_orig_nodes: usize,
    /// Number of unique edge midpoints (offsets the start of centroids).
    pub n_edge_nodes: usize,
    /// Number of triangle centroids — exactly one per original triangle.
    pub n_centroid_nodes: usize,
    /// For each original triangle index, the indices of its 6 child
    /// sub-triangles in `refined_triangles` (in the order described in
    /// the module docs).
    pub primal_to_children: Vec<[usize; 6]>,
    /// Map `(min_node, max_node)` → midpoint global index. Useful when
    /// later phases need to look up an edge's midpoint without rebuilding
    /// the topology.
    pub edge_to_midpoint: HashMap<(usize, usize), usize>,
}

impl BarycentricRefinement {
    /// Total node count of the refined mesh.
    pub fn n_refined_nodes(&self) -> usize {
        self.refined_nodes.len()
    }

    /// Total triangle count of the refined mesh: `6 × n_orig_tri`.
    pub fn n_refined_triangles(&self) -> usize {
        self.refined_triangles.len()
    }

    /// Global index of the midpoint node for an oriented edge `(a, b)`.
    /// Order-independent: `edge_midpoint(a, b) == edge_midpoint(b, a)`.
    pub fn edge_midpoint(&self, a: usize, b: usize) -> Option<usize> {
        let key = if a < b { (a, b) } else { (b, a) };
        self.edge_to_midpoint.get(&key).copied()
    }
}

/// Build the barycentric refinement from a slice of elements + the
/// node array directly. Use this when a `Mesh` value isn't readily
/// available (e.g. inside the BEM solver where `(elements, nodes)` is
/// the working form). Identical semantics to `refine_surface_mesh`.
pub fn refine_from_elements(elements: &[Element], nodes: &[[f64; 3]]) -> BarycentricRefinement {
    refine_inner(elements, nodes)
}

/// Build the barycentric refinement of a surface triangle mesh. Skips
/// non-triangular elements (the BEM mesh API allows quads but the
/// production tbem path is Tri3-only) and elements marked as evaluation-
/// only — they don't carry boundary data.
pub fn refine_surface_mesh(mesh: &Mesh) -> BarycentricRefinement {
    refine_inner(&mesh.elements, &mesh.nodes)
}

fn refine_inner(elements: &[Element], nodes_arr: &[[f64; 3]]) -> BarycentricRefinement {
    let mesh_nodes = nodes_arr;
    let n_orig_nodes = mesh_nodes.len();

    // Pass 1 — collect surface triangles from the input mesh and
    // discover unique edges. An edge is identified by its two endpoint
    // global indices, sorted ascending so we don't double-count when
    // two triangles share the edge.
    let mut surface_tris: Vec<[usize; 3]> = Vec::with_capacity(elements.len());
    for elem in elements {
        if elem.property.is_evaluation() {
            continue;
        }
        if !matches!(elem.element_type, ElementType::Tri3) {
            continue;
        }
        debug_assert_eq!(
            elem.connectivity.len(),
            3,
            "Tri3 element must have 3 connectivity entries"
        );
        surface_tris.push([
            elem.connectivity[0],
            elem.connectivity[1],
            elem.connectivity[2],
        ]);
    }
    let n_orig_tri = surface_tris.len();

    let mut edge_to_midpoint: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in &surface_tris {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            edge_to_midpoint.entry(key).or_insert(0); // placeholder, real index assigned below
        }
    }

    // Pass 2 — assign sequential global indices to each unique edge
    // midpoint, in deterministic key-sorted order so two builds of the
    // same mesh produce the same refinement (helpful for tests + cache).
    let mut edge_keys: Vec<(usize, usize)> = edge_to_midpoint.keys().copied().collect();
    edge_keys.sort();
    for (idx, key) in edge_keys.iter().enumerate() {
        edge_to_midpoint.insert(*key, n_orig_nodes + idx);
    }
    let n_edge_nodes = edge_keys.len();

    // Pass 3 — allocate the refined node array and fill in the new
    // points (edge midpoints first, then centroids).
    let n_centroid_nodes = n_orig_tri;
    let n_refined_nodes = n_orig_nodes + n_edge_nodes + n_centroid_nodes;
    let mut refined_nodes: Vec<[f64; 3]> = vec![[0.0; 3]; n_refined_nodes];

    // Copy original nodes verbatim.
    for i in 0..n_orig_nodes {
        for d in 0..3 {
            refined_nodes[i][d] = mesh_nodes[i][d];
        }
    }

    // Edge midpoints.
    for (key, &mid_idx) in edge_to_midpoint.iter() {
        let (a, b) = *key;
        for d in 0..3 {
            refined_nodes[mid_idx][d] = 0.5 * (mesh_nodes[a][d] + mesh_nodes[b][d]);
        }
    }

    // Centroids (one per surface triangle).
    let centroid_offset = n_orig_nodes + n_edge_nodes;
    for (t_idx, tri) in surface_tris.iter().enumerate() {
        let cen_idx = centroid_offset + t_idx;
        for d in 0..3 {
            refined_nodes[cen_idx][d] =
                (mesh_nodes[tri[0]][d] + mesh_nodes[tri[1]][d] + mesh_nodes[tri[2]][d]) / 3.0;
        }
    }

    // Pass 4 — emit the 6 child sub-triangles per original triangle.
    let mut refined_triangles: Vec<[usize; 3]> = Vec::with_capacity(6 * n_orig_tri);
    let mut primal_to_children: Vec<[usize; 6]> = Vec::with_capacity(n_orig_tri);

    for (t_idx, tri) in surface_tris.iter().enumerate() {
        let v0 = tri[0];
        let v1 = tri[1];
        let v2 = tri[2];
        let e01 = edge_to_midpoint[&if v0 < v1 { (v0, v1) } else { (v1, v0) }];
        let e12 = edge_to_midpoint[&if v1 < v2 { (v1, v2) } else { (v2, v1) }];
        let e20 = edge_to_midpoint[&if v2 < v0 { (v2, v0) } else { (v0, v2) }];
        let c = centroid_offset + t_idx;

        // Six sub-triangles, ordered (v0-e01-c), (e01-v1-c), (v1-e12-c),
        // (e12-v2-c), (v2-e20-c), (e20-v0-c). Winding inherits the
        // parent's: walking around the parent in order
        // (v0, e01, v1, e12, v2, e20, v0) and flipping inward to the
        // centroid keeps each sub-triangle's outward normal aligned
        // with the parent's.
        let base = refined_triangles.len();
        refined_triangles.push([v0, e01, c]);
        refined_triangles.push([e01, v1, c]);
        refined_triangles.push([v1, e12, c]);
        refined_triangles.push([e12, v2, c]);
        refined_triangles.push([v2, e20, c]);
        refined_triangles.push([e20, v0, c]);
        primal_to_children.push([base, base + 1, base + 2, base + 3, base + 4, base + 5]);
    }

    BarycentricRefinement {
        refined_nodes,
        refined_triangles,
        n_orig_nodes,
        n_edge_nodes,
        n_centroid_nodes,
        primal_to_children,
        edge_to_midpoint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;

    #[inline]
    fn tri_area(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> f64 {
        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cx = v1[1] * v2[2] - v1[2] * v2[1];
        let cy = v1[2] * v2[0] - v1[0] * v2[2];
        let cz = v1[0] * v2[1] - v1[1] * v2[0];
        0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
    }

    fn node(refined: &BarycentricRefinement, idx: usize) -> [f64; 3] {
        [
            refined.refined_nodes[idx][0],
            refined.refined_nodes[idx][1],
            refined.refined_nodes[idx][2],
        ]
    }

    #[test]
    fn refinement_node_and_triangle_counts_match_euler() {
        // Subdivision-2 icosphere: 80 original triangles, 42 vertices,
        // 120 edges (Euler V - E + F = 2 ⇒ E = V + F - 2 = 120).
        let mesh = generate_icosphere_mesh(1.0, 2);
        let n_orig_tri = mesh.elements.len();
        assert_eq!(
            n_orig_tri, 320,
            "subdiv-2 icosphere should have 320 elements"
        );
        let n_orig_nodes = mesh.nodes.len();

        let refined = refine_surface_mesh(&mesh);

        // Triangles: exactly 6 per parent.
        assert_eq!(refined.n_refined_triangles(), 6 * n_orig_tri);
        // Nodes: original vertices + unique edge midpoints + per-triangle
        // centroids. For a closed manifold triangulation Euler's relation
        // gives E = V + F - 2 (= 3F/2 since each tri has 3 edges and each
        // edge is shared by 2 tris). Check both consistency relations.
        let expected_edges = 3 * n_orig_tri / 2;
        assert_eq!(
            refined.n_edge_nodes, expected_edges,
            "edge dedup must hit Euler count: got {}, expected {}",
            refined.n_edge_nodes, expected_edges
        );
        assert_eq!(refined.n_centroid_nodes, n_orig_tri);
        assert_eq!(
            refined.n_refined_nodes(),
            n_orig_nodes + refined.n_edge_nodes + n_orig_tri
        );
    }

    #[test]
    fn refinement_preserves_total_surface_area() {
        // The barycentric refinement is a partition of each original
        // triangle into 6 sub-triangles by straight-line cuts at the
        // edge midpoints and centroid. Sum of sub-areas must equal the
        // sum of original areas to round-off.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let original_area: f64 = mesh.elements.iter().map(|e| e.area).sum();

        let refined = refine_surface_mesh(&mesh);
        let refined_area: f64 = refined
            .refined_triangles
            .iter()
            .map(|tri| {
                tri_area(
                    node(&refined, tri[0]),
                    node(&refined, tri[1]),
                    node(&refined, tri[2]),
                )
            })
            .sum();

        let rel = (refined_area - original_area).abs() / original_area.max(1e-30);
        assert!(
            rel < 1e-12,
            "refinement should preserve area: original {original_area:.6}, \
             refined {refined_area:.6}, rel {rel:.3e}"
        );
    }

    #[test]
    fn each_centroid_belongs_to_exactly_six_sub_triangles() {
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let centroid_start = refined.n_orig_nodes + refined.n_edge_nodes;
        for c_idx in centroid_start..refined.n_refined_nodes() {
            let count = refined
                .refined_triangles
                .iter()
                .filter(|tri| tri.contains(&c_idx))
                .count();
            assert_eq!(
                count, 6,
                "centroid node {c_idx} should be in exactly 6 sub-triangles, got {count}"
            );
        }
    }

    #[test]
    fn each_edge_midpoint_belongs_to_at_most_four_sub_triangles() {
        // Manifold property: each edge is shared by ≤ 2 parent triangles
        // (= 2 for closed surfaces; = 1 on a boundary). Each parent
        // contributes 2 of its 6 sub-triangles touching the midpoint
        // (the two whose long edge is half of this parent edge), so the
        // shared midpoint sits in 4 sub-triangles on a closed mesh.
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let centroid_start = refined.n_orig_nodes + refined.n_edge_nodes;
        for mid_idx in refined.n_orig_nodes..centroid_start {
            let count = refined
                .refined_triangles
                .iter()
                .filter(|tri| tri.contains(&mid_idx))
                .count();
            // Closed icosphere: every interior edge is shared by 2
            // triangles → 4 child sub-triangles.
            assert_eq!(
                count, 4,
                "edge-midpoint node {mid_idx} on a closed manifold should sit \
                 in exactly 4 sub-triangles, got {count}"
            );
        }
    }

    #[test]
    fn edge_midpoint_is_geometric_midpoint() {
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        for (&(a, b), &mid_idx) in refined.edge_to_midpoint.iter() {
            let pa = node(&refined, a);
            let pb = node(&refined, b);
            let pm = node(&refined, mid_idx);
            for d in 0..3 {
                assert!(
                    (pm[d] - 0.5 * (pa[d] + pb[d])).abs() < 1e-12,
                    "midpoint {mid_idx} of edge ({a}, {b}) is not geometric midpoint"
                );
            }
        }
    }

    #[test]
    fn primal_to_children_maps_into_refined_triangles() {
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let n_refined = refined.refined_triangles.len();
        for (parent_idx, kids) in refined.primal_to_children.iter().enumerate() {
            for &child_idx in kids {
                assert!(
                    child_idx < n_refined,
                    "parent {parent_idx} child idx {child_idx} out of range"
                );
            }
            // The 6 children must all be distinct.
            for i in 0..6 {
                for j in (i + 1)..6 {
                    assert_ne!(kids[i], kids[j]);
                }
            }
        }
    }

    #[test]
    fn edge_midpoint_lookup_is_order_independent() {
        let mesh = generate_icosphere_mesh(1.0, 2);
        let refined = refine_surface_mesh(&mesh);
        let elem = &mesh.elements[0];
        let v0 = elem.connectivity[0];
        let v1 = elem.connectivity[1];
        let m_ab = refined.edge_midpoint(v0, v1).unwrap();
        let m_ba = refined.edge_midpoint(v1, v0).unwrap();
        assert_eq!(m_ab, m_ba);
    }
}
