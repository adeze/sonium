//! P2 / Q2 (second-order Lagrange) element support
//!
//! ## Canonical FEM ordering (sonium internal)
//!
//! For every supported element type, P2/Q2 nodes are laid out in
//! `Element::nodes` as:
//!
//! 1. Vertices (in the existing P1/Q1 order).
//! 2. Edge mid-side nodes, in the order of [`ElementType::local_edges`].
//! 3. Face-centre nodes (Hex27 only), in the order of
//!    [`ElementType::local_faces`].
//! 4. The interior (volume / surface bubble) node, when present
//!    ([`ElementType::p2_has_interior`]).
//!
//! Per element type:
//!
//! | Element       | Total | Layout                                                     |
//! |---------------|-------|------------------------------------------------------------|
//! | Triangle      | 6     | 3 verts + 3 edge midpoints                                 |
//! | Quadrilateral | 9     | 4 verts + 4 edge midpoints + 1 face centre                 |
//! | Tetrahedron   | 10    | 4 verts + 6 edge midpoints                                 |
//! | Hexahedron    | 27    | 8 verts + 12 edge mids + 6 face centres + 1 volume centre  |
//!
//! This is the order the `sonium-fem` Lagrange basis functions are written in
//! (see `lagrange::p2_triangle` and friends), and the order the assembler
//! consumes via `Element::nodes`. Do not rearrange — the FEM crate has no
//! knowledge of mid-side node ordering beyond this contract.
//!
//! ## Gmsh ordering
//!
//! Gmsh writes Tri6 / Qua9 in the same order we use, but Tet10 and Hex27 use a
//! different mid-side ordering. The permutation tables
//! [`tet10_gmsh_to_fem`] / [`hex27_gmsh_to_fem`] convert from Gmsh node order
//! (as parsed by `mshio`) to our FEM order:
//!
//! ```text
//! fem_nodes[i] = gmsh_nodes[perm[i]]
//! ```
//!
//! ### Tet10 (Gmsh)
//!
//! ```text
//!         v
//!         |
//!         2
//!        ,/|`\
//!      ,6  |  `5
//!     ,/   |8   `\
//!    0----4|------1 --> u
//!     `\.  |    ,/
//!        `7.  ,9
//!           `3
//!           w
//! ```
//!
//! Gmsh mid-side node indices:
//! `4=(0,1)`, `5=(1,2)`, `6=(2,0)`, `7=(3,0)`, `8=(3,2)`, `9=(3,1)`.
//!
//! ### Hex27 (Gmsh)
//!
//! Gmsh mid-edge indices 8–19:
//! `8=(0,1)`, `9=(0,3)`, `10=(0,4)`, `11=(1,2)`, `12=(1,5)`, `13=(2,3)`,
//! `14=(2,6)`, `15=(3,7)`, `16=(4,5)`, `17=(4,7)`, `18=(5,6)`, `19=(6,7)`.
//!
//! Gmsh face indices 20–25 (face centres):
//! `20`=z− (0,3,2,1), `21`=y− (0,1,5,4), `22`=x− (0,4,7,3),
//! `23`=x+ (1,2,6,5), `24`=y+ (2,3,7,6), `25`=z+ (4,5,6,7).
//! `26` is the volume centre.

use std::collections::HashMap;

use super::types::{BoundaryFace, Element, ElementType, Mesh, Point};

// ---------------------------------------------------------------------------
// Gmsh ↔ FEM permutations
// ---------------------------------------------------------------------------

/// Permutation mapping Gmsh Tri6 node order onto sonium FEM P2 order.
///
/// Identity — both orders use vertices then edges (0,1), (1,2), (2,0).
pub const TRI6_GMSH_TO_FEM: [usize; 6] = [0, 1, 2, 3, 4, 5];

/// Permutation mapping Gmsh Qua9 node order onto sonium FEM Q2 order.
///
/// Identity — both orders use vertices, edges (0,1)(1,2)(2,3)(3,0), centre.
pub const QUA9_GMSH_TO_FEM: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

/// Permutation mapping Gmsh Tet10 node order onto sonium FEM P2 order.
///
/// FEM edge order: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
/// Gmsh edge order: (0,1) (1,2) (2,0) (3,0) (3,2) (3,1).
pub const TET10_GMSH_TO_FEM: [usize; 10] = [0, 1, 2, 3, 4, 6, 7, 5, 9, 8];

/// Permutation mapping Gmsh Hex27 node order onto sonium FEM Q2 order.
pub const HEX27_GMSH_TO_FEM: [usize; 27] = [
    // vertices (identical to Gmsh)
    0, 1, 2, 3, 4, 5, 6, 7,
    // edge mids: our edges (0,1)(1,2)(2,3)(3,0)(4,5)(5,6)(6,7)(7,4)(0,4)(1,5)(2,6)(3,7)
    // map onto gmsh edges  g8 g11 g13 g9  g16 g18 g19 g17 g10 g12 g14 g15
    8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
    // face centres: our face order
    // bottom (z−), top (z+), front (y−), back (y+), left (x−), right (x+)
    // → gmsh         20      25         21        24        22         23
    20, 25, 21, 24, 22, 23, // volume centre
    26,
];

/// Return the Gmsh→FEM P2 node permutation for an element type, or `None` if
/// the gmsh order coincides with FEM order (Tri6/Qua9).
pub fn gmsh_to_fem_p2_perm(et: ElementType) -> &'static [usize] {
    match et {
        ElementType::Triangle => &TRI6_GMSH_TO_FEM,
        ElementType::Quadrilateral => &QUA9_GMSH_TO_FEM,
        ElementType::Tetrahedron => &TET10_GMSH_TO_FEM,
        ElementType::Hexahedron => &HEX27_GMSH_TO_FEM,
    }
}

/// Reorder a Gmsh-ordered P2 node list into sonium FEM P2 order.
///
/// `gmsh_nodes.len()` must equal `et.num_nodes_p2()`.
pub fn reorder_gmsh_to_fem_p2(et: ElementType, gmsh_nodes: &[usize]) -> Vec<usize> {
    let perm = gmsh_to_fem_p2_perm(et);
    assert_eq!(
        gmsh_nodes.len(),
        perm.len(),
        "expected {} P2 nodes for {:?}, got {}",
        perm.len(),
        et,
        gmsh_nodes.len()
    );
    perm.iter().map(|&i| gmsh_nodes[i]).collect()
}

// ---------------------------------------------------------------------------
// P1 → P2 conversion
// ---------------------------------------------------------------------------

/// Cache for unique mid-side / face / volume nodes created during P2 promotion.
struct P2NodeCache {
    /// Unordered edge `(min,max)` → newly created mid-side node index.
    edge_mid: HashMap<(usize, usize), usize>,
    /// Sorted face-vertex tuple → face-centre node index.
    face_centre: HashMap<Vec<usize>, usize>,
}

impl P2NodeCache {
    fn new() -> Self {
        Self {
            edge_mid: HashMap::new(),
            face_centre: HashMap::new(),
        }
    }

    fn edge_midpoint(&mut self, mesh: &mut Mesh, a: usize, b: usize) -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&idx) = self.edge_mid.get(&key) {
            return idx;
        }
        let p = mesh.nodes[a].midpoint(&mesh.nodes[b]);
        let idx = mesh.nodes.len();
        mesh.nodes.push(p);
        self.edge_mid.insert(key, idx);
        idx
    }

    fn face_centre_node(&mut self, mesh: &mut Mesh, face_vertices: &[usize]) -> usize {
        let mut key: Vec<usize> = face_vertices.to_vec();
        key.sort_unstable();
        if let Some(&idx) = self.face_centre.get(&key) {
            return idx;
        }
        let n = face_vertices.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &v in face_vertices {
            cx += mesh.nodes[v].x;
            cy += mesh.nodes[v].y;
            cz += mesh.nodes[v].z;
        }
        let p = Point::new_3d(cx / n, cy / n, cz / n);
        let idx = mesh.nodes.len();
        mesh.nodes.push(p);
        self.face_centre.insert(key, idx);
        idx
    }
}

/// Promote a P1/Q1 [`Mesh`] to P2/Q2.
///
/// For each element, edge mid-side nodes are added (shared between elements
/// that touch the same edge), then face-centre nodes (shared across hexes that
/// share a face), then the per-element volume / surface bubble node where
/// applicable. The returned mesh has the same `elements.len()` and element IDs
/// as the input; only `nodes` and per-element `nodes` lists grow.
///
/// Boundary face/edge node lists are also promoted: 2-node boundary edges
/// become 3-node (vertices + edge midpoint), 3-node triangle boundaries become
/// 6-node, 4-node quad boundaries become 9-node. This keeps `BoundaryFace`
/// consistent with the volume element ordering, so an FEM trace operator can
/// pick up the higher-order DOFs by following the existing
/// `(element_idx, local_idx)` reference.
///
/// # Panics
///
/// Panics if any element is already P2 (mixed-order meshes aren't supported).
pub fn promote_to_p2(mesh: &Mesh) -> Mesh {
    for elem in &mesh.elements {
        assert_eq!(
            elem.polynomial_order(),
            1,
            "promote_to_p2: element {} is not P1",
            elem.id
        );
    }

    let mut out = Mesh::new(mesh.dimension);
    out.physical_groups = mesh.physical_groups.clone();
    // Copy original vertex nodes verbatim; they keep indices 0..mesh.num_nodes().
    out.nodes = mesh.nodes.clone();
    // Reset cached fields explicitly (we are rebuilding the topology).
    out.node_to_elements = None;
    out.topology = None;

    let mut cache = P2NodeCache::new();
    out.elements.reserve(mesh.elements.len());

    for elem in &mesh.elements {
        let et = elem.element_type;
        let verts: Vec<usize> = elem.vertices().to_vec();
        let mut p2_nodes: Vec<usize> = Vec::with_capacity(et.num_nodes_p2());
        p2_nodes.extend_from_slice(&verts);

        // Edge midpoints in canonical FEM order
        for &(la, lb) in et.local_edges() {
            let a = verts[la];
            let b = verts[lb];
            p2_nodes.push(cache.edge_midpoint(&mut out, a, b));
        }

        // Face centres (Hex27 only)
        if et.p2_has_face_centres() {
            for face_local in et.local_faces() {
                let face_global: Vec<usize> = face_local.iter().map(|&li| verts[li]).collect();
                p2_nodes.push(cache.face_centre_node(&mut out, &face_global));
            }
        }

        // Interior / volume bubble (Q2 quad and Hex27)
        if et.p2_has_interior() {
            let centre = element_centroid_from_vertices(&out.nodes, &verts);
            let idx = out.nodes.len();
            out.nodes.push(centre);
            p2_nodes.push(idx);
        }

        debug_assert_eq!(p2_nodes.len(), et.num_nodes_p2());

        let mut e = Element::new(et, p2_nodes, elem.id);
        e.parent_id = elem.parent_id;
        e.level = elem.level;
        e.marker = elem.marker;
        out.elements.push(e);
    }

    out.next_element_id = mesh.next_element_id;

    // Promote boundary faces/edges to carry higher-order nodes.
    out.boundaries.reserve(mesh.boundaries.len());
    for bf in &mesh.boundaries {
        let promoted = promote_boundary_face(bf, &mut out, &mut cache);
        out.boundaries.push(promoted);
    }

    out
}

fn element_centroid_from_vertices(nodes: &[Point], verts: &[usize]) -> Point {
    let n = verts.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for &v in verts {
        cx += nodes[v].x;
        cy += nodes[v].y;
        cz += nodes[v].z;
    }
    Point::new_3d(cx / n, cy / n, cz / n)
}

fn promote_boundary_face(
    bf: &BoundaryFace,
    out: &mut Mesh,
    cache: &mut P2NodeCache,
) -> BoundaryFace {
    let mut promoted_nodes = bf.nodes.clone();
    match bf.nodes.len() {
        // 2D boundary edge → Line3: append mid-edge node.
        2 => {
            let m = cache.edge_midpoint(out, bf.nodes[0], bf.nodes[1]);
            promoted_nodes.push(m);
        }
        // Triangle boundary face → Tri6.
        3 => {
            let m01 = cache.edge_midpoint(out, bf.nodes[0], bf.nodes[1]);
            let m12 = cache.edge_midpoint(out, bf.nodes[1], bf.nodes[2]);
            let m20 = cache.edge_midpoint(out, bf.nodes[2], bf.nodes[0]);
            promoted_nodes.extend_from_slice(&[m01, m12, m20]);
        }
        // Quad boundary face → Qua9: 4 edge mids + face centre.
        4 => {
            let m01 = cache.edge_midpoint(out, bf.nodes[0], bf.nodes[1]);
            let m12 = cache.edge_midpoint(out, bf.nodes[1], bf.nodes[2]);
            let m23 = cache.edge_midpoint(out, bf.nodes[2], bf.nodes[3]);
            let m30 = cache.edge_midpoint(out, bf.nodes[3], bf.nodes[0]);
            let centre = cache.face_centre_node(out, &bf.nodes);
            promoted_nodes.extend_from_slice(&[m01, m12, m23, m30, centre]);
        }
        _ => {
            panic!(
                "promote_boundary_face: unexpected boundary node count {}",
                bf.nodes.len()
            );
        }
    }
    BoundaryFace {
        nodes: promoted_nodes,
        boundary_type: bf.boundary_type,
        marker: bf.marker,
        element_idx: bf.element_idx,
        local_idx: bf.local_idx,
    }
}

impl Mesh {
    /// Promote this P1/Q1 mesh to P2/Q2. See [`promote_to_p2`].
    pub fn to_p2(&self) -> Mesh {
        promote_to_p2(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{
        rectangular_mesh_quads, rectangular_mesh_triangles, unit_cube_hexahedra,
        unit_cube_tetrahedra,
    };

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn p2_triangle_node_layout() {
        let m = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1).to_p2();
        // One quad cell → 2 triangles. P1: 4 vertices.
        // Unique edges in this mesh: 5 (4 outer + 1 diagonal). So P2 has 9 nodes.
        assert_eq!(m.elements.len(), 2);
        assert_eq!(m.num_nodes(), 4 + 5);
        for e in &m.elements {
            assert_eq!(e.nodes.len(), 6);
            assert!(e.is_p2());
        }
        // Verify edge midpoints lie at the actual edge midpoints in space.
        for e in &m.elements {
            let v = e.vertices().to_vec();
            for (k, &(la, lb)) in ElementType::Triangle.local_edges().iter().enumerate() {
                let mid_idx = e.nodes[3 + k];
                let pm = m.nodes[mid_idx];
                let pa = m.nodes[v[la]];
                let pb = m.nodes[v[lb]];
                assert!(approx_eq(pm.x, 0.5 * (pa.x + pb.x)));
                assert!(approx_eq(pm.y, 0.5 * (pa.y + pb.y)));
            }
        }
    }

    #[test]
    fn q2_quad_has_centre_node() {
        // 2×2 grid of quads → 9 vertices, 12 unique edges, 4 face centres.
        let m = rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 2, 2).to_p2();
        assert_eq!(m.elements.len(), 4);
        assert_eq!(m.num_nodes(), 9 + 12 + 4);
        for e in &m.elements {
            assert_eq!(e.nodes.len(), 9);
            // Last node is the face centre — should coincide with the
            // vertex centroid for an axis-aligned square.
            let v = e.vertices();
            let cx: f64 = v.iter().map(|&i| m.nodes[i].x).sum::<f64>() / 4.0;
            let cy: f64 = v.iter().map(|&i| m.nodes[i].y).sum::<f64>() / 4.0;
            let pc = m.nodes[e.nodes[8]];
            assert!(approx_eq(pc.x, cx));
            assert!(approx_eq(pc.y, cy));
        }
    }

    #[test]
    fn p2_tet_node_layout() {
        let m = unit_cube_tetrahedra(1).to_p2();
        // 6 tets per unit cube, P2 → 10 nodes each.
        assert_eq!(m.elements.len(), 6);
        for e in &m.elements {
            assert_eq!(e.element_type, ElementType::Tetrahedron);
            assert_eq!(e.nodes.len(), 10);
            // Verify each mid-side node is the midpoint of the corresponding
            // FEM-ordered edge.
            let v = e.vertices().to_vec();
            for (k, &(la, lb)) in ElementType::Tetrahedron.local_edges().iter().enumerate() {
                let mid_idx = e.nodes[4 + k];
                let pm = m.nodes[mid_idx];
                let pa = m.nodes[v[la]];
                let pb = m.nodes[v[lb]];
                assert!(approx_eq(pm.x, 0.5 * (pa.x + pb.x)));
                assert!(approx_eq(pm.y, 0.5 * (pa.y + pb.y)));
                assert!(approx_eq(pm.z, 0.5 * (pa.z + pb.z)));
            }
        }
    }

    #[test]
    fn q2_hex_full_layout() {
        let m = unit_cube_hexahedra(1).to_p2();
        // Single hex, P2 → 27 nodes total.
        assert_eq!(m.elements.len(), 1);
        assert_eq!(m.num_nodes(), 27);
        let e = &m.elements[0];
        assert_eq!(e.nodes.len(), 27);
        let v = e.vertices().to_vec();

        // Edge mids (8..20)
        for (k, &(la, lb)) in ElementType::Hexahedron.local_edges().iter().enumerate() {
            let pm = m.nodes[e.nodes[8 + k]];
            let pa = m.nodes[v[la]];
            let pb = m.nodes[v[lb]];
            assert!(approx_eq(pm.x, 0.5 * (pa.x + pb.x)));
            assert!(approx_eq(pm.y, 0.5 * (pa.y + pb.y)));
            assert!(approx_eq(pm.z, 0.5 * (pa.z + pb.z)));
        }
        // Face centres (20..26): one per local face
        for (k, face) in ElementType::Hexahedron.local_faces().iter().enumerate() {
            let pc = m.nodes[e.nodes[20 + k]];
            let cx: f64 = face.iter().map(|&li| m.nodes[v[li]].x).sum::<f64>() / 4.0;
            let cy: f64 = face.iter().map(|&li| m.nodes[v[li]].y).sum::<f64>() / 4.0;
            let cz: f64 = face.iter().map(|&li| m.nodes[v[li]].z).sum::<f64>() / 4.0;
            assert!(approx_eq(pc.x, cx));
            assert!(approx_eq(pc.y, cy));
            assert!(approx_eq(pc.z, cz));
        }
        // Volume centre (26): centroid
        let cv = m.nodes[e.nodes[26]];
        assert!(approx_eq(cv.x, 0.5));
        assert!(approx_eq(cv.y, 0.5));
        assert!(approx_eq(cv.z, 0.5));
    }

    #[test]
    fn shared_edges_use_single_midpoint() {
        // Two triangles sharing the diagonal must share the midpoint node.
        let m = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1).to_p2();
        let e0 = &m.elements[0];
        let e1 = &m.elements[1];
        // Both triangles share the diagonal edge (vertex 0 → vertex 2 in
        // rectangular_mesh_triangles' first cell).
        let v0 = e0.vertices();
        let shared_a = v0[0];
        let shared_b = v0[2];
        let find_mid = |e: &Element, a: usize, b: usize| -> usize {
            for (k, &(la, lb)) in ElementType::Triangle.local_edges().iter().enumerate() {
                let ea = e.nodes[la];
                let eb = e.nodes[lb];
                if (ea == a && eb == b) || (ea == b && eb == a) {
                    return e.nodes[3 + k];
                }
            }
            panic!("edge ({a},{b}) not found in element {:?}", e.nodes);
        };
        let m0 = find_mid(e0, shared_a, shared_b);
        let m1 = find_mid(e1, shared_a, shared_b);
        assert_eq!(m0, m1);
    }

    #[test]
    fn boundary_faces_are_promoted() {
        let m = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2).to_p2();
        for b in &m.boundaries {
            assert_eq!(b.nodes.len(), 3, "2D boundary edges should be Line3 in P2");
        }
    }

    #[test]
    fn tet10_permutation_round_trip() {
        // Identity should be a permutation: applying it to an indexed list
        // produces the same indexed list (we only use this when reading gmsh,
        // but the table must be a valid permutation).
        let p = &TET10_GMSH_TO_FEM;
        let mut seen = [false; 10];
        for &i in p {
            assert!(i < 10);
            assert!(!seen[i], "duplicate index {} in TET10 permutation", i);
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn hex27_permutation_is_a_permutation() {
        let p = &HEX27_GMSH_TO_FEM;
        let mut seen = [false; 27];
        for &i in p {
            assert!(i < 27);
            assert!(!seen[i], "duplicate index {} in HEX27 permutation", i);
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    /// Geometric round-trip: build a synthetic Gmsh-ordered Tet10 where every
    /// node has a position derived from its Gmsh role (vertex / specific
    /// mid-edge), apply [`reorder_gmsh_to_fem_p2`], and check that every FEM
    /// position lands on the correct sonium-FEM edge midpoint.
    ///
    /// This regresses against bugs in the [`TET10_GMSH_TO_FEM`] table: if any
    /// entry is wrong, the resulting FEM node will be at the wrong midpoint.
    #[test]
    fn tet10_perm_maps_gmsh_to_fem_geometrically() {
        // Reference tet vertex coordinates (arbitrary but distinct).
        let v: [(f64, f64, f64); 4] = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ];
        let mid = |a: usize, b: usize| -> (f64, f64, f64) {
            (
                0.5 * (v[a].0 + v[b].0),
                0.5 * (v[a].1 + v[b].1),
                0.5 * (v[a].2 + v[b].2),
            )
        };
        // Gmsh Tet10 ordering: 0..3 vertices, then
        // 4=(0,1), 5=(1,2), 6=(2,0), 7=(3,0), 8=(3,2), 9=(3,1).
        let gmsh_coords: [(f64, f64, f64); 10] = [
            v[0],
            v[1],
            v[2],
            v[3],
            mid(0, 1),
            mid(1, 2),
            mid(2, 0),
            mid(3, 0),
            mid(3, 2),
            mid(3, 1),
        ];
        // Use the node *index* as the "tag" — reorder operates on indices.
        let gmsh_nodes: Vec<usize> = (0..10).collect();
        let fem_nodes = reorder_gmsh_to_fem_p2(ElementType::Tetrahedron, &gmsh_nodes);

        // FEM expected positions: vertices, then mids in FEM edge order.
        let fem_expected: [(f64, f64, f64); 10] = [
            v[0],
            v[1],
            v[2],
            v[3],
            mid(0, 1),
            mid(0, 2),
            mid(0, 3),
            mid(1, 2),
            mid(1, 3),
            mid(2, 3),
        ];
        for (i, &g_idx) in fem_nodes.iter().enumerate() {
            let got = gmsh_coords[g_idx];
            let want = fem_expected[i];
            assert!(
                (got.0 - want.0).abs() < 1e-15
                    && (got.1 - want.1).abs() < 1e-15
                    && (got.2 - want.2).abs() < 1e-15,
                "Tet10 FEM[{i}] expected {want:?}, got {got:?}"
            );
        }
    }

    /// Same geometric round-trip for Hex27: gmsh-ordered positions go in, FEM
    /// ordering comes out, and every position must land on its sonium-FEM
    /// canonical site (vertex, edge midpoint, face centroid, volume centre).
    #[test]
    fn hex27_perm_maps_gmsh_to_fem_geometrically() {
        // Reference hex vertices (standard sonium/Gmsh ordering).
        let v: [(f64, f64, f64); 8] = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ];
        let mid = |a: usize, b: usize| -> (f64, f64, f64) {
            (
                0.5 * (v[a].0 + v[b].0),
                0.5 * (v[a].1 + v[b].1),
                0.5 * (v[a].2 + v[b].2),
            )
        };
        let centroid = |face: &[usize]| -> (f64, f64, f64) {
            let n = face.len() as f64;
            let mut s = (0.0, 0.0, 0.0);
            for &i in face {
                s.0 += v[i].0;
                s.1 += v[i].1;
                s.2 += v[i].2;
            }
            (s.0 / n, s.1 / n, s.2 / n)
        };
        let vol_centre = centroid(&[0, 1, 2, 3, 4, 5, 6, 7]);

        // Gmsh Hex27 ordering — see docs at top of this file.
        let gmsh_coords: [(f64, f64, f64); 27] = [
            v[0],
            v[1],
            v[2],
            v[3],
            v[4],
            v[5],
            v[6],
            v[7],
            // Edges 8..19
            mid(0, 1), // 8
            mid(0, 3), // 9
            mid(0, 4), // 10
            mid(1, 2), // 11
            mid(1, 5), // 12
            mid(2, 3), // 13
            mid(2, 6), // 14
            mid(3, 7), // 15
            mid(4, 5), // 16
            mid(4, 7), // 17
            mid(5, 6), // 18
            mid(6, 7), // 19
            // Faces 20..25
            centroid(&[0, 3, 2, 1]), // 20  z−
            centroid(&[0, 1, 5, 4]), // 21  y−
            centroid(&[0, 4, 7, 3]), // 22  x−
            centroid(&[1, 2, 6, 5]), // 23  x+
            centroid(&[2, 3, 7, 6]), // 24  y+
            centroid(&[4, 5, 6, 7]), // 25  z+
            // Volume
            vol_centre, // 26
        ];

        let gmsh_nodes: Vec<usize> = (0..27).collect();
        let fem_nodes = reorder_gmsh_to_fem_p2(ElementType::Hexahedron, &gmsh_nodes);

        // FEM expected positions, in canonical order.
        let mut fem_expected: Vec<(f64, f64, f64)> = Vec::with_capacity(27);
        fem_expected.extend_from_slice(&v);
        for &(a, b) in ElementType::Hexahedron.local_edges() {
            fem_expected.push(mid(a, b));
        }
        for face in ElementType::Hexahedron.local_faces() {
            fem_expected.push(centroid(face));
        }
        fem_expected.push(vol_centre);

        assert_eq!(fem_expected.len(), 27);
        for (i, &g_idx) in fem_nodes.iter().enumerate() {
            let got = gmsh_coords[g_idx];
            let want = fem_expected[i];
            assert!(
                (got.0 - want.0).abs() < 1e-15
                    && (got.1 - want.1).abs() < 1e-15
                    && (got.2 - want.2).abs() < 1e-15,
                "Hex27 FEM[{i}] expected {want:?}, got {got:?}"
            );
        }
    }

    /// Tri6 / Qua9 permutations are identity, but check that the reorder
    /// function honours them rather than silently re-interpreting node order.
    #[test]
    fn tri6_qua9_perm_geometric() {
        let nodes: Vec<usize> = (0..6).collect();
        let fem = reorder_gmsh_to_fem_p2(ElementType::Triangle, &nodes);
        assert_eq!(fem, nodes);

        let nodes: Vec<usize> = (0..9).collect();
        let fem = reorder_gmsh_to_fem_p2(ElementType::Quadrilateral, &nodes);
        assert_eq!(fem, nodes);
    }

    #[test]
    #[should_panic(expected = "expected 10")]
    fn reorder_rejects_wrong_node_count() {
        reorder_gmsh_to_fem_p2(ElementType::Tetrahedron, &[0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "is not P1")]
    fn promote_to_p2_rejects_p2_input() {
        let m = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1).to_p2();
        // Promoting twice should panic.
        let _ = m.to_p2();
    }
}
