//! Adversarial quality corpus: parametric defect families with auditable coverage.
//!
//! Coverage spec (bounded-hybrid decision, Stage 1): the published Thingi10K
//! defect taxonomy (Zhou & Jacobson 2016 — ~45% of models fail a
//! self-intersection-free assumption, 22.5% are non-manifold, plus
//! degenerate-triangle/aspect-ratio/coplanar-intersection quality measures)
//! mapped onto synthetic fixtures, so no corpus redistribution is needed.
//! Mapping (taxonomy class -> fixture -> assertion):
//!
//! - non-manifold edges -> `flap_mesh` (3 tris, one shared edge) ->
//!   `check_manifold` flags exactly the shared edge; `open_box_mesh`
//!   (boundary-only) stays manifold. Vertex-only non-manifolds (bowties)
//!   are invisible to edge-owner counting: documented, untested.
//! - degenerate triangles -> `degenerate_triangle_mesh` (coincident nodes) ->
//!   metrics stay finite and score the element worst.
//! - slivers (aspect-ratio tail) -> `sliver_mesh` + altitude sweep ->
//!   `mean_ratio` strictly decreases as altitude -> 0.
//! - thin gaps/walls -> `thin_gap_box` gap sweep 1e-3..1e-12 -> positive
//!   volumes and finite quality throughout (no NaN cliff).
//! - self-intersection (geometric, not topological) -> `interpenetrating_boxes`
//!   -> bounded termination + finite metrics; `check_manifold` cannot see
//!   geometric intersection (connectivity is still 2-manifold): documented.
//! - boundary conformity -> `backward_facing_step_triangles` -> every
//!   boundary face carries a nonzero marker from the documented set.
//! - disjoint components -> `two_disjoint_boxes` -> report stays clean
//!   (components do not confuse edge-owner counting).
//!
//! Known `check_manifold` limits (documented, not fixed here): the
//! infallible entry fail-open reports clean on dimension-mismatched meshes
//! (element dimension != mesh dimension) — use `try_check_manifold` for
//! untrusted meshes. Surface fixtures below are therefore `Mesh::new(2)`.
//!
//! Positive volumes on simple point clouds do not establish recovery of thin
//! gaps, boundary conformity or sliver quality on real CAD. Full-corpus
//! validation runs (meshing all of Thingi10K) are deferred: separate
//! compute/storage project.

use sonium_mesher::mesh::quality::per_element_quality;
use sonium_mesher::mesh::tetrahedralize::{TetrahedralizeConfig, tetrahedralize_points_checked};
use sonium_mesher::mesh::validate::check_manifold;
use sonium_mesher::mesh::{
    ElementType, Mesh, Point, backward_facing_step_triangles, box_mesh_tetrahedra,
    try_box_mesh_tetrahedra,
};

/// A sliver tetrahedron: unit base triangle, fourth vertex 1e-12 above the
/// base plane. Valid (positive volume) but arbitrarily bad quality.
fn sliver_mesh() -> Mesh {
    let mut mesh = Mesh::new(3);
    mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
    mesh.add_node(Point::new_3d(1.0 / 3.0, 1.0 / 3.0, 1e-12));
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh
}

/// Three triangles sharing one edge (a "flap"): the shared edge has three
/// owners, the textbook non-manifold edge from print-oriented CAD.
fn flap_mesh() -> Mesh {
    // NOTE: Mesh::new(2), not 3: the connectivity contract requires element
    // dimension == mesh dimension, so 3D-embedded surfaces are 2D meshes
    // with finite z-coordinates. Mesh::new(3) with triangles is invalid
    // connectivity, on which check_manifold fail-open reports clean.
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point::new_3d(0.0, 0.0, 0.0)); // 0
    mesh.add_node(Point::new_3d(1.0, 0.0, 0.0)); // 1
    mesh.add_node(Point::new_3d(0.0, 1.0, 0.0)); // 2
    mesh.add_node(Point::new_3d(0.0, 0.0, 1.0)); // 3
    mesh.add_node(Point::new_3d(0.5, 0.5, 0.5)); // 4
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 1, 3]);
    mesh.add_element(ElementType::Triangle, vec![0, 1, 4]);
    mesh
}

/// One square face of a box (two triangles, open boundary): manifold with
/// boundary — every edge has exactly one owner.
fn open_box_face_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(1.0, 1.0, 0.0));
    mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh.add_element(ElementType::Triangle, vec![0, 2, 3]);
    mesh
}

/// Degenerate triangle: two coincident nodes, zero area.
fn degenerate_triangle_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point::new_2d(0.0, 0.0));
    mesh.add_node(Point::new_2d(0.0, 0.0));
    mesh.add_node(Point::new_2d(1.0, 0.0));
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh
}

/// Sliver tetrahedron with parametric altitude over the unit base triangle.
fn sliver_altitude_mesh(altitude: f64) -> Mesh {
    let mut mesh = Mesh::new(3);
    mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
    mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
    mesh.add_node(Point::new_3d(1.0 / 3.0, 1.0 / 3.0, altitude));
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh
}

/// Corner points of two interpenetrating unit boxes: any surface
/// triangulation self-intersects geometrically. Point set for the
/// tetrahedralizer's bounded-termination contract.
fn interpenetrating_boxes_points() -> Vec<Point> {
    let mut points = Vec::new();
    for (ox, oy, oz) in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)] {
        for [x, y, z] in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ] {
            points.push(Point::new_3d(ox + x, oy + y, oz + z));
        }
    }
    points
}

/// Two disjoint unit-box tet meshes in one `Mesh`.
fn two_disjoint_boxes() -> Mesh {
    let mut mesh = Mesh::new(3);
    for ox in [0.0, 5.0] {
        let base = mesh.num_nodes();
        for [x, y, z] in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ] {
            mesh.add_node(Point::new_3d(ox + x, y, z));
        }
        // Kuhn-style split of the cube into 6 tets would be verbose; two
        // diagonal tets per box suffice for connectivity assertions.
        mesh.add_element(
            ElementType::Tetrahedron,
            vec![base, base + 1, base + 2, base + 4],
        );
        mesh.add_element(
            ElementType::Tetrahedron,
            vec![base + 1, base + 2, base + 6, base + 4],
        );
    }
    mesh
}

#[test]
fn sliver_tet_is_flagged_by_every_quality_metric() {
    let mesh = sliver_mesh();
    let volume = mesh.element_measure(0);
    assert!(
        volume > 0.0,
        "sliver must remain a valid positive-volume tet"
    );

    let quality = per_element_quality(&mesh);
    assert_eq!(quality.len(), 1);
    let q = quality[0];
    assert!(q.mean_ratio.is_finite() && q.aspect_quality.is_finite());
    assert!(q.min_dihedral.is_finite() && q.max_dihedral.is_finite());
    // A 1e-12 altitude over a unit base is far below any usable threshold.
    assert!(
        q.mean_ratio < 1e-6,
        "mean ratio must flag the sliver, got {}",
        q.mean_ratio
    );
    assert!(
        q.min_dihedral < 1e-6,
        "min dihedral must flag the sliver, got {}",
        q.min_dihedral
    );

    // Contrast: a Kuhn tet of the unit cube is orders of magnitude better.
    let reference = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
    let ref_quality = per_element_quality(&reference);
    let best_reference = ref_quality
        .iter()
        .map(|r| r.mean_ratio)
        .fold(0.0_f64, f64::max);
    assert!(
        best_reference > 100.0 * q.mean_ratio,
        "structured tets must be far better than the sliver"
    );
}

#[test]
fn thin_gap_box_keeps_positive_volumes_and_finite_quality() {
    // A 1e-9-thick gap: every element is a valid sliver-class tet, but the
    // generator must terminate with finite nodes, positive measures and
    // finite quality — never NaN from degenerate arithmetic.
    let mesh = try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1e-9, 2, 2, 1)
        .expect("thin gap is valid admitted input");
    // 2x2x1 cells, 6 Kuhn tets per cell.
    assert_eq!(mesh.num_elements(), 2 * 2 * 6);
    for index in 0..mesh.num_elements() {
        assert!(
            mesh.element_measure(index) > 0.0,
            "thin-gap element {index} must keep positive volume"
        );
    }
    let quality = per_element_quality(&mesh);
    assert!(quality.iter().all(|q| q.mean_ratio.is_finite()));
    assert!(quality.iter().all(|q| q.min_dihedral.is_finite()));
    assert!(
        quality.iter().all(|q| q.mean_ratio < 0.5),
        "thin-gap tets must score poorly, not as well-formed elements"
    );
}

#[test]
fn near_coplanar_cloud_terminates_bounded() {
    // Five points with 1e-12 out-of-plane perturbation: the robust kernel
    // may accept (slivers) or reject (no volume), but it must terminate
    // with finite quality or a typed error — never hang or panic.
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1e-12],
    ]
    .map(|[x, y, z]| Point::new_3d(x, y, z));
    let config = TetrahedralizeConfig::default();
    if let Ok(result) = tetrahedralize_points_checked(&points, &config) {
        let quality = per_element_quality(&result.mesh);
        assert!(quality.iter().all(|q| q.mean_ratio.is_finite()));
    }
}

#[cfg(feature = "mshio")]
#[test]
fn p2_multielement_cube_survives_gmsh_round_trip() {
    use sonium_mesher::mesh::gmsh::from_msh_bytes;
    use sonium_mesher::mesh::gmsh_write::write_gmsh;

    // Multi-element P2 mesh with shared mid-edge nodes: the single-tet unit
    // test cannot catch a broken shared-node identity or a permutation that
    // only fails with several elements.
    let p1 = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1);
    let p2 = p1.to_p2();
    assert!(p2.num_elements() > 1);

    let mut buf = Vec::new();
    write_gmsh(&p2, &mut buf).expect("write P2 mesh");
    let parsed = from_msh_bytes(&buf).expect("re-parse P2 mesh");
    assert_eq!(parsed.num_nodes(), p2.num_nodes());
    assert_eq!(parsed.num_elements(), p2.num_elements());

    // Every FEM-ordered node must land at the same coordinates: shared
    // mid-edge nodes included, so ordering is preserved element by element.
    for (original, round_tripped) in p2.elements.iter().zip(parsed.elements.iter()) {
        assert_eq!(original.element_type, round_tripped.element_type);
        assert_eq!(original.nodes.len(), round_tripped.nodes.len());
        for (&before, &after) in original.nodes.iter().zip(round_tripped.nodes.iter()) {
            let a = p2.nodes[before];
            let b = parsed.nodes[after];
            assert!(
                (a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12 && (a.z - b.z).abs() < 1e-12,
                "P2 node identity moved across the Gmsh round trip"
            );
        }
    }
}

#[test]
fn flap_edge_is_non_manifold_open_face_is_not() {
    let flap = flap_mesh();
    let report = check_manifold(&flap);
    assert!(!report.is_manifold(), "3-owner edge must be non-manifold");
    assert_eq!(report.non_manifold_faces.len(), 1);
    let (face, owners) = &report.non_manifold_faces[0];
    assert_eq!(face, &vec![0, 1], "exactly the shared edge is flagged");
    assert_eq!(owners.len(), 3);

    // Contrast: boundary-only mesh (every edge owned once) is manifold.
    let open = open_box_face_mesh();
    let open_report = check_manifold(&open);
    assert!(
        open_report.is_manifold(),
        "open boundary must stay manifold"
    );
}

#[test]
fn degenerate_triangle_scores_worst_finite() {
    let mesh = degenerate_triangle_mesh();
    assert_eq!(mesh.element_measure(0), 0.0);
    let quality = per_element_quality(&mesh);
    assert_eq!(quality.len(), 1);
    let q = quality[0];
    // Zero area must not produce NaN metrics; the element must score at the
    // bottom of the quality scale, never as well-formed.
    assert!(
        q.mean_ratio.is_finite(),
        "mean ratio must stay finite, got {}",
        q.mean_ratio
    );
    assert!(q.aspect_quality.is_finite());
    assert!(
        q.mean_ratio <= 1e-12,
        "degenerate triangle must score worst, got {}",
        q.mean_ratio
    );
}

#[test]
fn sliver_quality_degrades_monotonically_with_altitude() {
    // Quality sensitivity curve: four decades of altitude must produce four
    // strictly decreasing mean ratios, all finite.
    let mut previous = f64::INFINITY;
    for altitude in [1e-2, 1e-4, 1e-8, 1e-12] {
        let mesh = sliver_altitude_mesh(altitude);
        assert!(
            mesh.element_measure(0) > 0.0,
            "altitude {altitude} must stay valid"
        );
        let quality = per_element_quality(&mesh);
        let ratio = quality[0].mean_ratio;
        assert!(ratio.is_finite(), "altitude {altitude} must stay finite");
        assert!(
            ratio < previous,
            "quality must degrade monotonically: {ratio} !< {previous} at altitude {altitude}"
        );
        previous = ratio;
    }
}

#[test]
fn thin_gap_sweep_keeps_positive_volumes() {
    // No NaN cliff across four decades of gap thickness: every element keeps
    // positive volume and finite quality.
    for gap in [1e-3, 1e-6, 1e-9, 1e-12] {
        let mesh = try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, gap, 2, 2, 1)
            .expect("thin gap is valid admitted input");
        for index in 0..mesh.num_elements() {
            assert!(
                mesh.element_measure(index) > 0.0,
                "gap {gap}: element {index} lost volume"
            );
        }
        let quality = per_element_quality(&mesh);
        assert!(
            quality
                .iter()
                .all(|q| q.mean_ratio.is_finite() && q.min_dihedral.is_finite()),
            "gap {gap}: quality must stay finite"
        );
    }
}

#[test]
fn self_intersecting_point_set_terminates_bounded() {
    // Geometric self-intersection is invisible to edge-owner counting, so
    // the contract here is bounded termination: the tetrahedralizer must
    // return Ok with finite quality or a typed error — never hang or panic.
    // (Connectivity of any triangulation is still 2-manifold; detection of
    // geometric intersection is out of scope for check_manifold.)
    let points = interpenetrating_boxes_points();
    let config = TetrahedralizeConfig::default();
    if let Ok(result) = tetrahedralize_points_checked(&points, &config) {
        let quality = per_element_quality(&result.mesh);
        assert!(quality.iter().all(|q| q.mean_ratio.is_finite()));
    }
}

#[test]
fn step_domain_marks_every_boundary_face() {
    // Boundary conformity: the stepped domain must tag every boundary face
    // with a documented marker (1 inlet, 2 outlet, 3 walls) — no unmarked
    // leaks for solvers to misread as interior.
    let mesh = backward_facing_step_triangles(2.0, 1.0, 0.5, 0.25, 8, 8);
    assert!(!mesh.boundaries.is_empty());
    for face in &mesh.boundaries {
        assert!(
            [1, 2, 3].contains(&face.marker),
            "boundary face has undocumented marker {}",
            face.marker
        );
    }
    for marker in [1, 2, 3] {
        assert!(
            mesh.boundaries.iter().any(|face| face.marker == marker),
            "marker {marker} must appear at least once"
        );
    }
}

#[test]
fn disjoint_components_report_clean() {
    // Two disjoint closed components: edge-owner counting is per-face, so
    // components must not confuse the report.
    let mesh = two_disjoint_boxes();
    assert_eq!(mesh.num_elements(), 4);
    let report = check_manifold(&mesh);
    assert!(
        report.is_manifold(),
        "disjoint closed components must report clean"
    );
}
