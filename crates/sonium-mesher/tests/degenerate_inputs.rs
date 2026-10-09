//! Degenerate-input tests for every fallible (`*_checked` / `try_*`) entry
//! point in `sonium-mesher`.
//!
//! Each test feeds `NaN`, empty, coplanar, duplicate, out-of-range, or
//! otherwise malformed input and asserts a typed `Err` — never a panic.
//! Companion to the per-module unit tests, which focus on the happy path.

use sonium_mesher::mesh::topology::{Face, MeshTopology};
use sonium_mesher::mesh::validate::{
    MeshValidationError, try_find_duplicate_nodes, try_merge_duplicate_nodes,
};
use sonium_mesher::mesh::{
    Element, ElementType, LayeredShellError, Mesh, Point, compact_unused_nodes_checked,
    rectangular_mesh_triangles,
    tetrahedralize::{TetrahedralizeConfig, tetrahedralize_points_checked},
    try_layered_spherical_shell_mesh_tetrahedra,
};

fn point_cloud(points: &[[f64; 3]]) -> Vec<Point> {
    points
        .iter()
        .map(|&[x, y, z]| Point::new_3d(x, y, z))
        .collect()
}

#[test]
fn point_cloud_checked_rejects_degenerate_inputs() {
    let config = TetrahedralizeConfig::default();

    // Empty cloud.
    assert!(tetrahedralize_points_checked(&[], &config).is_err());

    // Fewer than four points can never span a volume.
    let trio = point_cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    assert!(tetrahedralize_points_checked(&trio, &config).is_err());

    // Non-finite coordinates.
    let nan = point_cloud(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [f64::NAN, 0.0, 1.0],
    ]);
    assert!(tetrahedralize_points_checked(&nan, &config).is_err());

    // Repeated vertices.
    let repeated = point_cloud(&[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]);
    assert!(tetrahedralize_points_checked(&repeated, &config).is_err());

    // Coplanar points span no volume.
    let coplanar = point_cloud(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ]);
    assert!(tetrahedralize_points_checked(&coplanar, &config).is_err());
}

#[test]
fn layered_shell_checked_reports_typed_errors() {
    // `Mesh` carries no `PartialEq`, so compare the unwrapped typed error.
    let err = |radii: &[f64], subdivisions: &[usize]| {
        try_layered_spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, radii, 1, subdivisions)
            .unwrap_err()
    };
    // Empty and singleton radii.
    assert_eq!(err(&[], &[]), LayeredShellError::TooFewRadii { actual: 0 });
    assert_eq!(
        err(&[1.0], &[]),
        LayeredShellError::TooFewRadii { actual: 1 }
    );
    // Non-increasing radii.
    assert_eq!(
        err(&[1.0, 0.5], &[1]),
        LayeredShellError::NonIncreasingRadii
    );
    // Non-positive inner radius.
    assert_eq!(
        err(&[0.0, 1.0], &[1]),
        LayeredShellError::NonIncreasingRadii
    );
    // Subdivision-count mismatch.
    assert_eq!(
        err(&[0.5, 1.0], &[1, 1]),
        LayeredShellError::SubdivisionCountMismatch {
            regions: 1,
            actual: 2
        }
    );
    // Zero radial subdivisions.
    assert_eq!(
        err(&[0.5, 1.0], &[0]),
        LayeredShellError::EmptyRadialSubdivision
    );
}

#[test]
fn boundary_and_topology_gates_reject_invalid_meshes() {
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);
    assert!(mesh.try_detect_boundaries().is_ok());
    assert!(mesh.try_topology().is_ok());

    // Corrupt one element with an out-of-range node index.
    mesh.elements.push(Element::new(
        ElementType::Triangle,
        vec![0, 1, mesh.num_nodes() + 99],
        999,
    ));
    assert!(
        mesh.try_detect_boundaries().is_err(),
        "boundaries must report invalid connectivity, not silently clear"
    );
    assert!(mesh.try_topology().is_err());
    assert!(MeshTopology::try_build(&mesh).is_err());

    // Non-finite coordinates fail admission too.
    let mut nan_mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1);
    nan_mesh.nodes.push(Point::new_3d(f64::NAN, 0.0, 0.0));
    assert!(nan_mesh.try_detect_boundaries().is_err());
}

#[test]
fn duplicate_node_gates_reject_bad_tolerance_and_magnitude() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);

    // NaN never equals itself, so match the variant instead of comparing
    // payloads.
    let is_invalid_tol = |result: Result<Vec<(usize, usize)>, MeshValidationError>| {
        matches!(result, Err(MeshValidationError::InvalidTolerance(_)))
    };
    for bad_tol in [f64::NAN, 0.0, -1e-9, f64::INFINITY] {
        assert!(
            is_invalid_tol(try_find_duplicate_nodes(&mesh, bad_tol)),
            "tol {bad_tol} must be rejected"
        );
        assert!(
            matches!(
                try_merge_duplicate_nodes(
                    &mut rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2),
                    bad_tol
                ),
                Err(MeshValidationError::InvalidTolerance(_))
            ),
            "tol {bad_tol} must be rejected"
        );
    }

    // Huge coordinates with a tiny tolerance would saturate the i64 spatial
    // hash; the gate must fail closed instead of merging the whole mesh.
    let mut huge = Mesh::new(3);
    huge.add_node(Point::new_3d(1e18, 0.0, 0.0));
    huge.add_node(Point::new_3d(-1e18, 0.0, 0.0));
    huge.add_node(Point::new_3d(0.0, 1e18, 0.0));
    huge.add_node(Point::new_3d(0.0, 0.0, 1e18));
    huge.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    assert_eq!(
        try_find_duplicate_nodes(&huge, 1e-12),
        Err(MeshValidationError::InvalidTolerance(1e-12))
    );
}

#[test]
fn compaction_and_small_constructors_reject_malformed_input() {
    // Orphaned nodes compact cleanly on a valid mesh.
    let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1);
    mesh.add_node(Point::new_2d(42.0, 42.0));
    let report = compact_unused_nodes_checked(&mut mesh).expect("valid mesh");
    assert_eq!(report.removed_nodes, 1);

    // Invalid connectivity fails instead of remapping garbage.
    let mut bad = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, 1);
    bad.elements.push(Element::new(
        ElementType::Triangle,
        vec![0, 1, bad.num_nodes() + 7],
        4242,
    ));
    assert!(compact_unused_nodes_checked(&mut bad).is_err());

    // Small fallible constructors report arity errors.
    assert!(Face::try_from_nodes(&[0, 1]).is_err());
    assert!(Face::try_from_nodes(&[0, 1, 2, 3, 4]).is_err());
    assert!(Face::try_from_nodes(&[0, 1, 2]).is_ok());
    let odd = Element::new(ElementType::Triangle, vec![0, 1, 2, 3, 4], 7);
    assert!(odd.try_polynomial_order().is_err());
    assert!(!odd.is_p2(), "is_p2 must return false, not panic");
}
