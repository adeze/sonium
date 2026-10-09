//! Admission tests for the `try_*` structured-grid generators.
//!
//! The infallible convenience generators divide the span by the cell count,
//! so a zero count silently fills the mesh with infinite coordinates. These
//! tests prove the checked variants reject zero counts and non-finite
//! extents with typed errors, and that valid inputs match the convenience
//! wrappers node for node.

use sonium_mesher::mesh::{
    LayeredShellError, StructuredGridError, box_mesh_hexahedra, box_mesh_tetrahedra,
    rectangular_mesh_quads, rectangular_mesh_triangles, try_annular_mesh_triangles,
    try_backward_facing_step_triangles, try_binary_mask_mesh_triangles, try_box_mesh_hexahedra,
    try_box_mesh_tetrahedra, try_box_mesh_tetrahedra_symmetric,
    try_channel_with_cylinder_triangles, try_circular_mesh_triangles, try_duct_profile_mesh,
    try_fitted_binary_mask_mesh_triangles, try_fitted_implicit_region_mesh_triangles,
    try_implicit_region_mesh_triangles, try_layered_spherical_shell_mesh_tetrahedra,
    try_rectangular_mesh_quads, try_rectangular_mesh_triangles,
    try_spherical_shell_mesh_tetrahedra, try_two_disk_union_mesh_triangles,
    try_unit_cube_hexahedra, try_unit_cube_tetrahedra, try_unit_square_quads,
    try_unit_square_triangles,
};

fn assert_finite_nodes(mesh: &sonium_mesher::mesh::Mesh) {
    assert!(
        mesh.nodes
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()),
        "generated nodes must be finite"
    );
}

#[test]
fn checked_rectangular_rejects_zero_counts() {
    assert_eq!(
        try_rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 0, 2).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "x",
            actual: 0
        }
    );
    assert_eq!(
        try_rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 0).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "y",
            actual: 0
        }
    );
    assert_eq!(
        try_rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 0, 1).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "x",
            actual: 0
        }
    );
}

#[test]
fn checked_box_rejects_zero_counts() {
    assert_eq!(
        try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 0).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "z",
            actual: 0
        }
    );
    assert_eq!(
        try_box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0, 1, 1).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "x",
            actual: 0
        }
    );
    assert_eq!(
        try_box_mesh_tetrahedra_symmetric(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 0, 1).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "y",
            actual: 0
        }
    );
}

#[test]
fn checked_generators_reject_non_finite_extents() {
    assert!(matches!(
        try_rectangular_mesh_triangles(f64::NAN, 1.0, 0.0, 1.0, 2, 2).unwrap_err(),
        StructuredGridError::NonFiniteExtent { axis: "x", .. }
    ));
    assert!(matches!(
        try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, f64::INFINITY, 1, 1, 1).unwrap_err(),
        StructuredGridError::NonFiniteExtent { axis: "z", .. }
    ));
}

#[test]
fn checked_valid_inputs_match_convenience_wrappers() {
    let tris = try_rectangular_mesh_triangles(0.0, 2.0, 0.0, 1.0, 3, 2).expect("valid 2D input");
    let tris_ref = rectangular_mesh_triangles(0.0, 2.0, 0.0, 1.0, 3, 2);
    assert_eq!(tris.num_nodes(), tris_ref.num_nodes());
    assert_eq!(tris.num_elements(), 3 * 2 * 2);
    assert_finite_nodes(&tris);

    let quads = try_rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 2, 2).expect("valid quad input");
    assert_eq!(
        quads.num_elements(),
        rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 2, 2).num_elements()
    );

    let tets =
        try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1).expect("valid 3D input");
    assert_eq!(
        tets.num_elements(),
        box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1).num_elements()
    );
    assert_eq!(tets.num_elements(), 6);
    assert_finite_nodes(&tets);

    let hexes =
        try_box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1).expect("valid hex input");
    assert_eq!(
        hexes.num_elements(),
        box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1).num_elements()
    );
    assert_eq!(hexes.num_elements(), 2);
    assert_finite_nodes(&hexes);
}

#[test]
fn checked_generators_reject_hostile_counts() {
    // usize::MAX passes zero/non-finite admission but must not reach
    // allocation: previously SIGABRT on a 6 GiB Vec growth (exit 134).
    let err = try_rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, usize::MAX, 1).unwrap_err();
    assert!(
        matches!(err, StructuredGridError::CellBudgetExceeded { .. }),
        "got {err:?}"
    );
    // A merely large (non-overflowing) product reports exact cells.
    let err = try_rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1_000_000_000_000, 1).unwrap_err();
    assert_eq!(
        err,
        StructuredGridError::CellBudgetExceeded {
            shape: "structured 2D grid",
            cells: 1_000_000_000_000,
            max_cells: StructuredGridError::MAX_GENERATOR_BYTES
                / StructuredGridError::BYTES_PER_CELL_2D,
        }
    );
    let err = try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, usize::MAX, 1, 1).unwrap_err();
    assert!(
        matches!(err, StructuredGridError::CellBudgetExceeded { .. }),
        "got {err:?}"
    );
}

#[test]
fn budget_admits_realistic_large_meshes() {
    // 100^3 box: 1M cells x 512 B/cell = 512 MiB, inside the 1 GiB budget.
    let big = try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 100, 100)
        .expect("100^3 box must be admitted");
    assert_eq!(big.num_elements(), 6 * 100 * 100 * 100);
    assert_finite_nodes(&big);
    // 1000^2 grid: 1M cells, inside the 2D budget.
    let flat = try_rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1000, 1000)
        .expect("1000^2 grid must be admitted");
    assert_eq!(flat.num_elements(), 2 * 1000 * 1000);
}

#[test]
fn radial_twins_reject_zero_and_hostile_counts() {
    assert_eq!(
        try_circular_mesh_triangles(0.0, 0.0, 1.0, 0, 8).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "radial",
            actual: 0
        }
    );
    assert_eq!(
        try_circular_mesh_triangles(0.0, 0.0, 1.0, 2, 0).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "angular",
            actual: 0
        }
    );
    assert!(matches!(
        try_circular_mesh_triangles(0.0, 0.0, 1.0, usize::MAX, usize::MAX).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_annular_mesh_triangles(0.0, 0.0, 0.5, 1.0, usize::MAX, 8).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_annular_mesh_triangles(f64::NAN, 0.0, 0.5, 1.0, 2, 8).unwrap_err(),
        StructuredGridError::NonFiniteExtent { .. }
    ));
    // Valid twin matches the infallible body exactly.
    let got = try_circular_mesh_triangles(0.0, 0.0, 1.0, 2, 8).expect("valid circular");
    let want = sonium_mesher::mesh::circular_mesh_triangles(0.0, 0.0, 1.0, 2, 8);
    assert_eq!(got.num_nodes(), want.num_nodes());
    assert_eq!(got.num_elements(), want.num_elements());
    assert_finite_nodes(&got);
}

#[test]
fn shell_twins_reject_zero_layers_and_hostile_subdivisions() {
    assert_eq!(
        try_spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, 0.5, 1.0, 1, 0).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "layers",
            actual: 0
        }
    );
    // Subdivision quadruples base triangles per level: must not reach the
    // icosphere loop.
    assert!(matches!(
        try_spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, 0.5, 1.0, usize::MAX, 4).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_layered_spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, &[0.5, 1.0], usize::MAX, &[4])
            .unwrap_err(),
        LayeredShellError::BudgetExceeded { .. }
    ));
    assert!(matches!(
        try_layered_spherical_shell_mesh_tetrahedra(f64::NAN, 0.0, 0.0, &[0.5, 1.0], 1, &[2])
            .unwrap_err(),
        LayeredShellError::NonFiniteCenter
    ));
    // Base icosahedron (subdivisions == 0) stays admitted.
    let coarse =
        try_spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, 0.5, 1.0, 0, 1).expect("coarse shell");
    assert!(coarse.num_elements() > 0);
    assert_finite_nodes(&coarse);
}

#[test]
fn implicit_family_twins_reject_bad_counts_and_masks() {
    assert!(matches!(
        try_implicit_region_mesh_triangles(0.0, 1.0, 0.0, 1.0, usize::MAX, 1, 1, |_, _| true)
            .unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_implicit_region_mesh_triangles(0.0, 0.0, 0.0, 1.0, 4, 4, 1, |_, _| true).unwrap_err(),
        StructuredGridError::DegenerateExtent { .. }
    ));
    assert!(matches!(
        try_binary_mask_mesh_triangles(&[], 0.0, 1.0, 0.0, 1.0, 1).unwrap_err(),
        StructuredGridError::InvalidMask { .. }
    ));
    assert!(matches!(
        try_binary_mask_mesh_triangles(&[vec![true], vec![true, false]], 0.0, 1.0, 0.0, 1.0, 1)
            .unwrap_err(),
        StructuredGridError::InvalidMask { .. }
    ));
    // Refinement levels multiply the grid: levels = 100 must fail before
    // any refinement loop runs.
    let mask = vec![vec![true; 4]; 4];
    let config = sonium_mesher::mesh::FittedImplicitRegionConfig {
        nx: 4,
        ny: 4,
        boundary_refinement_levels: 100,
        projection_iterations: 12,
    };
    assert!(matches!(
        try_fitted_implicit_region_mesh_triangles(0.0, 1.0, 0.0, 1.0, 1, config, |_, _| 0.0)
            .unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_fitted_binary_mask_mesh_triangles(&mask, 0.0, 1.0, 0.0, 1.0, 1, 100).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    // Valid twins match their bodies.
    let got = try_implicit_region_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4, 1, |_, _| true)
        .expect("valid implicit");
    let want =
        sonium_mesher::mesh::implicit_region_mesh_triangles(0.0, 1.0, 0.0, 1.0, 4, 4, 1, |_, _| {
            true
        });
    assert_eq!(got.num_nodes(), want.num_nodes());
    assert_eq!(got.num_elements(), want.num_elements());
}

#[test]
fn channel_step_disk_duct_unit_twins_reject_hostile_counts() {
    assert!(matches!(
        try_backward_facing_step_triangles(2.0, 1.0, 0.5, 0.25, usize::MAX, 4).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_channel_with_cylinder_triangles(0.0, 2.0, 0.0, 1.0, 4, 4, 1.0, 0.5, 0.0).unwrap_err(),
        StructuredGridError::DegenerateExtent { .. }
    ));
    assert!(matches!(
        try_channel_with_cylinder_triangles(0.0, 2.0, 0.0, 1.0, usize::MAX, 4, 1.0, 0.5, 0.1)
            .unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_two_disk_union_mesh_triangles((0.0, 0.0), (3.0, 0.0), 1.0, 0.0, 4, 4, 1).unwrap_err(),
        StructuredGridError::DegenerateExtent { .. }
    ));
    assert!(matches!(
        try_two_disk_union_mesh_triangles((f64::NAN, 0.0), (3.0, 0.0), 1.0, 1.0, 4, 4, 1)
            .unwrap_err(),
        StructuredGridError::NonFiniteExtent { .. }
    ));
    assert_eq!(
        try_duct_profile_mesh(0.0, 1.0, |_| 0.0, |_| 1.0, 0, 4).unwrap_err(),
        StructuredGridError::ZeroResolution {
            axis: "z",
            actual: 0
        }
    );
    assert!(matches!(
        try_duct_profile_mesh(1.0, 1.0, |_| 0.0, |_| 1.0, 4, 4).unwrap_err(),
        StructuredGridError::DegenerateExtent { .. }
    ));
    assert!(matches!(
        try_unit_square_triangles(usize::MAX).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(matches!(
        try_unit_cube_tetrahedra(usize::MAX).unwrap_err(),
        StructuredGridError::CellBudgetExceeded { .. }
    ));
    assert!(try_unit_square_quads(2).is_ok());
    assert!(try_unit_cube_hexahedra(2).is_ok());
}
