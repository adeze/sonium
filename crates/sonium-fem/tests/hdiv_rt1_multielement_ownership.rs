use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{PeriodicTransform, SectorPeriodicMap};
use sonium_fem::mesh::Face;
use sonium_fem::mesh::{unit_cube_hexahedra, unit_cube_tetrahedra};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};

#[test]
fn rt1_tetrahedral_layout_reuses_shared_face_modes() {
    let mesh = unit_cube_tetrahedra(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let layout = space
        .rt1_dof_layout_3d(&mesh)
        .expect("generated tetrahedral mesh has valid RT1 ownership");

    assert_eq!(layout.num_dofs(), space.num_global_dofs(&mesh));
    assert_eq!(layout.face_offsets.len(), layout.face_owners.len() + 1);
    assert!(layout.face_owners.iter().any(|owners| owners.len() == 2));

    for (face, owners) in layout.face_owners.iter().enumerate() {
        let range = layout.face_dofs(face).expect("face range");
        assert_eq!(range.len(), 3, "tetrahedral RT1 face modes");
        assert!((1..=2).contains(&owners.len()));
        for &element in owners {
            assert!(
                layout.element_dofs[element][..]
                    .windows(range.len())
                    .any(|window| window == (range.clone()).collect::<Vec<_>>().as_slice())
            );
        }
    }

    for (element, dofs) in layout.element_dofs.iter().enumerate() {
        assert_eq!(dofs.len(), 15);
        assert_eq!(layout.cell_dofs(element).expect("cell range").len(), 3);
        assert!(dofs.windows(2).all(|window| window[0] != window[1]));
    }
}

#[test]
fn rt1_hexahedral_layout_accounts_for_four_mode_faces_and_cell_modes() {
    let mesh = unit_cube_hexahedra(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let layout = space
        .rt1_dof_layout_3d(&mesh)
        .expect("generated hexahedral mesh has valid RT1 ownership");

    assert_eq!(layout.num_dofs(), space.num_global_dofs(&mesh));
    assert!(
        layout
            .face_offsets
            .windows(2)
            .all(|window| window[1] - window[0] == 4)
    );
    assert_eq!(
        layout.cell_dofs(0).expect("hexahedral cell range").len(),
        12
    );
    assert_eq!(layout.element_dofs[0].len(), 36);
}

#[test]
fn rt1_3d_layout_rejects_wrong_order_and_dimension() {
    let mesh = unit_cube_tetrahedra(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    assert!(HdivSpace::new().rt1_dof_layout_3d(&mesh).is_none());

    let mut two_dimensional = mesh.clone();
    two_dimensional.dimension = 2;
    assert!(space.rt1_dof_layout_3d(&two_dimensional).is_none());

    let mut non_manifold = mesh;
    let duplicate = non_manifold.elements[0].clone();
    non_manifold.elements.push(duplicate.clone());
    non_manifold.elements.push(duplicate);
    assert!(space.rt1_dof_layout_3d(&non_manifold).is_none());
}

#[test]
fn rt1_layout_ranges_are_checked_at_boundaries() {
    let mesh = unit_cube_tetrahedra(1);
    let layout = HdivSpace::with_order(HdivOrder::Rt1)
        .rt1_dof_layout_3d(&mesh)
        .expect("valid layout");
    assert!(layout.face_dofs(layout.face_owners.len()).is_none());
    assert!(layout.cell_dofs(layout.element_dofs.len()).is_none());
}

#[test]
fn layout_backed_mixed_periodic_consumer_matches_explicit_offsets() {
    let mesh = unit_cube_tetrahedra(1);
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let layout = space.rt1_dof_layout_3d(&mesh).expect("valid layout");
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 3), (1, 4), (2, 5)],
        edge_pairs: Vec::new(),
        face_pairs: Vec::new(),
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let master_faces = [(0, Face::Triangle([0, 1, 2]))];
    let slave_faces = [(1, Face::Triangle([3, 4, 5]))];

    let explicit = map
        .hdiv_rt1_mixed_face_constraints_from_faces(
            &master_faces,
            &slave_faces,
            &layout.face_offsets,
            &layout.face_offsets,
        )
        .expect("explicit offsets");
    let from_layout = map
        .hdiv_rt1_mixed_face_constraints_from_layout(&master_faces, &slave_faces, &layout)
        .expect("layout offsets");
    assert_eq!(from_layout, explicit);

    let num_dofs = layout.num_dofs();
    let mut matrix = TripletMatrix::new(num_dofs, num_dofs);
    for dof in 0..num_dofs {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
    }
    let system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); num_dofs]);
    let condensed = map
        .condense_hdiv_rt1_mixed_fem_system_complex_from_layout(
            &system,
            &master_faces,
            &slave_faces,
            &layout,
        )
        .expect("layout-backed condensation");
    assert_eq!(condensed.matrix.num_rows, num_dofs - 3);
    assert_eq!(condensed.matrix.num_cols, num_dofs - 3);
}
