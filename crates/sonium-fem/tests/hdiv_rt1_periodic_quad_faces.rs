use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{
    HdivRt1PeriodicQuadFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_hdiv_rt1_periodic_quad_face_constraints, pair_periodic_hdiv_rt1_quadrilateral_faces,
};
use sonium_fem::mesh::Face;
use sonium_fem::mesh::unit_cube_hexahedra;
use sonium_fem::spaces::{HdivOrder, HdivSpace};

fn terms(
    constraint: &sonium_fem::boundary::periodic::MultiPointConstraint,
) -> Vec<(usize, Complex64)> {
    constraint
        .masters
        .iter()
        .map(|term| (term.dof, term.coefficient))
        .collect()
}

#[test]
fn rt1_periodic_quad_face_modes_follow_identity_rotation_and_reflection_maps() {
    let identity = expand_hdiv_rt1_periodic_quad_face_constraints(
        &[HdivRt1PeriodicQuadFacePair {
            master_face: 2,
            slave_face: 5,
            vertex_permutation: [0, 1, 2, 3],
        }],
        Complex64::new(1.0, 0.0),
    )
    .expect("identity quadrilateral map must expand");
    assert_eq!(identity.len(), 4);
    for (row, constraint) in identity.iter().enumerate() {
        assert_eq!(constraint.slave, 20 + row);
        assert_eq!(terms(constraint), vec![(8 + row, Complex64::new(1.0, 0.0))]);
    }

    let quarter_turn = expand_hdiv_rt1_periodic_quad_face_constraints(
        &[HdivRt1PeriodicQuadFacePair {
            master_face: 1,
            slave_face: 3,
            vertex_permutation: [1, 2, 3, 0],
        }],
        Complex64::new(1.0, 0.0),
    )
    .expect("quarter-turn quadrilateral map must expand");
    assert_eq!(terms(&quarter_turn[0]), vec![(4, Complex64::new(1.0, 0.0))]);
    assert_eq!(
        terms(&quarter_turn[1]),
        vec![(6, Complex64::new(-1.0, 0.0))]
    );
    assert_eq!(terms(&quarter_turn[2]), vec![(5, Complex64::new(1.0, 0.0))]);
    assert_eq!(
        terms(&quarter_turn[3]),
        vec![(7, Complex64::new(-1.0, 0.0))]
    );

    let reflection = expand_hdiv_rt1_periodic_quad_face_constraints(
        &[HdivRt1PeriodicQuadFacePair {
            master_face: 0,
            slave_face: 1,
            vertex_permutation: [3, 2, 1, 0],
        }],
        Complex64::new(0.6, 0.8),
    )
    .expect("reflected quadrilateral map must expand");
    assert_eq!(terms(&reflection[0]), vec![(0, Complex64::new(-0.6, -0.8))]);
    assert_eq!(terms(&reflection[1]), vec![(1, Complex64::new(-0.6, -0.8))]);
    assert_eq!(terms(&reflection[2]), vec![(2, Complex64::new(0.6, 0.8))]);
    assert_eq!(terms(&reflection[3]), vec![(3, Complex64::new(0.6, 0.8))]);
}

#[test]
fn rt1_periodic_quad_face_constraints_recover_complex_mpc_values() {
    let constraints = expand_hdiv_rt1_periodic_quad_face_constraints(
        &[HdivRt1PeriodicQuadFacePair {
            master_face: 0,
            slave_face: 1,
            vertex_permutation: [1, 2, 3, 0],
        }],
        Complex64::new(0.6, 0.8),
    )
    .expect("RT1 quadrilateral constraints must expand");
    let system = MpcSystem::new(8, &constraints).expect("constraints must form an MPC system");
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.125, -0.75);
    }
    let recovered = system.recover(&reduced).expect("MPC recovery must succeed");
    for constraint in &constraints {
        let expected = constraint
            .masters
            .iter()
            .map(|term| term.coefficient * recovered[term.dof])
            .sum::<Complex64>();
        assert!((recovered[constraint.slave] - expected).norm() < 1.0e-12);
    }
}

#[test]
fn rt1_periodic_quad_faces_condense_complex_fem_system_and_recover() {
    let master_faces = [(0, Face::Quad([0, 1, 2, 3]))];
    let slave_faces = [(1, Face::Quad([7, 6, 5, 4]))];
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 4), (1, 5), (2, 6), (3, 7)],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let mut matrix = TripletMatrix::new(8, 8);
    for dof in 0..8 {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
    }
    let system = FemSystem::new(
        matrix,
        (1..=8)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect(),
    );
    let reduced = map
        .condense_hdiv_rt1_quadrilateral_fem_system_complex(&system, &master_faces, &slave_faces)
        .expect("quadrilateral periodic condensation must succeed");
    assert_eq!(reduced.matrix.num_rows, 4);
    assert_eq!(reduced.matrix.num_cols, 4);
    assert_eq!(
        reduced.rhs,
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(10.0, 0.0),
            Complex64::new(12.0, 0.0),
        ]
    );
    let reduced_matrix = reduced.matrix.to_csr();
    for dof in 0..4 {
        assert_eq!(reduced_matrix.get(dof, dof), Complex64::new(2.0, 0.0));
    }
}

#[test]
fn rt1_periodic_quad_face_pairing_derives_square_permutation() {
    let master_faces = [(12, Face::Quad([0, 1, 2, 3]))];
    let slave_faces = [(20, Face::Quad([4, 7, 6, 5]))];
    let pairs = pair_periodic_hdiv_rt1_quadrilateral_faces(
        &master_faces,
        &slave_faces,
        &[(0, 4), (1, 5), (2, 6), (3, 7)],
    )
    .expect("geometric quadrilateral pairing must derive a square symmetry");
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].master_face, 12);
    assert_eq!(pairs[0].slave_face, 20);
    assert_eq!(pairs[0].vertex_permutation, [0, 3, 2, 1]);

    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 4), (1, 5), (2, 6), (3, 7)],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let constraints = map
        .hdiv_rt1_quadrilateral_face_constraints_from_faces(&master_faces, &slave_faces)
        .expect("periodic map must expand quadrilateral RT1 constraints");
    assert_eq!(constraints.len(), 4);
}

#[test]
fn rt1_periodic_quad_faces_use_explicit_mixed_face_offsets() {
    let space = HdivSpace::with_order(HdivOrder::Rt1);
    let offsets = space
        .rt1_face_dof_offsets_3d(&unit_cube_hexahedra(1))
        .expect("hexahedral RT1 face offsets must be available");
    assert_eq!(offsets, vec![0, 4, 8, 12, 16, 20, 24]);

    let master_faces = [(1, Face::Quad([0, 1, 2, 3]))];
    let slave_faces = [(2, Face::Quad([4, 7, 6, 5]))];
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 4), (1, 5), (2, 6), (3, 7)],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let constraints = map
        .hdiv_rt1_quadrilateral_face_constraints_from_faces_with_offsets(
            &master_faces,
            &slave_faces,
            &[0, 3],
            &[0, 4, 7],
        )
        .expect("explicit offsets must support mixed face numbering");
    assert_eq!(constraints.len(), 4);
    assert_eq!(constraints[0].slave, 7);
    assert_eq!(terms(&constraints[0])[0].0, 3);
}

#[test]
fn rt1_periodic_quad_face_rejects_bad_maps_phase_and_overflow() {
    assert!(matches!(
        expand_hdiv_rt1_periodic_quad_face_constraints(
            &[HdivRt1PeriodicQuadFacePair {
                master_face: 0,
                slave_face: 1,
                vertex_permutation: [0, 0, 2, 3],
            }],
            Complex64::new(1.0, 0.0),
        ),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ));
    assert!(matches!(
        expand_hdiv_rt1_periodic_quad_face_constraints(
            &[HdivRt1PeriodicQuadFacePair {
                master_face: 0,
                slave_face: 1,
                vertex_permutation: [0, 2, 1, 3],
            }],
            Complex64::new(1.0, 0.0),
        ),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ));
    assert!(matches!(
        expand_hdiv_rt1_periodic_quad_face_constraints(
            &[HdivRt1PeriodicQuadFacePair {
                master_face: 0,
                slave_face: 1,
                vertex_permutation: [0, 1, 2, 3],
            }],
            Complex64::new(f64::NAN, 0.0),
        ),
        Err(MpcError::NonFiniteCoefficient { .. })
    ));
    assert!(matches!(
        expand_hdiv_rt1_periodic_quad_face_constraints(
            &[HdivRt1PeriodicQuadFacePair {
                master_face: usize::MAX,
                slave_face: 1,
                vertex_permutation: [0, 1, 2, 3],
            }],
            Complex64::new(1.0, 0.0),
        ),
        Err(MpcError::DofIndexOverflow(usize::MAX))
    ));
    assert!(matches!(
        pair_periodic_hdiv_rt1_quadrilateral_faces(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[(1, Face::Quad([3, 4, 5, 6]))],
            &[(0, 3), (1, 4), (2, 5)],
        ),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ));
}
