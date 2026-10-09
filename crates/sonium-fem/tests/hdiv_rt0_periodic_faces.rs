use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap, pair_periodic_faces_oriented,
};
use sonium_fem::mesh::Face;

#[test]
fn rt0_periodic_face_pairing_preserves_triangle_and_quad_orientation() {
    let node_pairs = [
        (0, 3),
        (1, 4),
        (2, 5),
        (6, 9),
        (7, 10),
        (8, 11),
        (12, 15),
        (13, 16),
        (14, 17),
        (15, 18),
        (18, 21),
        (19, 22),
        (20, 23),
    ];
    let pairs = pair_periodic_faces_oriented(
        &[
            (40, Face::Triangle([0, 1, 2])),
            (41, Face::Triangle([6, 7, 8])),
            (42, Face::Quad([12, 13, 14, 15])),
        ],
        &[
            (50, Face::Triangle([3, 4, 5])),
            (51, Face::Triangle([9, 11, 10])),
            (52, Face::Quad([15, 16, 17, 18])),
        ],
        &node_pairs,
    )
    .expect("periodic face pairing must recover cyclic/reversed maps");
    assert_eq!(pairs, vec![(40, 50, 1), (41, 51, -1), (42, 52, 1)]);
}

#[test]
fn rt0_periodic_face_constraints_recover_complex_mpc_values() {
    let phase = Complex64::new(0.0, 1.0);
    let map = SectorPeriodicMap {
        node_pairs: vec![],
        edge_pairs: vec![],
        face_pairs: vec![(0, 1, 1), (2, 3, -1)],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let constraints = map
        .hdiv_rt0_face_constraints_3d()
        .expect("RT0 face constraints must expand");
    assert_eq!(constraints.len(), 2);
    assert_eq!(constraints[0].phase, phase);
    assert_eq!(constraints[1].phase, -phase);

    let mpc = MpcSystem::from_periodic(4, &constraints).expect("MPC must be valid");
    let mut reduced = vec![Complex64::new(0.0, 0.0); mpc.num_reduced_dofs()];
    for (index, &full_dof) in mpc.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.25, -0.5);
    }
    let recovered = mpc.recover(&reduced).expect("MPC recovery must succeed");
    for constraint in &constraints {
        assert!(constraint.residual(&recovered).norm() < 1.0e-12);
    }
}

#[test]
fn rt0_periodic_face_pairing_rejects_unpaired_and_ambiguous_inputs() {
    assert!(matches!(
        pair_periodic_faces_oriented(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[(1, Face::Triangle([3, 4, 5]))],
            &[(0, 3), (1, 4)],
        ),
        Err(MpcError::UnpairedFaceNode(2))
    ));

    assert!(matches!(
        pair_periodic_faces_oriented(
            &[(0, Face::Triangle([0, 1, 2]))],
            &[
                (1, Face::Triangle([3, 4, 5])),
                (2, Face::Triangle([3, 5, 4])),
            ],
            &[(0, 3), (1, 4), (2, 5)],
        ),
        Err(MpcError::PairCountMismatch { .. })
    ));
}
