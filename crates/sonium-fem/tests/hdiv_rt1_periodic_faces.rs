use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    HdivRt1PeriodicFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_hdiv_rt1_periodic_face_constraints, pair_periodic_hdiv_rt1_faces,
};
use sonium_fem::mesh::Face;

#[test]
fn rt1_periodic_face_modes_follow_cyclic_and_reversed_vertex_maps() {
    let cyclic = expand_hdiv_rt1_periodic_face_constraints(
        &[HdivRt1PeriodicFacePair {
            master_face: 2,
            slave_face: 5,
            vertex_permutation: [1, 2, 0],
        }],
        Complex64::new(1.0, 0.0),
    )
    .expect("cyclic RT1 face map must expand");
    assert_eq!(cyclic.len(), 3);
    assert_eq!(cyclic[0].slave, 15);
    assert_eq!(cyclic[0].masters[0].dof, 6);
    assert_eq!(cyclic[0].masters[0].coefficient, Complex64::new(1.0, 0.0));
    assert_eq!(cyclic[1].slave, 16);
    assert_eq!(
        cyclic[1]
            .masters
            .iter()
            .map(|term| (term.dof, term.coefficient))
            .collect::<Vec<_>>(),
        vec![
            (6, Complex64::new(1.0, 0.0)),
            (7, Complex64::new(-1.0, 0.0)),
            (8, Complex64::new(-1.0, 0.0)),
        ]
    );

    let reversed = expand_hdiv_rt1_periodic_face_constraints(
        &[HdivRt1PeriodicFacePair {
            master_face: 0,
            slave_face: 1,
            vertex_permutation: [0, 2, 1],
        }],
        Complex64::new(0.6, 0.8),
    )
    .expect("reversed RT1 face map must expand");
    assert_eq!(
        reversed[0].masters[0].coefficient,
        Complex64::new(-0.6, -0.8)
    );
    assert_eq!(reversed[1].masters.len(), 1);
    assert_eq!(reversed[1].masters[0].dof, 2);
    assert_eq!(
        reversed[1].masters[0].coefficient,
        Complex64::new(-0.6, -0.8)
    );
    assert_eq!(reversed[2].masters.len(), 1);
    assert_eq!(reversed[2].masters[0].dof, 1);
    assert_eq!(
        reversed[2].masters[0].coefficient,
        Complex64::new(-0.6, -0.8)
    );
}

#[test]
fn rt1_periodic_face_modes_recover_complex_mpc_values() {
    let phase = Complex64::new(0.0, 1.0);
    let map = SectorPeriodicMap {
        node_pairs: vec![],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let pair = HdivRt1PeriodicFacePair {
        master_face: 0,
        slave_face: 1,
        vertex_permutation: [0, 2, 1],
    };
    let constraints = map
        .hdiv_rt1_face_constraints_3d(&[pair])
        .expect("RT1 periodic face constraints must expand");
    let mpcs = constraints.clone();
    let system = MpcSystem::new(6, &mpcs).expect("RT1 periodic MPC must be valid");
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.125, -0.75);
    }
    let recovered = system
        .recover(&reduced)
        .expect("RT1 MPC recovery must succeed");
    for constraint in &constraints {
        let expected: Complex64 = constraint
            .masters
            .iter()
            .map(|term| term.coefficient * recovered[term.dof])
            .sum();
        assert!((recovered[constraint.slave] - expected).norm() < 1.0e-12);
    }
}

#[test]
fn rt1_periodic_face_modes_reject_bad_permutations_phase_and_indices() {
    let invalid = HdivRt1PeriodicFacePair {
        master_face: 0,
        slave_face: 1,
        vertex_permutation: [0, 0, 2],
    };
    assert!(matches!(
        expand_hdiv_rt1_periodic_face_constraints(&[invalid], Complex64::new(1.0, 0.0)),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ));
    let valid = HdivRt1PeriodicFacePair {
        master_face: 0,
        slave_face: 1,
        vertex_permutation: [0, 1, 2],
    };
    assert!(matches!(
        expand_hdiv_rt1_periodic_face_constraints(&[valid], Complex64::new(f64::NAN, 0.0)),
        Err(MpcError::NonFiniteCoefficient { .. })
    ));
    let overflow = HdivRt1PeriodicFacePair {
        master_face: usize::MAX,
        slave_face: 1,
        vertex_permutation: [0, 1, 2],
    };
    assert!(matches!(
        expand_hdiv_rt1_periodic_face_constraints(&[overflow], Complex64::new(1.0, 0.0)),
        Err(MpcError::DofIndexOverflow(usize::MAX))
    ));
}

#[test]
fn rt1_periodic_face_pairing_derives_vertex_permutation_from_node_pairs() {
    let master_faces = [(12, Face::Triangle([0, 1, 2]))];
    let slave_faces = [(20, Face::Triangle([4, 3, 5]))];
    let node_pairs = [(0, 3), (1, 4), (2, 5)];
    let pairs = pair_periodic_hdiv_rt1_faces(&master_faces, &slave_faces, &node_pairs)
        .expect("RT1 face pairing must derive the permutation");
    assert_eq!(pairs[0].master_face, 12);
    assert_eq!(pairs[0].slave_face, 20);
    assert_eq!(pairs[0].vertex_permutation, [1, 0, 2]);

    let map = SectorPeriodicMap {
        node_pairs: node_pairs.to_vec(),
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let constraints = map
        .hdiv_rt1_face_constraints_from_faces(&master_faces, &slave_faces)
        .expect("map must expand paired RT1 faces");
    assert_eq!(constraints.len(), 3);
}
