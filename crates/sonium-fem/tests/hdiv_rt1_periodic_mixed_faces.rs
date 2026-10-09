use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{
    HdivRt1PeriodicMixedFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_hdiv_rt1_periodic_mixed_face_constraints, pair_periodic_hdiv_rt1_mixed_faces,
};
use sonium_fem::mesh::Face;

fn mixed_faces() -> (Vec<(usize, Face)>, Vec<(usize, Face)>, Vec<(usize, usize)>) {
    (
        vec![
            (0, Face::Triangle([0, 1, 2])),
            (1, Face::Quad([3, 4, 5, 6])),
        ],
        vec![
            (2, Face::Triangle([7, 8, 9])),
            (3, Face::Quad([10, 13, 12, 11])),
        ],
        vec![(0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13)],
    )
}

fn map() -> SectorPeriodicMap {
    SectorPeriodicMap {
        node_pairs: mixed_faces().2,
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    }
}

#[test]
fn mixed_rt1_periodic_faces_pair_by_topology_and_expand_all_modes() {
    let (master_faces, slave_faces, node_pairs) = mixed_faces();
    let pairs = pair_periodic_hdiv_rt1_mixed_faces(&master_faces, &slave_faces, &node_pairs)
        .expect("mixed triangular/quadrilateral faces must pair");
    assert_eq!(pairs.len(), 2);
    assert!(matches!(
        pairs[0],
        HdivRt1PeriodicMixedFacePair::Triangle(_)
    ));
    assert!(matches!(
        pairs[1],
        HdivRt1PeriodicMixedFacePair::Quadrilateral(_)
    ));

    let constraints = expand_hdiv_rt1_periodic_mixed_face_constraints(
        &pairs,
        &[0, 3],
        &[0, 3, 7, 10],
        Complex64::new(1.0, 0.0),
    )
    .expect("mixed RT1 modes must expand");
    assert_eq!(constraints.len(), 7);
    assert_eq!(constraints[0].slave, 7);
    assert_eq!(constraints[0].masters[0].dof, 0);
    assert_eq!(constraints[3].slave, 10);
    assert_eq!(constraints[3].masters[0].dof, 3);
}

#[test]
fn mixed_rt1_periodic_faces_recover_and_condense_complex_system() {
    let (master_faces, slave_faces, _) = mixed_faces();
    let constraints = map()
        .hdiv_rt1_mixed_face_constraints_from_faces(
            &master_faces,
            &slave_faces,
            &[0, 3],
            &[0, 3, 7, 10],
        )
        .expect("mixed constraints must build");
    let system = MpcSystem::new(14, &constraints).expect("mixed constraints must be acyclic");
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.5, -0.25);
    }
    let recovered = system
        .recover(&reduced)
        .expect("mixed recovery must succeed");
    for constraint in &constraints {
        let expected = constraint
            .masters
            .iter()
            .map(|term| term.coefficient * recovered[term.dof])
            .sum::<Complex64>();
        assert!((recovered[constraint.slave] - expected).norm() < 1.0e-12);
    }

    let mut matrix = TripletMatrix::new(14, 14);
    for dof in 0..14 {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
    }
    let fem_system = FemSystem::new(
        matrix,
        (1..=14)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect(),
    );
    let condensed = map()
        .condense_hdiv_rt1_mixed_fem_system_complex(
            &fem_system,
            &master_faces,
            &slave_faces,
            &[0, 3],
            &[0, 3, 7, 10],
        )
        .expect("mixed condensation must succeed");
    assert_eq!(condensed.matrix.num_rows, 7);
    assert_eq!(condensed.matrix.num_cols, 7);
    let csr = condensed.matrix.to_csr();
    for dof in 0..7 {
        assert_eq!(csr.get(dof, dof), Complex64::new(2.0, 0.0));
    }
}

#[test]
fn mixed_rt1_periodic_faces_reject_topology_offsets_and_phase_errors() {
    let (master_faces, _, node_pairs) = mixed_faces();
    let mismatch = [(2, Face::Quad([7, 8, 9, 10]))];
    assert!(matches!(
        pair_periodic_hdiv_rt1_mixed_faces(&master_faces[..1], &mismatch, &node_pairs),
        Err(MpcError::PairCountMismatch { .. })
    ));

    let triangle_pair = HdivRt1PeriodicMixedFacePair::Triangle(
        sonium_fem::boundary::periodic::HdivRt1PeriodicFacePair {
            master_face: 0,
            slave_face: 2,
            vertex_permutation: [0, 1, 2],
        },
    );
    assert!(matches!(
        expand_hdiv_rt1_periodic_mixed_face_constraints(
            &[triangle_pair],
            &[0],
            &[0],
            Complex64::new(1.0, 0.0),
        ),
        Err(MpcError::DofOutOfRange { .. })
    ));

    assert!(matches!(
        expand_hdiv_rt1_periodic_mixed_face_constraints(
            &[],
            &[],
            &[],
            Complex64::new(f64::NAN, 0.0),
        ),
        Err(MpcError::NonFiniteCoefficient { .. })
    ));
}
