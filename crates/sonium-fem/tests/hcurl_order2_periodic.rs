use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::boundary::periodic::{
    MpcError, MpcSystem, PeriodicDofConstraint, PeriodicTransform, SectorPeriodicMap,
    expand_order2_oriented_hcurl_constraints,
};

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn order2_periodic_edges_expand_signed_and_even_modes() {
    let phase = Complex64::new(0.6, 0.8);
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 3), (1, 4)],
        edge_pairs: vec![(2, 7, -1), (5, 9, 1)],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };

    let constraints = map.hcurl_order2_constraints().unwrap();
    assert_eq!(constraints.len(), 4);
    assert_eq!(constraints[0].master, 4);
    assert_eq!(constraints[0].slave, 14);
    assert_complex_close(constraints[0].phase, -phase);
    assert_eq!(constraints[1].master, 5);
    assert_eq!(constraints[1].slave, 15);
    assert_complex_close(constraints[1].phase, phase);
    assert_complex_close(constraints[2].phase, phase);
    assert_complex_close(constraints[3].phase, phase);
}

#[test]
fn order2_periodic_edges_recover_complex_mpc_values() {
    let phase = Complex64::new(0.0, 1.0);
    let constraints = expand_order2_oriented_hcurl_constraints(&[(1, 6, -1)], phase).unwrap();
    let mpcs: Vec<_> = constraints.iter().copied().map(Into::into).collect();
    let system = MpcSystem::new(14, &mpcs).unwrap();

    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.25, -0.5);
    }
    let recovered = system.recover(&reduced).unwrap();

    assert_complex_close(recovered[12], -phase * recovered[2]);
    assert_complex_close(recovered[13], phase * recovered[3]);
    for constraint in &constraints {
        assert_complex_close(constraint.residual(&recovered), Complex64::new(0.0, 0.0));
    }
}

#[test]
fn order2_periodic_edges_reject_invalid_inputs() {
    assert!(matches!(
        expand_order2_oriented_hcurl_constraints(&[(0, 1, 0)], Complex64::new(1.0, 0.0)),
        Err(MpcError::InvalidOrientation(0))
    ));
    assert!(matches!(
        expand_order2_oriented_hcurl_constraints(&[(0, 1, 1)], Complex64::new(f64::NAN, 0.0),),
        Err(MpcError::NonFiniteCoefficient { .. })
    ));
    assert!(matches!(
        expand_order2_oriented_hcurl_constraints(&[(usize::MAX, 1, 1)], Complex64::new(1.0, 0.0),),
        Err(MpcError::DofIndexOverflow(usize::MAX))
    ));

    let duplicate = vec![
        PeriodicDofConstraint {
            master: 0,
            slave: 2,
            phase: Complex64::new(1.0, 0.0),
        },
        PeriodicDofConstraint {
            master: 1,
            slave: 2,
            phase: Complex64::new(1.0, 0.0),
        },
    ];
    let mpcs: Vec<_> = duplicate.iter().copied().map(Into::into).collect();
    assert!(matches!(
        MpcSystem::new(3, &mpcs),
        Err(MpcError::DuplicateSlave(2))
    ));
}

#[test]
fn order2_periodic_edges_condense_complex_fem_system_and_recover_full_field() {
    let map = SectorPeriodicMap {
        node_pairs: vec![],
        edge_pairs: vec![(0, 2, -1)],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase: Complex64::new(1.0, 0.0),
    };
    let mut matrix = TripletMatrix::new(6, 6);
    for dof in 0..6 {
        matrix.push(dof, dof, Complex64::new(1.0, 0.0));
    }
    let system = FemSystem::new(
        matrix,
        (1..=6)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect(),
    );

    let reduced_system = map
        .condense_hcurl_order2_fem_system_complex(&system, &[], 2)
        .unwrap();
    assert_eq!(reduced_system.matrix.num_rows, 4);
    assert_eq!(reduced_system.rhs[0], Complex64::new(-4.0, 0.0));
    assert_eq!(reduced_system.rhs[1], Complex64::new(8.0, 0.0));
    assert_eq!(reduced_system.rhs[2], Complex64::new(3.0, 0.0));
    assert_eq!(reduced_system.rhs[3], Complex64::new(4.0, 0.0));
    let reduced_matrix = reduced_system.matrix.to_csr();
    assert_eq!(reduced_matrix.get(0, 0), Complex64::new(2.0, 0.0));
    assert_eq!(reduced_matrix.get(1, 1), Complex64::new(2.0, 0.0));
    assert_eq!(reduced_matrix.get(2, 2), Complex64::new(1.0, 0.0));
    assert_eq!(reduced_matrix.get(3, 3), Complex64::new(1.0, 0.0));

    let mpc = map.hcurl_order2_mpc_system(6, &[], 2).unwrap();
    let full = mpc
        .recover(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ])
        .unwrap();
    assert_eq!(full[4], Complex64::new(-1.0, 0.0));
    assert_eq!(full[5], Complex64::new(2.0, 0.0));
    for constraint in map.hcurl_order2_constraints().unwrap() {
        assert_complex_close(constraint.residual(&full), Complex64::new(0.0, 0.0));
    }
}
