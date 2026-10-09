use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    HcurlOrder2PeriodicFacePair, MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_order2_hcurl_face_constraints,
};

fn assert_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "{actual:?} != {expected:?}"
    );
}

fn pair(slave_vertices: [usize; 3]) -> HcurlOrder2PeriodicFacePair {
    HcurlOrder2PeriodicFacePair {
        master_face: 1,
        slave_face: 5,
        master_vertices: [10, 20, 30],
        slave_vertices,
    }
}

#[test]
fn order2_periodic_face_modes_match_identity_cyclic_and_reversed_maps() {
    let phase = Complex64::new(0.6, 0.8);

    let identity =
        expand_order2_hcurl_face_constraints(&[pair([110, 120, 130])], 12, phase).unwrap();
    assert_eq!(identity[0].slave, 22);
    assert_eq!(identity[1].slave, 23);
    assert_eq!(identity[0].masters.len(), 1);
    assert_eq!(identity[1].masters.len(), 1);
    assert_eq!(identity[0].masters[0].dof, 14);
    assert_eq!(identity[1].masters[0].dof, 15);
    assert_close(identity[0].masters[0].coefficient, phase);
    assert_close(identity[1].masters[0].coefficient, phase);

    // Master [10,20,30] maps to slave canonical positions [2,0,1].
    let cyclic = expand_order2_hcurl_face_constraints(&[pair([120, 130, 110])], 12, phase).unwrap();
    assert_eq!(cyclic[0].masters.len(), 2);
    assert_eq!(cyclic[1].masters.len(), 1);
    assert_eq!(cyclic[0].masters[0].dof, 14);
    assert_eq!(cyclic[0].masters[1].dof, 15);
    assert_close(cyclic[0].masters[0].coefficient, -phase);
    assert_close(cyclic[0].masters[1].coefficient, -phase);
    assert_eq!(cyclic[1].masters[0].dof, 14);
    assert_close(cyclic[1].masters[0].coefficient, phase);

    // Master [10,20,30] maps to slave canonical positions [0,2,1].
    let reversed =
        expand_order2_hcurl_face_constraints(&[pair([110, 130, 120])], 12, phase).unwrap();
    assert_eq!(reversed[0].masters.len(), 1);
    assert_eq!(reversed[1].masters.len(), 2);
    assert_eq!(reversed[0].masters[0].dof, 14);
    assert_close(reversed[0].masters[0].coefficient, -phase);
    assert_eq!(reversed[1].masters[0].dof, 14);
    assert_eq!(reversed[1].masters[1].dof, 15);
    assert_close(reversed[1].masters[0].coefficient, phase);
    assert_close(reversed[1].masters[1].coefficient, phase);
}

#[test]
fn order2_periodic_face_modes_recover_complex_mpc_values() {
    let phase = Complex64::new(0.0, 1.0);
    let constraints =
        expand_order2_hcurl_face_constraints(&[pair([120, 130, 110])], 12, phase).unwrap();
    let system = MpcSystem::new(24, &constraints).unwrap();
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.25, -0.5);
    }
    let recovered = system.recover(&reduced).unwrap();

    for mpc in &constraints {
        let expected: Complex64 = mpc
            .masters
            .iter()
            .map(|term| term.coefficient * recovered[term.dof])
            .sum();
        assert_close(recovered[mpc.slave], expected);
    }

    let map = SectorPeriodicMap {
        node_pairs: vec![],
        edge_pairs: vec![],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let through_map = map
        .hcurl_order2_face_constraints(&[pair([110, 120, 130])], 12)
        .unwrap();
    assert_eq!(
        through_map,
        expand_order2_hcurl_face_constraints(&[pair([110, 120, 130])], 12, phase).unwrap()
    );
}

#[test]
fn order2_periodic_face_modes_reject_invalid_metadata() {
    let phase = Complex64::new(1.0, 0.0);
    assert!(matches!(
        expand_order2_hcurl_face_constraints(&[pair([110, 110, 130])], 12, phase),
        Err(MpcError::InvalidFaceVertexMap { .. })
    ));
    assert!(matches!(
        expand_order2_hcurl_face_constraints(
            &[pair([110, 120, 130])],
            12,
            Complex64::new(f64::NAN, 0.0),
        ),
        Err(MpcError::NonFiniteCoefficient { .. })
    ));
    assert!(matches!(
        expand_order2_hcurl_face_constraints(
            &[HcurlOrder2PeriodicFacePair {
                master_face: usize::MAX,
                slave_face: 0,
                master_vertices: [10, 20, 30],
                slave_vertices: [110, 120, 130],
            }],
            usize::MAX,
            phase,
        ),
        Err(MpcError::DofIndexOverflow(_))
    ));
}
