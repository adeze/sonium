use sonium_bem::{
    ACOUSTIC_CONVENTION, BEM_CONVENTION_CONTRACT_VERSION, BemConvention, BemConventionError,
    BemDomain, BurtonMillerSign, NormalVelocityDirection, PressureTrace,
};

#[test]
fn canonical_exterior_contract_freezes_signs_traces_tau_and_burton_miller() {
    let contract = ACOUSTIC_CONVENTION
        .typed_contract(BemDomain::Exterior, PressureTrace::Total)
        .expect("canonical BEM convention is valid");

    assert_eq!(contract.version, BEM_CONVENTION_CONTRACT_VERSION);
    assert_eq!((contract.time_sign, contract.spatial_sign), (-1, 1));
    assert_eq!(contract.domain, BemDomain::Exterior);
    assert_eq!(contract.tau, 1.0);
    assert_eq!(
        contract.traces.normal_velocity,
        NormalVelocityDirection::Outward
    );
    assert_eq!(contract.traces.pressure, PressureTrace::Total);
    assert_eq!(
        contract.burton_miller_sign,
        BurtonMillerSign::PositiveImaginary
    );
}

#[test]
fn domain_tau_is_typed_and_interior_trace_adaptation_is_explicit() {
    assert_eq!(BemDomain::Exterior.tau(), 1.0);
    assert_eq!(BemDomain::Interior.tau(), -1.0);

    let contract = ACOUSTIC_CONVENTION
        .typed_contract(BemDomain::Interior, PressureTrace::Scattered)
        .expect("interior domain uses the same checked harmonic signs");
    assert_eq!(contract.tau, -1.0);
    assert_eq!(contract.traces.pressure, PressureTrace::Scattered);
}

#[test]
fn invalid_signs_and_inward_normals_cannot_cross_the_bem_boundary() {
    let invalid_sign = BemConvention {
        time_sign: 0,
        ..ACOUSTIC_CONVENTION
    };
    assert!(matches!(
        invalid_sign.typed_contract(BemDomain::Exterior, PressureTrace::Total),
        Err(BemConventionError::InvalidSign { field: "time", .. })
    ));

    let inward = BemConvention {
        outward_normals: false,
        ..ACOUSTIC_CONVENTION
    };
    assert_eq!(
        inward.typed_contract(BemDomain::Exterior, PressureTrace::Total),
        Err(BemConventionError::InwardNormals)
    );
}
