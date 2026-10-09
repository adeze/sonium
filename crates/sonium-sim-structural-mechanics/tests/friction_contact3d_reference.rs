use sonium_sim_structural_mechanics::{
    FRICTIONAL_3D_CONTACT_SCHEMA_VERSION, Frictional3DContactStudy, run_frictional_3d_contact,
};

fn assert_close(left: f64, right: f64) {
    let tolerance = 1.0e-12 * left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right}"
    );
}

#[test]
fn prescribed_3d_friction_contact_is_finite_and_capped() {
    let result = run_frictional_3d_contact(&Frictional3DContactStudy::introductory())
        .expect("bounded frictional contact reference should converge");

    assert_eq!(result.schema_version, FRICTIONAL_3D_CONTACT_SCHEMA_VERSION);
    assert!(result.active_pairs > 0);
    assert!(result.max_penetration_m > 0.0);
    assert!(result.friction_active);
    assert!(result.total_normal_reaction_n.is_finite());
    assert!(result.total_tangential_reaction_n > 0.0);
    assert!(result.coulomb_utilization <= 1.0 + 1.0e-8);
    assert!(result.net_contact_force_n < 1.0e-8);
}

#[test]
fn prescribed_3d_friction_contact_round_trips() {
    let result = run_frictional_3d_contact(&Frictional3DContactStudy::introductory())
        .expect("bounded frictional contact reference should converge");
    let encoded = serde_json::to_string(&result).expect("serialize result");
    let decoded: sonium_sim_structural_mechanics::Frictional3DContactResult =
        serde_json::from_str(&encoded).expect("deserialize result");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_close(decoded.study.length_m, result.study.length_m);
    assert_close(decoded.study.width_m, result.study.width_m);
    assert_close(decoded.study.height_m, result.study.height_m);
    assert_eq!(decoded.study.nx, result.study.nx);
    assert_eq!(decoded.study.ny, result.study.ny);
    assert_eq!(decoded.study.nz, result.study.nz);
    assert_close(
        decoded.study.normal_compression_m,
        result.study.normal_compression_m,
    );
    assert_close(
        decoded.study.tangential_shear_m,
        result.study.tangential_shear_m,
    );
    assert_close(decoded.study.contact_epsilon, result.study.contact_epsilon);
    assert_close(
        decoded.study.friction_coefficient,
        result.study.friction_coefficient,
    );
    assert_close(
        decoded.study.contact_search_radius_m,
        result.study.contact_search_radius_m,
    );
    assert_close(
        decoded.study.contact_bvh_margin_m,
        result.study.contact_bvh_margin_m,
    );
    assert_eq!(decoded.nodes, result.nodes);
    assert_eq!(decoded.elements, result.elements);
    assert_eq!(decoded.active_pairs, result.active_pairs);
    assert_close(decoded.max_penetration_m, result.max_penetration_m);
    assert_close(
        decoded.total_normal_reaction_n,
        result.total_normal_reaction_n,
    );
    assert_close(
        decoded.total_tangential_reaction_n,
        result.total_tangential_reaction_n,
    );
    assert_close(decoded.coulomb_utilization, result.coulomb_utilization);
    assert_close(decoded.net_contact_force_n, result.net_contact_force_n);
    assert_eq!(decoded.friction_active, result.friction_active);
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.execution, result.execution);
}

#[test]
fn prescribed_3d_friction_contact_rejects_invalid_coefficient() {
    let mut study = Frictional3DContactStudy::introductory();
    study.friction_coefficient = -0.1;
    assert!(run_frictional_3d_contact(&study).is_err());
}
