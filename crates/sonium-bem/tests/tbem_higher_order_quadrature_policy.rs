use sonium_bem::core::assembly::{TbemQuadraturePolicy, validate_tbem_quadrature_for_basis};
use sonium_bem::core::mesh::generate_sphere_mesh;
use sonium_bem::core::{BasisOrder, PhysicsParams};

#[test]
fn basis_weighted_quadrature_gate_covers_linear_and_quadratic_panels() {
    let mesh = generate_sphere_mesh(0.1, 3, 4);
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let permissive = TbemQuadraturePolicy {
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
        ..TbemQuadraturePolicy::default()
    };

    for basis in [BasisOrder::Linear, BasisOrder::Quadratic] {
        validate_tbem_quadrature_for_basis(
            &mesh.elements,
            &mesh.nodes,
            &physics,
            basis,
            &permissive,
        )
        .expect("permissive basis-weighted policy should pass");
    }
}

#[test]
fn basis_weighted_quadrature_gate_rejects_zero_budget() {
    let mesh = generate_sphere_mesh(0.1, 3, 4);
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let strict = TbemQuadraturePolicy {
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..TbemQuadraturePolicy::default()
    };

    let error = validate_tbem_quadrature_for_basis(
        &mesh.elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Linear,
        &strict,
    )
    .expect_err("zero budget must reject a non-identical basis estimate");
    assert!(error.error.to_string().contains("quadrature"));
}
