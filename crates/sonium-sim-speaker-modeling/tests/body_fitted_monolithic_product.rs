//! Bounded W2-02 production-path integration evidence.
//!
//! Stable case ID: `w2-02.body-fitted-monolithic-product-integration.v1`
//!
//! This intentionally stays as an application integration test rather than
//! changing the shared QA manifest.  It exercises the public product path:
//! body-fitted sealed interior mesh -> exterior BEM bridge -> monolithic
//! FEM/BEM/electromechanical solve.

use num_complex::Complex64;
use sonium_sim_speaker_modeling::fem_bem_coupling::MonolithicSealedProductCoupling;
use sonium_sim_speaker_modeling::{
    BraceCrossSection, BraceDesign, CabinetDesign, CabinetMaterial, CrossoverBranch,
    CrossoverNetwork, PanelAxis, PanelId, ProductDesign, ProductDriver, ProductInteriorMeshConfig,
    ProductInteriorPatchKind, build_body_fitted_sealed_product_interior_mesh,
};

const CASE_ID: &str = "w2-02.body-fitted-monolithic-product-integration.v1";

fn product_fixture() -> ProductDesign {
    ProductDesign {
        cabinet: CabinetDesign {
            box_width_m: 0.40,
            box_height_m: 0.30,
            box_depth_m: 0.30,
            wall_material: CabinetMaterial::Mdf18mm,
            port: None,
            baffle_width_m: 0.44,
            baffle_height_m: 0.34,
            braces: Vec::new(),
        },
        drivers: vec![ProductDriver {
            id: "w2-02-driver".to_string(),
            radius_m: 0.08,
            position_m: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            force_factor_tm: 5.0,
            coil_resistance_ohm: 6.0,
            coil_reactance_ohm: 1.0,
            mechanical_resistance_n_s_per_m: 1.0,
            mechanical_reactance_n_s_per_m: 2.0,
            crossover_branch: 0,
        }],
        crossover: CrossoverNetwork {
            branches: vec![CrossoverBranch::default()],
        },
        waveguide: None,
        structural_acoustic: None,
    }
}

#[test]
fn w2_02_body_fitted_mesh_reaches_monolithic_fem_bem_with_evidence() {
    assert_eq!(
        CASE_ID,
        "w2-02.body-fitted-monolithic-product-integration.v1"
    );

    let design = product_fixture();
    design.validate().expect("bounded product fixture is valid");
    let mesh = build_body_fitted_sealed_product_interior_mesh(
        &design,
        ProductInteriorMeshConfig::default(),
    )
    .expect("body-fitted sealed product mesh");

    // Mesh evidence carried across the production handoff.
    assert!(!mesh.geometry_fingerprint.is_empty());
    assert!(mesh.diagnostics.node_count > 0);
    assert!(mesh.diagnostics.tetrahedron_count > 0);
    assert!(mesh.diagnostics.boundary_face_count > 0);
    assert!(mesh.diagnostics.fluid_volume_m3.is_finite());
    assert!(mesh.diagnostics.expected_box_volume_m3 > 0.0);
    let volume_relative_error =
        (mesh.diagnostics.fluid_volume_m3 / mesh.diagnostics.expected_box_volume_m3 - 1.0).abs();
    assert!(
        volume_relative_error < 1.0e-10,
        "{CASE_ID}: fluid volume relative error={volume_relative_error:e}"
    );

    let driver_patch = mesh
        .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
        .expect("body-fitted driver rear patch");
    assert!(!driver_patch.boundary_faces.is_empty());
    let expected_driver_area_m2 = std::f64::consts::PI * design.drivers[0].radius_m.powi(2);
    let driver_area_relative_error = (driver_patch.area_m2 / expected_driver_area_m2 - 1.0).abs();
    assert!(
        driver_area_relative_error < 2.0e-2,
        "{CASE_ID}: driver patch area relative error={driver_area_relative_error:e}"
    );
    assert!(
        mesh.patches
            .iter()
            .filter(|patch| matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)))
            .all(|patch| !patch.boundary_faces.is_empty() && patch.area_m2.is_finite())
    );
    let expected_box_surface_area_m2 = 2.0
        * (design.cabinet.box_width_m * design.cabinet.box_height_m
            + design.cabinet.box_width_m * design.cabinet.box_depth_m
            + design.cabinet.box_height_m * design.cabinet.box_depth_m);
    let patch_area_relative_error = (mesh.patches.iter().map(|patch| patch.area_m2).sum::<f64>()
        / expected_box_surface_area_m2
        - 1.0)
        .abs();
    assert!(
        patch_area_relative_error < 1.0e-10,
        "{CASE_ID}: patch-area conservation error={patch_area_relative_error:e}"
    );
    let interior_node_count = mesh.mesh.num_nodes();

    let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
        .expect("body-fitted sealed product FEM/BEM coupling");
    let solution = coupling
        .solve_frequency(&design, 120.0, Complex64::new(2.0, 0.0))
        .expect("body-fitted sealed product monolithic solve");

    // Coupling evidence: FEM and BEM unknowns both participate in the solve,
    // and every production residual channel is retained and closed.
    assert_eq!(solution.interior_pressure.len(), interior_node_count);
    assert!(!solution.exterior_pressure.is_empty());
    assert_eq!(solution.driver_velocity_m_per_s.len(), 1);
    assert_eq!(solution.driver_current_a.len(), 1);
    assert!(solution.driver_velocity_m_per_s[0].norm() > 0.0);
    assert!(solution.driver_current_a[0].norm() > 0.0);
    assert!(solution.interior_relative_residual < 1.0e-10);
    assert!(solution.exterior_relative_residual < 1.0e-10);
    assert!(solution.mechanical_relative_residual < 1.0e-10);
    assert!(solution.electrical_relative_residual < 1.0e-10);
    assert!(solution.relative_residual < 1.0e-10);
    assert!(solution.electrical_input_power_w.is_finite());
    assert!(solution.motor_mechanical_power_w.is_finite());
    assert!(solution.electrical_power_residual_w.abs() < 1.0e-10);
    assert!(solution.mechanical_power_residual_w.abs() < 1.0e-10);
}

#[test]
fn w2_02_body_fitted_brace_is_lowered_into_typed_patch() {
    let mut product = product_fixture();
    product.cabinet.braces.push(BraceDesign {
        panel: PanelId::Top,
        split_along: PanelAxis::Width,
        position_fraction: 0.5,
        cross_section: Some(BraceCrossSection {
            width_m: 0.03,
            depth_m: 0.03,
        }),
    });

    let mesh = build_body_fitted_sealed_product_interior_mesh(
        &product,
        ProductInteriorMeshConfig::default(),
    )
    .expect("body-fitted braces should use the multiwall lowering path");

    assert!(
        mesh.patch(ProductInteriorPatchKind::BraceWall { brace_index: 0 })
            .is_some_and(|patch| patch.area_m2 > 0.0)
    );
}

#[test]
fn w2_02_body_fitted_ported_product_closes_port_loading() {
    let mut design = product_fixture();
    let mut port = CabinetDesign::default()
        .port
        .expect("default cabinet port fixture");
    port.position_m = [0.15, 0.0];
    design.cabinet.port = Some(port);
    design.validate().expect("ported product fixture is valid");

    let mesh = build_body_fitted_sealed_product_interior_mesh(
        &design,
        ProductInteriorMeshConfig::default(),
    )
    .expect("body-fitted ported product mesh");
    assert!(
        mesh.patch(ProductInteriorPatchKind::PortMouth)
            .is_some_and(|patch| patch.area_m2 > 0.0 && !patch.boundary_faces.is_empty())
    );
    assert!(
        mesh.patch(ProductInteriorPatchKind::PortWall)
            .is_some_and(|patch| patch.area_m2 > 0.0 && !patch.boundary_faces.is_empty())
    );

    let coupling = MonolithicSealedProductCoupling::new(&design, mesh, 343.0, 1.204)
        .expect("body-fitted ported FEM/BEM coupling");
    let solution = coupling
        .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
        .expect("body-fitted ported monolithic solve");

    assert!(!solution.port_velocity_m_per_s.is_empty());
    assert!(solution.port_volume_velocity_m3_per_s.norm() > 0.0);
    assert!(
        solution
            .port_pressure_jump_pa
            .iter()
            .all(|value| value.norm().is_finite())
    );
    assert!(solution.relative_residual < 1.0e-10);
    assert!(solution.interior_relative_residual < 1.0e-10);
    assert!(solution.exterior_relative_residual < 1.0e-10);
    assert!(solution.mechanical_relative_residual < 1.0e-10);
    assert!(solution.electrical_relative_residual < 1.0e-10);
}
