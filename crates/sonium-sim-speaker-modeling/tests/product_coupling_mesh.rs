use sonium_sim_speaker_modeling::fem_bem_coupling::{
    MonolithicProductCouplingError, MonolithicSealedProductCoupling, ProductCouplingMeshError,
};
use sonium_sim_speaker_modeling::{
    BraceCrossSection, BraceDesign, CabinetDesign, CabinetMaterial, CrossoverBranch,
    CrossoverNetwork, PanelAxis, PanelId, ProductDesign, ProductDriver, ProductInteriorMeshConfig,
    ProductInteriorMeshError, ProductInteriorPatchKind, ProductWaveguideDesign,
    build_body_fitted_sealed_product_interior_mesh, build_product_coupling_mesh,
};

use num_complex::Complex64;
use sonium_sim_speaker_modeling::{
    ArtifactProvenance, InterfaceReferenceEvidence, MonolithicProductReferenceEvidence,
    ProductReferencePoint, ProductReferenceSweep,
};

fn ported_product() -> ProductDesign {
    ProductDesign {
        cabinet: CabinetDesign {
            box_width_m: 0.40,
            box_height_m: 0.30,
            box_depth_m: 0.30,
            wall_material: CabinetMaterial::Mdf18mm,
            port: Some({
                let mut port = CabinetDesign::default().port.expect("default port");
                port.position_m = [0.06, 0.0];
                port
            }),
            baffle_width_m: 0.44,
            baffle_height_m: 0.34,
            braces: vec![BraceDesign {
                panel: PanelId::Top,
                split_along: PanelAxis::Width,
                position_fraction: 0.5,
                cross_section: Some(BraceCrossSection {
                    width_m: 0.05,
                    depth_m: 0.05,
                }),
            }],
        },
        drivers: vec![ProductDriver {
            id: "canonical-port-driver".into(),
            radius_m: 0.045,
            position_m: [-0.06, 0.0, 0.0],
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

fn waveguide_for(driver_id: &str) -> ProductWaveguideDesign {
    ProductWaveguideDesign {
        driver_id: driver_id.into(),
        horizontal_spline: vec![[0.0, 0.04], [0.5, 0.03], [1.0, 0.045]],
        vertical_spline: vec![[0.0, 0.04], [0.5, 0.03], [1.0, 0.045]],
        depth_m: 0.02,
        surface_specific_impedance_pa_s_per_m: 2.5e6,
    }
}

#[test]
fn body_fitted_waveguide_rejects_tilted_driver() {
    let mut design = ported_product();
    design.cabinet.port = None;
    design.cabinet.braces.clear();
    design.drivers[0].normal = [0.0, 2.0_f64.sqrt().recip(), 2.0_f64.sqrt().recip()];
    design.waveguide = Some(waveguide_for(&design.drivers[0].id));

    let error = build_body_fitted_sealed_product_interior_mesh(
        &design,
        ProductInteriorMeshConfig::default(),
    )
    .expect_err("tilted waveguide should be rejected");
    assert!(
        matches!(error, ProductInteriorMeshError::WaveguideLayoutUnsupported),
        "unexpected tilted-waveguide error: {error:?}"
    );
}

#[test]
fn body_fitted_waveguide_composes_with_port_and_brace() {
    let mut design = ported_product();
    design.waveguide = Some(waveguide_for(&design.drivers[0].id));

    assert!(matches!(
        build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default()),
        Err(ProductCouplingMeshError::Interior(
            ProductInteriorMeshError::WaveguideLayoutUnsupported
        ))
    ));
}

#[test]
fn body_fitted_waveguide_rejects_multiple_drivers() {
    let mut design = ported_product();
    design.cabinet.port = None;
    design.cabinet.braces.clear();
    let mut second_driver = design.drivers[0].clone();
    second_driver.id = "second-driver".into();
    second_driver.position_m = [0.06, 0.0, 0.0];
    second_driver.crossover_branch = 1;
    design.drivers.push(second_driver);
    design.crossover.branches.push(CrossoverBranch::default());
    design.waveguide = Some(waveguide_for(&design.drivers[0].id));

    assert!(matches!(
        build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default()),
        Err(ProductCouplingMeshError::Interior(
            ProductInteriorMeshError::WaveguideLayoutUnsupported
        ))
    ));
}

#[test]
fn canonical_product_mesh_retains_port_driver_and_brace_traces() {
    let mesh = build_product_coupling_mesh(&ported_product(), ProductInteriorMeshConfig::default())
        .expect("ported product coupling mesh should build");

    assert!(
        mesh.geometry_fingerprint
            .starts_with("product-coupling-v1:")
    );
    assert_eq!(
        mesh.exterior.bem_to_fem_boundary.len(),
        mesh.exterior.mesh.elements.len()
    );
    for kind in [
        ProductInteriorPatchKind::DriverRear { driver_index: 0 },
        ProductInteriorPatchKind::PortMouth,
        ProductInteriorPatchKind::PortWall,
        ProductInteriorPatchKind::BraceWall { brace_index: 0 },
    ] {
        let trace = mesh
            .trace(kind)
            .expect("canonical patch trace should be retained");
        assert!(!trace.fem_boundary_faces.is_empty());
        if !matches!(
            kind,
            ProductInteriorPatchKind::PortWall | ProductInteriorPatchKind::BraceWall { .. }
        ) {
            assert!(!trace.bem_elements.is_empty());
        }
        assert!(trace.area_m2.is_finite() && trace.area_m2 > 0.0);
        assert!(
            trace
                .bem_elements
                .iter()
                .all(|&element| element < mesh.exterior.mesh.elements.len())
        );
    }
}

#[test]
fn monolithic_product_coupling_consumes_canonical_trace_maps() {
    let design = ported_product();
    let mesh = build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default())
        .expect("ported product coupling mesh should build");
    MonolithicSealedProductCoupling::from_coupling_mesh(&design, mesh, 343.0, 1.204)
        .expect("canonical product coupling mesh should construct monolithic reference");

    let mut stale_mesh = build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default())
        .expect("ported product coupling mesh should build");
    let invalid_element = stale_mesh.exterior.mesh.elements.len();
    stale_mesh
        .traces
        .get_mut(&ProductInteriorPatchKind::DriverRear { driver_index: 0 })
        .expect("driver trace")
        .bem_elements[0] = invalid_element;

    let error = match MonolithicSealedProductCoupling::from_coupling_mesh(
        &design, stale_mesh, 343.0, 1.204,
    ) {
        Ok(_) => panic!("stale canonical trace must be rejected before solving"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        MonolithicProductCouplingError::InvalidTraceElement {
            patch: ProductInteriorPatchKind::DriverRear { driver_index: 0 },
            element,
        } if element == invalid_element
    ));
}

#[test]
fn product_coupling_evidence_round_trips_and_rejects_tampering() {
    let design = ported_product();
    let mesh = build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default())
        .expect("ported product coupling mesh should build");
    let evidence = mesh.evidence();
    evidence.validate().expect("fresh evidence should be valid");

    let encoded = serde_json::to_string(&evidence).expect("mesh evidence should serialize");
    let decoded = serde_json::from_str(&encoded).expect("mesh evidence should deserialize");
    assert_eq!(decoded, evidence);
    mesh.validate_evidence(&decoded)
        .expect("round-tripped evidence should match mesh topology");

    let mut tampered = decoded.clone();
    let trace_index = tampered
        .traces
        .iter()
        .position(|trace| !trace.bem_elements.is_empty())
        .expect("canonical evidence should contain a mapped interface");
    tampered.traces[trace_index].bem_elements[0] = mesh.exterior.mesh.elements.len();
    assert!(matches!(
        mesh.validate_evidence(&tampered),
        Err(
            sonium_sim_speaker_modeling::ProductCouplingMeshEvidenceError::InvalidTraceElement { .. }
        )
    ));

    let mut invalid_face = decoded.clone();
    let trace_index = invalid_face
        .traces
        .iter()
        .position(|trace| !trace.fem_boundary_faces.is_empty())
        .expect("canonical evidence contains a FEM interface");
    invalid_face.traces[trace_index].fem_boundary_faces[0] =
        invalid_face.interior_boundary_face_count;
    assert!(matches!(
        invalid_face.validate(),
        Err(sonium_sim_speaker_modeling::ProductCouplingMeshEvidenceError::InvalidTraceFace { .. })
    ));

    let mut wrong_schema = decoded;
    wrong_schema.schema_version += 1;
    assert!(matches!(
        mesh.validate_evidence(&wrong_schema),
        Err(sonium_sim_speaker_modeling::ProductCouplingMeshEvidenceError::SchemaVersion { .. })
    ));
}

#[test]
fn canonical_product_mesh_rejects_unresolved_port_wall_geometry() {
    let mut design = ported_product();
    design
        .cabinet
        .port
        .as_mut()
        .expect("ported fixture")
        .wall_thickness_m = None;
    let error = build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default())
        .expect_err("unresolved port wall must fail before BEM construction");
    assert!(error.to_string().contains("explicit tube-wall thickness"));
}

fn reference_sweep_for_canonical_mesh(
    mesh: &sonium_sim_speaker_modeling::ProductCouplingMesh,
) -> ProductReferenceSweep {
    let interface = |interface_id: &str| InterfaceReferenceEvidence {
        interface_id: interface_id.into(),
        relative_residual: 0.0,
        power_residual_w: 0.0,
        volume_flow_m3_per_s: Complex64::new(0.0, 0.0),
        mesh_convergence_error: Some(0.0),
        port_flow_promotion: None,
    };
    let driver_velocity = Complex64::new(1.0e-3, 0.0);
    let point = ProductReferencePoint {
        frequency_hz: 90.0,
        drive_velocity_rms_m_per_s: driver_velocity,
        driver_velocities_rms_m_per_s: vec![driver_velocity],
        aperture_volume_velocity_m3_per_s: Complex64::new(0.0, 0.0),
        mean_pressure_rms_pa: Complex64::new(1.0, 0.0),
        mean_abs_pressure_pa: 1.0,
        solver_iterations: 1,
        relative_residual: 0.0,
        monolithic_evidence: Some(MonolithicProductReferenceEvidence {
            interior_relative_residual: 0.0,
            exterior_relative_residual: 0.0,
            mechanical_relative_residual: 0.0,
            electrical_relative_residual: 0.0,
            electrical_power_residual_w: 0.0,
            mechanical_power_residual_w: 0.0,
            port_volume_velocity_m3_per_s: Complex64::new(0.0, 0.0),
            maximum_port_pressure_jump_pa: 0.0,
            mesh_node_count: mesh.interior.diagnostics.node_count,
            mesh_tetrahedron_count: mesh.interior.diagnostics.tetrahedron_count,
            mesh_geometry_fingerprint: mesh.interior.geometry_fingerprint.clone(),
            independent_validation: None,
        }),
        interface_evidence: Some(vec![
            interface("driver-aperture-0"),
            interface("port-mouth"),
        ]),
        thermal_evidence: None,
        structural_evidence: None,
    };
    ProductReferenceSweep {
        speed_of_sound_m_per_s: 343.0,
        density_kg_per_m3: 1.204,
        interior_dofs: mesh.interior.diagnostics.node_count,
        aperture_elements: mesh.exterior.aperture_elements.len().max(1),
        points: vec![point],
        driver_count: 1,
        provenance: ArtifactProvenance::native_golden(),
        rom_promotion: None,
        mesh_convergence: Vec::new(),
    }
}

#[test]
fn reference_sweep_binds_to_canonical_mesh_topology_and_interfaces() {
    let design = ported_product();
    let mesh = build_product_coupling_mesh(&design, ProductInteriorMeshConfig::default())
        .expect("ported product coupling mesh should build");
    let sweep = reference_sweep_for_canonical_mesh(&mesh);
    sweep
        .validate_against_product_coupling_mesh(&mesh)
        .expect("reference sweep should bind to its canonical mesh");

    let mut stale_fingerprint = sweep.clone();
    stale_fingerprint.points[0]
        .monolithic_evidence
        .as_mut()
        .expect("monolithic evidence")
        .mesh_geometry_fingerprint = "stale-product-geometry".into();
    assert!(matches!(
        stale_fingerprint.validate_against_product_coupling_mesh(&mesh),
        Err(sonium_sim_speaker_modeling::ProductDesignError::InvalidReferenceEvidence(message))
            if message.contains("fingerprint")
    ));

    let mut missing_interface = sweep.clone();
    missing_interface.points[0]
        .interface_evidence
        .as_mut()
        .expect("interface evidence")
        .retain(|interface| interface.interface_id != "port-mouth");
    assert!(matches!(
        missing_interface.validate_against_product_coupling_mesh(&mesh),
        Err(sonium_sim_speaker_modeling::ProductDesignError::InvalidReferenceEvidence(message))
            if message.contains("missing canonical interface port-mouth")
    ));

    let mut extra_interface = sweep.clone();
    extra_interface.points[0]
        .interface_evidence
        .as_mut()
        .expect("interface evidence")
        .push(InterfaceReferenceEvidence {
            interface_id: "stale-interface".into(),
            relative_residual: 0.0,
            power_residual_w: 0.0,
            volume_flow_m3_per_s: Complex64::new(0.0, 0.0),
            mesh_convergence_error: Some(0.0),
            port_flow_promotion: None,
        });
    assert!(matches!(
        extra_interface.validate_against_product_coupling_mesh(&mesh),
        Err(sonium_sim_speaker_modeling::ProductDesignError::InvalidReferenceEvidence(message))
            if message.contains("interface identities differ")
    ));

    let mut wrong_topology = sweep;
    wrong_topology.interior_dofs += 1;
    assert!(matches!(
        wrong_topology.validate_against_product_coupling_mesh(&mesh),
        Err(sonium_sim_speaker_modeling::ProductDesignError::InvalidReferenceEvidence(message))
            if message.contains("DOF count")
    ));
}
