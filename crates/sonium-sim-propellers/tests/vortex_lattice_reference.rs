use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    Blade3dConfig, BladePanel, Point3, PropellerDesign, VortexLatticeError, VortexLatticeOptions,
    evaluate_vortex_lattice_reference,
};

fn radial_panel(root_x: f64, tip_x: f64, chord_m: f64) -> BladePanel {
    BladePanel {
        vertices: [
            Point3 {
                x: root_x,
                y: chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: tip_x,
                y: chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: tip_x,
                y: -chord_m * 0.5,
                z: 0.0,
            },
            Point3 {
                x: root_x,
                y: -chord_m * 0.5,
                z: 0.0,
            },
        ],
        blade_index: 0,
        radius_fraction: 0.5,
        chord_fraction: chord_m,
        area: (tip_x - root_x) * chord_m,
    }
}

fn rotating_options() -> VortexLatticeOptions {
    VortexLatticeOptions {
        freestream_m_s: Point3 {
            x: 0.0,
            y: 0.0,
            z: 6.0,
        },
        angular_velocity_rad_s: 180.0,
        wake_length_m: 0.8,
        wake_segments: 6,
        ..VortexLatticeOptions::default()
    }
}

#[test]
fn public_reference_returns_rotation_axis_force_and_moment() {
    let result =
        evaluate_vortex_lattice_reference(&[radial_panel(0.04, 0.09, 0.018)], rotating_options())
            .expect("rotating panel should solve");
    assert!(result.is_finite());
    assert!(result.thrust_n.abs() > 1.0e-12);
    assert!(result.total_moment_nm.y.abs() > 1.0e-12);
    assert!(result.panels[0].force_n.z.abs() > 1.0e-12);
    assert!(result.maximum_normal_residual_m_s < 1.0e-8);
}

#[test]
fn public_reference_is_linear_in_kinematic_input() {
    let first =
        evaluate_vortex_lattice_reference(&[radial_panel(0.04, 0.09, 0.018)], rotating_options())
            .expect("first solve");
    let second = evaluate_vortex_lattice_reference(
        &[radial_panel(0.04, 0.09, 0.018)],
        VortexLatticeOptions {
            freestream_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 12.0,
            },
            angular_velocity_rad_s: 360.0,
            wake_length_m: 0.8,
            wake_segments: 6,
            ..VortexLatticeOptions::default()
        },
    )
    .expect("scaled solve");
    assert!(
        (second.panels[0].circulation_m2_s / first.panels[0].circulation_m2_s - 2.0).abs()
            < 1.0e-11
    );
    assert!((second.thrust_n / first.thrust_n - 4.0).abs() < 1.0e-9);
}

#[test]
fn public_reference_handles_multiple_panels_and_wake_refinement() {
    let panels = vec![
        radial_panel(0.04, 0.065, 0.018),
        radial_panel(0.065, 0.09, 0.014),
    ];
    let coarse = evaluate_vortex_lattice_reference(
        &panels,
        VortexLatticeOptions {
            wake_segments: 2,
            ..rotating_options()
        },
    )
    .expect("coarse wake should solve");
    let refined = evaluate_vortex_lattice_reference(
        &panels,
        VortexLatticeOptions {
            wake_segments: 10,
            ..rotating_options()
        },
    )
    .expect("refined wake should solve");
    assert!(coarse.is_finite());
    assert!(refined.is_finite());
    assert_eq!(coarse.matrix_size, 2);
    assert_eq!(refined.wake_segment_count_per_leg, 10);
    assert!(coarse.panels.iter().all(|panel| panel.is_finite()));
    assert!(refined.panels.iter().all(|panel| panel.is_finite()));
}

#[test]
fn generated_conventional_mesh_is_admitted_as_panel_flow_input() {
    let source = PropellerSource::drone_5inch(12_000.0);
    let design = PropellerDesign::conventional(source);
    let config = Blade3dConfig {
        radial_segments: 3,
        chord_segments: 2,
        loop_segments: 4,
        ..Blade3dConfig::default()
    };
    let mesh = config
        .try_build_mesh(&design)
        .expect("small conventional mesh should build");
    let result = evaluate_vortex_lattice_reference(
        &mesh.panels,
        VortexLatticeOptions {
            freestream_m_s: Point3 {
                x: 0.0,
                y: 0.0,
                z: 10.0,
            },
            max_panels: mesh.panels.len(),
            ..VortexLatticeOptions::default()
        },
    )
    .expect("generated mesh should solve");
    assert_eq!(result.panels.len(), mesh.panels.len());
    assert!(result.is_finite());
    assert!(result.maximum_normal_residual_m_s < 1.0e-7);
}

#[test]
fn invalid_public_options_and_geometry_fail_closed() {
    let valid_panel = radial_panel(0.04, 0.09, 0.018);
    assert!(matches!(
        evaluate_vortex_lattice_reference(
            &[valid_panel.clone()],
            VortexLatticeOptions {
                max_panels: 0,
                ..VortexLatticeOptions::default()
            }
        ),
        Err(VortexLatticeError::InvalidOption {
            field: "max_panels",
            ..
        })
    ));
    assert!(matches!(
        evaluate_vortex_lattice_reference(
            &[valid_panel.clone()],
            VortexLatticeOptions {
                residual_tolerance_m_s: f64::NAN,
                ..VortexLatticeOptions::default()
            }
        ),
        Err(VortexLatticeError::InvalidOption {
            field: "residual_tolerance_m_s",
            ..
        })
    ));
    let mut invalid = valid_panel;
    invalid.vertices[2].x = f64::NAN;
    assert!(matches!(
        evaluate_vortex_lattice_reference(&[invalid], VortexLatticeOptions::default()),
        Err(VortexLatticeError::InvalidPanel {
            field: "vertices",
            ..
        })
    ));
}

#[test]
fn nonzero_rotation_axis_is_required_even_without_rotation() {
    assert!(matches!(
        evaluate_vortex_lattice_reference(
            &[radial_panel(0.04, 0.09, 0.018)],
            VortexLatticeOptions {
                rotation_axis: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..VortexLatticeOptions::default()
            }
        ),
        Err(VortexLatticeError::InvalidOption {
            field: "rotation_axis",
            ..
        })
    ));
}
