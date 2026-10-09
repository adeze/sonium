use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    AirfoilPolar, AirfoilPolarPoint, Blade3dConfig, BladeElementBemOptions, BladeElementModel,
    BladeElementOperatingPoint, BladeElementStation, PanelLoadProjectionError,
    PanelLoadProjectionOptions, PropellerDesign, project_bem_loads_to_panel_mesh,
};

fn source() -> PropellerSource {
    PropellerSource::drone_5inch(12_000.0)
}

fn model() -> BladeElementModel {
    let source = source();
    let stations = vec![
        BladeElementStation {
            radius_m: 0.018,
            chord_m: 0.018,
            pitch_deg: 24.0,
        },
        BladeElementStation {
            radius_m: 0.036,
            chord_m: 0.014,
            pitch_deg: 21.0,
        },
        BladeElementStation {
            radius_m: 0.054,
            chord_m: 0.010,
            pitch_deg: 18.0,
        },
    ];
    let polar = AirfoilPolar::try_new(vec![
        AirfoilPolarPoint {
            angle_of_attack_deg: -20.0,
            lift_coefficient: -0.8,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 0.0,
            lift_coefficient: 0.0,
            drag_coefficient: 0.02,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 20.0,
            lift_coefficient: 0.8,
            drag_coefficient: 0.08,
        },
        AirfoilPolarPoint {
            angle_of_attack_deg: 40.0,
            lift_coefficient: 0.9,
            drag_coefficient: 0.16,
        },
    ])
    .expect("valid polar");
    BladeElementModel::try_from_source(&source, stations, polar).expect("valid BEM model")
}

fn bem_result() -> sonium_sim_propellers::BladeElementBemResult {
    model()
        .evaluate_bem(
            BladeElementOperatingPoint {
                rpm: 12_000.0,
                freestream_m_s: 10.0,
                density_kg_m3: 1.204,
            },
            BladeElementBemOptions {
                max_iterations: 150,
                convergence_tolerance: 1.0e-8,
                relaxation: 0.35,
                ..BladeElementBemOptions::default()
            },
        )
        .expect("valid converged BEM result")
}

fn config() -> Blade3dConfig {
    Blade3dConfig {
        radial_segments: 8,
        chord_segments: 4,
        loop_segments: 12,
        ..Blade3dConfig::default()
    }
}

fn projection_options(blade_count: usize) -> PanelLoadProjectionOptions {
    let source = source();
    PanelLoadProjectionOptions {
        hub_radius_m: source.hub_radius,
        tip_radius_m: source.tip_radius,
        blade_count,
        ..PanelLoadProjectionOptions::default()
    }
}

#[test]
fn conventional_mesh_projection_preserves_bem_thrust_and_torque() {
    let source = source();
    let mesh = config()
        .try_build_mesh(&PropellerDesign::conventional(source.clone()))
        .expect("conventional mesh should build");
    let bem = bem_result();
    let result = project_bem_loads_to_panel_mesh(&mesh, &bem, projection_options(source.n_blades))
        .expect("conventional load projection should pass");

    assert_eq!(result.panels.len(), mesh.panels.len());
    assert!(result.closure_passed);
    assert!(result.thrust_closure_error_n.abs() < 1.0e-10);
    assert!(result.torque_closure_error_nm.abs() < 1.0e-10);
    assert!(result.panels.iter().any(|panel| panel.force_n[2] > 0.0));
    assert!(result.is_finite());
}

#[test]
fn toroidal_mesh_receives_finite_closed_load_ledger() {
    let source = source();
    let mesh = config()
        .try_build_mesh(&PropellerDesign::toroidal_serrated(source.clone()))
        .expect("toroidal mesh should build");
    let result =
        project_bem_loads_to_panel_mesh(&mesh, &bem_result(), projection_options(source.n_blades))
            .expect("toroidal load projection should pass");

    assert!(!mesh.panels.is_empty());
    assert!(result.closure_passed);
    assert!(
        result
            .panels
            .iter()
            .all(|panel| panel.force_n.iter().all(|value| value.is_finite()))
    );
    assert!(result.is_finite());
}

#[test]
fn blade_count_changes_per_blade_distribution_but_not_total_ledger() {
    let source = source();
    let mesh = config()
        .try_build_mesh(&PropellerDesign::conventional(source.clone()))
        .expect("conventional mesh should build");
    let bem = bem_result();
    let one_blade = project_bem_loads_to_panel_mesh(&mesh, &bem, projection_options(1))
        .expect("one-blade projection should pass");
    let two_blade = project_bem_loads_to_panel_mesh(&mesh, &bem, projection_options(2))
        .expect("two-blade projection should pass");

    assert!((one_blade.total_thrust_n - two_blade.total_thrust_n).abs() < 1.0e-10);
    assert!((one_blade.total_torque_nm - two_blade.total_torque_nm).abs() < 1.0e-10);
    assert!(
        (one_blade.panels[0].axial_force_n - 2.0 * two_blade.panels[0].axial_force_n).abs()
            < 1.0e-10
    );
}

#[test]
fn unordered_bem_strips_fail_closed() {
    let source = source();
    let mesh = config()
        .try_build_mesh(&PropellerDesign::conventional(source.clone()))
        .expect("conventional mesh should build");
    let mut unordered_bem = bem_result();
    unordered_bem.loads.strips.swap(0, 1);
    let result =
        project_bem_loads_to_panel_mesh(&mesh, &unordered_bem, projection_options(source.n_blades));
    assert!(matches!(
        result,
        Err(PanelLoadProjectionError::InvalidBem {
            field: "loads.strips.radius_m",
            ..
        })
    ));
}

#[test]
fn invalid_options_and_unconverged_bem_fail_closed() {
    let source = source();
    let mesh = config()
        .try_build_mesh(&PropellerDesign::conventional(source.clone()))
        .expect("conventional mesh should build");
    let invalid_axis = project_bem_loads_to_panel_mesh(
        &mesh,
        &bem_result(),
        PanelLoadProjectionOptions {
            axis: [0.0, 0.0, 0.0],
            ..projection_options(source.n_blades)
        },
    );
    assert!(matches!(
        invalid_axis,
        Err(PanelLoadProjectionError::InvalidOptions { field: "axis", .. })
    ));

    let mut unconverged = bem_result();
    unconverged.converged = false;
    let invalid_bem =
        project_bem_loads_to_panel_mesh(&mesh, &unconverged, projection_options(source.n_blades));
    assert!(matches!(
        invalid_bem,
        Err(PanelLoadProjectionError::InvalidBem {
            field: "converged",
            ..
        })
    ));
}
