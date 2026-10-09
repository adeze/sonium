use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use sonium_sim_propellers::{
    Blade3dConfig, PanelStateTransferError, PanelStateTransferOptions, PropellerDesign,
    transfer_panel_scalar_state,
};

fn mesh(radial_segments: usize, chord_segments: usize) -> sonium_sim_propellers::BladeMesh {
    let design = PropellerDesign::conventional(PropellerSource::drone_5inch(18_000.0));
    Blade3dConfig {
        radial_segments,
        chord_segments,
        ..Blade3dConfig::default()
    }
    .try_build_mesh(&design)
    .expect("valid conventional mesh")
}

fn options() -> PanelStateTransferOptions {
    PanelStateTransferOptions {
        maximum_source_distance_m: 0.05,
        ..PanelStateTransferOptions::default()
    }
}

#[test]
fn constant_state_is_preserved_across_resolution_change() {
    let source_mesh = mesh(8, 3);
    let target_mesh = mesh(17, 5);
    let source_values = vec![2.75; source_mesh.panels.len()];

    let result = transfer_panel_scalar_state(&source_mesh, &source_values, &target_mesh, options())
        .expect("constant state transfer");

    assert!(result.is_finite());
    assert_eq!(result.values.len(), target_mesh.panels.len());
    assert!(
        result
            .values
            .iter()
            .all(|value| (*value - 2.75).abs() < 1.0e-12)
    );
    assert!(result.area_weighted_mean_delta.abs() < 1.0e-12);
}

#[test]
fn exact_mesh_transfer_reproduces_a_geometry_field() {
    let source_mesh = mesh(9, 4);
    let source_values = source_mesh
        .panels
        .iter()
        .map(|panel| {
            let center = panel.vertices.iter().fold([0.0; 3], |mut sum, point| {
                sum[0] += point.x * 0.25;
                sum[1] += point.y * 0.25;
                sum[2] += point.z * 0.25;
                sum
            });
            center[0] + 2.0 * center[1] - center[2]
        })
        .collect::<Vec<_>>();

    let result = transfer_panel_scalar_state(&source_mesh, &source_values, &source_mesh, options())
        .expect("identity transfer");

    assert_eq!(result.values.len(), source_values.len());
    for (actual, expected) in result.values.iter().zip(source_values) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    assert_eq!(result.maximum_source_distance_m, 0.0);
}

#[test]
fn transfer_keeps_replicated_blades_isolated() {
    let source_mesh = mesh(6, 3);
    let target_mesh = mesh(13, 4);
    let source_values = source_mesh
        .panels
        .iter()
        .map(|panel| 10.0 + panel.blade_index as f64)
        .collect::<Vec<_>>();

    let result = transfer_panel_scalar_state(&source_mesh, &source_values, &target_mesh, options())
        .expect("per-blade state transfer");

    for (panel, value) in target_mesh.panels.iter().zip(result.values) {
        assert!((value - (10.0 + panel.blade_index as f64)).abs() < 1.0e-12);
    }
}

#[test]
fn transfer_is_deterministic_and_reports_distance_metrics() {
    let source_mesh = mesh(7, 3);
    let target_mesh = mesh(12, 4);
    let source_values = source_mesh
        .panels
        .iter()
        .enumerate()
        .map(|(index, panel)| index as f64 * 0.01 + panel.radius_fraction)
        .collect::<Vec<_>>();

    let first = transfer_panel_scalar_state(&source_mesh, &source_values, &target_mesh, options())
        .expect("first transfer");
    let second = transfer_panel_scalar_state(&source_mesh, &source_values, &target_mesh, options())
        .expect("replayed transfer");

    assert_eq!(first, second);
    assert!(first.maximum_source_distance_m >= first.mean_source_distance_m);
    assert!(first.maximum_source_distance_m <= options().maximum_source_distance_m);
}

#[test]
fn invalid_lengths_values_topology_and_budgets_fail_closed() {
    let source_mesh = mesh(5, 2);
    let target_mesh = mesh(8, 3);
    let values = vec![1.0; source_mesh.panels.len()];

    assert!(matches!(
        transfer_panel_scalar_state(
            &source_mesh,
            &values[..values.len() - 1],
            &target_mesh,
            options()
        ),
        Err(PanelStateTransferError::StateLengthMismatch { mesh: "source", .. })
    ));

    let mut non_finite = values.clone();
    non_finite[0] = f64::NAN;
    assert!(matches!(
        transfer_panel_scalar_state(&source_mesh, &non_finite, &target_mesh, options()),
        Err(PanelStateTransferError::InvalidPanel { mesh: "source", .. })
    ));

    let toroidal_design =
        PropellerDesign::toroidal_serrated(PropellerSource::drone_5inch(18_000.0));
    let toroidal_mesh = Blade3dConfig::default()
        .try_build_mesh(&toroidal_design)
        .expect("valid toroidal mesh");
    assert!(matches!(
        transfer_panel_scalar_state(&source_mesh, &values, &toroidal_mesh, options()),
        Err(PanelStateTransferError::TopologyMismatch { field: "planform" })
    ));

    let budget = PanelStateTransferOptions {
        maximum_distance_evaluations: 1,
        ..options()
    };
    assert!(matches!(
        transfer_panel_scalar_state(&source_mesh, &values, &target_mesh, budget),
        Err(PanelStateTransferError::DistanceBudgetExceeded { .. })
    ));
}

#[test]
fn invalid_options_and_distance_gate_fail_closed() {
    let source_mesh = mesh(4, 2);
    let target_mesh = mesh(11, 3);
    let values = vec![0.0; source_mesh.panels.len()];

    let invalid_options = PanelStateTransferOptions {
        maximum_neighbors: 0,
        ..options()
    };
    assert!(matches!(
        transfer_panel_scalar_state(&source_mesh, &values, &target_mesh, invalid_options),
        Err(PanelStateTransferError::InvalidOptions {
            field: "maximum_neighbors",
            ..
        })
    ));

    let strict_distance = PanelStateTransferOptions {
        maximum_source_distance_m: 1.0e-12,
        ..options()
    };
    assert!(matches!(
        transfer_panel_scalar_state(&source_mesh, &values, &target_mesh, strict_distance),
        Err(PanelStateTransferError::SourceDistanceExceeded { .. })
    ));
}
