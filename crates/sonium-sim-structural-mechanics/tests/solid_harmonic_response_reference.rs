use sonium_sim_structural_mechanics::{
    SOLID3D_HARMONIC_RESPONSE_SCHEMA_VERSION, Solid3DHarmonicResponseResult,
    Solid3DHarmonicResponseStudy, StructuralModelAssumption, run_solid3d_harmonic_response,
};

fn study() -> Solid3DHarmonicResponseStudy {
    let mut study = Solid3DHarmonicResponseStudy::introductory();
    study.base.geometry.nx = 1;
    study.base.geometry.ny = 1;
    study.base.geometry.nz = 1;
    study.frequencies_hz = vec![1.0, 10.0, 100.0];
    study.observation_node = 5;
    study.observation_component = 0;
    study.residual_tolerance = 1.0e-8;
    study.allowable_displacement_m = 1.0;
    study
}

#[test]
fn direct_solid3d_harmonic_response_reports_finite_complex_fem_samples() {
    let result = run_solid3d_harmonic_response(&study()).expect("direct harmonic response");

    assert_eq!(
        result.schema_version,
        SOLID3D_HARMONIC_RESPONSE_SCHEMA_VERSION
    );
    assert_eq!(
        result.model_assumption,
        StructuralModelAssumption::Solid3DHarmonicResponse
    );
    assert_eq!(
        result.polynomial_order,
        sonium_sim_structural_mechanics::StructuralPolynomialOrder::Linear
    );
    assert_eq!(result.nodes, 8);
    assert_eq!(result.elements, 6);
    assert_eq!(result.dofs, 24);
    assert_eq!(result.points.len(), 3);
    assert_eq!(result.displacement_field_phasors_m.len(), 3);
    assert!(result.displacement_field_phasors_m.iter().all(|field| {
        field.len() == result.nodes
            && field
                .iter()
                .flatten()
                .flatten()
                .all(|value| value.is_finite())
    }));
    for (point, field) in result
        .points
        .iter()
        .zip(&result.displacement_field_phasors_m)
    {
        let observed = field[study().observation_node][study().observation_component];
        assert!((observed[0] - point.displacement_phasor_m[0]).abs() < 1.0e-20);
        assert!((observed[1] - point.displacement_phasor_m[1]).abs() < 1.0e-20);
    }
    assert_eq!(result.factorizations, 3);
    assert!(result.max_factorization_memory_bytes > 0);
    assert!(result.converged);
    assert!(result.feasible);
    assert!(result.peak_displacement_abs_m.is_finite());
    assert!(result.points.iter().all(|point| {
        point.displacement_abs_m.is_finite()
            && point.phase_lag_rad.is_finite()
            && point.relative_solver_residual <= study().residual_tolerance
    }));
    assert!(
        result
            .points
            .iter()
            .any(|point| point.displacement_abs_m > 0.0)
    );
    assert!(
        result
            .points
            .iter()
            .any(|point| point.frequency_hz == result.peak_frequency_hz)
    );
}

#[test]
fn direct_solid3d_harmonic_response_scales_with_force_phasor() {
    let baseline = run_solid3d_harmonic_response(&study()).unwrap();
    let mut doubled = study();
    for force in &mut doubled.force_phasors_n {
        force[0] *= 2.0;
        force[1] *= 2.0;
    }
    let doubled = run_solid3d_harmonic_response(&doubled).unwrap();

    assert!(
        (doubled.peak_displacement_abs_m - 2.0 * baseline.peak_displacement_abs_m).abs() < 1.0e-14
    );
    for (base, scaled) in baseline.points.iter().zip(&doubled.points) {
        assert!((scaled.displacement_abs_m - 2.0 * base.displacement_abs_m).abs() < 1.0e-14);
        assert!((scaled.phase_lag_rad - base.phase_lag_rad).abs() < 1.0e-12);
    }
    for (base_field, scaled_field) in baseline
        .displacement_field_phasors_m
        .iter()
        .zip(&doubled.displacement_field_phasors_m)
    {
        for (base_node, scaled_node) in base_field.iter().zip(scaled_field) {
            for (base_component, scaled_component) in base_node.iter().zip(scaled_node) {
                assert!((scaled_component[0] - 2.0 * base_component[0]).abs() < 1.0e-20);
                assert!((scaled_component[1] - 2.0 * base_component[1]).abs() < 1.0e-20);
            }
        }
    }
}

#[test]
fn direct_solid3d_harmonic_response_preserves_force_phase() {
    let mut quadrature_force = study();
    quadrature_force.force_phasors_n = [[0.0, 100.0], [0.0, 0.0], [0.0, 0.0]];
    let result = run_solid3d_harmonic_response(&quadrature_force).unwrap();

    assert!(result.points.iter().all(|point| {
        point.displacement_phasor_m[1].abs() > 0.0
            && point.displacement_phasor_m[0].is_finite()
            && point.displacement_phasor_m[1].is_finite()
    }));
}

#[test]
fn direct_solid3d_harmonic_response_rejects_invalid_controls_before_or_at_admission() {
    let mut zero_force = study();
    zero_force.force_phasors_n = [[0.0; 2]; 3];
    assert!(run_solid3d_harmonic_response(&zero_force).is_err());

    let mut bad_grid = study();
    bad_grid.frequencies_hz[1] = bad_grid.frequencies_hz[0];
    assert!(run_solid3d_harmonic_response(&bad_grid).is_err());

    let mut bad_component = study();
    bad_component.observation_component = 3;
    assert!(run_solid3d_harmonic_response(&bad_component).is_err());

    let mut bad_node = study();
    bad_node.observation_node = 8;
    assert!(run_solid3d_harmonic_response(&bad_node).is_err());

    let mut bad_damping = study();
    bad_damping.damping_ratio = -0.01;
    assert!(run_solid3d_harmonic_response(&bad_damping).is_err());

    let mut bad_tolerance = study();
    bad_tolerance.residual_tolerance = 0.0;
    assert!(run_solid3d_harmonic_response(&bad_tolerance).is_err());

    let mut bad_mesh = study();
    bad_mesh.base.geometry.nx = 0;
    assert!(run_solid3d_harmonic_response(&bad_mesh).is_err());
}

#[test]
fn direct_solid3d_harmonic_response_rejects_unresolved_undamped_factorization() {
    let mut undamped = study();
    undamped.damping_ratio = 0.0;
    undamped.frequencies_hz = vec![1.0];
    let result = run_solid3d_harmonic_response(&undamped);
    assert!(result.is_err());
}

#[test]
fn direct_solid3d_harmonic_response_round_trips_through_serde() {
    let result = run_solid3d_harmonic_response(&study()).unwrap();
    let encoded = serde_json::to_string(&result).unwrap();
    let decoded: Solid3DHarmonicResponseResult = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study, result.study);
    assert_eq!(decoded.nodes, result.nodes);
    assert_eq!(decoded.elements, result.elements);
    assert_eq!(decoded.dofs, result.dofs);
    assert_eq!(decoded.points.len(), result.points.len());
    for (decoded_field, result_field) in decoded
        .displacement_field_phasors_m
        .iter()
        .zip(&result.displacement_field_phasors_m)
    {
        for (decoded_node, result_node) in decoded_field.iter().zip(result_field) {
            for (decoded_component, result_component) in decoded_node.iter().zip(result_node) {
                assert!((decoded_component[0] - result_component[0]).abs() < 1.0e-20);
                assert!((decoded_component[1] - result_component[1]).abs() < 1.0e-20);
            }
        }
    }
    assert_eq!(decoded.factorizations, result.factorizations);
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.polynomial_order, result.polynomial_order);
    assert_eq!(decoded.execution, result.execution);
    assert!((decoded.peak_displacement_abs_m - result.peak_displacement_abs_m).abs() < 1.0e-14);
    assert_eq!(decoded.peak_frequency_hz, result.peak_frequency_hz);
    assert_eq!(decoded.converged, result.converged);
    assert_eq!(decoded.feasible, result.feasible);
    for (decoded_point, result_point) in decoded.points.iter().zip(&result.points) {
        assert_eq!(decoded_point.frequency_hz, result_point.frequency_hz);
        assert!(
            (decoded_point.displacement_abs_m - result_point.displacement_abs_m).abs() < 1.0e-14
        );
        assert!((decoded_point.phase_lag_rad - result_point.phase_lag_rad).abs() < 1.0e-14);
        assert!(
            (decoded_point.relative_solver_residual - result_point.relative_solver_residual).abs()
                < 1.0e-14
        );
    }
}
