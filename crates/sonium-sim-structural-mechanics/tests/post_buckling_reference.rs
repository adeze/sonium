use sonium_sim_structural_mechanics::{
    POST_BUCKLING_SCHEMA_VERSION, PostBucklingBranch, PostBucklingModelAssumption,
    PostBucklingStudy, run_post_buckling,
};

fn assert_close(left: f64, right: f64) {
    let tolerance = 1.0e-12 * left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right}"
    );
}

#[test]
fn scalar_post_buckling_crosses_limit_point_and_snap_back() {
    let result = run_post_buckling(&PostBucklingStudy::introductory())
        .expect("bounded post-buckling reference should converge");

    assert_eq!(result.schema_version, POST_BUCKLING_SCHEMA_VERSION);
    assert_eq!(
        result.model_assumption,
        PostBucklingModelAssumption::ScalarSofteningArch
    );
    assert!(result.turning_point_step > 0);
    assert!(result.turning_point_step + 1 < result.points.len());
    assert!(result.peak_load_factor > result.terminal_load_factor);
    assert!(result.terminal_displacement_m >= result.study.terminal_displacement_m);
    assert!(
        result.terminal_displacement_m - result.study.terminal_displacement_m
            <= result.study.arc_length_step * 2.0
    );
    assert!(result.max_equilibrium_residual_n <= result.study.residual_tolerance_n);
    assert!(
        result.max_arc_constraint_residual
            <= result.study.residual_tolerance_n / result.study.reference_load_n
    );

    let peak = &result.points[result.turning_point_step];
    assert_eq!(peak.branch, PostBucklingBranch::PrePeak);
    let first_snap_back = &result.points[result.turning_point_step + 1];
    assert_eq!(first_snap_back.branch, PostBucklingBranch::SnapBack);
    let terminal = result.points.last().expect("terminal point");
    assert_eq!(terminal.branch, PostBucklingBranch::SnapBack);
    assert!(terminal.load_factor < peak.load_factor);
    assert!(terminal.displacement_m > peak.displacement_m);
}

#[test]
fn scalar_post_buckling_result_round_trips() {
    let result = run_post_buckling(&PostBucklingStudy::introductory())
        .expect("bounded post-buckling reference should converge");
    let encoded = serde_json::to_string(&result).expect("serialize result");
    let decoded: sonium_sim_structural_mechanics::PostBucklingResult =
        serde_json::from_str(&encoded).expect("deserialize result");

    assert_eq!(decoded.schema_version, result.schema_version);
    assert_eq!(decoded.study.max_steps, result.study.max_steps);
    assert_eq!(decoded.points.len(), result.points.len());
    assert_eq!(decoded.turning_point_step, result.turning_point_step);
    assert_eq!(decoded.model_assumption, result.model_assumption);
    assert_eq!(decoded.execution, result.execution);
    assert_close(decoded.peak_load_factor, result.peak_load_factor);
    assert_close(
        decoded.terminal_displacement_m,
        result.terminal_displacement_m,
    );
    assert_close(decoded.terminal_load_factor, result.terminal_load_factor);
    for (decoded_point, result_point) in decoded.points.iter().zip(&result.points) {
        assert_eq!(decoded_point.step, result_point.step);
        assert_eq!(decoded_point.branch, result_point.branch);
        assert_close(decoded_point.displacement_m, result_point.displacement_m);
        assert_close(decoded_point.load_factor, result_point.load_factor);
        assert_close(
            decoded_point.equilibrium_residual_n,
            result_point.equilibrium_residual_n,
        );
    }
}

#[test]
fn scalar_post_buckling_rejects_non_snap_back_terminal_path() {
    let mut study = PostBucklingStudy::introductory();
    study.terminal_displacement_m = 0.05;
    assert!(run_post_buckling(&study).is_err());
}
