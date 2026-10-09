use sonium_sim_structural_mechanics::{BucklingEndCondition, BucklingStudy, run_linear_buckling};

fn independent_euler_load(study: &BucklingStudy) -> f64 {
    let second_moment_m4 = study.geometry.width_m * study.geometry.height_m.powi(3) / 12.0;
    let effective_length_factor = match study.end_condition {
        BucklingEndCondition::PinnedPinned => 1.0,
        BucklingEndCondition::Cantilever => 2.0,
    };
    std::f64::consts::PI.powi(2) * study.youngs_modulus_pa * second_moment_m4
        / (effective_length_factor * study.geometry.length_m).powi(2)
}

#[test]
fn pinned_and_cantilever_buckling_match_independent_euler_reference() {
    for end_condition in [
        BucklingEndCondition::PinnedPinned,
        BucklingEndCondition::Cantilever,
    ] {
        let mut study = BucklingStudy::introductory();
        study.geometry.elements = 8;
        study.end_condition = end_condition;

        let independent_reference = independent_euler_load(&study);
        let result = run_linear_buckling(&study).expect("linear buckling solve");

        assert!(result.converged);
        assert!(result.eigenvalue_n.is_finite());
        assert!(result.residual_norm.is_finite());
        assert!(result.residual_norm < 1.0e-8);
        assert!(
            (result.analytical_euler_load_n - independent_reference).abs() / independent_reference
                < 1.0e-12
        );
        assert!(
            (result.eigenvalue_n - independent_reference).abs() / independent_reference < 5.0e-2,
            "{end_condition:?}: eigenvalue={} reference={independent_reference}",
            result.eigenvalue_n
        );
        assert!(
            result
                .mode_shape
                .iter()
                .all(|mode| mode.iter().all(|value| value.is_finite()))
        );
        assert!(result.mode_shape.first().expect("first mode node")[0].abs() < 1.0e-10);
        if end_condition == BucklingEndCondition::PinnedPinned {
            assert!(result.mode_shape.last().expect("last mode node")[0].abs() < 1.0e-10);
        }
    }
}

#[test]
fn analytical_euler_reference_validates_input_before_evaluation() {
    let mut study = BucklingStudy::introductory();
    study.geometry.elements = 1;

    assert!(study.analytical_euler_load_n().is_err());
}
