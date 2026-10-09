use num_complex::Complex64;
use sonium_bem::core::{BemError, BemProblem, BemSolver, IncidentField};

#[test]
fn malformed_incident_fields_fail_before_solver_assembly() {
    let cases = [
        IncidentField::PlaneWave {
            direction: [f64::NAN, 0.0, 1.0],
            amplitude: Complex64::new(1.0, 0.0),
        },
        IncidentField::PointSource {
            position: [0.0, f64::INFINITY, 0.0],
            strength: Complex64::new(1.0, 0.0),
        },
        IncidentField::MultiplePlaneWaves(vec![([0.0, 0.0, 1.0], Complex64::new(f64::NAN, 0.0))]),
        IncidentField::MultiplePointSources(vec![(
            [0.0, 0.0, 1.0],
            Complex64::new(0.0, f64::INFINITY),
        )]),
    ];

    for field in cases {
        let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6);
        problem.incident_field = field;
        let error = BemSolver::new()
            .solve(&problem)
            .expect_err("malformed incident field reached assembly");
        assert!(
            matches!(error, BemError::InvalidParameters(message) if message.contains("incident field"))
        );
    }
}
