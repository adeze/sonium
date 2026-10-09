//! Release-facing discontinuous-linear BEM execution and topology contract.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::{BemProblem, BemSolver};
use sonium_bem::core::types::BasisOrder;
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "bem.linear-basis-execution.v1";

fn finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let expected_dofs: usize = problem
            .mesh
            .elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .map(|element| element.connectivity.len())
            .sum();
        let topology_ok = problem
            .mesh
            .elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .all(|element| element.connectivity.len() == 3);

        let solution = BemSolver::new()
            .with_basis_order(BasisOrder::Linear)
            .solve(&problem)
            .expect("discontinuous-linear TBEM solve");
        let coefficient_finite = solution
            .surface_pressure
            .iter()
            .copied()
            .all(finite_complex);
        let evaluation = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);

        assert!(topology_ok, "fixture must contain only Tri3 surface panels");
        assert_eq!(solution.num_dofs(), expected_dofs);
        assert!(
            coefficient_finite,
            "linear solution has non-finite coefficients"
        );
        assert!(
            finite_complex(evaluation),
            "linear field evaluation is non-finite"
        );
        assert!(evaluation.norm() > 0.0, "linear field evaluation is empty");

        (expected_dofs, coefficient_finite, evaluation)
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok((expected_dofs, coefficient_finite, evaluation)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent surface-panel topology and finite-field execution contract")
            .reference_checksum("analytic/deterministic-fixture:bem-linear-basis-execution-v1")
            .input_checksum("bem:rigid-sphere:linear-tbem:ka-0.183")
            .dimension("3-D rigid sphere; Tri3 discontinuous linear TBEM; 4x8 angular mesh")
            .formulation("production TBEM linear basis with Burton-Miller rigid boundary")
            .tolerance("exact linear DOF count; all coefficients and point evaluation finite")
            .metric("linear_dofs", expected_dofs as f64)
            .minimum("linear_dofs", 1.0)
            .metric("topology_and_dof_contract", 1.0)
            .minimum("topology_and_dof_contract", 1.0)
            .metric(
                "finite_solution_coefficients",
                if coefficient_finite { 1.0 } else { 0.0 },
            )
            .minimum("finite_solution_coefficients", 1.0)
            .metric(
                "finite_nonzero_field_evaluation",
                if finite_complex(evaluation) && evaluation.norm() > 0.0 {
                    1.0
                } else {
                    0.0
                },
            )
            .minimum("finite_nonzero_field_evaluation", 1.0)
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(
            CASE_ID,
            format!("linear-basis execution contract panicked: {payload:?}"),
        )
        .elapsed_ms(elapsed_ms),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
