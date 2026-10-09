//! Release-facing checked P1 scalar Neumann/Robin boundary contract.

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::boundary::generic::{
    BoundaryConditionError, FemSystem, try_apply_neumann_real, try_apply_robin_complex,
};
use sonium_fem::mesh::{unit_cube_tetrahedra, unit_square_triangles};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.natural-boundary-contract.v1";

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("edge_resultant_checks", metrics.edge_resultant_checks as f64)
            .minimum("edge_resultant_checks", 1.0)
            .metric("face_resultant_checks", metrics.face_resultant_checks as f64)
            .minimum("face_resultant_checks", 1.0)
            .metric("robin_checks", metrics.robin_checks as f64)
            .minimum("robin_checks", 1.0)
            .metric("rollback_checks", metrics.rollback_checks as f64)
            .minimum("rollback_checks", 2.0)
            .reference(
                "checked P1 scalar natural-boundary integration with typed rejection and transactional rollback",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-natural-boundary-contract-v1")
            .dimension("2-D marked edges and 3-D triangular faces")
            .formulation("Neumann resultant and P1 Robin boundary mass/RHS contract")
            .tolerance("finite callbacks, positive boundary measure, and no mutation on rejection")
            .input_checksum("mesh:unit-square-triangles-and-unit-cube-tetrahedra:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FEM natural-boundary QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, sonium_qa::QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    edge_resultant_checks: usize,
    face_resultant_checks: usize,
    robin_checks: usize,
    rollback_checks: usize,
}

fn run() -> Result<Metrics, String> {
    let mut square = unit_square_triangles(1);
    square.detect_boundaries();
    let n = square.num_nodes();
    let mut neumann = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);
    try_apply_neumann_real(&mut neumann, &square, 0, &|_, _, _| 1.0)
        .map_err(|error| format!("valid edge Neumann application failed: {error}"))?;
    if (neumann.rhs.iter().sum::<f64>() - 4.0).abs() > 1.0e-12 {
        return Err(format!(
            "unit-square edge resultant is {}, expected 4",
            neumann.rhs.iter().sum::<f64>()
        ));
    }

    let mut cube = unit_cube_tetrahedra(1);
    cube.detect_boundaries();
    let n = cube.num_nodes();
    let mut face_neumann = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);
    try_apply_neumann_real(&mut face_neumann, &cube, 0, &|_, _, _| 1.0)
        .map_err(|error| format!("valid face Neumann application failed: {error}"))?;
    if (face_neumann.rhs.iter().sum::<f64>() - 6.0).abs() > 1.0e-12 {
        return Err(format!(
            "unit-cube face resultant is {}, expected 6",
            face_neumann.rhs.iter().sum::<f64>()
        ));
    }

    let mut robin = FemSystem::new(
        TripletMatrix::new(square.num_nodes(), square.num_nodes()),
        vec![Complex64::new(0.0, 0.0); square.num_nodes()],
    );
    try_apply_robin_complex(
        &mut robin,
        &square,
        0,
        &|_, _, _| Complex64::new(1.0, 0.0),
        &|_, _, _| Complex64::new(2.0, 0.0),
    )
    .map_err(|error| format!("valid Robin application failed: {error}"))?;
    let matrix_integral: f64 = robin.matrix.values.iter().map(|value| value.re).sum();
    let rhs_integral: f64 = robin.rhs.iter().map(|value| value.re).sum();
    if (matrix_integral - 4.0).abs() > 1.0e-12 || (rhs_integral - 8.0).abs() > 1.0e-12 {
        return Err(format!(
            "Robin integrals are matrix={matrix_integral}, rhs={rhs_integral}; expected 4, 8"
        ));
    }

    let original_rhs = robin.rhs.clone();
    let original_nnz = robin.matrix.nnz();
    let error = try_apply_robin_complex(
        &mut robin,
        &square,
        0,
        &|_, _, _| Complex64::new(f64::NAN, 0.0),
        &|_, _, _| Complex64::new(0.0, 0.0),
    )
    .expect_err("non-finite Robin callback must be rejected");
    if !matches!(error, BoundaryConditionError::NonFiniteBoundaryValue { .. }) {
        return Err(format!("unexpected Robin rejection: {error}"));
    }
    if robin.rhs != original_rhs || robin.matrix.nnz() != original_nnz {
        return Err("Robin rejection mutated the system".into());
    }

    let mut bad_neumann = FemSystem::new(TripletMatrix::new(1, 1), vec![0.0]);
    let error = try_apply_neumann_real(&mut bad_neumann, &square, 0, &|_, _, _| 1.0)
        .expect_err("out-of-range boundary node must be rejected");
    if !matches!(
        error,
        BoundaryConditionError::BoundaryNodeOutOfBounds { .. }
    ) {
        return Err(format!("unexpected node rejection: {error}"));
    }
    if bad_neumann.rhs != vec![0.0] || bad_neumann.matrix.nnz() != 0 {
        return Err("node rejection mutated the system".into());
    }

    Ok(Metrics {
        edge_resultant_checks: 1,
        face_resultant_checks: 1,
        robin_checks: 1,
        rollback_checks: 2,
    })
}
