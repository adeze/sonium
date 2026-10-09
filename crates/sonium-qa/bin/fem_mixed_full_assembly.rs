//! CPU QA for full checked mixed-system assembly.

use sonium_fem::assembly::{MixedSystem, assemble_mixed_full_checked};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::spaces::{BasisTabulation, DGSpace, H1Space};
use sonium_qa::QaResult;
use sonium_solvers::LinearOperator;

const CASE_ID: &str = "fem.mixed-full-assembly.v1";

#[derive(Debug, Clone, Copy)]
struct WeightedMass(f64);

impl BilinearForm<f64> for WeightedMass {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let factor = self.0 * weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, factor * test.value(i) * trial.value(j));
            }
        }
    }
}

fn run() -> Result<(usize, usize, usize, f64, usize), String> {
    let mesh = unit_square_triangles(4);
    let space_0 = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);
    let space_1 = DGSpace::p0();
    let system: MixedSystem<f64> = assemble_mixed_full_checked(
        &WeightedMass(1.0),
        &WeightedMass(2.0),
        &WeightedMass(3.0),
        &WeightedMass(4.0),
        &space_0,
        &space_1,
        &mesh,
        2,
    )
    .map_err(|error| format!("full mixed assembly rejected valid spaces: {error:?}"))?;

    let total = system.n_dofs_0 + system.n_dofs_1;
    let x: Vec<f64> = (0..total).map(|index| 0.25 + 0.13 * index as f64).collect();
    let block_result = system.to_block_operator().apply(&x);
    let monolithic_result = system.to_monolithic().to_csr().apply(&x);
    let max_apply_error = block_result
        .iter()
        .zip(&monolithic_result)
        .map(|(block, mono)| (block - mono).abs())
        .fold(0.0, f64::max);

    let mut rejection_checks = 0;
    let mut malformed = unit_square_triangles(1);
    malformed.elements[0].nodes.pop();
    if assemble_mixed_full_checked(
        &WeightedMass(1.0),
        &WeightedMass(2.0),
        &WeightedMass(3.0),
        &WeightedMass(4.0),
        &space_0,
        &space_1,
        &malformed,
        2,
    )
    .is_err()
    {
        rejection_checks += 1;
    }

    Ok((
        system.n_dofs_0,
        system.n_dofs_1,
        system.block_11.nnz(),
        max_apply_error,
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((space_0_dofs, space_1_dofs, lower_right_nnz, max_apply_error, rejection_checks)) => {
            QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("space_0_dofs", space_0_dofs as f64)
                .minimum("space_0_dofs", 1.0)
                .metric("space_1_dofs", space_1_dofs as f64)
                .minimum("space_1_dofs", 1.0)
                .metric("lower_right_nnz", lower_right_nnz as f64)
                .minimum("lower_right_nnz", 1.0)
                .metric("block_monolithic_apply_error", max_apply_error)
                .maximum("block_monolithic_apply_error", 2.0e-12)
                .metric("rejection_checks", rejection_checks as f64)
                .minimum("rejection_checks", 1.0)
                .dimension("2-D affine triangular H1 P1 / DG P0 mixed system")
                .formulation("checked four-block mixed assembly with nonzero lower-right block")
                .reference(
                    "independent weighted element mass forms and monolithic/block apply identity",
                )
                .reference_checksum("analytic/weighted-four-block-mass:fem-mixed-full-v1")
                .input_checksum("unit-square-triangles-4:h1-p1-dg-p0-four-block:v1")
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
