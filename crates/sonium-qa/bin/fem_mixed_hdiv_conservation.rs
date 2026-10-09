//! Release-backed mixed RT0/P1 divergence-conservation reference.

use sonium_fem::assembly::assemble_mixed_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{HdivDivergenceForm, MassForm};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, HdivSpace};
use sonium_qa::QaResult;
use std::time::Instant;

const CASE_ID: &str = "fem.mixed-hdiv-conservation.v1";

fn conservation_error(refinement: usize) -> Result<f64, String> {
    let mesh = unit_square_triangles(refinement);
    let hdiv = HdivSpace::new();
    let scalar = H1Space::new(PolynomialDegree::P1);
    let mixed = assemble_mixed_checked(&MassForm, &HdivDivergenceForm, &hdiv, &scalar, &mesh, 4)
        .map_err(|error| format!("mixed space validation failed: {error}"))?;

    let mut expected = vec![0.0; hdiv.num_global_dofs(&mesh)];
    for element_idx in 0..mesh.num_elements() {
        let quadrature = QuadratureRule::new(mesh.elements[element_idx].element_type, 4);
        let local_to_global = hdiv.local_to_global(&mesh, element_idx);
        for point in quadrature.iter() {
            let tabulation =
                hdiv.tabulate(&mesh, element_idx, point.xi(), point.eta(), point.zeta());
            for (local, &global) in local_to_global.iter().enumerate() {
                expected[global] -= point.weight * tabulation.det_j * tabulation.divergence(local);
            }
        }
    }

    let mut actual = vec![0.0; expected.len()];
    for ((&row, &column), &value) in mixed
        .block_01
        .rows
        .iter()
        .zip(&mixed.block_01.cols)
        .zip(&mixed.block_01.values)
    {
        if column >= scalar.num_global_dofs(&mesh) {
            return Err(format!("coupling column {column} exceeds scalar DOFs"));
        }
        actual[row] += value;
    }

    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max))
}

fn main() {
    let started = Instant::now();
    let result = (|| -> Result<QaResult, String> {
        let errors = [1usize, 2, 4]
            .into_iter()
            .map(|refinement| conservation_error(refinement).map(|error| (refinement, error)))
            .collect::<Result<Vec<_>, _>>()?;
        if errors
            .iter()
            .any(|(_, error)| !error.is_finite() || *error > 2.0e-12)
        {
            return Err(format!(
                "mixed conservation errors exceeded tolerance: {errors:?}"
            ));
        }
        let mut result = QaResult::passed(CASE_ID)
            .reference("independently integrated RT0 divergence and P1 partition of unity")
            .reference_checksum("analytic/deterministic-fixture:fem-mixed-hdiv-conservation-v1")
            .input_checksum("analytic/deterministic-fixture:fem-mixed-hdiv-conservation-v1")
            .tolerance("maximum conservation error <= 2e-12 at refinements 1, 2, 4")
            .elapsed_ms(started.elapsed().as_millis() as u64);
        for (refinement, error) in errors {
            result = result.metric(format!("refinement_{refinement}_max_error"), error);
        }
        Ok(result)
    })();

    match result {
        Ok(result) => {
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("FEM MIXED H(div) CONSERVATION FAILED");
            std::process::exit(1);
        }
    }
}
