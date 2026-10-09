//! CPU QA for checked generic assembly with distinct test and trial spaces.

use sonium_fem::assembly::assemble_bilinear_two_spaces_checked;
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::{Mesh, unit_square_triangles};
use sonium_fem::spaces::{BasisTabulation, DGSpace, FunctionSpace, H1Space};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.cross-space-assembly.v1";

#[derive(Debug, Clone, Copy)]
struct CrossMassForm;

impl BilinearForm<f64> for CrossMassForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let factor = weight * test.det_j;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                local_matrix.add(i, j, test.value(i) * trial.value(j) * factor);
            }
        }
    }
}

fn triangle_area(mesh: &Mesh, element: usize) -> f64 {
    let vertices = mesh.elements[element].vertices();
    let a = &mesh.nodes[vertices[0]];
    let b = &mesh.nodes[vertices[1]];
    let c = &mesh.nodes[vertices[2]];
    0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
}

fn run() -> Result<(usize, usize, f64, usize), String> {
    let mesh = unit_square_triangles(4);
    let test_space = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);
    let trial_space = DGSpace::p0();
    let matrix =
        assemble_bilinear_two_spaces_checked(&CrossMassForm, &test_space, &trial_space, &mesh, 2)
            .map_err(|error| format!("cross-space assembly rejected valid spaces: {error:?}"))?;

    let mut max_error: f64 = 0.0;
    let mut rejection_checks = 0;
    for element in 0..mesh.num_elements() {
        let area = triangle_area(&mesh, element);
        let mut sum = 0.0;
        for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
            if column == element {
                max_error = max_error.max((value - area / 3.0).abs());
                if row >= matrix.num_rows || value <= 0.0 {
                    return Err("cross-space local block has invalid entries".into());
                }
                sum += value;
            }
        }
        max_error = max_error.max((sum - area).abs());
    }

    let mut malformed = unit_square_triangles(1);
    malformed.elements[0].nodes.pop();
    if assemble_bilinear_two_spaces_checked(
        &CrossMassForm,
        &test_space,
        &trial_space,
        &malformed,
        2,
    )
    .is_err()
    {
        rejection_checks += 1;
    }

    Ok((
        matrix.num_rows,
        matrix.num_cols,
        max_error,
        rejection_checks,
    ))
}

fn main() {
    let result = match run() {
        Ok((rows, columns, max_error, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("rows", rows as f64)
            .minimum("rows", 1.0)
            .metric("columns", columns as f64)
            .minimum("columns", 1.0)
            .metric("max_local_mass_error", max_error)
            .maximum("max_local_mass_error", 2.0e-12)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 1.0)
            .dimension("2-D affine triangular H1 test / DG P0 trial block")
            .formulation("checked rectangular cross-space mass assembly")
            .reference("analytic P1 basis integral against element-local constant trial state")
            .reference_checksum("analytic/triangle-p1-dg-p0-cross-mass:fem-cross-space-v1")
            .input_checksum("unit-square-triangles-4:h1-p1-dg-p0:v1"),
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
