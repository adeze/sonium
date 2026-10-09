//! Independent finite-difference gate for a production nonlinear FEM tangent.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{LocalMatrix, LocalVector, NonlinearForm};
use sonium_fem::mesh::{ElementType, unit_square_triangles};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_fem::validation::check_tangent_consistency;
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.nonlinear-tangent.v1";
const MINIMUM_ORDER: f64 = 1.8;

fn main() {
    match run() {
        Ok((order, first_remainder, last_remainder)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("observed_order", order)
                .metric("first_remainder_norm", first_remainder)
                .metric("last_remainder_norm", last_remainder)
                .minimum("observed_order", MINIMUM_ORDER)
                .seed("unit-square-triangle:p1:neo-hookean:central-taylor");
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("FEM NONLINEAR TANGENT FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(f64, f64, f64), String> {
    let mesh = unit_square_triangles(2);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let form = HyperelasticForm::new(NeoHookean::from_young(1.0, 0.3, 2));
    let quadrature = QuadratureRule::new(ElementType::Triangle, 2);
    let state = vec![0.01, -0.02, 0.005, 0.01, -0.004, 0.003];
    let direction = vec![0.2, -0.1, 0.05, 0.03, -0.07, 0.11];

    let residual = |current: &[f64]| {
        let mut result = LocalVector::zeros(current.len());
        for point in quadrature.iter() {
            let tab = space.tabulate(&mesh, 0, point.xi(), point.eta(), point.zeta());
            form.element_residual(&tab, current, point.weight, &mut result);
        }
        result.values
    };
    let tangent = |current: &[f64], vector: &[f64]| {
        let mut matrix = LocalMatrix::zeros(current.len());
        for point in quadrature.iter() {
            let tab = space.tabulate(&mesh, 0, point.xi(), point.eta(), point.zeta());
            form.element_tangent(&tab, &tab, current, point.weight, &mut matrix);
        }
        (0..current.len())
            .map(|row| {
                (0..current.len())
                    .map(|column| matrix.get(row, column) * vector[column])
                    .sum()
            })
            .collect()
    };

    let report = check_tangent_consistency(
        residual,
        tangent,
        &state,
        &direction,
        &[1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4],
    )
    .map_err(|error| error.to_string())?;
    let first = report
        .remainder_norms
        .first()
        .copied()
        .ok_or_else(|| "tangent check returned no remainder norms".to_string())?;
    let last = report
        .remainder_norms
        .last()
        .copied()
        .ok_or_else(|| "tangent check returned no final remainder norm".to_string())?;
    if !report.observed_order.is_finite() || !first.is_finite() || !last.is_finite() {
        return Err("nonlinear tangent report contained a non-finite metric".into());
    }
    if report.observed_order < MINIMUM_ORDER {
        return Err(format!(
            "nonlinear tangent Taylor remainder order {:.3} is below {:.3}",
            report.observed_order, MINIMUM_ORDER
        ));
    }
    Ok((report.observed_order, first, last))
}
