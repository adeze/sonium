//! Independent CPU reference for Maxwell's explicit H(curl) mass coefficient.

use num_complex::Complex64;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::physics::electromagnetics::maxwell::MaxwellForm;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

const CASE_ID: &str = "fem.maxwell-coefficient-reference.v1";
const QUADRATURE_ORDER: usize = 7;
const MASS_COEFFICIENT: Complex64 = Complex64::new(1.75, -0.5);
const MAX_RELATIVE_ERROR: f64 = 5.0e-12;

fn aggregate(matrix: &TripletMatrix<Complex64>) -> BTreeMap<(usize, usize), Complex64> {
    let mut aggregate = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *aggregate
            .entry((row, column))
            .or_insert(Complex64::new(0.0, 0.0)) += value;
    }
    aggregate
}

/// Integrate the public H(curl) basis directly, independently of MaxwellForm.
fn independent_edge_mass(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HcurlSpace::new();
    let mut reference = BTreeMap::new();

    for element in 0..mesh.num_elements() {
        let element_type = mesh.elements[element].element_type;
        let quadrature = QuadratureRule::new(element_type, QUADRATURE_ORDER);
        let dofs = space.local_to_global(mesh, element);

        for point in quadrature.iter() {
            let tabulation = space.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * tabulation.det_j;
            for (local_row, &row) in dofs.iter().enumerate() {
                for (local_column, &column) in dofs.iter().enumerate() {
                    let value = (0..tabulation.spatial_dim)
                        .map(|component| {
                            tabulation.vector_value(local_row, component)
                                * tabulation.vector_value(local_column, component)
                        })
                        .sum::<f64>()
                        * weight;
                    *reference.entry((row, column)).or_insert(0.0) += value;
                }
            }
        }
    }
    reference
}

fn run() -> Result<(f64, usize, usize), String> {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let form = MaxwellForm::from_coefficients(0.0, MASS_COEFFICIENT);
    let assembled = assemble_bilinear_checked(&form, &space, &space, &mesh, QUADRATURE_ORDER)
        .map_err(|error| format!("checked Maxwell assembly failed: {error:?}"))?;
    let actual = aggregate(&assembled);
    let reference = independent_edge_mass(&mesh);
    let keys: BTreeSet<_> = actual.keys().chain(reference.keys()).copied().collect();
    let max_error = keys
        .iter()
        .map(|key| {
            let actual_value = actual.get(key).copied().unwrap_or_default();
            let expected = reference.get(key).copied().unwrap_or_default();
            (actual_value - MASS_COEFFICIENT * expected).norm()
        })
        .fold(0.0, f64::max);
    let reference_scale = reference
        .values()
        .map(|value| value.abs())
        .sum::<f64>()
        .max(1.0e-12);
    let relative_error = max_error / (MASS_COEFFICIENT.norm() * reference_scale);
    if !relative_error.is_finite() || relative_error > MAX_RELATIVE_ERROR {
        return Err(format!(
            "explicit Maxwell mass coefficient mismatch: relative error={relative_error:e}"
        ));
    }
    Ok((relative_error, actual.len(), reference.len()))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((relative_error, actual_entries, reference_entries)) => {
            let result = QaResult::passed(CASE_ID)
                .reference("independent H(curl) edge-basis quadrature")
                .reference_checksum("analytic/deterministic-fixture:fem-maxwell-coefficient-v1")
                .dimension("2D unit-square P1 triangles; H(curl) edge space")
                .formulation("MaxwellForm explicit complex mass coefficient through checked H(curl) assembly")
                .tolerance("relative complex mass-matrix error <= 5e-12")
                .metric("relative_complex_mass_error", relative_error)
                .maximum("relative_complex_mass_error", MAX_RELATIVE_ERROR)
                .metric("assembled_entries", actual_entries as f64)
                .metric("reference_entries", reference_entries as f64)
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            println!("SUCCESS");
        }
        Err(error) => {
            let result = QaResult::failed(CASE_ID, error);
            println!("QA_RESULT: {}", result.to_json_line().unwrap());
            eprintln!("MAXWELL COEFFICIENT REFERENCE FAILED");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
