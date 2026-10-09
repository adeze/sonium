//! Independent CPU evidence for MaxwellForm's explicit coefficient constructor.
//!
//! The reference integrates the public H(curl) basis values directly with the
//! public quadrature rule. It does not call MaxwellForm or reuse its assembly
//! kernel, so a dropped explicit mass coefficient is observable.

use std::collections::{BTreeMap, BTreeSet};

use num_complex::Complex64;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear_checked};
use sonium_fem::mesh::unit_square_triangles;
use sonium_fem::physics::electromagnetics::maxwell::MaxwellForm;
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};

fn aggregate(matrix: &TripletMatrix<Complex64>) -> BTreeMap<(usize, usize), Complex64> {
    let mut aggregate = BTreeMap::new();
    for ((&row, &column), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
        *aggregate
            .entry((row, column))
            .or_insert(Complex64::new(0.0, 0.0)) += value;
    }
    aggregate
}

fn independent_edge_mass(mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<(usize, usize), f64> {
    let space = HcurlSpace::new();
    let mut reference = BTreeMap::new();

    for element in 0..mesh.num_elements() {
        let element_type = mesh.elements[element].element_type;
        let quadrature = QuadratureRule::new(element_type, 7);
        let dofs = space.local_to_global(mesh, element);

        for point in quadrature.iter() {
            let tabulation = space.tabulate(mesh, element, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * tabulation.det_j;
            for (local_row, &row) in dofs.iter().enumerate() {
                for (local_column, &column) in dofs.iter().enumerate() {
                    let dot = (0..tabulation.spatial_dim)
                        .map(|component| {
                            tabulation.vector_value(local_row, component)
                                * tabulation.vector_value(local_column, component)
                        })
                        .sum::<f64>();
                    *reference.entry((row, column)).or_insert(0.0) += weight * dot;
                }
            }
        }
    }

    reference
}

#[test]
fn from_coefficients_preserves_explicit_complex_mass_term_on_checked_hcurl_assembly() {
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let mass_coeff = Complex64::new(1.75, -0.5);
    let form = MaxwellForm::from_coefficients(0.0, mass_coeff);

    let assembled = assemble_bilinear_checked(&form, &space, &space, &mesh, 7)
        .expect("checked Maxwell assembly must admit the H(curl) space");
    let actual = aggregate(&assembled);
    let reference = independent_edge_mass(&mesh);
    let keys: BTreeSet<_> = actual.keys().chain(reference.keys()).copied().collect();

    let max_error = keys
        .iter()
        .map(|key| {
            let actual = actual.get(key).copied().unwrap_or_default();
            let expected = mass_coeff * reference.get(key).copied().unwrap_or_default();
            (actual - expected).norm()
        })
        .fold(0.0, f64::max);

    assert!(
        reference.values().map(|value| value.abs()).sum::<f64>() > 1.0e-12,
        "independent edge-mass reference unexpectedly vanished"
    );
    assert!(
        max_error < 5.0e-12,
        "Maxwell explicit mass coefficient residual was {max_error:.3e}"
    );
}
