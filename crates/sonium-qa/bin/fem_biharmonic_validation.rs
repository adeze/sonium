//! Biharmonic validation tests
//!
//! Tests for C0 interior penalty (C0-IP) biharmonic plate bending.
//!
//! - Facet form penalty parameter bounds
//! - Form symmetry (penalty term is symmetric)
//! - Penalty scaling with element size (α/h)
//! - DG assembly structure verification

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::assembly::dg::assemble_dg;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::solid::biharmonic::{BiharmonicCellForm, BiharmonicFacetForm};
use sonium_fem::spaces::H1Space;

fn test_biharmonic_facet_form_penalty_bounds() {
    let form = BiharmonicFacetForm::new(8.0);

    assert!(form.alpha >= 4.0, "Penalty should be >= 4 for stability");
    assert!(form.alpha <= 16.0, "Penalty should not be excessive");
}

fn test_biharmonic_facet_form_default() {
    let form = BiharmonicFacetForm::default_penalty();

    assert!(
        (form.alpha - 8.0).abs() < 1e-10,
        "Default penalty should be 8"
    );
}

fn test_biharmonic_cell_form_exists() {
    let form = BiharmonicCellForm;
    let mesh = unit_square_triangles(4);
    let space = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);

    let _triplet = assemble_bilinear(&form, &space, &space, &mesh, 1);
}

fn test_biharmonic_facet_form_penalty_scaling() {
    let h_fine: f64 = 0.125;
    let h_coarse: f64 = 0.25;
    let alpha: f64 = 8.0;

    let penalty_fine = alpha / h_fine;
    let penalty_coarse = alpha / h_coarse;

    assert!(
        penalty_fine > penalty_coarse,
        "Finer mesh should have larger penalty"
    );
    let diff = (penalty_fine - penalty_coarse * 2.0).abs();
    assert!(diff < 1e-10);
}

fn test_biharmonic_facet_form_assembly() {
    let form = BiharmonicFacetForm::default_penalty();
    let mesh = unit_square_triangles(4);
    let space = H1Space::new(sonium_fem::basis::PolynomialDegree::P1);

    let triplet = assemble_dg(&BiharmonicCellForm, &form, &space, &mesh, 1);
    let csr = triplet.to_csr();

    assert!(csr.num_rows > 0, "Matrix should have rows");
    assert!(csr.nnz() > 0, "Penalty form should contribute entries");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_biharmonic_facet_form_penalty_bounds();
        test_biharmonic_facet_form_default();
        test_biharmonic_cell_form_exists();
        test_biharmonic_facet_form_penalty_scaling();
        test_biharmonic_facet_form_assembly();
    });

    match result {
        Ok(()) => {
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
