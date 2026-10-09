//! Electroacoustics validation tests
//!
//! Tests for coupled fluid-structure (acoustic + elastic) systems.
//!
//! - Block system assembly dimensions
//! - Coupling matrix structure verification
//! - Frequency-dependent assembly
//! - Coupled system symmetry (when applicable)

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::acoustics::electroacoustics::{CoupledSystem, assemble_coupling_matrix};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

fn test_coupled_system_block_dimensions() {
    let mesh = unit_square_triangles(4);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let nf = p_space.num_global_dofs(&mesh);
    let ns = u_space.num_global_dofs(&mesh);

    let fluid_stiffness: TripletMatrix<f64> = TripletMatrix::new(nf, nf);
    let fluid_mass = assemble_bilinear(&LaplacianForm, &p_space, &p_space, &mesh, 1);

    let material = IsotropicMaterial::new(1.0, 0.3);
    let elastic_form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let solid_stiffness = assemble_bilinear(&elastic_form, &u_space, &u_space, &mesh, 1);

    let coupling = assemble_coupling_matrix(&mesh, 1, nf, ns, 2);

    let system = CoupledSystem {
        n_fluid: nf,
        n_solid: ns,
        fluid_stiffness,
        fluid_mass,
        fluid_damping: TripletMatrix::new(nf, nf),
        solid_stiffness,
        solid_mass: TripletMatrix::new(ns, ns),
        solid_damping: TripletMatrix::new(ns, ns),
        coupling,
        interface_load: None,
        electrical: None,
    };

    let total = system.total_dofs();
    assert!(total > 0, "Total DOFs should be positive");

    let assembled = system.assemble_block(
        Complex64::new(25.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-25.0, 0.0),
    );

    assert_eq!(assembled.num_rows, total);
    assert_eq!(assembled.num_cols, total);
}

fn test_coupled_system_coupling_block() {
    let mesh = unit_square_triangles(4);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let nf = p_space.num_global_dofs(&mesh);
    let ns = u_space.num_global_dofs(&mesh);

    let fluid_stiffness: TripletMatrix<f64> = TripletMatrix::new(nf, nf);
    let fluid_mass = assemble_bilinear(&LaplacianForm, &p_space, &p_space, &mesh, 1);

    let material = IsotropicMaterial::new(1.0, 0.3);
    let elastic_form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let solid_stiffness = assemble_bilinear(&elastic_form, &u_space, &u_space, &mesh, 1);

    let coupling = assemble_coupling_matrix(&mesh, 1, nf, ns, 2);

    let system = CoupledSystem {
        n_fluid: nf,
        n_solid: ns,
        fluid_stiffness,
        fluid_mass,
        fluid_damping: TripletMatrix::new(nf, nf),
        solid_stiffness,
        solid_mass: TripletMatrix::new(ns, ns),
        solid_damping: TripletMatrix::new(ns, ns),
        coupling,
        interface_load: None,
        electrical: None,
    };

    let assembled = system.assemble_block(
        Complex64::new(25.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-25.0, 0.0),
    );

    assert!(
        assembled.nnz() > system.fluid_stiffness.nnz() + system.solid_stiffness.nnz(),
        "Coupled system should have more entries than sum of blocks"
    );
}

fn test_coupled_system_frequency_dependence() {
    let mesh = unit_square_triangles(4);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let nf = p_space.num_global_dofs(&mesh);
    let ns = u_space.num_global_dofs(&mesh);

    let fluid_stiffness1 = TripletMatrix::new(nf, nf);
    let fluid_mass = assemble_bilinear(&LaplacianForm, &p_space, &p_space, &mesh, 1);

    let material = IsotropicMaterial::new(1.0, 0.3);
    let elastic_form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let solid_stiffness = assemble_bilinear(&elastic_form, &u_space, &u_space, &mesh, 1);

    let coupling = assemble_coupling_matrix(&mesh, 1, nf, ns, 2);

    let system = CoupledSystem {
        n_fluid: nf,
        n_solid: ns,
        fluid_stiffness: fluid_stiffness1,
        fluid_mass,
        fluid_damping: TripletMatrix::new(nf, nf),
        solid_stiffness,
        solid_mass: TripletMatrix::new(ns, ns),
        solid_damping: TripletMatrix::new(ns, ns),
        coupling,
        interface_load: None,
        electrical: None,
    };

    let k1_sq = Complex64::new(25.0, 0.0);
    let k2_sq = Complex64::new(100.0, 0.0);

    let assembled1 =
        system.assemble_block(k1_sq, Complex64::new(1.0, 0.0), Complex64::new(-25.0, 0.0));
    let assembled2 =
        system.assemble_block(k2_sq, Complex64::new(1.0, 0.0), Complex64::new(-100.0, 0.0));

    let sum1: f64 = assembled1.values.iter().map(|v| v.norm_sqr()).sum();
    let sum2: f64 = assembled2.values.iter().map(|v| v.norm_sqr()).sum();

    assert!(
        sum2 > sum1,
        "Higher frequency should give larger matrix norm (more mass contribution)"
    );
}

fn test_coupled_system_coupling_matrix_structure() {
    let mesh = unit_square_triangles(4);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let nf = p_space.num_global_dofs(&mesh);
    let ns = u_space.num_global_dofs(&mesh);

    let _fluid_stiffness: TripletMatrix<f64> = TripletMatrix::new(nf, nf);
    let _fluid_mass = assemble_bilinear(&LaplacianForm, &p_space, &p_space, &mesh, 1);

    let material = IsotropicMaterial::new(1.0, 0.3);
    let elastic_form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let _solid_stiffness = assemble_bilinear(&elastic_form, &u_space, &u_space, &mesh, 1);

    let coupling = assemble_coupling_matrix(&mesh, 1, nf, ns, 2);

    assert!(coupling.nnz() > 0, "Coupling matrix should be non-empty");
    assert_eq!(coupling.num_rows, nf);
    assert_eq!(coupling.num_cols, ns);
}

fn test_coupled_system_real_when_k_squared_real() {
    let mesh = unit_square_triangles(4);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_space = VectorH1Space::new(PolynomialDegree::P1, 2);

    let nf = p_space.num_global_dofs(&mesh);
    let ns = u_space.num_global_dofs(&mesh);

    let k_squared = Complex64::new(25.0, 0.0);
    let fluid_stiffness: TripletMatrix<f64> = TripletMatrix::new(nf, nf);
    let fluid_mass = assemble_bilinear(&LaplacianForm, &p_space, &p_space, &mesh, 1);

    let material = IsotropicMaterial::new(1.0, 0.3);
    let elastic_form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let solid_stiffness = assemble_bilinear(&elastic_form, &u_space, &u_space, &mesh, 1);

    let coupling = assemble_coupling_matrix(&mesh, 1, nf, ns, 2);

    let system = CoupledSystem {
        n_fluid: nf,
        n_solid: ns,
        fluid_stiffness,
        fluid_mass,
        fluid_damping: TripletMatrix::new(nf, nf),
        solid_stiffness,
        solid_mass: TripletMatrix::new(ns, ns),
        solid_damping: TripletMatrix::new(ns, ns),
        coupling,
        interface_load: None,
        electrical: None,
    };

    let assembled = system.assemble_block(
        k_squared,
        Complex64::new(1.0, 0.0),
        Complex64::new(-25.0, 0.0),
    );

    let max_imag = assembled
        .values
        .iter()
        .map(|v| v.im.abs())
        .fold(0.0, f64::max);
    assert!(max_imag < 1e-10, "Real k² should give real matrix");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_coupled_system_block_dimensions();
        test_coupled_system_coupling_block();
        test_coupled_system_frequency_dependence();
        test_coupled_system_coupling_matrix_structure();
        test_coupled_system_real_when_k_squared_real();
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
