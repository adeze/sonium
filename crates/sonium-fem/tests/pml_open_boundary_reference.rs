//! Independent CPU reference for a one-dimensional acoustic open boundary.
//!
//! The physical domain is terminated by a polynomial PML.  The reference
//! pressure is the outgoing plane wave analytically continued through the
//! stretched coordinate, so this test checks the assembled FEM operator and
//! the sparse solve without reusing an FEM solution as its oracle.

use num_complex::Complex64;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::{PmlProfile, PmlRegion},
    mesh::{Point, rectangular_mesh_triangles},
    physics::acoustics::pressure::PmlHelmholtzForm,
    spaces::H1Space,
};
use sonium_solvers::{CsrMatrix, SparseLuFactorization};

const PHYSICAL_RIGHT_M: f64 = 0.5;
const PML_WIDTH_M: f64 = 0.25;
const OUTER_RIGHT_M: f64 = PHYSICAL_RIGHT_M + PML_WIDTH_M;
const STRIP_WIDTH_M: f64 = 0.01;
const SOUND_SPEED_M_S: f64 = 343.2;
const ANGULAR_FREQUENCY_RAD_S: f64 = 2745.6;
const SIGMA_MAX_M_INV: f64 = 100.0;

fn wavenumber() -> f64 {
    ANGULAR_FREQUENCY_RAD_S / SOUND_SPEED_M_S
}

fn attenuation_integral(x: f64) -> f64 {
    if x <= PHYSICAL_RIGHT_M {
        return 0.0;
    }

    let distance = (x - PHYSICAL_RIGHT_M).clamp(0.0, PML_WIDTH_M);
    SIGMA_MAX_M_INV * distance * distance / (2.0 * PML_WIDTH_M)
}

fn exact_pressure(x: f64) -> Complex64 {
    Complex64::new(-attenuation_integral(x), wavenumber() * x).exp()
}

fn impose_dirichlet_values(
    matrix: &mut CsrMatrix<Complex64>,
    rhs: &mut [Complex64],
    values: &[Option<Complex64>],
) {
    assert_eq!(matrix.num_rows, values.len());
    assert_eq!(rhs.len(), values.len());

    for row in 0..matrix.num_rows {
        if values[row].is_some() {
            continue;
        }
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if let Some(value) = values[matrix.col_indices[entry]] {
                rhs[row] -= matrix.values[entry] * value;
                matrix.values[entry] = Complex64::default();
            }
        }
    }

    for (dof, value) in values.iter().enumerate() {
        let Some(value) = value else { continue };
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::default()
            };
        }
        rhs[dof] = *value;
    }
}

fn column_average(nodes: &[Point], solution: &[Complex64], x: f64) -> Complex64 {
    let values: Vec<_> = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, point)| ((point.x - x).abs() < 1.0e-12).then_some(solution[index]))
        .collect();
    assert_eq!(
        values.len(),
        2,
        "one-cell strip must have two nodes per x column"
    );
    0.5 * (values[0] + values[1])
}

pub fn run_cpu_fem_pml_outgoing_acoustic_reference() {
    let mesh = rectangular_mesh_triangles(0.0, OUTER_RIGHT_M, 0.0, STRIP_WIDTH_M, 160, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let k = wavenumber();
    let mut pml = PmlRegion::x_positive(PHYSICAL_RIGHT_M, PML_WIDTH_M, SIGMA_MAX_M_INV, k);
    pml.profile = PmlProfile::Polynomial { power: 1 };

    let mut matrix = assemble_bilinear(
        &PmlHelmholtzForm::from_real(k, vec![pml]),
        &space,
        &space,
        &mesh,
        3,
    )
    .to_csr();
    let mut rhs = vec![Complex64::default(); mesh.num_nodes()];
    let mut dirichlet = vec![None; mesh.num_nodes()];
    for (index, point) in mesh.nodes.iter().enumerate() {
        if point.x.abs() < 1.0e-12 {
            dirichlet[index] = Some(Complex64::new(1.0, 0.0));
        } else if (point.x - OUTER_RIGHT_M).abs() < 1.0e-12 {
            // The exact outgoing wave is already below 4e-6 here.
            dirichlet[index] = Some(Complex64::default());
        }
    }
    impose_dirichlet_values(&mut matrix, &mut rhs, &dirichlet);

    let factorization = SparseLuFactorization::factorize(&matrix)
        .expect("CPU FEM PML reference operator must factorize");
    let pressure = factorization
        .solve(&rhs)
        .expect("CPU FEM PML reference solve must succeed");

    let residual = matrix
        .matvec(&pressure)
        .iter()
        .zip(&rhs)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    assert!(residual / rhs_norm < 1.0e-11);

    let max_error = mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (pressure[index] - exact_pressure(point.x)).norm())
        .fold(0.0, f64::max);
    let physical_error = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| point.x <= PHYSICAL_RIGHT_M + 1.0e-12)
        .map(|(index, point)| (pressure[index] - exact_pressure(point.x)).norm())
        .fold(0.0, f64::max);

    let fit_x = 0.3; // aligned with the 160-cell mesh
    let fit_pressure = column_average(&mesh.nodes, &pressure, fit_x);
    let forward = Complex64::new(0.0, k * fit_x).exp();
    let backward = Complex64::new(0.0, -k * fit_x).exp();
    let reflected = (fit_pressure - forward) / (backward - forward);
    let incident = Complex64::new(1.0, 0.0) - reflected;
    let reflection_ratio = reflected.norm() / incident.norm();

    eprintln!(
        "CPU FEM acoustic PML reference: residual={:.3e}, max_error={:.3e}, physical_error={:.3e}, reflection={:.3e}",
        residual / rhs_norm,
        max_error,
        physical_error,
        reflection_ratio,
    );

    assert!(exact_pressure(OUTER_RIGHT_M).norm() < 4.0e-6);
    assert!(reflection_ratio < 2.0e-3);
    assert!(physical_error < 3.0e-3);
    assert!(max_error < 3.0e-3);
}

#[test]
fn cpu_fem_pml_recovers_outgoing_acoustic_reference() {
    run_cpu_fem_pml_outgoing_acoustic_reference();
}
