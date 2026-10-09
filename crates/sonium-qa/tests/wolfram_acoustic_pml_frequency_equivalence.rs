//! Frequency-domain equivalence check for Wolfram's concrete 1D transient
//! acoustic PML example.
//!
//! Wolfram solves first-order pressure/velocity equations with equal damping
//! `c sigma(x)` in a 0.25 m PML appended to a 0.5 m physical interval. For a
//! harmonic outgoing characteristic, those equations give the complex stretch
//! `s_x = 1 + i sigma/k` used by Sonium's `PmlHelmholtzForm`. This test uses
//! Wolfram's exact dimensions, frequency, material constants, and linear
//! `sigma` profile, then reconstructs its pressure snapshot at 0.0025 s.
//!
//! This does not implement Wolfram's transient auxiliary-field solver. It
//! validates the shared harmonic limit and keeps transient PML as a separate
//! API gap.

use num_complex::Complex64;
use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::{PmlDirection, PmlProfile, PmlRegion},
    mesh::rectangular_mesh_triangles,
    physics::acoustics::pressure::PmlHelmholtzForm,
    spaces::H1Space,
};
use sonium_solvers::{CsrMatrix, SparseLuFactorization, traits::Preconditioner};

const PHYSICAL_RIGHT_M: f64 = 0.5;
const PML_WIDTH_M: f64 = 0.25;
const OUTER_RIGHT_M: f64 = PHYSICAL_RIGHT_M + PML_WIDTH_M;
const STRIP_WIDTH_M: f64 = 0.01;
const SOUND_SPEED_M_S: f64 = 343.2;
const AIR_DENSITY_KG_M3: f64 = 1.2041;
const ANGULAR_FREQUENCY_RAD_S: f64 = 5025.0;
const SIGMA_MAX_M_INV: f64 = 100.0;
const FINAL_TIME_S: f64 = 0.0025;

fn attenuation_integral(x: f64) -> f64 {
    if x <= PHYSICAL_RIGHT_M {
        0.0
    } else {
        let distance = (x - PHYSICAL_RIGHT_M).min(PML_WIDTH_M);
        SIGMA_MAX_M_INV * distance * distance / (2.0 * PML_WIDTH_M)
    }
}

fn exact_complex_pressure(x: f64) -> Complex64 {
    let wavenumber = ANGULAR_FREQUENCY_RAD_S / SOUND_SPEED_M_S;
    Complex64::new(-attenuation_integral(x), wavenumber * x).exp()
}

fn impose_dirichlet_values(
    matrix: &mut CsrMatrix<Complex64>,
    rhs: &mut [Complex64],
    values: &[Option<Complex64>],
) {
    for row in 0..matrix.num_rows {
        if values[row].is_some() {
            continue;
        }
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            if let Some(value) = values[matrix.col_indices[entry]] {
                rhs[row] -= matrix.values[entry] * value;
                matrix.values[entry] = Complex64::new(0.0, 0.0);
            }
        }
    }
    for (dof, value) in values.iter().enumerate() {
        let Some(value) = value else { continue };
        for entry in matrix.row_ptrs[dof]..matrix.row_ptrs[dof + 1] {
            matrix.values[entry] = if matrix.col_indices[entry] == dof {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
        rhs[dof] = *value;
    }
}

fn column_average(nodes: &[sonium_fem::mesh::Point], solution: &[Complex64], x: f64) -> Complex64 {
    let values: Vec<_> = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, point)| ((point.x - x).abs() < 1.0e-12).then_some(solution[index]))
        .collect();
    assert_eq!(values.len(), 2);
    0.5 * (values[0] + values[1])
}

#[derive(Debug)]
struct ResultSummary {
    max_complex_error: f64,
    max_physical_domain_error: f64,
    max_final_snapshot_error_pa: f64,
    reflection_ratio: f64,
    outer_exact_amplitude: f64,
    relative_residual: f64,
    samples: [Complex64; 4],
}

fn solve_case(cells_x: usize) -> ResultSummary {
    let mesh = rectangular_mesh_triangles(0.0, OUTER_RIGHT_M, 0.0, STRIP_WIDTH_M, cells_x, 1);
    let space = H1Space::new(PolynomialDegree::P1);
    let wavenumber = ANGULAR_FREQUENCY_RAD_S / SOUND_SPEED_M_S;
    let mut pml = PmlRegion::x_positive(PHYSICAL_RIGHT_M, PML_WIDTH_M, SIGMA_MAX_M_INV, wavenumber);
    pml.profile = PmlProfile::Polynomial { power: 1 };
    let mut matrix = assemble_bilinear(
        &PmlHelmholtzForm::from_real(wavenumber, vec![pml]),
        &space,
        &space,
        &mesh,
        3,
    )
    .to_csr();
    let mut rhs = vec![Complex64::new(0.0, 0.0); mesh.num_nodes()];
    let mut dirichlet = vec![None; mesh.num_nodes()];
    for (index, point) in mesh.nodes.iter().enumerate() {
        if point.x.abs() < 1.0e-12 {
            dirichlet[index] = Some(Complex64::new(1.0, 0.0));
        } else if (point.x - OUTER_RIGHT_M).abs() < 1.0e-12 {
            dirichlet[index] = Some(Complex64::new(0.0, 0.0));
        }
    }
    impose_dirichlet_values(&mut matrix, &mut rhs, &dirichlet);
    let factorization =
        SparseLuFactorization::factorize(&matrix).expect("1D acoustic PML operator must factorize");
    let pressure = factorization.apply(&rhs);
    let lhs = matrix.matvec(&pressure);
    let residual = lhs
        .iter()
        .zip(&rhs)
        .map(|(left, right)| (*left - *right).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();

    let mut max_complex_error = 0.0_f64;
    let mut max_physical_domain_error = 0.0_f64;
    let mut max_final_snapshot_error_pa = 0.0_f64;
    let time_phase = Complex64::new(0.0, -ANGULAR_FREQUENCY_RAD_S * FINAL_TIME_S).exp();
    for (index, point) in mesh.nodes.iter().enumerate() {
        let exact = exact_complex_pressure(point.x);
        let error = (pressure[index] - exact).norm();
        max_complex_error = max_complex_error.max(error);
        if point.x <= PHYSICAL_RIGHT_M + 1.0e-12 {
            max_physical_domain_error = max_physical_domain_error.max(error);
        }
        let computed_snapshot = (Complex64::i() * pressure[index] * time_phase).re;
        let exact_snapshot = (ANGULAR_FREQUENCY_RAD_S * (FINAL_TIME_S - point.x / SOUND_SPEED_M_S))
            .sin()
            * (-attenuation_integral(point.x)).exp();
        max_final_snapshot_error_pa =
            max_final_snapshot_error_pa.max((computed_snapshot - exact_snapshot).abs());
    }

    let fit_x = 0.25;
    let fit_pressure = column_average(&mesh.nodes, &pressure, fit_x);
    let forward = Complex64::new(0.0, wavenumber * fit_x).exp();
    let backward = Complex64::new(0.0, -wavenumber * fit_x).exp();
    let reflected = (fit_pressure - forward) / (backward - forward);
    let incident = Complex64::new(1.0, 0.0) - reflected;
    let samples = [0.0, 0.25, 0.5, 0.625].map(|x| column_average(&mesh.nodes, &pressure, x));

    ResultSummary {
        max_complex_error,
        max_physical_domain_error,
        max_final_snapshot_error_pa,
        reflection_ratio: reflected.norm() / incident.norm(),
        outer_exact_amplitude: exact_complex_pressure(OUTER_RIGHT_M).norm(),
        relative_residual: residual / rhs_norm,
        samples,
    }
}

#[test]
fn sonium_pml_matches_wolfram_1d_harmonic_limit() {
    let coarse = solve_case(600);
    let fine = solve_case(1200);
    eprintln!("Wolfram 1D PML harmonic equivalence: coarse={coarse:?}, fine={fine:?}");

    assert!(fine.outer_exact_amplitude < 4.0e-6);
    assert!(fine.relative_residual < 1.0e-11);
    assert!(fine.reflection_ratio < 2.0e-4);
    assert!(fine.max_physical_domain_error < 2.0e-4);
    assert!(fine.max_complex_error < 2.0e-4);
    assert!(fine.max_final_snapshot_error_pa < 2.0e-4);
    for (sample, (&fine_value, &coarse_value)) in
        fine.samples.iter().zip(&coarse.samples).enumerate()
    {
        assert!(
            (fine_value - coarse_value).norm() < 3.0e-4,
            "sample {sample}: fine={fine_value:?}, coarse={coarse_value:?}"
        );
    }

    let impedance = AIR_DENSITY_KG_M3 * SOUND_SPEED_M_S;
    let velocity_samples = fine.samples.map(|pressure| pressure / impedance);
    for (pressure, velocity) in fine.samples.iter().zip(&velocity_samples) {
        assert!((*pressure - impedance * *velocity).norm() < 1.0e-14);
    }
}

fn exact_diagonal_pressure(x: f64, y: f64) -> Complex64 {
    let wavenumber = ANGULAR_FREQUENCY_RAD_S / SOUND_SPEED_M_S;
    let direction = 1.0 / 2.0_f64.sqrt();
    let attenuation = direction * (attenuation_integral(x) + attenuation_integral(y));
    Complex64::new(-attenuation, wavenumber * direction * (x + y)).exp()
}

fn point_value(
    nodes: &[sonium_fem::mesh::Point],
    solution: &[Complex64],
    x: f64,
    y: f64,
) -> Complex64 {
    let index = nodes
        .iter()
        .position(|point| (point.x - x).abs() < 1.0e-12 && (point.y - y).abs() < 1.0e-12)
        .expect("sample must align with a mesh node");
    solution[index]
}

#[derive(Debug)]
struct CornerResultSummary {
    max_physical_domain_error: f64,
    max_pml_error: f64,
    corner_amplitude: f64,
    exact_corner_amplitude: f64,
    relative_residual: f64,
    samples: [Complex64; 3],
}

fn solve_corner_case(cells_per_axis: usize) -> CornerResultSummary {
    let mesh = rectangular_mesh_triangles(
        0.0,
        OUTER_RIGHT_M,
        0.0,
        OUTER_RIGHT_M,
        cells_per_axis,
        cells_per_axis,
    );
    let space = H1Space::new(PolynomialDegree::P1);
    let wavenumber = ANGULAR_FREQUENCY_RAD_S / SOUND_SPEED_M_S;
    let mut corner = PmlRegion::new(
        PmlDirection::XY,
        [PHYSICAL_RIGHT_M, PHYSICAL_RIGHT_M, f64::NEG_INFINITY],
        [OUTER_RIGHT_M, OUTER_RIGHT_M, f64::INFINITY],
        SIGMA_MAX_M_INV,
        wavenumber,
    );
    corner.profile = PmlProfile::Polynomial { power: 1 };
    let mut x_side =
        PmlRegion::x_positive(PHYSICAL_RIGHT_M, PML_WIDTH_M, SIGMA_MAX_M_INV, wavenumber);
    x_side.profile = PmlProfile::Polynomial { power: 1 };
    let mut y_side =
        PmlRegion::y_positive(PHYSICAL_RIGHT_M, PML_WIDTH_M, SIGMA_MAX_M_INV, wavenumber);
    y_side.profile = PmlProfile::Polynomial { power: 1 };
    // `PmlHelmholtzForm` uses the first containing region. The corner must be
    // listed first so both stretches are active there.
    let mut matrix = assemble_bilinear(
        &PmlHelmholtzForm::from_real(wavenumber, vec![corner, x_side, y_side]),
        &space,
        &space,
        &mesh,
        3,
    )
    .to_csr();
    let mut rhs = vec![Complex64::new(0.0, 0.0); mesh.num_nodes()];
    let mut dirichlet = vec![None; mesh.num_nodes()];
    for (index, point) in mesh.nodes.iter().enumerate() {
        if (point.x - OUTER_RIGHT_M).abs() < 1.0e-12 || (point.y - OUTER_RIGHT_M).abs() < 1.0e-12 {
            dirichlet[index] = Some(Complex64::new(0.0, 0.0));
        } else if point.x.abs() < 1.0e-12 || point.y.abs() < 1.0e-12 {
            dirichlet[index] = Some(exact_diagonal_pressure(point.x, point.y));
        }
    }
    impose_dirichlet_values(&mut matrix, &mut rhs, &dirichlet);
    let factorization = SparseLuFactorization::factorize(&matrix)
        .expect("2D acoustic corner-PML operator must factorize");
    let pressure = factorization.apply(&rhs);
    let lhs = matrix.matvec(&pressure);
    let residual = lhs
        .iter()
        .zip(&rhs)
        .map(|(left, right)| (*left - *right).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    let mut max_physical_domain_error = 0.0_f64;
    let mut max_pml_error = 0.0_f64;
    for (index, point) in mesh.nodes.iter().enumerate() {
        let error = (pressure[index] - exact_diagonal_pressure(point.x, point.y)).norm();
        if point.x <= PHYSICAL_RIGHT_M + 1.0e-12 && point.y <= PHYSICAL_RIGHT_M + 1.0e-12 {
            max_physical_domain_error = max_physical_domain_error.max(error);
        } else {
            max_pml_error = max_pml_error.max(error);
        }
    }
    let samples = [[0.25, 0.25], [0.5, 0.5], [0.625, 0.625]]
        .map(|point| point_value(&mesh.nodes, &pressure, point[0], point[1]));
    let corner_amplitude = samples[2].norm();
    let exact_corner_amplitude = exact_diagonal_pressure(0.625, 0.625).norm();

    CornerResultSummary {
        max_physical_domain_error,
        max_pml_error,
        corner_amplitude,
        exact_corner_amplitude,
        relative_residual: residual / rhs_norm,
        samples,
    }
}

#[test]
fn sonium_composes_two_dimensional_corner_pml_stretches() {
    let coarse = solve_corner_case(60);
    let fine = solve_corner_case(120);
    eprintln!("Wolfram 2D corner-PML harmonic equivalence: coarse={coarse:?}, fine={fine:?}");

    assert!(fine.relative_residual < 1.0e-10);
    assert!(fine.max_physical_domain_error < 0.015);
    assert!(fine.max_pml_error < 0.015);
    assert!(
        (fine.corner_amplitude - fine.exact_corner_amplitude).abs() / fine.exact_corner_amplitude
            < 0.03
    );
    for (sample, (&fine_value, &coarse_value)) in
        fine.samples.iter().zip(&coarse.samples).enumerate()
    {
        assert!(
            (fine_value - coarse_value).norm() < 0.023,
            "sample {sample}: fine={fine_value:?}, coarse={coarse_value:?}"
        );
    }
}
