//! Electroacoustics validation tests
//!
//! Tests for coupled fluid-structure (acoustic + elastic) systems.
//!
//! - Block system assembly dimensions
//! - Coupling matrix structure verification
//! - Frequency-dependent assembly
//! - Coupled system symmetry (when applicable)
//! - Driven coupled solve with pressure/displacement response
//! - Decoupled-limit reference against independent block solves
//! - Physical reciprocity under swapped forward drives
//! - Independent cavity/bar pressure and displacement convergence
//! - Input/dissipated power balance

use num_complex::Complex64;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{LaplacianForm, MassForm};
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::acoustics::electroacoustics::{
    CoupledSystem, HarmonicCoupledPoint, assemble_coupling_matrix,
};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.electroacoustics-validation.v1";

// Driven coupled problem parameters.
const MESH_SUBDIVISIONS: usize = 4;
const DRIVE_FREQUENCY_HZ: f64 = 5.0;
const SOUND_SPEED: f64 = 343.0;
const FLUID_DENSITY: f64 = 1.204;
const SOLID_YOUNG: f64 = 1000.0;
const SOLID_POISSON: f64 = 0.3;
const FLUID_DAMPING_RATIO: f64 = 0.05;
const SOLID_LUMPED_MASS: f64 = 1.0;
const SOLID_LUMPED_DAMPING: f64 = 0.5;
const DRIVE_NODE: usize = 10;
const SENSE_NODE: usize = 12;

// Acceptance tolerances.
const TOL_RESIDUAL: f64 = 1e-10;
const TOL_DECOUPLED: f64 = 1e-10;
const TOL_RECIPROCITY: f64 = 1e-9;
const TOL_POWER: f64 = 1e-9;
const MIN_COUPLING_TRANSFER: f64 = 1e-6;

fn check(condition: bool, message: String) -> Result<(), String> {
    if condition { Ok(()) } else { Err(message) }
}

/// Dense complex Gaussian elimination with partial pivoting.
fn dense_complex_solve(matrix: &mut [Complex64], rhs: &mut [Complex64]) -> Result<(), String> {
    let n = rhs.len();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_value = matrix[col * n + col].norm();
        for row in (col + 1)..n {
            let value = matrix[row * n + col].norm();
            if value > pivot_value {
                pivot_value = value;
                pivot = row;
            }
        }
        if pivot_value == 0.0 {
            return Err(format!("singular matrix at column {col}"));
        }
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diagonal = matrix[col * n + col];
        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / diagonal;
            matrix[row * n + col] = Complex64::new(0.0, 0.0);
            for k in (col + 1)..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
    Ok(())
}

fn dense_from_triplet(triplet: &TripletMatrix<Complex64>) -> Vec<Complex64> {
    let n = triplet.num_rows;
    let mut dense = vec![Complex64::new(0.0, 0.0); n * n];
    for k in 0..triplet.values.len() {
        dense[triplet.rows[k] * n + triplet.cols[k]] += triplet.values[k];
    }
    dense
}

fn relative_residual(
    matrix: &[Complex64],
    solution: &[Complex64],
    rhs: &[Complex64],
    n: usize,
) -> f64 {
    let mut residual = 0.0;
    let mut rhs_norm = 0.0;
    for row in 0..n {
        let mut row_sum = Complex64::new(0.0, 0.0);
        for col in 0..n {
            row_sum += matrix[row * n + col] * solution[col];
        }
        residual += (row_sum - rhs[row]).norm_sqr();
        rhs_norm += rhs[row].norm_sqr();
    }
    (residual / rhs_norm).sqrt()
}

/// Build the driven coupled test system with physical damping.
fn coupled_system() -> (CoupledSystem, HarmonicCoupledPoint, usize, usize) {
    let mesh = unit_square_triangles(MESH_SUBDIVISIONS);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let nf = p_space.num_global_dofs(&mesh);
    let ns = u_space.num_global_dofs(&mesh);

    let fluid_stiffness = assemble_bilinear(&LaplacianForm, &p_space, &p_space, &mesh, 2);
    let fluid_mass = assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);
    let mut fluid_damping = fluid_mass.clone();
    for value in &mut fluid_damping.values {
        *value *= FLUID_DAMPING_RATIO;
    }

    let material = IsotropicMaterial::new(SOLID_YOUNG, SOLID_POISSON);
    let elastic_form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStrain);
    let solid_stiffness = assemble_bilinear(&elastic_form, &u_space, &u_space, &mesh, 2);
    let mut solid_mass = TripletMatrix::new(ns, ns);
    let mut solid_damping = TripletMatrix::new(ns, ns);
    for dof in 0..ns {
        solid_mass.push(dof, dof, SOLID_LUMPED_MASS);
        solid_damping.push(dof, dof, SOLID_LUMPED_DAMPING);
    }

    let coupling = assemble_coupling_matrix(&mesh, 1, nf, ns, 2);
    let system = CoupledSystem {
        n_fluid: nf,
        n_solid: ns,
        fluid_stiffness,
        fluid_mass,
        fluid_damping,
        solid_stiffness,
        solid_mass,
        solid_damping,
        coupling,
        interface_load: None,
        electrical: None,
    };
    let point = HarmonicCoupledPoint {
        omega_rad_per_s: 2.0 * std::f64::consts::PI * DRIVE_FREQUENCY_HZ,
        sound_speed_m_per_s: SOUND_SPEED,
        fluid_density_kg_per_m3: FLUID_DENSITY,
    };
    (system, point, nf, ns)
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    assembly_checks: usize,
    coupled_residual: f64,
    pressure_norm: f64,
    displacement_norm: f64,
    coupling_transfer: f64,
    decoupled_fluid_leakage: f64,
    decoupled_solid_error: f64,
    reciprocity_error: f64,
    input_power: f64,
    dissipated_power: f64,
    power_error: f64,
}

fn run() -> Result<Metrics, String> {
    let mut assembly_checks = 0usize;
    let (system, point, nf, ns) = coupled_system();
    let total = system.total_dofs();
    check(
        total == nf + ns && total > 0,
        "total DOFs mismatch".to_string(),
    )?;
    assembly_checks += 1;

    let assembled = system
        .assemble_harmonic(point)
        .map_err(|error| format!("harmonic assembly failed: {error:?}"))?;
    check(
        assembled.num_rows == total && assembled.num_cols == total,
        "assembled block dimensions mismatch".to_string(),
    )?;
    assembly_checks += 1;
    check(
        assembled.nnz() > system.fluid_stiffness.nnz() + system.solid_stiffness.nnz(),
        "coupled system should add coupling entries".to_string(),
    )?;
    assembly_checks += 1;

    // Frequency dependence: a higher drive frequency changes the assembly.
    let high_point = HarmonicCoupledPoint {
        omega_rad_per_s: 2.0 * point.omega_rad_per_s,
        ..point
    };
    let assembled_high = system
        .assemble_harmonic(high_point)
        .map_err(|error| format!("high frequency assembly failed: {error:?}"))?;
    let sum_low: f64 = assembled.values.iter().map(|value| value.norm_sqr()).sum();
    let sum_high: f64 = assembled_high
        .values
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    check(
        (sum_high - sum_low).abs() / sum_low > 1e-6,
        "assembly should depend on frequency".to_string(),
    )?;
    assembly_checks += 1;

    check(
        system.coupling.nnz() > 0
            && system.coupling.num_rows == nf
            && system.coupling.num_cols == ns,
        "coupling matrix structure mismatch".to_string(),
    )?;
    assembly_checks += 1;

    // Driven coupled solve with a unit transverse force on the interface edge.
    let drive_dof = nf + DRIVE_NODE * 2;
    let mut load = vec![Complex64::new(0.0, 0.0); total];
    load[drive_dof] = Complex64::new(1.0, 0.0);
    let dense = dense_from_triplet(&assembled);
    let mut solve_matrix = dense.clone();
    let mut solution = load.clone();
    dense_complex_solve(&mut solve_matrix, &mut solution)?;
    let coupled_residual = relative_residual(&dense, &solution, &load, total);
    check(
        coupled_residual <= TOL_RESIDUAL,
        format!("coupled residual {coupled_residual:.3e} exceeds {TOL_RESIDUAL:.1e}"),
    )?;

    let pressure = &solution[..nf];
    let displacement = &solution[nf..];
    let pressure_norm: f64 = pressure
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let displacement_norm: f64 = displacement
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    check(
        displacement_norm > 0.0,
        "driven displacement should be nonzero".to_string(),
    )?;
    let coupling_transfer = pressure_norm / displacement_norm;
    check(
        coupling_transfer >= MIN_COUPLING_TRANSFER,
        format!("coupling transfer {coupling_transfer:.3e} below {MIN_COUPLING_TRANSFER:.1e}"),
    )?;

    // Decoupled limit: with the coupling removed, the fluid stays quiet and
    // the solid matches an independently assembled block solve.
    let mut decoupled = coupled_system().0;
    decoupled.coupling = TripletMatrix::new(nf, ns);
    let decoupled_triplet = decoupled
        .assemble_harmonic(point)
        .map_err(|error| format!("decoupled assembly failed: {error:?}"))?;
    let decoupled_dense = dense_from_triplet(&decoupled_triplet);
    let mut decoupled_matrix = decoupled_dense.clone();
    let mut decoupled_solution = load.clone();
    dense_complex_solve(&mut decoupled_matrix, &mut decoupled_solution)?;
    let decoupled_pressure_norm: f64 = decoupled_solution[..nf]
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let decoupled_solid_norm: f64 = decoupled_solution[nf..]
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let decoupled_fluid_leakage = decoupled_pressure_norm / decoupled_solid_norm;
    check(
        decoupled_fluid_leakage <= TOL_DECOUPLED,
        format!(
            "decoupled fluid leakage {decoupled_fluid_leakage:.3e} exceeds {TOL_DECOUPLED:.1e}"
        ),
    )?;

    // Independent solid block: K_s - omega^2 M_s + i omega C_s.
    let omega = point.omega_rad_per_s;
    let mut solid_block = vec![Complex64::new(0.0, 0.0); ns * ns];
    for k in 0..decoupled.solid_stiffness.values.len() {
        solid_block[decoupled.solid_stiffness.rows[k] * ns + decoupled.solid_stiffness.cols[k]] +=
            Complex64::new(decoupled.solid_stiffness.values[k], 0.0);
    }
    for k in 0..decoupled.solid_mass.values.len() {
        solid_block[decoupled.solid_mass.rows[k] * ns + decoupled.solid_mass.cols[k]] +=
            Complex64::new(-omega * omega * decoupled.solid_mass.values[k], 0.0);
    }
    for k in 0..decoupled.solid_damping.values.len() {
        solid_block[decoupled.solid_damping.rows[k] * ns + decoupled.solid_damping.cols[k]] +=
            Complex64::new(0.0, omega * decoupled.solid_damping.values[k]);
    }
    let mut solid_load = vec![Complex64::new(0.0, 0.0); ns];
    solid_load[DRIVE_NODE * 2] = Complex64::new(1.0, 0.0);
    let mut solid_matrix = solid_block.clone();
    let mut solid_solution = solid_load.clone();
    dense_complex_solve(&mut solid_matrix, &mut solid_solution)?;
    let mut solid_difference = 0.0;
    let mut solid_norm = 0.0;
    for dof in 0..ns {
        solid_difference += (decoupled_solution[nf + dof] - solid_solution[dof]).norm_sqr();
        solid_norm += solid_solution[dof].norm_sqr();
    }
    let decoupled_solid_error = (solid_difference / solid_norm).sqrt();
    check(
        decoupled_solid_error <= TOL_DECOUPLED,
        format!("decoupled solid error {decoupled_solid_error:.3e} exceeds {TOL_DECOUPLED:.1e}"),
    )?;

    // Physical reciprocity uses the same forward operator for swapped drives.
    // Dividing fluid rows by rho*omega^2 symmetrizes the pressure/displacement
    // coupling, hence H_ps/(rho*omega^2) = H_sp.
    let sense_dof = SENSE_NODE;
    let mut reciprocal_matrix = dense.clone();
    let mut reciprocal_solution = vec![Complex64::new(0.0, 0.0); total];
    reciprocal_solution[sense_dof] = Complex64::new(1.0, 0.0);
    dense_complex_solve(&mut reciprocal_matrix, &mut reciprocal_solution)?;
    let scaled_pressure = solution[sense_dof] / (FLUID_DENSITY * omega * omega);
    let reciprocity_error = (scaled_pressure - reciprocal_solution[drive_dof]).norm()
        / scaled_pressure.norm().max(1e-30);
    check(
        reciprocity_error <= TOL_RECIPROCITY,
        format!("reciprocity error {reciprocity_error:.3e} exceeds {TOL_RECIPROCITY:.1e}"),
    )?;

    // Power balance: input power equals damped dissipation for the passive
    // driven system with conservative interface coupling.
    let drive_response = solution[drive_dof];
    let input_power = -0.5 * omega * drive_response.im;
    check(
        input_power > 0.0,
        format!("input power {input_power:.3e} should be positive"),
    )?;
    let mut fluid_dissipation = 0.0;
    for k in 0..system.fluid_damping.values.len() {
        let row = system.fluid_damping.rows[k];
        let col = system.fluid_damping.cols[k];
        fluid_dissipation +=
            system.fluid_damping.values[k] * (pressure[row].conj() * pressure[col]).re;
    }
    let mut solid_dissipation = 0.0;
    for k in 0..system.solid_damping.values.len() {
        let row = system.solid_damping.rows[k];
        let col = system.solid_damping.cols[k];
        solid_dissipation +=
            system.solid_damping.values[k] * (displacement[row].conj() * displacement[col]).re;
    }
    // Power balance: input power equals damped dissipation. The solid block
    // is a force equation, so its dissipation is (omega^2 / 2) u^H C_s u.
    // The fluid block is a pressure equation; converting its damping source
    // through the linearized Euler velocity v = g / (i omega rho) gives a
    // dissipation of p^H C_f p / (2 rho). The interface exchange cancels.
    let solid_dissipated = 0.5 * omega * omega * solid_dissipation;
    let fluid_dissipated = fluid_dissipation / (2.0 * FLUID_DENSITY);
    let dissipated_power = solid_dissipated + fluid_dissipated;
    let power_error = (input_power - dissipated_power).abs() / input_power;
    check(
        power_error <= TOL_POWER,
        format!("power error {power_error:.3e} exceeds {TOL_POWER:.1e}"),
    )?;

    Ok(Metrics {
        assembly_checks,
        coupled_residual,
        pressure_norm,
        displacement_norm,
        coupling_transfer,
        decoupled_fluid_leakage,
        decoupled_solid_error,
        reciprocity_error,
        input_power,
        dissipated_power,
        power_error,
    })
}

/// Restrict assembled 2D forms to uniform cross-section pressure and axial motion.
///
/// This is a cavity coupled to a clamped axial bar, each of unit length and
/// cross-section. Coordinates in each domain point away from the shared x=0
/// interface; the bar's global axial displacement follows the fluid x axis.
/// Transverse displacement is constrained and Poisson's ratio is zero.
fn duct_system(n: usize) -> (CoupledSystem, Vec<usize>) {
    let mesh = unit_square_triangles(n);
    let scalar = H1Space::new(PolynomialDegree::P1);
    let vector = VectorH1Space::new(PolynomialDegree::P1, 2);
    let nf = scalar.num_global_dofs(&mesh);
    let ns = vector.num_global_dofs(&mesh);
    let axial: Vec<usize> = mesh
        .nodes
        .iter()
        .map(|p| (p.x * n as f64).round() as usize)
        .collect();
    let reduce = |a: &TripletMatrix<f64>, vector_rows: bool, vector_cols: bool| {
        let mut b = TripletMatrix::new(n + 1, n + 1);
        for k in 0..a.nnz() {
            let row = a.rows[k];
            let col = a.cols[k];
            if (vector_rows && !row.is_multiple_of(2)) || (vector_cols && !col.is_multiple_of(2)) {
                continue;
            }
            b.push(
                axial[if vector_rows { row / 2 } else { row }],
                axial[if vector_cols { col / 2 } else { col }],
                a.values[k],
            );
        }
        b
    };
    let mass = reduce(
        &assemble_bilinear(&MassForm, &scalar, &scalar, &mesh, 2),
        false,
        false,
    );
    let stiffness = reduce(
        &assemble_bilinear(&LaplacianForm, &scalar, &scalar, &mesh, 2),
        false,
        false,
    );
    let elastic = ElasticityForm::new_2d(
        IsotropicMaterial::new(20_000.0, 0.0),
        PlaneAssumption::PlaneStress,
    );
    let solid_stiffness = reduce(
        &assemble_bilinear(&elastic, &vector, &vector, &mesh, 2),
        true,
        true,
    );
    let scaled_mass = |coefficient: f64| {
        let mut a = mass.clone();
        for value in &mut a.values {
            *value *= coefficient;
        }
        a
    };
    let coupling = reduce(&assemble_coupling_matrix(&mesh, 1, nf, ns, 2), false, true);
    (
        CoupledSystem {
            n_fluid: n + 1,
            n_solid: n + 1,
            fluid_stiffness: stiffness,
            fluid_mass: mass.clone(),
            fluid_damping: scaled_mass(1e-5),
            solid_stiffness,
            solid_mass: scaled_mass(2.0),
            solid_damping: scaled_mass(0.2),
            coupling,
            interface_load: None,
            electrical: None,
        },
        vec![2 * n + 1],
    )
}

fn quadratic(a: &TripletMatrix<f64>, x: &[Complex64]) -> f64 {
    (0..a.nnz())
        .map(|k| a.values[k] * (x[a.rows[k]].conj() * x[a.cols[k]]).re)
        .sum()
}

fn duct_reference_gate() -> QaResult {
    let run_reference = || -> Result<QaResult, String> {
        let mut fine_pressure = 0.0_f64;
        let mut fine_displacement = 0.0_f64;
        let mut rate = f64::INFINITY;
        let mut residual = 0.0_f64;
        let mut power_error = 0.0_f64;
        let mut reciprocity = 0.0_f64;
        let mut minimum_fluid_power_fraction = f64::INFINITY;
        let mut negative_error = f64::INFINITY;
        let mut executed = 0usize;
        let mut free_pressure = usize::MAX;
        let mut free_displacement = usize::MAX;
        for frequency in [10.0, 20.0, 30.0] {
            let omega = 2.0 * std::f64::consts::PI * frequency;
            // Independent continuum equations and interface force balance.
            // p''+q^2*p=0, p'(1)=0, p'(0)=rho*omega^2*u(0).
            // E*u''+(rho_s*omega^2-i*omega*d_s)*u=0, u(1)=0.
            // -E*u'(0)+p(0)=1, with exp(+i*omega*t).
            let q = Complex64::new((omega / SOUND_SPEED).powi(2), -omega * 1e-5).sqrt();
            let s = (Complex64::new(2.0 * omega * omega, -omega * 0.2) / 20_000.0).sqrt();
            let fluid_impedance = FLUID_DENSITY * omega * omega * q.cos() / (q * q.sin());
            let bar_impedance = 20_000.0 * s * s.cos() / s.sin();
            let u0 = Complex64::new(1.0, 0.0) / (bar_impedance + fluid_impedance);
            let amplitude = FLUID_DENSITY * omega * omega * u0 / (q * q.sin());
            let mut previous: Option<(f64, f64)> = None;
            for n in [8, 16, 32] {
                let (system, constrained) = duct_system(n);
                let nf = n + 1;
                let total = 2 * nf;
                let pressure_dofs = (0..nf).filter(|dof| !constrained.contains(dof)).count();
                let displacement_dofs =
                    (nf..total).filter(|dof| !constrained.contains(dof)).count();
                check(
                    pressure_dofs > 0 && displacement_dofs > 0,
                    "reference case has no free field unknowns".into(),
                )?;
                free_pressure = free_pressure.min(pressure_dofs);
                free_displacement = free_displacement.min(displacement_dofs);
                let point = HarmonicCoupledPoint {
                    omega_rad_per_s: omega,
                    sound_speed_m_per_s: SOUND_SPEED,
                    fluid_density_kg_per_m3: FLUID_DENSITY,
                };
                let interface_integral: f64 = system.coupling.values.iter().sum();
                check(
                    (interface_integral + 1.0).abs() < 1e-12,
                    "interface normal/area differs from independent reference".into(),
                )?;
                let assembled = system
                    .assemble_harmonic(point)
                    .map_err(|e| format!("{e:?}"))?;
                let mut matrix = dense_from_triplet(&assembled);
                for &dof in &constrained {
                    for k in 0..total {
                        matrix[dof * total + k] = 0.0.into();
                        matrix[k * total + dof] = 0.0.into();
                    }
                    matrix[dof * total + dof] = 1.0.into();
                }
                let mut rhs = vec![Complex64::new(0.0, 0.0); total];
                rhs[nf] = 1.0.into();
                let mut solution = rhs.clone();
                dense_complex_solve(&mut matrix.clone(), &mut solution)?;
                residual = residual.max(relative_residual(&matrix, &solution, &rhs, total));
                // Integrate squared error of piecewise-linear solved fields with
                // independent 3-point Gauss quadrature in the axial coordinate.
                let mut errors = [0.0_f64; 2];
                let mut norms = [0.0_f64; 2];
                for element in 0..n {
                    for (z, weight) in [
                        (-0.7745966692414834, 5.0 / 9.0),
                        (0.0, 8.0 / 9.0),
                        (0.7745966692414834, 5.0 / 9.0),
                    ] {
                        let t = 0.5 * (z + 1.0);
                        let x = (element as f64 + t) / n as f64;
                        let exact = [
                            amplitude * (q * (1.0 - x)).cos(),
                            u0 * (s * (1.0 - x)).sin() / s.sin(),
                        ];
                        for field in 0..2 {
                            let offset = field * nf;
                            let value = solution[offset + element] * (1.0 - t)
                                + solution[offset + element + 1] * t;
                            errors[field] += weight * (value - exact[field]).norm_sqr();
                            norms[field] += weight * exact[field].norm_sqr();
                        }
                    }
                }
                let p_error = (errors[0] / norms[0]).sqrt();
                let u_error = (errors[1] / norms[1]).sqrt();
                check(
                    p_error.is_finite() && u_error.is_finite(),
                    "nonfinite solved reference error".into(),
                )?;
                executed += 1;
                if let Some((p, u)) = previous {
                    rate = rate.min((p / p_error).log2()).min((u / u_error).log2());
                }
                previous = Some((p_error, u_error));
                if n == 32 {
                    fine_pressure = fine_pressure.max(p_error);
                    fine_displacement = fine_displacement.max(u_error);
                }
                let p = &solution[..nf];
                let u = &solution[nf..];
                let input = -0.5 * omega * u[0].im;
                let fluid_power = quadratic(&system.fluid_damping, p) / (2.0 * FLUID_DENSITY);
                let solid_power = 0.5 * omega * omega * quadratic(&system.solid_damping, u);
                check(
                    input > 0.0 && fluid_power > 0.0 && solid_power > 0.0,
                    "nonpassive cavity/bar response".into(),
                )?;
                power_error = power_error.max((input - fluid_power - solid_power).abs() / input);
                minimum_fluid_power_fraction =
                    minimum_fluid_power_fraction.min(fluid_power / input);
                let sense = n / 2;
                let mut swapped = vec![Complex64::new(0.0, 0.0); total];
                swapped[sense] = 1.0.into();
                dense_complex_solve(&mut matrix.clone(), &mut swapped)?;
                let scaled = p[sense] / (FLUID_DENSITY * omega * omega);
                reciprocity = reciprocity.max((scaled - swapped[nf]).norm() / scaled.norm());
                if n == 32 {
                    // Missing coupling must destroy the independent pressure reference.
                    let mut uncoupled = system;
                    uncoupled.coupling = TripletMatrix::new(nf, nf);
                    let a = uncoupled
                        .assemble_harmonic(point)
                        .map_err(|e| format!("{e:?}"))?;
                    let mut a = dense_from_triplet(&a);
                    for &dof in &constrained {
                        for k in 0..total {
                            a[dof * total + k] = 0.0.into();
                            a[k * total + dof] = 0.0.into();
                        }
                        a[dof * total + dof] = 1.0.into();
                    }
                    let mut control = rhs.clone();
                    dense_complex_solve(&mut a, &mut control)?;
                    negative_error = negative_error.min(
                        (control[0] - amplitude * q.cos()).norm() / (amplitude * q.cos()).norm(),
                    );
                }
                println!(
                    "Cavity/bar {frequency}Hz n={n}: pressure={p_error:.6e}, displacement={u_error:.6e}, power={:.6e}",
                    (input - fluid_power - solid_power).abs() / input
                );
            }
        }
        let mut result = QaResult::passed("fem.electroacoustics-cavity-bar.v1")
            .requested_backend("cpu")
            .actual_backend("cpu");
        for (name, value, bound) in [
            ("fine_pressure_error", fine_pressure, 0.003),
            ("fine_displacement_error", fine_displacement, 0.003),
            ("true_residual", residual, 1e-9),
            ("power_error", power_error, 1e-9),
            ("physical_reciprocity_error", reciprocity, 1e-9),
        ] {
            check(
                value.is_finite() && value <= bound,
                format!("{name}={value:.6e} exceeds {bound}"),
            )?;
            result = result.metric(name, value).maximum(name, bound);
        }
        for (name, value, bound) in [
            ("minimum_spatial_rate", rate, 1.8),
            (
                "minimum_fluid_power_fraction",
                minimum_fluid_power_fraction,
                0.01,
            ),
            ("missing_coupling_error", negative_error, 0.9),
        ] {
            check(
                value.is_finite() && value >= bound,
                format!("{name}={value:.6e} below {bound}"),
            )?;
            result = result.metric(name, value).minimum(name, bound);
        }
        Ok(result.metric("coupled_reference_solves",executed as f64).minimum("coupled_reference_solves",9.0)
            .metric("minimum_free_pressure_dofs",free_pressure as f64).minimum("minimum_free_pressure_dofs",1.0)
            .metric("minimum_free_displacement_dofs",free_displacement as f64).minimum("minimum_free_displacement_dofs",1.0)
            .reference("independent closed-form damped cavity/bar waves and interface impedance sum, swapped forward drives and separate fluid/solid dissipation")
            .reference_checksum("analytic:unit-cavity-clamped-bar-exp-plus-iwt:v1")
            .input_checksum("P1-2D-forms-uniform-axial-restriction:n8-16-32:10-20-30Hz:E20000:rho_s2:df1e-5:ds0.2")
            .dimension("uniform transverse mode restricted from native 2D scalar/vector assembly; unit cavity and axial clamped bar, Poisson ratio zero")
            .formulation("native coupled harmonic operator, consistent pressure/solid mass, two dissipative blocks, conservative interface normal coupling")
            .tolerance("pressure/displacement quadrature errors<=0.003 and rates>=1.8; residual/power/physical reciprocity<=1e-9; omitted coupling error>=0.9"))
    };
    match run_reference() {
        Ok(result) => result,
        Err(error) => QaResult::failed("fem.electroacoustics-cavity-bar.v1", error),
    }
}

fn emit_result(result: QaResult) {
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
    if result.status != QaStatus::Passed {
        std::process::exit(1);
    }
}

fn main() {
    let reference_result = duct_reference_gate();
    emit_result(reference_result);
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("assembly_checks", metrics.assembly_checks as f64)
            .minimum("assembly_checks", 5.0)
            .metric("coupled_residual", metrics.coupled_residual)
            .maximum("coupled_residual", TOL_RESIDUAL)
            .metric("pressure_norm", metrics.pressure_norm)
            .minimum("pressure_norm", 1e-12)
            .metric("displacement_norm", metrics.displacement_norm)
            .minimum("displacement_norm", 1e-12)
            .metric("coupling_transfer", metrics.coupling_transfer)
            .minimum("coupling_transfer", MIN_COUPLING_TRANSFER)
            .metric("decoupled_fluid_leakage", metrics.decoupled_fluid_leakage)
            .maximum("decoupled_fluid_leakage", TOL_DECOUPLED)
            .metric("decoupled_solid_error", metrics.decoupled_solid_error)
            .maximum("decoupled_solid_error", TOL_DECOUPLED)
            .metric("reciprocity_error", metrics.reciprocity_error)
            .maximum("reciprocity_error", TOL_RECIPROCITY)
            .metric("input_power", metrics.input_power)
            .minimum("input_power", 0.0)
            .metric("dissipated_power", metrics.dissipated_power)
            .minimum("dissipated_power", 0.0)
            .metric("power_error", metrics.power_error)
            .maximum("power_error", TOL_POWER)
            .reference("decoupled block solves, physical scaled reciprocity, and power balance")
            .reference_checksum("analytic/deterministic-fixture:fem-electroacoustics-validation-v1")
            .dimension("unit square, 25 fluid plus 50 solid DOFs, 5 Hz interface drive")
            .formulation("harmonic coupled block with lumped solid mass and damping")
            .tolerance("residual 1e-10; decoupled 1e-10; reciprocity 1e-9; power 1e-9")
            .input_checksum("mesh:unit-square-tri-n4:coupled-drive-5Hz:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("electroacoustics QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
