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
//! - Transpose reciprocity identity
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

    // Transpose reciprocity: x[j] from A x = e_i equals y[i] from A^T y = e_j.
    let sense_dof = SENSE_NODE;
    let mut transpose = vec![Complex64::new(0.0, 0.0); total * total];
    for row in 0..total {
        for col in 0..total {
            transpose[col * total + row] = dense[row * total + col];
        }
    }
    let mut reciprocal_load = vec![Complex64::new(0.0, 0.0); total];
    reciprocal_load[sense_dof] = Complex64::new(1.0, 0.0);
    let mut reciprocal_solution = reciprocal_load.clone();
    dense_complex_solve(&mut transpose, &mut reciprocal_solution)?;
    let reciprocity_error = (solution[sense_dof] - reciprocal_solution[drive_dof]).norm()
        / solution[sense_dof].norm().max(1e-30);
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

fn main() {
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
            .reference("decoupled block solves, transpose reciprocity, and power balance")
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
