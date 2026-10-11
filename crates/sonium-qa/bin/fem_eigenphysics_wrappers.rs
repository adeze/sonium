use sonium_fem::assembly::TripletMatrix;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::electromagnetics::maxwell::EmMaterial;
use sonium_fem::physics::solid::eigenmodes::{
    AcousticEigenmodeConfig, ElectromagneticEigenmodeConfig, StructuralModalConfig,
    ThermalStabilityConfig, build_acoustic_eigenmode_problem,
    build_electromagnetic_eigenmode_problem, build_linear_buckling_problem,
    build_structural_modal_problem, build_thermal_stability_problem, critical_loads,
    euler_column_buckling_load, thermal_decay_rates,
};
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_fem::physics::thermal::heat_conduction::ThermalMaterial;
use sonium_qa::QaResult;
use sonium_solvers::{CsrMatrix, LinearOperator, LobpcgConfig, ShiftInvertConfig};

const CASE_ID: &str = "fem.eigenphysics-wrappers.v1";
// First cantilever root of cos(beta L) cosh(beta L) = -1.
const BETA1L: f64 = 1.87510406871196;
// Independent relative eigen-residual bound applied to every solved leg:
// ||K x - lambda M x|| / ||K x|| recomputed from the assembled operators.
const EIGEN_RESIDUAL_TOLERANCE: f64 = 1e-6;
// Structural coarse/fine (32x8 vs 64x16) eigenvalue self-convergence bound.
// P1 eigenvalues converge as O(h^2), so the coarse/fine gap is about three
// times the fine-mesh error; the bound holds margin above that.
const STRUCTURAL_SELF_CONVERGENCE_TOLERANCE: f64 = 0.06;
// Cross-theory consistency of the fine-mesh structural frequency against
// Euler-Bernoulli beam theory. The 2D strip carries shear deformation that
// beam theory omits (a few percent at L/h = 5), so the bound covers that
// modeling gap plus residual P1 discretization error.
const STRUCTURAL_BEAM_THEORY_TOLERANCE: f64 = 0.05;
// Buckling eigenvalue accuracy on the 2x2 diagonal pencil.
const BUCKLING_EIGENVALUE_TOLERANCE: f64 = 1e-8;
// Thermal decay-rate accuracy against the analytic rod rate.
const THERMAL_RATE_TOLERANCE: f64 = 0.02;
// Acoustic fundamental accuracy against c/2.
const ACOUSTIC_FREQUENCY_TOLERANCE: f64 = 0.01;
// EM transverse wavenumber accuracy against the TE10-like cutoff pi/a.
// The 16x8 mesh reaches ~5e-4 with clean O(h^2) contraction from the 8x4
// mesh, so the bound holds an order of magnitude of margin.
const EM_WAVENUMBER_TOLERANCE: f64 = 0.005;
// Consistency between the reported EM eigenvalue and the Rayleigh quotient
// of the reported eigenvector.
const EM_RAYLEIGH_TOLERANCE: f64 = 1e-8;
const SPEED_OF_LIGHT: f64 = 299_792_458.0;

fn lobpcg_one() -> LobpcgConfig {
    LobpcgConfig {
        num_eigenvalues: 1,
        max_iterations: 80,
        tolerance: 1.0e-7,
        ..Default::default()
    }
}

/// Independently recomputed relative true residual for one eigenpair.
fn relative_eigen_residual(
    stiffness: &CsrMatrix<f64>,
    mass: &CsrMatrix<f64>,
    lambda: f64,
    vector: &[f64],
) -> f64 {
    let k_x = stiffness.apply(vector);
    let m_x = mass.apply(vector);
    let residual_norm = k_x
        .iter()
        .zip(m_x.iter())
        .map(|(&kv, &mv)| (kv - lambda * mv).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = k_x
        .iter()
        .map(|v| v.powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-300);
    residual_norm / scale
}

fn gate_residual(
    leg: &str,
    stiffness: &CsrMatrix<f64>,
    mass: &CsrMatrix<f64>,
    lambda: f64,
    vector: &[f64],
) -> Result<f64, String> {
    let residual = relative_eigen_residual(stiffness, mass, lambda, vector);
    if !residual.is_finite() || residual >= EIGEN_RESIDUAL_TOLERANCE {
        return Err(format!(
            "{leg} eigen-residual {residual:.2e} exceeds bound {EIGEN_RESIDUAL_TOLERANCE:.1e}"
        ));
    }
    Ok(residual)
}

fn test_structural_modal_wrapper() -> Result<(f64, f64, f64, f64), String> {
    // Euler-Bernoulli reference for a unit-thickness strip
    // (E = 1000, rho = 2, L = 1, h = 0.2); also used as the shift-invert
    // target. LOBPCG stalls on the finer pencils, so the wrapper accuracy
    // legs use the direct shift-invert path (convergence still required).
    let young: f64 = 1000.0;
    let density: f64 = 2.0;
    let length: f64 = 1.0;
    let height: f64 = 0.2;
    let second_moment = height.powi(3) / 12.0;
    let area = height;
    let expected = BETA1L.powi(2) / (2.0 * std::f64::consts::PI * length.powi(2))
        * ((young * second_moment) / (density * area)).sqrt();
    let shift_target = (2.0 * std::f64::consts::PI * expected).powi(2);

    // Coarse and fine cantilever strips; both legs must converge so the
    // self-convergence comparison below is meaningful.
    let mut eigenvalues = Vec::new();
    let mut residuals = Vec::new();
    for &(nx, ny, tag) in &[(32_usize, 8_usize, "coarse"), (64_usize, 16_usize, "fine")] {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, nx, ny);
        let problem = build_structural_modal_problem(
            &mesh,
            IsotropicMaterial::new(1000.0, 0.25),
            &StructuralModalConfig::plane_stress(2.0, vec![1]),
        );
        let config = ShiftInvertConfig {
            target: shift_target,
            num_eigenvalues: 1,
            krylov_dimension: 60,
            tolerance: 1.0e-8,
            breakdown_tolerance: 1.0e-14,
        };
        let result = problem
            .solve_near(&config)
            .map_err(|error| format!("structural {tag} shift-invert failed: {error:?}"))?;
        if !result.converged {
            return Err(format!("structural {tag} shift-invert did not converge"));
        }
        let lambda = result.eigenvalues[0];
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "structural {tag} eigenvalue is not positive: {lambda}"
            ));
        }
        // Sanity: the shift targets the fundamental, so a returned value far
        // above the reference would indicate a wrong (higher) mode.
        if lambda > 4.0 * shift_target {
            return Err(format!(
                "structural {tag} eigenvalue {lambda:.2e} is far from the fundamental shift {shift_target:.2e}"
            ));
        }
        let vector = result.eigenvectors.column_to_vec(0);
        let residual = gate_residual(
            &format!("structural {tag}"),
            &problem.stiffness_like,
            &problem.mass_like,
            lambda,
            &vector,
        )?;
        println!(
            "Structural modal wrapper ({tag}): lambda={lambda:.4e}, residual={residual:.2e}, converged={}",
            result.converged
        );
        eigenvalues.push(lambda);
        residuals.push(residual);
    }

    let problem = build_structural_modal_problem(
        &rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 64, 16),
        IsotropicMaterial::new(1000.0, 0.25),
        &StructuralModalConfig::plane_stress(2.0, vec![1]),
    );
    let frequencies = problem.eigenvalues_to_frequencies_hz(&[eigenvalues[1]]);
    let self_difference = (eigenvalues[1] - eigenvalues[0]).abs() / eigenvalues[1];
    if !self_difference.is_finite() || self_difference >= STRUCTURAL_SELF_CONVERGENCE_TOLERANCE {
        return Err(format!(
            "structural coarse/fine eigenvalue difference {self_difference:.2e} exceeds bound {STRUCTURAL_SELF_CONVERGENCE_TOLERANCE:.1e}"
        ));
    }

    // Cross-theory check against the Euler-Bernoulli reference.
    let beam_error = (frequencies[0] - expected).abs() / expected;
    println!(
        "Structural modal wrapper: f_fine={:.4e} Hz, beam theory={:.4e} Hz, rel diff={:.2e}",
        frequencies[0], expected, beam_error
    );
    if !beam_error.is_finite() || beam_error >= STRUCTURAL_BEAM_THEORY_TOLERANCE {
        return Err(format!(
            "structural frequency deviates {beam_error:.2e} from beam theory (bound {STRUCTURAL_BEAM_THEORY_TOLERANCE:.1e})"
        ));
    }
    Ok((frequencies[0], residuals[1], self_difference, beam_error))
}

fn test_linear_buckling_wrapper() -> Result<(f64, f64), String> {
    let mut elastic = TripletMatrix::new(2, 2);
    elastic.push(0, 0, 4.0);
    elastic.push(1, 1, 9.0);
    let mut geometric = TripletMatrix::new(2, 2);
    geometric.push(0, 0, 1.0);
    geometric.push(1, 1, 1.0);
    let problem = build_linear_buckling_problem(&elastic, &geometric, &[]);
    let result = problem
        .solve(&lobpcg_one())
        .map_err(|error| format!("buckling LOBPCG failed: {error:?}"))?;
    if !result.converged {
        return Err("buckling LOBPCG did not converge".to_string());
    }
    let loads = critical_loads(&result.eigenvalues);
    let euler = euler_column_buckling_load(200.0e9, 1.0e-8, 2.0, 1.0);

    println!(
        "Buckling wrapper: lambda={:.4e}, Euler load={:.4e}",
        loads[0], euler
    );
    let error = (loads[0] - 4.0).abs();
    if !error.is_finite() || error >= BUCKLING_EIGENVALUE_TOLERANCE {
        return Err(format!(
            "buckling eigenvalue error {error:.2e} exceeds bound {BUCKLING_EIGENVALUE_TOLERANCE:.1e}"
        ));
    }
    if !euler.is_finite() || euler <= 0.0 {
        return Err("Euler buckling reference is not positive".to_string());
    }
    let vector = result.eigenvectors.column_to_vec(0);
    let residual = gate_residual(
        "buckling",
        &problem.stiffness_like,
        &problem.mass_like,
        result.eigenvalues[0],
        &vector,
    )?;
    Ok((error, residual))
}

fn test_thermal_stability_wrapper() -> Result<(f64, f64), String> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 8, 1);
    let material = ThermalMaterial::new(2.0, 4.0, 5.0);
    let problem = build_thermal_stability_problem(
        &mesh,
        material,
        &ThermalStabilityConfig {
            degree: PolynomialDegree::P1,
            reaction_coefficient: 0.0,
            fixed_temperature_boundary_markers: vec![1, 2],
        },
    );
    let result = problem
        .solve(&lobpcg_one())
        .map_err(|error| format!("thermal LOBPCG failed: {error:?}"))?;
    if !result.converged {
        return Err("thermal LOBPCG did not converge".to_string());
    }
    let rates = thermal_decay_rates(&result.eigenvalues);
    let expected = material.diffusivity() * std::f64::consts::PI.powi(2);

    println!(
        "Thermal stability wrapper: rate={:.4e}, expected~{:.4e}",
        rates[0], expected
    );
    if !rates[0].is_finite() || rates[0] <= 0.0 {
        return Err("thermal decay rate is not positive".to_string());
    }
    let error = (rates[0] - expected).abs() / expected;
    if error >= THERMAL_RATE_TOLERANCE {
        return Err(format!(
            "thermal rate error {error:.2e} exceeds bound {THERMAL_RATE_TOLERANCE:.1e}"
        ));
    }
    let vector = result.eigenvectors.column_to_vec(0);
    let residual = gate_residual(
        "thermal",
        &problem.stiffness_like,
        &problem.mass_like,
        result.eigenvalues[0],
        &vector,
    )?;
    Ok((error, residual))
}

fn test_acoustic_eigenmode_wrapper() -> Result<(f64, f64), String> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 8, 1);
    let c = 343.0;
    let problem = build_acoustic_eigenmode_problem(
        &mesh,
        &AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: c,
            pressure_release_boundary_markers: vec![1, 2],
        },
    );
    let result = problem
        .solve(&lobpcg_one())
        .map_err(|error| format!("acoustic LOBPCG failed: {error:?}"))?;
    if !result.converged {
        return Err("acoustic LOBPCG did not converge".to_string());
    }
    let frequencies = problem.eigenvalues_to_frequencies_hz(&result.eigenvalues);
    let expected = c / 2.0;

    println!(
        "Acoustic eigenmode wrapper: f={:.4e} Hz, expected~{:.4e} Hz",
        frequencies[0], expected
    );
    if !frequencies[0].is_finite() || frequencies[0] <= 0.0 {
        return Err("acoustic frequency is not positive".to_string());
    }
    let error = (frequencies[0] - expected).abs() / expected;
    if error >= ACOUSTIC_FREQUENCY_TOLERANCE {
        return Err(format!(
            "acoustic frequency error {error:.2e} exceeds bound {ACOUSTIC_FREQUENCY_TOLERANCE:.1e}"
        ));
    }
    let vector = result.eigenvectors.column_to_vec(0);
    let residual = gate_residual(
        "acoustic",
        &problem.stiffness_like,
        &problem.mass_like,
        result.eigenvalues[0],
        &vector,
    )?;
    Ok((error, residual))
}

fn shift_invert_config(target: f64) -> ShiftInvertConfig {
    ShiftInvertConfig {
        target,
        num_eigenvalues: 1,
        krylov_dimension: 60,
        tolerance: 1.0e-8,
        breakdown_tolerance: 1.0e-14,
    }
}

/// Solve the PEC rectangular-cavity transverse mode nearest k = pi/a and
/// return (wavenumber error, residual, solved k, Rayleigh quotient check).
fn solve_em_transverse_mode(nx: usize, ny: usize) -> Result<(f64, f64, f64, f64), String> {
    let width = 1.0;
    let height = 0.5;
    let mesh = rectangular_mesh_triangles(0.0, width, 0.0, height, nx, ny);
    let problem = build_electromagnetic_eigenmode_problem(
        &mesh,
        EmMaterial::vacuum(),
        &ElectromagneticEigenmodeConfig {
            order: 1,
            pec_boundary_markers: vec![1, 2, 3, 4],
        },
    );
    if problem.stiffness_like.num_rows == 0 {
        return Err("EM eigenproblem has no free DOFs".to_string());
    }
    // Lowest transverse mode of the PEC rectangle: k = pi / a. The pencil is
    // K e = omega^2 M e, so the shift is (c k)^2.
    let expected_k = std::f64::consts::PI / width;
    let target = (SPEED_OF_LIGHT * expected_k).powi(2);
    let result = problem
        .solve_near(&shift_invert_config(target))
        .map_err(|error| format!("EM shift-invert failed at {nx}x{ny}: {error:?}"))?;
    if !result.converged {
        return Err(format!("EM shift-invert did not converge at {nx}x{ny}"));
    }
    let lambda = result.eigenvalues[0];
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(format!(
            "EM eigenvalue is not positive at {nx}x{ny}: {lambda}"
        ));
    }
    let solved_k = lambda.sqrt() / SPEED_OF_LIGHT;
    let error = (solved_k - expected_k).abs() / expected_k;
    let vector = result.eigenvectors.column_to_vec(0);
    let residual =
        relative_eigen_residual(&problem.stiffness_like, &problem.mass_like, lambda, &vector);
    // Rayleigh quotient consistency proves the reported eigenvalue belongs to
    // the reported vector rather than being a copied reference value.
    let k_x = problem.stiffness_like.apply(&vector);
    let m_x = problem.mass_like.apply(&vector);
    let rayleigh = vector
        .iter()
        .zip(k_x.iter())
        .map(|(&a, &b)| a * b)
        .sum::<f64>()
        / vector
            .iter()
            .zip(m_x.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            .max(1e-300);
    let rayleigh_check = (rayleigh - lambda).abs() / lambda;
    println!(
        "EM eigenmode wrapper ({nx}x{ny}): k={solved_k:.6e} 1/m, expected={expected_k:.6e}, rel err={error:.2e}, residual={residual:.2e}"
    );
    if !residual.is_finite() || residual >= EIGEN_RESIDUAL_TOLERANCE {
        return Err(format!(
            "EM eigen-residual {residual:.2e} exceeds bound {EIGEN_RESIDUAL_TOLERANCE:.1e} at {nx}x{ny}"
        ));
    }
    Ok((error, residual, solved_k, rayleigh_check))
}

fn test_em_eigenmode_wrapper() -> Result<(f64, f64, f64, f64, f64), String> {
    let (coarse_error, _, _, _) = solve_em_transverse_mode(8, 4)?;
    let (fine_error, fine_residual, solved_k, rayleigh_check) = solve_em_transverse_mode(16, 8)?;
    // Contraction: the fine mesh must improve on the coarse mesh.
    if fine_error >= coarse_error {
        return Err(format!(
            "EM wavenumber error does not contract: coarse={coarse_error:.2e}, fine={fine_error:.2e}"
        ));
    }
    if !fine_error.is_finite() || fine_error >= EM_WAVENUMBER_TOLERANCE {
        return Err(format!(
            "EM wavenumber error {fine_error:.2e} exceeds bound {EM_WAVENUMBER_TOLERANCE:.1e}"
        ));
    }
    if !rayleigh_check.is_finite() || rayleigh_check >= EM_RAYLEIGH_TOLERANCE {
        return Err(format!(
            "EM Rayleigh quotient inconsistent with reported eigenvalue: {rayleigh_check:.2e}"
        ));
    }
    Ok((
        coarse_error,
        fine_error,
        fine_residual,
        solved_k,
        rayleigh_check,
    ))
}

struct WrapperMetrics {
    structural_frequency: f64,
    structural_residual: f64,
    structural_self_difference: f64,
    structural_beam_error: f64,
    buckling_error: f64,
    buckling_residual: f64,
    thermal_error: f64,
    thermal_residual: f64,
    acoustic_error: f64,
    acoustic_residual: f64,
    em_coarse_error: f64,
    em_fine_error: f64,
    em_residual: f64,
    em_wavenumber: f64,
    em_rayleigh_check: f64,
}

fn run() -> Result<WrapperMetrics, String> {
    println!("=== QA: FEM eigenphysics wrappers ===");
    let (
        structural_frequency,
        structural_residual,
        structural_self_difference,
        structural_beam_error,
    ) = test_structural_modal_wrapper()?;
    let (buckling_error, buckling_residual) = test_linear_buckling_wrapper()?;
    let (thermal_error, thermal_residual) = test_thermal_stability_wrapper()?;
    let (acoustic_error, acoustic_residual) = test_acoustic_eigenmode_wrapper()?;
    let (em_coarse_error, em_fine_error, em_residual, em_wavenumber, em_rayleigh_check) =
        test_em_eigenmode_wrapper()?;
    Ok(WrapperMetrics {
        structural_frequency,
        structural_residual,
        structural_self_difference,
        structural_beam_error,
        buckling_error,
        buckling_residual,
        thermal_error,
        thermal_residual,
        acoustic_error,
        acoustic_residual,
        em_coarse_error,
        em_fine_error,
        em_residual,
        em_wavenumber,
        em_rayleigh_check,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("structural_frequency_hz", metrics.structural_frequency)
                .metric("structural_residual", metrics.structural_residual)
                .metric(
                    "structural_self_difference",
                    metrics.structural_self_difference,
                )
                .metric("structural_beam_error", metrics.structural_beam_error)
                .metric("buckling_error", metrics.buckling_error)
                .metric("buckling_residual", metrics.buckling_residual)
                .metric("thermal_error", metrics.thermal_error)
                .metric("thermal_residual", metrics.thermal_residual)
                .metric("acoustic_error", metrics.acoustic_error)
                .metric("acoustic_residual", metrics.acoustic_residual)
                .metric("em_coarse_error", metrics.em_coarse_error)
                .metric("em_fine_error", metrics.em_fine_error)
                .metric("em_residual", metrics.em_residual)
                .metric("em_wavenumber", metrics.em_wavenumber)
                .metric("em_rayleigh_check", metrics.em_rayleigh_check)
                .maximum("structural_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum(
                    "structural_self_difference",
                    STRUCTURAL_SELF_CONVERGENCE_TOLERANCE,
                )
                .maximum("structural_beam_error", STRUCTURAL_BEAM_THEORY_TOLERANCE)
                .maximum("buckling_error", BUCKLING_EIGENVALUE_TOLERANCE)
                .maximum("buckling_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum("thermal_error", THERMAL_RATE_TOLERANCE)
                .maximum("thermal_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum("acoustic_error", ACOUSTIC_FREQUENCY_TOLERANCE)
                .maximum("acoustic_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum("em_fine_error", EM_WAVENUMBER_TOLERANCE)
                .maximum("em_residual", EIGEN_RESIDUAL_TOLERANCE)
                .maximum("em_rayleigh_check", EM_RAYLEIGH_TOLERANCE)
                .tolerance(
                    "every solved leg requires convergence plus an independent eigen-residual gate",
                )
                .seed("fem-eigenphysics-wrappers:cpu:v1");
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
            eprintln!("FEM EIGENPHYSICS WRAPPERS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
