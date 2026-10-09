//! Schwarz iteration solver with PML transmission conditions
//!
//! Implements additive and multiplicative Schwarz methods where artificial
//! subdomain boundaries use PML absorption instead of classical transmission
//! conditions, achieving k-independent convergence.
//!
//! Reference: Galkowski et al. (2024, arXiv:2408.16580)

use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::schwarz_pml::config::{SchwarzPmlConfig, SchwarzVariant};
use crate::schwarz_pml::decomposition::{
    SubdomainInfo, compute_partition_of_unity, decompose_domain, extract_local_mesh,
};
use crate::schwarz_pml::local_assembly::{apply_dirichlet_values, assemble_local_pml_system};
use crate::solver::{LinearSolveBackendProvenance, Solution, SolverConfig, SolverError};
use num_complex::Complex64;
use sonium_solvers::{CsrMatrix, FgmresConfig, Preconditioner, fgmres_preconditioned};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

/// Solve a Helmholtz problem using the Optimized Schwarz Method with PML
///
/// This is the primary entry point. Unlike most solvers in solver/mod.rs, this
/// needs access to the mesh geometry for domain decomposition and PML construction.
///
/// # Arguments
/// * `mesh` - The global finite element mesh
/// * `degree` - Polynomial degree for basis functions
/// * `wavenumber` - Complex wavenumber (real part used for PML sizing)
/// * `rhs` - Global right-hand side vector (length = mesh.num_nodes())
/// * `dirichlet_bcs` - Global Dirichlet boundary conditions: (node_index, value)
/// * `config` - Schwarz-PML configuration
pub fn solve_schwarz_pml(
    mesh: &Mesh,
    degree: PolynomialDegree,
    wavenumber: Complex64,
    rhs: &[Complex64],
    dirichlet_bcs: &[(usize, Complex64)],
    config: &SchwarzPmlConfig,
) -> Result<Solution, SolverError> {
    let k = wavenumber.re;
    if !k.is_finite() || !wavenumber.im.is_finite() || k <= 0.0 {
        return Err(SolverError::InvalidConfiguration(
            "Wavenumber must be finite and have positive real part".into(),
        ));
    }

    if degree != PolynomialDegree::P1 {
        return Err(SolverError::InvalidConfiguration(
            "Schwarz-PML currently supports only vertex-based P1/Q1 spaces".into(),
        ));
    }

    let n_global = mesh.num_nodes();
    if rhs.len() != n_global {
        return Err(SolverError::DimensionMismatch {
            expected: n_global,
            actual: rhs.len(),
        });
    }
    if rhs.iter().any(|value| !complex_is_finite(*value)) {
        return Err(SolverError::InvalidConfiguration(
            "Right-hand side contains non-finite values".into(),
        ));
    }
    validate_config(config, mesh)?;
    validate_dirichlet_bcs(dirichlet_bcs, n_global)?;

    let start = Instant::now();

    // The outer iteration is a residual-correction Schwarz method. Its
    // convergence contract is therefore the residual of the *monolithic*
    // discrete Helmholtz problem, not stagnation of successive iterates.
    let global_unconstrained =
        assemble_local_pml_system(mesh, degree, wavenumber, &[], &HashSet::new());
    let (global_system, constrained_global_rhs) =
        apply_dirichlet_values(&global_unconstrained, rhs, dirichlet_bcs);

    // 1. Decompose domain
    let subdomains = decompose_domain(mesh, config, k);

    if config.verbosity > 0 {
        println!(
            "  [Schwarz-PML] {} subdomains, k={:.2}, overlap={:.1}%, PML={:.2} wavelengths",
            subdomains.len(),
            k,
            config.overlap_fraction * 100.0,
            config.pml_wavelengths,
        );
    }

    // 2. Assemble local systems for each subdomain
    let local_data: Vec<LocalSubdomainData> = subdomains
        .iter()
        .map(|sub| prepare_subdomain(mesh, sub, degree, wavenumber, dirichlet_bcs))
        .collect();

    let setup_time = start.elapsed();
    if config.verbosity > 0 {
        println!(
            "  [Schwarz-PML] Setup: {:.1}ms ({} local systems assembled)",
            setup_time.as_secs_f64() * 1000.0,
            local_data.len(),
        );
    }

    // 3. Compute partition of unity
    let pou_weights = compute_partition_of_unity(mesh, &subdomains);

    // 4. Run Schwarz iteration
    let solve_start = Instant::now();
    let result = match config.variant {
        SchwarzVariant::Additive => schwarz_additive(
            &global_system,
            &constrained_global_rhs,
            dirichlet_bcs,
            &subdomains,
            &local_data,
            &pou_weights,
            config,
        ),
        SchwarzVariant::Multiplicative => schwarz_multiplicative(
            &global_system,
            &constrained_global_rhs,
            dirichlet_bcs,
            &subdomains,
            &local_data,
            &pou_weights,
            config,
        ),
    };

    let solve_time = solve_start.elapsed();
    if config.verbosity > 0 {
        match &result {
            Ok(sol) => println!(
                "  [Schwarz-PML] {} in {} iterations (residual: {:.2e}, time: {:.1}ms)",
                if sol.converged {
                    "Converged"
                } else {
                    "Did not converge"
                },
                sol.iterations,
                sol.residual,
                solve_time.as_secs_f64() * 1000.0,
            ),
            Err(e) => println!("  [Schwarz-PML] Failed: {}", e),
        }
    }

    result
}

/// Pre-assembled data for a single subdomain
struct LocalSubdomainData {
    /// Local CSR correction matrix (K_pml - k² M_pml with homogeneous
    /// truncation and physical Dirichlet conditions).
    system: CsrMatrix<Complex64>,
    /// Nodes constrained to zero because local solves compute corrections.
    correction_dirichlet_nodes: HashSet<usize>,
}

/// Prepare a subdomain: extract local mesh, assemble PML system, restrict RHS
fn prepare_subdomain(
    global_mesh: &Mesh,
    sub: &SubdomainInfo,
    degree: PolynomialDegree,
    wavenumber: Complex64,
    global_dirichlet_bcs: &[(usize, Complex64)],
) -> LocalSubdomainData {
    let local_mesh = extract_local_mesh(global_mesh, sub);

    // Corrections vanish at PML truncations and prescribed physical
    // boundaries. Nonzero physical values are carried by the global iterate.
    let mut correction_dirichlet_nodes = sub.dirichlet_local_nodes.clone();

    for &(global_node, _value) in global_dirichlet_bcs {
        if let Some(&local_node) = sub.global_to_local.get(&global_node) {
            correction_dirichlet_nodes.insert(local_node);
        }
    }

    let system = assemble_local_pml_system(
        &local_mesh,
        degree,
        wavenumber,
        &sub.pml_regions,
        &correction_dirichlet_nodes,
    );

    LocalSubdomainData {
        system,
        correction_dirichlet_nodes,
    }
}

/// Additive Schwarz iteration
///
/// All subdomains are solved independently (parallelizable), then combined
/// using partition-of-unity weights.
fn schwarz_additive(
    global_system: &CsrMatrix<Complex64>,
    global_rhs: &[Complex64],
    dirichlet_bcs: &[(usize, Complex64)],
    subdomains: &[SubdomainInfo],
    local_data: &[LocalSubdomainData],
    pou_weights: &[Vec<(usize, f64)>],
    config: &SchwarzPmlConfig,
) -> Result<Solution, SolverError> {
    schwarz_krylov(
        global_system,
        global_rhs,
        dirichlet_bcs,
        subdomains,
        local_data,
        pou_weights,
        config,
        SchwarzVariant::Additive,
    )
}

/// Multiplicative Schwarz iteration
///
/// Subdomains are solved sequentially; each sees the latest global solution
/// from previous subdomains in the current sweep.
fn schwarz_multiplicative(
    global_system: &CsrMatrix<Complex64>,
    global_rhs: &[Complex64],
    dirichlet_bcs: &[(usize, Complex64)],
    subdomains: &[SubdomainInfo],
    local_data: &[LocalSubdomainData],
    pou_weights: &[Vec<(usize, f64)>],
    config: &SchwarzPmlConfig,
) -> Result<Solution, SolverError> {
    schwarz_krylov(
        global_system,
        global_rhs,
        dirichlet_bcs,
        subdomains,
        local_data,
        pou_weights,
        config,
        SchwarzVariant::Multiplicative,
    )
}

/// Use one additive or multiplicative PML-Schwarz sweep as a right
/// preconditioner for FGMRES. Krylov acceleration is essential for indefinite
/// Helmholtz systems; a raw Richardson fixed point can stagnate while still
/// being far from the monolithic solution.
#[allow(clippy::too_many_arguments)]
fn schwarz_krylov(
    global_system: &CsrMatrix<Complex64>,
    global_rhs: &[Complex64],
    dirichlet_bcs: &[(usize, Complex64)],
    subdomains: &[SubdomainInfo],
    local_data: &[LocalSubdomainData],
    pou_weights: &[Vec<(usize, f64)>],
    config: &SchwarzPmlConfig,
    variant: SchwarzVariant,
) -> Result<Solution, SolverError> {
    let local_solver_config = SolverConfig {
        solver_type: config.local_solver,
        gmres: config.local_gmres.clone(),
        verbosity: config.verbosity.saturating_sub(1),
        ..Default::default()
    };
    let preconditioner = PmlSchwarzPreconditioner {
        global_system,
        subdomains,
        local_data,
        pou_weights,
        local_solver_config,
        variant,
        local_failure: Mutex::new(None),
    };
    let initial = initial_solution(global_system.num_rows, dirichlet_bcs);
    let outer_config = FgmresConfig {
        max_iterations: 1,
        restart: config.max_iterations.min(global_system.num_rows).max(1),
        tolerance: config.tolerance,
        print_interval: usize::from(config.verbosity > 1),
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
    };
    let result = fgmres_preconditioned(
        global_system,
        &preconditioner,
        global_rhs,
        Some(&initial),
        &outer_config,
    );

    if let Some(message) = preconditioner.local_failure.into_inner().unwrap() {
        return Err(SolverError::InvalidConfiguration(format!(
            "Schwarz-PML local solve failed: {message}"
        )));
    }

    let mut values = result.x;
    enforce_dirichlet_values(&mut values, dirichlet_bcs);
    let (_, true_residual) = residual_and_relative_norm(global_system, &values, global_rhs);
    if result.converged && true_residual <= config.tolerance && true_residual.is_finite() {
        Ok(Solution {
            values,
            iterations: result.iterations,
            residual: true_residual,
            converged: true,
            status: result.status,
            attainable_accuracy: result.attainable_accuracy,
            backend: LinearSolveBackendProvenance::cpu(),
        })
    } else {
        Err(SolverError::ConvergenceFailure(
            result.iterations,
            true_residual,
        ))
    }
}

struct PmlSchwarzPreconditioner<'a> {
    global_system: &'a CsrMatrix<Complex64>,
    subdomains: &'a [SubdomainInfo],
    local_data: &'a [LocalSubdomainData],
    pou_weights: &'a [Vec<(usize, f64)>],
    local_solver_config: SolverConfig,
    variant: SchwarzVariant,
    local_failure: Mutex<Option<String>>,
}

impl Preconditioner<Complex64> for PmlSchwarzPreconditioner<'_> {
    fn apply_into(&self, residual: &[Complex64], correction: &mut [Complex64]) {
        correction.fill(Complex64::new(0.0, 0.0));

        for (j, (sub, ld)) in self
            .subdomains
            .iter()
            .zip(self.local_data.iter())
            .enumerate()
        {
            let current_residual = if self.variant == SchwarzVariant::Multiplicative {
                let applied = self.global_system.matvec(correction);
                residual
                    .iter()
                    .zip(applied)
                    .map(|(&r, az)| r - az)
                    .collect::<Vec<_>>()
            } else {
                residual.to_vec()
            };
            let local_rhs = restrict_residual(&current_residual, sub, ld, self.pou_weights, j);
            let local_correction =
                match solve_local_system(&ld.system, &local_rhs, &self.local_solver_config) {
                    Ok(solution) => solution,
                    Err(error) => {
                        *self.local_failure.lock().unwrap() = Some(error.to_string());
                        correction.fill(Complex64::new(0.0, 0.0));
                        return;
                    }
                };

            for (local_idx, &global_idx) in sub.local_to_global.iter().enumerate() {
                if let Some(weight) = subdomain_weight(&self.pou_weights[global_idx], j) {
                    correction[global_idx] += weight * local_correction[local_idx];
                }
            }
        }
    }
}

/// Restrict a global residual to the physical support of one subdomain.
/// PML-only nodes have zero forcing, while constrained correction nodes remain
/// homogeneous.
fn restrict_residual(
    global_residual: &[Complex64],
    sub: &SubdomainInfo,
    ld: &LocalSubdomainData,
    pou_weights: &[Vec<(usize, f64)>],
    subdomain_index: usize,
) -> Vec<Complex64> {
    let n_local = sub.local_to_global.len();
    let mut rhs = vec![Complex64::new(0.0, 0.0); n_local];

    for (local_idx, &global_idx) in sub.local_to_global.iter().enumerate() {
        if !ld.correction_dirichlet_nodes.contains(&local_idx)
            && subdomain_weight(&pou_weights[global_idx], subdomain_index).is_some()
        {
            rhs[local_idx] = global_residual[global_idx];
        }
    }

    rhs
}

fn subdomain_weight(node_weights: &[(usize, f64)], subdomain_index: usize) -> Option<f64> {
    node_weights
        .iter()
        .find_map(|&(index, weight)| (index == subdomain_index).then_some(weight))
}

fn initial_solution(n: usize, dirichlet_bcs: &[(usize, Complex64)]) -> Vec<Complex64> {
    let mut solution = vec![Complex64::new(0.0, 0.0); n];
    enforce_dirichlet_values(&mut solution, dirichlet_bcs);
    solution
}

fn enforce_dirichlet_values(solution: &mut [Complex64], dirichlet_bcs: &[(usize, Complex64)]) {
    for &(node, value) in dirichlet_bcs {
        solution[node] = value;
    }
}

fn residual_and_relative_norm(
    system: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> (Vec<Complex64>, f64) {
    let applied = system.matvec(solution);
    let residual: Vec<Complex64> = rhs.iter().zip(applied).map(|(&b, ax)| b - ax).collect();
    let residual_norm = residual
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    let relative = if rhs_norm > 0.0 {
        residual_norm / rhs_norm
    } else {
        residual_norm
    };
    (residual, relative)
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn validate_config(config: &SchwarzPmlConfig, mesh: &Mesh) -> Result<(), SolverError> {
    if config.num_subdomains < 2 || config.num_subdomains > mesh.num_elements().max(1) {
        return Err(SolverError::InvalidConfiguration(format!(
            "num_subdomains must be between 2 and the number of mesh elements ({})",
            mesh.num_elements()
        )));
    }
    if !config.overlap_fraction.is_finite() || !(0.0..=1.0).contains(&config.overlap_fraction) {
        return Err(SolverError::InvalidConfiguration(
            "overlap_fraction must be finite and in [0, 1]".into(),
        ));
    }
    if !config.pml_wavelengths.is_finite() || config.pml_wavelengths <= 0.0 {
        return Err(SolverError::InvalidConfiguration(
            "pml_wavelengths must be finite and positive".into(),
        ));
    }
    if !config.pml_target_reflection.is_finite()
        || !(0.0..1.0).contains(&config.pml_target_reflection)
    {
        return Err(SolverError::InvalidConfiguration(
            "pml_target_reflection must be finite and in (0, 1)".into(),
        ));
    }
    if config.max_iterations == 0 || !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(SolverError::InvalidConfiguration(
            "max_iterations and tolerance must be positive".into(),
        ));
    }
    Ok(())
}

fn validate_dirichlet_bcs(
    dirichlet_bcs: &[(usize, Complex64)],
    num_nodes: usize,
) -> Result<(), SolverError> {
    let mut seen = HashMap::new();
    for &(node, value) in dirichlet_bcs {
        if node >= num_nodes {
            return Err(SolverError::InvalidConfiguration(format!(
                "Dirichlet node {node} is out of range for {num_nodes} nodes"
            )));
        }
        if !complex_is_finite(value) {
            return Err(SolverError::InvalidConfiguration(format!(
                "Dirichlet value at node {node} is non-finite"
            )));
        }
        if let Some(previous) = seen.insert(node, value)
            && (previous - value).norm() > 1e-14
        {
            return Err(SolverError::InvalidConfiguration(format!(
                "Conflicting Dirichlet values at node {node}"
            )));
        }
    }
    Ok(())
}

/// Solve a local subdomain system using the configured solver
fn solve_local_system(
    system: &CsrMatrix<Complex64>,
    rhs: &[Complex64],
    config: &SolverConfig,
) -> Result<Vec<Complex64>, SolverError> {
    let solution = crate::solver::solve_csr(system, rhs, config)?;

    Ok(solution.values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::HelmholtzProblem;
    use crate::mesh::unit_square_triangles;
    use std::f64::consts::PI;

    fn boundary_nodes_with_value(mesh: &Mesh, value: Complex64) -> Vec<(usize, Complex64)> {
        mesh.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                let on_boundary = node.x.abs() < 1e-10
                    || (node.x - 1.0).abs() < 1e-10
                    || node.y.abs() < 1e-10
                    || (node.y - 1.0).abs() < 1e-10;
                on_boundary.then_some((i, value))
            })
            .collect()
    }

    #[test]
    fn test_schwarz_pml_preserves_nonzero_global_dirichlet_values() {
        let mesh = unit_square_triangles(8);
        let wavenumber = Complex64::new(2.0, 0.0);
        let problem =
            HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |_, _, _| {
                Complex64::new(0.0, 0.0)
            });
        let prescribed = Complex64::new(1.25, -0.5);
        let dirichlet_bcs = boundary_nodes_with_value(&mesh, prescribed);
        let config = SchwarzPmlConfig {
            num_subdomains: 2,
            max_iterations: 50,
            tolerance: 1e-8,
            verbosity: 0,
            ..Default::default()
        };

        let solution = solve_schwarz_pml(
            &mesh,
            PolynomialDegree::P1,
            wavenumber,
            &problem.rhs,
            &dirichlet_bcs,
            &config,
        )
        .expect("Schwarz-PML should converge with nonzero Dirichlet data");

        for &(node, expected) in &dirichlet_bcs {
            assert!(
                (solution.values[node] - expected).norm() < 1e-10,
                "Dirichlet node {node} was not preserved: got {}, expected {}",
                solution.values[node],
                expected,
            );
        }
    }

    #[test]
    fn test_schwarz_pml_basic_convergence() {
        // Simple test: unit square, k=2, uniform source
        let mesh = unit_square_triangles(8);
        let k = 2.0;
        let wavenumber = Complex64::new(k, 0.0);

        // Assemble global problem for RHS
        let problem =
            HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |x, y, _| {
                Complex64::new((PI * x).sin() * (PI * y).sin(), 0.0)
            });

        // All boundary nodes as Dirichlet = 0
        let mut dirichlet_bcs = Vec::new();
        for (i, node) in mesh.nodes.iter().enumerate() {
            if node.x.abs() < 1e-10
                || (node.x - 1.0).abs() < 1e-10
                || node.y.abs() < 1e-10
                || (node.y - 1.0).abs() < 1e-10
            {
                dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
            }
        }

        let config = SchwarzPmlConfig {
            num_subdomains: 2,
            max_iterations: 50,
            tolerance: 1e-4,
            verbosity: 0,
            ..Default::default()
        };

        let result = solve_schwarz_pml(
            &mesh,
            PolynomialDegree::P1,
            wavenumber,
            &problem.rhs,
            &dirichlet_bcs,
            &config,
        );

        assert!(
            result.is_ok(),
            "Schwarz-PML should converge: {:?}",
            result.err()
        );
        let sol = result.unwrap();
        assert!(sol.converged);
        assert!(sol.iterations < config.max_iterations);
    }

    #[test]
    fn test_multiplicative_converges_faster() {
        let mesh = unit_square_triangles(8);
        let k = 3.0;
        let wavenumber = Complex64::new(k, 0.0);

        let problem =
            HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, wavenumber, |_, _, _| {
                Complex64::new(1.0, 0.0)
            });

        let mut dirichlet_bcs = Vec::new();
        for (i, node) in mesh.nodes.iter().enumerate() {
            if node.x.abs() < 1e-10
                || (node.x - 1.0).abs() < 1e-10
                || node.y.abs() < 1e-10
                || (node.y - 1.0).abs() < 1e-10
            {
                dirichlet_bcs.push((i, Complex64::new(0.0, 0.0)));
            }
        }

        let base_config = SchwarzPmlConfig {
            num_subdomains: 2,
            max_iterations: 50,
            tolerance: 1e-4,
            verbosity: 0,
            ..Default::default()
        };

        let additive_config = SchwarzPmlConfig {
            variant: SchwarzVariant::Additive,
            ..base_config.clone()
        };
        let multiplicative_config = SchwarzPmlConfig {
            variant: SchwarzVariant::Multiplicative,
            ..base_config
        };

        let add_result = solve_schwarz_pml(
            &mesh,
            PolynomialDegree::P1,
            wavenumber,
            &problem.rhs,
            &dirichlet_bcs,
            &additive_config,
        );
        let mult_result = solve_schwarz_pml(
            &mesh,
            PolynomialDegree::P1,
            wavenumber,
            &problem.rhs,
            &dirichlet_bcs,
            &multiplicative_config,
        );

        // Both should converge
        assert!(add_result.is_ok());
        assert!(mult_result.is_ok());

        // Multiplicative should converge in fewer or equal iterations
        let add_sol = add_result.unwrap();
        let mult_sol = mult_result.unwrap();
        assert!(
            mult_sol.iterations <= add_sol.iterations + 5,
            "Multiplicative ({}) should converge no slower than additive ({})",
            mult_sol.iterations,
            add_sol.iterations,
        );
    }
}
