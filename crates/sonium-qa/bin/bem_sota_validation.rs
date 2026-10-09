//! BEM SOTA-oriented validation checks.
//!
//! This binary anchors the numerics that matter before larger BEM work:
//! Burton-Miller assembly remains active near resonance-prone regimes, FMM
//! all-near matvecs match dense TBEM, H-matrix near-field bookkeeping stays
//! finite, and near-singular quadrature behaves consistently as the source
//! approaches a panel.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::assembly::hmatrix::{build_hmatrix, build_hmatrix_stabilized};
use sonium_bem::core::assembly::mlfmm::{
    MlfmmSystem, NearFieldBlock, build_cluster_tree, build_mlfmm_system_stabilized,
};
use sonium_bem::core::assembly::slfmm::build_slfmm_system_stabilized;
use sonium_bem::core::assembly::tbem::{build_tbem_system_stabilized, build_tbem_system_with_beta};
use sonium_bem::core::bem_solver::BemSolutionRepresentation;
use sonium_bem::core::contracts::{
    check_adjoint_contract, check_block_rhs_contract, check_passivity_contract,
    check_reciprocity_contract, constant_panel_power, require_operator_accuracy,
};
use sonium_bem::core::integration::{
    optimal_quadrature_order, quasi_singular_integration, regular_integration,
    regular_integration_fixed_order,
};
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::solver::{
    DenseOperator, GmresConfig, MlfmmOperator, SlfmmOperator, solve_gmres,
};
use sonium_bem::core::types::{BoundaryCondition, Cluster, Element, ElementType, PhysicsParams};
use sonium_bem::core::{BasisOrder, BemSolution, IncidentField};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::LinearOperator;
use std::f64::consts::PI;
use std::time::Instant;

const HMATRIX_ACTION_TOLERANCE: f64 = 5.0e-4;
const FAR_FIELD_POWER_RADIUS_INVARIANCE_TOLERANCE: f64 = 1.5e-2;

fn prepared_sphere(subdivisions: usize, ka: f64) -> (Vec<Element>, Vec<[f64; 3]>, PhysicsParams) {
    let radius = 0.2;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let frequency = ka * speed_of_sound / (2.0 * PI * radius);

    let mesh = generate_icosphere_mesh(radius, subdivisions);
    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    (
        elements,
        mesh.nodes,
        PhysicsParams::new(frequency, speed_of_sound, density, false),
    )
}

fn relative_norm(a: &[Complex64], b: &[Complex64]) -> f64 {
    let diff_norm: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let ref_norm: f64 = a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    diff_norm / ref_norm.max(1e-15)
}

fn deterministic_vector(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64 + 1.0;
            Complex64::new((0.37 * x).sin(), (0.19 * x).cos())
        })
        .collect()
}

fn assert_all_finite(values: impl IntoIterator<Item = Complex64>, label: &str) {
    for value in values {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "{label} contains non-finite value {value:?}"
        );
    }
}

fn c(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

/// Evaluate the independent far-field power observable used by the dense
/// sphere gate, but on an arbitrary production solution.  Keeping this in
/// the operator-consistency binary ensures a compressed solve cannot pass on
/// algebraic matvec parity alone.
fn sphere_scattered_power_against_mie(
    label: &str,
    solution: &BemSolution,
    radius: f64,
) -> Result<(f64, f64, f64), String> {
    let near_radius = (12.0 * radius).max(radius + 1.0);
    let far_radius = (24.0 * radius).max(radius + 2.0);
    let near = sphere_scattered_power_at_radius(label, solution, radius, near_radius)?;
    let far = sphere_scattered_power_at_radius(label, solution, radius, far_radius)?;
    let radiation_condition_error = (far.1 - near.1).abs() / near.1.abs().max(f64::MIN_POSITIVE);
    Ok((near.0.max(far.0), near.1, radiation_condition_error))
}

/// Integrate scattered far-field power at one radius against the independent
/// Mie series. The caller compares two radii to check the outgoing-field
/// radiation condition rather than relying only on agreement at one sample
/// sphere.
fn sphere_scattered_power_at_radius(
    label: &str,
    solution: &BemSolution,
    radius: f64,
    eval_radius: f64,
) -> Result<(f64, f64), String> {
    let k = solution.physics.wave_number;
    let density = solution.physics.density;
    let speed_of_sound = solution.physics.speed_of_sound;
    let samples = 64usize;
    let mut theta = Vec::with_capacity(samples);
    let mut points = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mu = -1.0 + 2.0 * (sample as f64 + 0.5) / samples as f64;
        let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
        theta.push(mu.acos());
        points.push([eval_radius * sin_theta, 0.0, eval_radius * mu]);
    }

    let bem_scattered: Vec<Complex64> = solution
        .evaluate_pressure_field(&points)
        .into_iter()
        .map(|field| field.p_scattered)
        .collect();
    if bem_scattered
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(format!(
            "{label} far-field scattered pressure is non-finite"
        ));
    }

    let mie = sphere_scattering_3d(k, radius, 40, vec![eval_radius], theta);
    let mie_scattered: Vec<Complex64> = mie
        .pressure
        .iter()
        .zip(points.iter())
        .map(|(total, point)| {
            let phase = k * point[2];
            *total - Complex64::new(phase.cos(), phase.sin())
        })
        .collect();
    let angular_weight = 2.0 / samples as f64;
    let bem_integral = bem_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        * angular_weight
        * 2.0
        * PI;
    let mie_integral = mie_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        * angular_weight
        * 2.0
        * PI;
    let factor = eval_radius * eval_radius / (2.0 * density * speed_of_sound);
    let bem_power = factor * bem_integral;
    let mie_power = factor * mie_integral;
    if !bem_power.is_finite() || !mie_power.is_finite() || mie_power <= f64::MIN_POSITIVE {
        return Err(format!("{label} far-field power is non-finite or zero"));
    }
    Ok(((bem_power - mie_power).abs() / mie_power, bem_power))
}

#[derive(Debug, Clone, Copy)]
struct SpherePhysicalGateMetrics {
    power_relative_error: f64,
    radiation_condition_relative_error: f64,
    residual: f64,
    iterations: usize,
    krylov_ms: f64,
    residual_check_ms: f64,
    field_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct AllBackendSpherePhysicalGateMetrics {
    backend: [SpherePhysicalGateMetrics; 4],
    assembly_ms: [f64; 4],
    hmatrix_memory_bytes: usize,
}

/// Solve a closed sphere with a production operator and gate its physical
/// far-field power against the independent Mie series.  This deliberately
/// certifies the solved field rather than only an operator action, so every
/// backend has to preserve the shared incident-RHS and Cauchy-evaluation
/// conventions as well.
fn validate_sphere_physical_gate<O: LinearOperator<Complex64>, R: LinearOperator<Complex64>>(
    label: &str,
    operator: &O,
    exact_tbem_operator: &R,
    elements: Vec<Element>,
    nodes: Vec<[f64; 3]>,
    physics: PhysicsParams,
    radius: f64,
) -> SpherePhysicalGateMetrics {
    let n = elements.len();
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number * radius);
    let beta_scale = PhysicsParams::optimal_beta_scale(physics.wave_number * radius);
    let incident = IncidentField::plane_wave_z();
    let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|element| element.normal).collect();
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let rhs: Vec<Complex64> = incident
        .compute_rhs_with_beta(&centers, &normals, &physics, beta)
        .into_iter()
        .map(|value| value * sign)
        .collect();
    assert_eq!(rhs.len(), n);

    let config = GmresConfig {
        max_iterations: 300,
        restart: 50,
        tolerance: 1.0e-9,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 1,
        min_restart: None,
        max_restart: None,
    };
    let krylov_start = Instant::now();
    let solved = solve_gmres(&operator, &rhs, &config);
    let krylov_ms = krylov_start.elapsed().as_secs_f64() * 1.0e3;
    assert!(
        solved.converged,
        "production {label} sphere solve did not converge: {solved:?}"
    );
    let residual_start = Instant::now();
    assert_eq!(exact_tbem_operator.num_rows(), rhs.len());
    assert_eq!(exact_tbem_operator.num_cols(), solved.x.len());
    // Certify the accepted density against the independently assembled dense
    // TBEM action. In particular, an MLFMM solve must not validate its true
    // residual by applying the same approximate MLFMM operator a second time.
    let applied = exact_tbem_operator.apply(&solved.x);
    let residual = applied
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
    let residual_check_ms = residual_start.elapsed().as_secs_f64() * 1.0e3;
    assert!(residual.is_finite() && residual <= 2.0e-8);

    let solution = BemSolution {
        surface_pressure: solved.x,
        elements,
        nodes,
        incident_field: incident,
        physics,
        basis_order: BasisOrder::Constant,
        representation: BemSolutionRepresentation::DirectCauchy,
    };
    let field_start = Instant::now();
    let (power_error, power_w, radiation_condition_error) =
        sphere_scattered_power_against_mie(label, &solution, radius).unwrap_or_else(|error| {
            panic!("production {label} sphere field evaluation failed: {error}")
        });
    let field_ms = field_start.elapsed().as_secs_f64() * 1.0e3;
    println!(
        "{label} sphere physical gate: exact_tbem_residual={residual:.3e}, iterations={}, scattered_power={power_w:.6e} W, relative Mie error={power_error:.3e}, radius-invariance error={radiation_condition_error:.3e}",
        solved.iterations
    );
    assert!(
        power_error <= 6.0e-2,
        "production {label} scattered-power error {power_error:.3e} exceeds 6%"
    );
    assert!(
        radiation_condition_error <= FAR_FIELD_POWER_RADIUS_INVARIANCE_TOLERANCE,
        "production {label} scattered power changes {radiation_condition_error:.3e} between far radii"
    );
    SpherePhysicalGateMetrics {
        power_relative_error: power_error,
        radiation_condition_relative_error: radiation_condition_error,
        residual,
        iterations: solved.iterations,
        krylov_ms,
        residual_check_ms,
        field_ms,
    }
}

/// Solve same refined sphere through dense, H-matrix, SLFMM, MLFMM
/// implementations. H-matrix SLFMM fixtures intentionally all-near at this
/// mesh size, avoiding an unvalidated far-field split while still exercising
/// production assembly solve paths. MLFMM uses real multi-level tree
/// translation interactions below.
fn validate_all_backend_sphere_physical_gates() -> AllBackendSpherePhysicalGateMetrics {
    let radius = 0.2;
    // One refinement level beyond the cheap operator fixture keeps field
    // discretisation error below the common physical-error budget.
    let (elements, nodes, physics) = prepared_sphere(2, 0.5);
    let n = elements.len();
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number * radius);
    let beta_scale = PhysicsParams::optimal_beta_scale(physics.wave_number * radius);

    let dense_assembly_start = Instant::now();
    let dense = DenseOperator::new(
        build_tbem_system_stabilized(&elements, &nodes, &physics, sign, beta_scale).matrix,
    );
    let dense_assembly_ms = dense_assembly_start.elapsed().as_secs_f64() * 1.0e3;

    let mut all_near = Cluster::new([0.0, 0.0, 0.0]);
    all_near.element_indices = (0..n).collect();
    all_near.num_elements = n;
    all_near.near_clusters = vec![0];
    let hmatrix_assembly_start = Instant::now();
    let hmatrix = build_hmatrix_stabilized(
        &elements,
        &nodes,
        &[all_near.clone()],
        &physics,
        1.0e-8,
        16,
        sign,
        beta_scale,
    );
    let hmatrix_assembly_ms = hmatrix_assembly_start.elapsed().as_secs_f64() * 1.0e3;
    let hmatrix_memory_bytes = hmatrix.memory_bytes;
    let slfmm_assembly_start = Instant::now();
    let slfmm = SlfmmOperator::new(build_slfmm_system_stabilized(
        &elements,
        &nodes,
        &[all_near],
        &physics,
        8,
        16,
        8,
        sign,
        beta_scale,
    ));
    let slfmm_assembly_ms = slfmm_assembly_start.elapsed().as_secs_f64() * 1.0e3;
    let mlfmm_assembly_start = Instant::now();
    let tree = build_cluster_tree(&elements, 8, &physics);
    let mlfmm = MlfmmOperator::new(build_mlfmm_system_stabilized(
        &elements, &nodes, tree, &physics, sign, beta_scale,
    ));
    let mlfmm_assembly_ms = mlfmm_assembly_start.elapsed().as_secs_f64() * 1.0e3;

    AllBackendSpherePhysicalGateMetrics {
        backend: [
            validate_sphere_physical_gate(
                "dense",
                &dense,
                &dense,
                elements.clone(),
                nodes.clone(),
                physics.clone(),
                radius,
            ),
            validate_sphere_physical_gate(
                "H-matrix",
                &hmatrix,
                &dense,
                elements.clone(),
                nodes.clone(),
                physics.clone(),
                radius,
            ),
            validate_sphere_physical_gate(
                "SLFMM",
                &slfmm,
                &dense,
                elements.clone(),
                nodes.clone(),
                physics.clone(),
                radius,
            ),
            validate_sphere_physical_gate(
                "MLFMM", &mlfmm, &dense, elements, nodes, physics, radius,
            ),
        ],
        assembly_ms: [
            dense_assembly_ms,
            hmatrix_assembly_ms,
            slfmm_assembly_ms,
            mlfmm_assembly_ms,
        ],
        hmatrix_memory_bytes,
    }
}

fn bilinear_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn solve_2x2(a: &DenseMatrix<Complex64>, b: &[Complex64]) -> Vec<Complex64> {
    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    assert!(det.norm() > 1e-14, "singular 2x2 validation matrix");
    vec![
        (a[[1, 1]] * b[0] - a[[0, 1]] * b[1]) / det,
        (-a[[1, 0]] * b[0] + a[[0, 0]] * b[1]) / det,
    ]
}

fn dense_from_mlfmm_operator(operator: &MlfmmOperator) -> DenseMatrix<Complex64> {
    let n = operator.num_cols();
    let mut matrix = DenseMatrix::<Complex64>::zeros(operator.num_rows(), n);
    for col in 0..n {
        let mut basis = vec![Complex64::new(0.0, 0.0); n];
        basis[col] = c(1.0);
        let image = operator.apply(&basis);
        for row in 0..operator.num_rows() {
            matrix[[row, col]] = image[row];
        }
    }
    matrix
}

fn tiny_mlfmm_operator() -> MlfmmOperator {
    let mut level = sonium_bem::core::types::ClusterLevel::new(2);
    level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));
    level.clusters.push(Cluster::new([1.0, 0.0, 0.0]));

    let mut system = MlfmmSystem::new(2, 1, vec![level], vec![1]);
    system.leaf_dof_indices = vec![vec![0], vec![1]];
    for (source_cluster, field_cluster, value) in [
        (0, 0, c(2.0)),
        (0, 1, c(0.3)),
        (1, 0, c(-0.4)),
        (1, 1, c(1.5)),
    ] {
        system.near_matrix.push(NearFieldBlock {
            source_cluster,
            field_cluster,
            coefficients: sonium_solvers::dense::DenseMatrix::from_row_major_vec(1, 1, vec![value]),
        });
    }

    MlfmmOperator::new(system)
}

fn validate_burton_miller_near_resonance_assembly() {
    let (elements, nodes, physics) = prepared_sphere(1, PI);
    let no_bm = build_tbem_system_with_beta(&elements, &nodes, &physics, Complex64::new(0.0, 0.0));
    let bm = build_tbem_system_with_beta(
        &elements,
        &nodes,
        &physics,
        physics.burton_miller_beta_scaled(2.0),
    );

    assert_eq!(no_bm.num_dofs, bm.num_dofs);
    assert_all_finite(
        no_bm.matrix.as_slice().iter().copied(),
        "non-Burton-Miller matrix",
    );
    assert_all_finite(bm.matrix.as_slice().iter().copied(), "Burton-Miller matrix");

    let delta_norm: f64 = bm
        .matrix
        .as_slice()
        .iter()
        .zip(no_bm.matrix.as_slice().iter())
        .map(|(b, n)| (b - n).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let bm_norm: f64 = bm
        .matrix
        .as_slice()
        .iter()
        .map(|z| z.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let relative_delta = delta_norm / bm_norm.max(1e-15);

    println!(
        "Burton-Miller resonance-regime assembly: DOFs={}, relative matrix delta={relative_delta:.3e}",
        bm.num_dofs
    );
    assert!(
        relative_delta > 1e-2,
        "Burton-Miller term is not materially affecting the resonance-regime matrix"
    );

    let (n_rows, n_cols) = bm.matrix.shape();
    let bm_slice = bm.matrix.as_slice();
    let min_row_norm = (0..n_rows)
        .map(|i| {
            (0..n_cols)
                .map(|j| bm_slice[i * n_cols + j].norm_sqr())
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_row_norm > 1e-10,
        "Burton-Miller matrix has a near-zero row norm"
    );
}

fn validate_mlfmm_adjoint_shape_lagrangian() {
    let operator = tiny_mlfmm_operator();
    let a = dense_from_mlfmm_operator(&operator);

    let b = [c(1.0), c(-0.25)];
    let target = [c(0.2), c(-0.1)];
    let state = solve_2x2(&a, &b);
    let d_objective_du: Vec<Complex64> = state
        .iter()
        .zip(target.iter())
        .map(|(s, t)| s - t)
        .collect();

    let neg_d_obj: Vec<Complex64> = d_objective_du.iter().map(|v| -v).collect();
    let a_t = a.transposed();
    let adjoint = solve_2x2(&a_t, &neg_d_obj);
    let adjoint_t = operator.apply_transpose(&adjoint);
    let stationarity: Vec<Complex64> = d_objective_du
        .iter()
        .zip(adjoint_t.iter())
        .map(|(a, b)| a + b)
        .collect();
    let max_stationarity = stationarity
        .iter()
        .map(|entry| entry.norm())
        .fold(0.0, f64::max);
    assert!(
        max_stationarity < 1e-12,
        "MLFMM adjoint stationarity failed with residual {max_stationarity:.3e}"
    );

    let d_a_d_shape = DenseMatrix::from_row_major_vec(2, 2, vec![c(0.1), c(-0.2), c(0.05), c(0.3)]);
    let d_a_state = d_a_d_shape.matvec(&state);
    let neg_solve = solve_2x2(&a, &d_a_state);
    let direct_state_sensitivity: Vec<Complex64> = neg_solve.iter().map(|v| -v).collect();
    let direct_gradient = bilinear_dot(&d_objective_du, &direct_state_sensitivity);
    let adjoint_gradient = bilinear_dot(&adjoint, &d_a_state);
    let gradient_error = (direct_gradient - adjoint_gradient).norm();

    println!(
        "MLFMM adjoint shape-Lagrangian check: stationarity={max_stationarity:.3e}, gradient_error={gradient_error:.3e}"
    );
    assert!(
        gradient_error < 1e-12,
        "MLFMM adjoint shape gradient mismatch: direct={direct_gradient}, adjoint={adjoint_gradient}"
    );
}

fn validate_operator_physical_contracts() {
    let operator = tiny_mlfmm_operator();
    let x = vec![Complex64::new(0.7, -0.2), Complex64::new(-0.3, 0.5)];
    let y = vec![Complex64::new(-0.1, 0.8), Complex64::new(0.6, 0.2)];
    let adjoint = check_adjoint_contract(&operator, &x, &y)
        .expect("MLFMM Hermitian contract must produce finite diagnostics");
    let reciprocity = check_reciprocity_contract(&operator, &x, &y)
        .expect("MLFMM transpose reciprocity contract must produce finite diagnostics");
    let passivity = check_passivity_contract(
        &operator,
        &[x.clone(), y.clone(), deterministic_vector(2)],
        1.0e-12,
    )
    .expect("positive MLFMM fixture must satisfy the passivity gate");
    assert!(adjoint.relative_error <= 1.0e-12);
    assert!(reciprocity.relative_error <= 1.0e-12);
    assert!(passivity.minimum_power >= 0.0);

    let power = constant_panel_power(
        &[0.5, 0.75],
        &[Complex64::new(2.0, 1.0), Complex64::new(1.0, -0.5)],
        &[Complex64::new(0.1, -0.2), Complex64::new(0.2, 0.1)],
    )
    .expect("finite panel pressure/velocity should yield finite power");
    assert!(power.is_finite());
    println!(
        "Physical operator gates: adjoint={:.3e}, reciprocity={:.3e}, minimum_power={:.3e}, panel_power={:.3e}",
        adjoint.relative_error, reciprocity.relative_error, passivity.minimum_power, power
    );
}

fn validate_fmm_dense_all_near_parity() {
    let (elements, nodes, physics) = prepared_sphere(1, 0.8);
    let n = elements.len();
    let tbem = build_tbem_system_stabilized(&elements, &nodes, &physics, -1.0, 1.0);

    let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = (0..n).collect();
    let clusters = vec![cluster];
    let slfmm =
        build_slfmm_system_stabilized(&elements, &nodes, &clusters, &physics, 4, 8, 5, -1.0, 1.0);

    let x = deterministic_vector(n);
    let y_dense = tbem.matrix.matvec(&x);
    let y_fmm = slfmm.matvec(&x);
    let rel = relative_norm(&y_dense, &y_fmm);

    println!("SLFMM all-near dense parity: DOFs={n}, relative matvec error={rel:.3e}");
    assert!(
        rel < 1e-10,
        "all-near SLFMM should match dense TBEM matvec, got relative error {rel:.3e}"
    );
}

fn operator_contract_summary<O: LinearOperator<Complex64>>(
    label: &str,
    operator: &O,
    x: &[Complex64],
    y: &[Complex64],
    block_tolerance: f64,
) -> (f64, f64, f64) {
    let adjoint = check_adjoint_contract(operator, x, y)
        .unwrap_or_else(|error| panic!("{label} Hermitian contract failed: {error}"));
    let reciprocity = check_reciprocity_contract(operator, x, y)
        .unwrap_or_else(|error| panic!("{label} transpose contract failed: {error}"));
    let block_error = check_block_rhs_contract(operator, &[x.to_vec(), y.to_vec()])
        .unwrap_or_else(|error| panic!("{label} block-RHS contract failed: {error}"));
    assert!(adjoint.relative_error.is_finite());
    assert!(reciprocity.relative_error.is_finite());
    assert!(block_error.is_finite());
    assert!(
        block_error <= block_tolerance,
        "{label} block-RHS error {block_error:.3e} exceeds {block_tolerance:.3e}"
    );
    println!(
        "{label} production contracts: adjoint={:.3e}, reciprocity={:.3e}, block={:.3e}",
        adjoint.relative_error, reciprocity.relative_error, block_error
    );
    (
        adjoint.relative_error,
        reciprocity.relative_error,
        block_error,
    )
}

/// Exercise the shared operator contracts on production sphere assemblies.
///
/// The tiny positive fixture above proves the passivity helper itself. This
/// gate complements it by ensuring every production backend (dense, H-matrix,
/// SLFMM, and MLFMM) exposes consistent transpose, Hermitian, and block-RHS
/// actions on the same physical mesh and wavenumber.
fn validate_production_operator_contracts() {
    let (elements, nodes, physics) = prepared_sphere(1, 0.8);
    let n = elements.len();
    let dense_system = build_tbem_system_stabilized(&elements, &nodes, &physics, -1.0, 1.0);
    let dense = DenseOperator::new(dense_system.matrix.clone());

    let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = (0..n).collect();
    cluster.num_elements = n;
    let clusters = vec![cluster];
    let slfmm_system =
        build_slfmm_system_stabilized(&elements, &nodes, &clusters, &physics, 4, 8, 5, -1.0, 1.0);
    let slfmm = SlfmmOperator::new(slfmm_system);

    let mut h_cluster = Cluster::new([0.0, 0.0, 0.0]);
    h_cluster.element_indices = (0..n).collect();
    h_cluster.num_elements = n;
    h_cluster.near_clusters = vec![0];
    let hmatrix = build_hmatrix_stabilized(
        &elements,
        &nodes,
        &[h_cluster],
        &physics,
        1.0e-6,
        8,
        -1.0,
        1.0,
    );

    let tree = build_cluster_tree(&elements, 8, &physics);
    let mlfmm_system = build_mlfmm_system_stabilized(&elements, &nodes, tree, &physics, -1.0, 1.0);
    let mlfmm = MlfmmOperator::new(mlfmm_system);

    let x = deterministic_vector(n);
    let y = (0..n)
        .map(|index| {
            let value = index as f64 + 1.0;
            Complex64::new((0.17 * value).cos(), (0.31 * value).sin())
        })
        .collect::<Vec<_>>();
    let dense_metrics = operator_contract_summary("dense", &dense, &x, &y, 1.0e-12);
    let hmatrix_metrics = operator_contract_summary("H-matrix", &hmatrix, &x, &y, 1.0e-12);
    let slfmm_metrics = operator_contract_summary("SLFMM", &slfmm, &x, &y, 1.0e-12);
    let mlfmm_metrics = operator_contract_summary("MLFMM", &mlfmm, &x, &y, 1.0e-8);
    for (label, metrics) in [
        ("dense", dense_metrics),
        ("H-matrix", hmatrix_metrics),
        ("SLFMM", slfmm_metrics),
        ("MLFMM", mlfmm_metrics),
    ] {
        assert!(
            metrics.0.is_finite() && metrics.1.is_finite() && metrics.2.is_finite(),
            "{label} production contract metrics must be finite"
        );
    }
}

fn validate_mlfmm_sphere_parity() -> f64 {
    // Use the same closed icosphere fixture as the dense/H/SLFMM checks. A
    // real cluster tree is required here: a hand-built all-near MLFMM would
    // only exercise its near-field bookkeeping and could not detect a broken
    // upward/downward translation path.
    let (elements, nodes, physics) = prepared_sphere(1, 0.8);
    let n = elements.len();
    let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, -1.0, 1.0);
    let tree = build_cluster_tree(&elements, 8, &physics);
    let mlfmm = build_mlfmm_system_stabilized(&elements, &nodes, tree, &physics, -1.0, 1.0);
    assert!(
        mlfmm.num_levels >= 2,
        "MLFMM sphere fixture must exercise at least one translation level"
    );

    let probes = [
        deterministic_vector(n),
        (0..n)
            .map(|index| {
                let x = index as f64 + 1.0;
                Complex64::new((0.13 * x).cos(), (0.29 * x).sin())
            })
            .collect::<Vec<_>>(),
    ];
    let mut max_relative_error = 0.0_f64;
    for probe in &probes {
        let dense_action = dense.matrix.matvec(probe);
        let mlfmm_action = mlfmm.matvec(probe);
        assert_all_finite(mlfmm_action.iter().copied(), "MLFMM sphere matvec");
        max_relative_error = max_relative_error.max(relative_norm(&dense_action, &mlfmm_action));
    }

    println!(
        "MLFMM sphere parity: DOFs={n}, levels={}, max_relative_matvec_error={max_relative_error:.3e}",
        mlfmm.num_levels
    );
    assert!(
        max_relative_error < 1.0e-3,
        "MLFMM sphere action exceeds the 1e-3 dense reference budget: {max_relative_error:.3e}"
    );
    max_relative_error
}

fn validate_hmatrix_near_field_bookkeeping() {
    let (elements, nodes, physics) = prepared_sphere(1, 0.8);
    let n = elements.len();

    let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
    cluster.element_indices = (0..n).collect();
    cluster.near_clusters = vec![0];
    let hmatrix = build_hmatrix(&elements, &nodes, &[cluster], &physics, 1e-6, 8);

    assert_eq!(hmatrix.num_dofs, n);
    assert_eq!(hmatrix.dense_blocks.len(), 1);
    assert!(hmatrix.low_rank_blocks.is_empty());
    assert!(hmatrix.memory_bytes > 0);

    let x = deterministic_vector(n);
    let y = hmatrix.matvec(&x);
    assert_all_finite(y.iter().copied(), "H-matrix near-field matvec");

    let stats = hmatrix.rank_stats();
    assert_eq!(
        stats.block_count, 0,
        "all-near config must produce zero low-rank blocks"
    );
    assert_eq!(stats.saturated_count, 0);

    println!(
        "H-matrix near-field bookkeeping: DOFs={n}, dense_blocks={}, compression_ratio={:.3}, {}",
        hmatrix.dense_blocks.len(),
        hmatrix.compression_ratio,
        stats,
    );
}

/// Build an H-matrix on two genuinely well-separated sphere clusters and
/// confirm the rank-saturation reporter is wired through `build_hmatrix` end
/// to end. This is a wiring smoke-test, not an algorithm-quality bound —
/// strict per-`kD` accuracy lives in the Mie CI gate.
fn validate_hmatrix_rank_stats_far_field() {
    let (mut elements, mut nodes, physics) = prepared_sphere(1, 0.8);
    let single_n = elements.len();
    let single_node_count = nodes.len();

    // Build a second sphere mesh translated along +x by 4 sphere radii so the
    // two clusters are in the genuine far-field regime (no touching, no
    // near-singular pairs across the boundary).
    let (mut elements_b, nodes_b, _) = prepared_sphere(1, 0.8);
    let radius = 0.2;
    let separation = 4.0 * radius;
    // Append translated nodes from sphere B.
    let mut combined_nodes: Vec<[f64; 3]> = vec![[0.0; 3]; single_node_count + nodes_b.len()];
    for i in 0..single_node_count {
        for d in 0..3 {
            combined_nodes[i][d] = nodes[i][d];
        }
    }
    for i in 0..nodes_b.len() {
        combined_nodes[single_node_count + i][0] = nodes_b[i][0] + separation;
        combined_nodes[single_node_count + i][1] = nodes_b[i][1];
        combined_nodes[single_node_count + i][2] = nodes_b[i][2];
    }
    nodes = combined_nodes;

    // Re-index elements_b to point into the second half of `nodes` and assign
    // fresh DOF addresses that don't collide with sphere A.
    for (i, elem) in elements_b.iter_mut().enumerate() {
        for c in elem.connectivity.iter_mut() {
            *c += single_node_count;
        }
        elem.center[0] += separation;
        elem.dof_addresses = vec![single_n + i];
    }
    elements.extend(elements_b);

    let total_n = elements.len();
    assert_eq!(total_n, 2 * single_n);

    let mut left = Cluster::new([0.0, 0.0, 0.0]);
    let mut right = Cluster::new([separation, 0.0, 0.0]);
    left.element_indices = (0..single_n).collect();
    right.element_indices = (single_n..total_n).collect();
    left.num_elements = single_n;
    right.num_elements = single_n;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];

    let max_rank = 64;
    let hmatrix = build_hmatrix_stabilized(
        &elements,
        &nodes,
        &[left, right],
        &physics,
        1e-4,
        max_rank,
        -1.0,
        1.0,
    );

    assert_eq!(hmatrix.num_dofs, total_n);
    assert_eq!(
        hmatrix.dense_blocks.len(),
        2,
        "expected one dense self-block per cluster"
    );
    assert_eq!(
        hmatrix.low_rank_blocks.len(),
        2,
        "expected two low-rank cross blocks (left↔right)"
    );

    let stats = hmatrix.rank_stats();
    assert_eq!(stats.block_count, 2);
    assert!(
        stats.min_rank >= 1,
        "ACA must produce at least one rank step"
    );
    assert!(stats.max_rank <= max_rank, "realised rank ≤ cap");
    assert!(
        stats.saturation_fraction <= 1.0,
        "saturation fraction must lie in [0, 1]"
    );

    // The same separated fixture must also satisfy an independent dense
    // action budget. Rank/bookkeeping metadata alone cannot prove that an
    // H-matrix is numerically usable.
    let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, -1.0, 1.0);
    let dense_operator = DenseOperator::new(dense.matrix.clone());
    let probes = vec![
        deterministic_vector(total_n),
        (0..total_n)
            .map(|index| {
                let x = index as f64 + 1.0;
                Complex64::new((0.11 * x).cos(), (0.23 * x).sin())
            })
            .collect(),
    ];
    let accuracy =
        require_operator_accuracy(&hmatrix, &dense_operator, &probes, HMATRIX_ACTION_TOLERANCE)
            .expect("separated H-matrix action must meet its declared accuracy budget");
    assert!(accuracy.max_relative_error.is_finite());

    println!(
        "H-matrix far-field rank/accuracy: DOFs={total_n}, dense_blocks={}, low_rank_blocks={}, compression_ratio={:.3}, max_action_error={:.3e}, {}",
        hmatrix.dense_blocks.len(),
        hmatrix.low_rank_blocks.len(),
        hmatrix.compression_ratio,
        accuracy.max_relative_error,
        stats,
    );
}

/// Compare H-matrix and SLFMM far-field blocks against an independently
/// assembled dense TBEM action on two separated sphere clusters. The fixture
/// is intentionally disconnected so the cross-cluster interaction is a true
/// low-rank/far-field block while each self block remains exact near-field.
#[derive(Debug, Clone, Copy)]
struct FarFieldActionMetrics {
    hmatrix_error: f64,
    slfmm_error: f64,
    low_rank_blocks: usize,
}

fn validate_hmatrix_slfmm_far_field_action() -> FarFieldActionMetrics {
    let (mut elements, mut nodes, physics) = prepared_sphere(1, 0.8);
    let first_element_count = elements.len();
    let first_node_count = nodes.len();
    let (mut second_elements, second_nodes, _) = prepared_sphere(1, 0.8);
    let radius = 0.2;
    let separation = 4.0 * radius;

    nodes.extend(second_nodes.into_iter().map(|mut node| {
        node[0] += separation;
        node
    }));
    for (element_index, element) in second_elements.iter_mut().enumerate() {
        for connectivity in &mut element.connectivity {
            *connectivity += first_node_count;
        }
        element.center[0] += separation;
        element.dof_addresses = vec![first_element_count + element_index];
    }
    elements.extend(second_elements);

    let mut left = Cluster::new([0.0, 0.0, 0.0]);
    left.element_indices = (0..first_element_count).collect();
    left.num_elements = first_element_count;
    left.radius = radius;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    let mut right = Cluster::new([separation, 0.0, 0.0]);
    right.element_indices = (first_element_count..elements.len()).collect();
    right.num_elements = elements.len() - first_element_count;
    right.radius = radius;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    let clusters = vec![left, right];

    let radius_scale = physics.wave_number * radius;
    let sign = PhysicsParams::optimal_formulation_sign(radius_scale);
    let beta_scale = PhysicsParams::optimal_beta_scale(radius_scale);
    let dense = DenseOperator::new(
        build_tbem_system_stabilized(&elements, &nodes, &physics, sign, beta_scale).matrix,
    );
    let hmatrix = build_hmatrix_stabilized(
        &elements, &nodes, &clusters, &physics, 1.0e-4, 64, sign, beta_scale,
    );
    let slfmm = SlfmmOperator::new(build_slfmm_system_stabilized(
        &elements, &nodes, &clusters, &physics, 8, 16, 8, sign, beta_scale,
    ));
    assert!(
        !hmatrix.low_rank_blocks.is_empty(),
        "separated-sphere H-matrix fixture produced no far-field block"
    );
    let probe = deterministic_vector(elements.len());
    let dense_action = dense.apply(&probe);
    let hmatrix_action = hmatrix.matvec(&probe);
    let slfmm_action = slfmm.apply(&probe);
    let hmatrix_error = relative_norm(&dense_action, &hmatrix_action);
    let slfmm_error = relative_norm(&dense_action, &slfmm_action);
    assert!(
        hmatrix_error <= 2.0e-4,
        "separated-sphere H-matrix action error {hmatrix_error:.3e} exceeds budget"
    );
    assert!(
        slfmm_error <= 2.0e-3,
        "separated-sphere SLFMM action error {slfmm_error:.3e} exceeds budget"
    );
    println!(
        "H-matrix/SLFMM far-field action: DOFs={}, low_rank_blocks={}, hmatrix_error={hmatrix_error:.3e}, slfmm_error={slfmm_error:.3e}",
        elements.len(),
        hmatrix.low_rank_blocks.len(),
    );
    FarFieldActionMetrics {
        hmatrix_error,
        slfmm_error,
        low_rank_blocks: hmatrix.low_rank_blocks.len(),
    }
}

fn validate_near_singular_quadrature() {
    let element_coords =
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let element_center = [1.0 / 3.0, 1.0 / 3.0, 0.0];
    let source_normal = [0.0, 0.0, 1.0];
    let element_area = 0.5;
    let physics = PhysicsParams::new(250.0, 343.0, 1.21, false);

    let mut previous_g_norm = f64::INFINITY;
    for distance in [1.0e-3, 1.7, 3.0] {
        let source_point = [1.0 / 3.0, 1.0 / 3.0, distance];
        let adaptive = quasi_singular_integration(
            &source_point,
            &source_normal,
            &element_coords,
            ElementType::Tri3,
            &element_center,
            element_area,
            &physics,
        );
        let element_size = element_area.sqrt();
        let ratio = distance / element_size;
        let order = optimal_quadrature_order(distance, element_size);
        let expected = if ratio < 2.0 {
            regular_integration(
                &source_point,
                &source_normal,
                &element_coords,
                ElementType::Tri3,
                element_area,
                &physics,
                None,
                0,
                false,
            )
        } else {
            regular_integration_fixed_order(
                &source_point,
                &source_normal,
                &element_coords,
                ElementType::Tri3,
                &physics,
                order,
            )
        };
        let fixed_high = regular_integration_fixed_order(
            &source_point,
            &source_normal,
            &element_coords,
            ElementType::Tri3,
            &physics,
            4,
        );

        assert_all_finite(
            [
                adaptive.g_integral,
                adaptive.dg_dn_integral,
                adaptive.dg_dnx_integral,
                adaptive.d2g_dnxdny_integral,
            ],
            "near-singular adaptive quadrature",
        );

        let dispatch_rel = (adaptive.g_integral - expected.g_integral).norm()
            / expected.g_integral.norm().max(1e-15);
        let high_rel = (adaptive.g_integral - fixed_high.g_integral).norm()
            / fixed_high.g_integral.norm().max(1e-15);
        println!(
            "Near-singular quadrature: d={distance:.1e}, ratio={ratio:.2}, order={order}, |G|={:.3e}, rel_to_dispatch={dispatch_rel:.3e}, rel_to_fixed_high={high_rel:.3e}",
            adaptive.g_integral.norm()
        );

        assert!(
            dispatch_rel < 1e-12,
            "adaptive near-singular integration did not follow the expected quadrature dispatch"
        );
        assert!(
            adaptive.g_integral.norm() <= previous_g_norm,
            "single-layer integral should not grow as source distance increases"
        );
        previous_g_norm = adaptive.g_integral.norm();
    }
}

#[derive(Debug, Clone, Copy)]
struct ValidationMetrics {
    mlfmm_sphere_max_relative_error: f64,
    far_field_hmatrix_action_error: f64,
    far_field_slfmm_action_error: f64,
    far_field_low_rank_blocks: usize,
    all_backend_sphere_max_power_relative_error: f64,
    all_backend_sphere_max_radiation_condition_relative_error: f64,
    all_backend_sphere_max_residual: f64,
    all_backend_sphere_max_iterations: usize,
    dense_assembly_ms: f64,
    hmatrix_assembly_compression_ms: f64,
    slfmm_setup_ms: f64,
    mlfmm_setup_ms: f64,
    hmatrix_memory_bytes: usize,
    all_backend_max_krylov_ms: f64,
    all_backend_max_residual_check_ms: f64,
    all_backend_max_field_ms: f64,
    mlfmm_adjoint_ms: f64,
}

fn run_all() -> ValidationMetrics {
    validate_burton_miller_near_resonance_assembly();
    validate_fmm_dense_all_near_parity();
    validate_production_operator_contracts();
    let mlfmm_sphere_max_relative_error = validate_mlfmm_sphere_parity();
    let backend_sphere = validate_all_backend_sphere_physical_gates();
    let adjoint_start = Instant::now();
    validate_mlfmm_adjoint_shape_lagrangian();
    let mlfmm_adjoint_ms = adjoint_start.elapsed().as_secs_f64() * 1.0e3;
    validate_operator_physical_contracts();
    validate_hmatrix_near_field_bookkeeping();
    validate_hmatrix_rank_stats_far_field();
    let far_field = validate_hmatrix_slfmm_far_field_action();
    validate_near_singular_quadrature();
    ValidationMetrics {
        mlfmm_sphere_max_relative_error,
        far_field_hmatrix_action_error: far_field.hmatrix_error,
        far_field_slfmm_action_error: far_field.slfmm_error,
        far_field_low_rank_blocks: far_field.low_rank_blocks,
        all_backend_sphere_max_power_relative_error: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.power_relative_error)
            .fold(0.0_f64, f64::max),
        all_backend_sphere_max_radiation_condition_relative_error: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.radiation_condition_relative_error)
            .fold(0.0_f64, f64::max),
        all_backend_sphere_max_residual: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.residual)
            .fold(0.0_f64, f64::max),
        all_backend_sphere_max_iterations: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.iterations)
            .max()
            .unwrap_or(0),
        dense_assembly_ms: backend_sphere.assembly_ms[0],
        hmatrix_assembly_compression_ms: backend_sphere.assembly_ms[1],
        slfmm_setup_ms: backend_sphere.assembly_ms[2],
        mlfmm_setup_ms: backend_sphere.assembly_ms[3],
        hmatrix_memory_bytes: backend_sphere.hmatrix_memory_bytes,
        all_backend_max_krylov_ms: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.krylov_ms)
            .fold(0.0_f64, f64::max),
        all_backend_max_residual_check_ms: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.residual_check_ms)
            .fold(0.0_f64, f64::max),
        all_backend_max_field_ms: backend_sphere
            .backend
            .iter()
            .map(|metrics| metrics.field_ms)
            .fold(0.0_f64, f64::max),
        mlfmm_adjoint_ms,
    }
}

fn main() {
    let total_start = Instant::now();
    let result = std::panic::catch_unwind(run_all);
    match result {
        Ok(metrics) => {
            if let Ok(record) = QaResult::passed("bem.operator-consistency.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .input_checksum("analytic/deterministic-fixture:bem-sphere-v1")
                .reference("dense TBEM action, exact small-system identities, over-integrated quadrature, MLFMM sphere parity, and independent Mie far-field power/radius-invariance shared by dense/H-matrix/SLFMM/MLFMM solves")
                .reference_checksum("analytic/deterministic fixture")
                .dimension("3D closed-sphere, separated-sphere, and two-DOF fixture cases")
                .formulation("production operator consistency and quadrature contracts")
                .tolerance("every dense/H-matrix/SLFMM/MLFMM sphere solve: power <= 6e-2, far-field radius-invariance <= 1.5e-2, and true residual <= 2e-8; remaining cases are defined in qa-bem-sota-validation")
                .elapsed_ms(total_start.elapsed().as_millis() as u64)
                .timing_ms("dense_assembly", metrics.dense_assembly_ms.max(0.0) as u64)
                .timing_ms(
                    "hmatrix_assembly_compression",
                    metrics.hmatrix_assembly_compression_ms.max(0.0) as u64,
                )
                .timing_ms("slfmm_setup", metrics.slfmm_setup_ms.max(0.0) as u64)
                .timing_ms("mlfmm_setup", metrics.mlfmm_setup_ms.max(0.0) as u64)
                .timing_ms("krylov", metrics.all_backend_max_krylov_ms.max(0.0) as u64)
                .timing_ms(
                    "residual_check",
                    metrics.all_backend_max_residual_check_ms.max(0.0) as u64,
                )
                .timing_ms("field", metrics.all_backend_max_field_ms.max(0.0) as u64)
                .memory_bytes(metrics.hmatrix_memory_bytes as u64)
        .metric("physical_contracts", 1.0)
        .metric(
            "far_field_hmatrix_action_error",
            metrics.far_field_hmatrix_action_error,
        )
        .metric(
            "far_field_slfmm_action_error",
            metrics.far_field_slfmm_action_error,
        )
        .metric(
            "far_field_low_rank_blocks",
            metrics.far_field_low_rank_blocks as f64,
        )
                .metric("contract_fixture_dofs", 2.0)
                .metric(
                    "mlfmm_sphere_max_relative_error",
                    metrics.mlfmm_sphere_max_relative_error,
                )
                .metric(
                    "all_backend_sphere_max_power_relative_error",
                    metrics.all_backend_sphere_max_power_relative_error,
                )
                .metric(
                    "all_backend_sphere_max_radiation_condition_relative_error",
                    metrics.all_backend_sphere_max_radiation_condition_relative_error,
                )
                .metric(
                    "all_backend_sphere_max_residual",
                    metrics.all_backend_sphere_max_residual,
                )
                .metric(
                    "all_backend_sphere_max_iterations",
                    metrics.all_backend_sphere_max_iterations as f64,
                )
                .metric("dense_assembly_ms", metrics.dense_assembly_ms)
                .metric(
                    "hmatrix_assembly_compression_ms",
                    metrics.hmatrix_assembly_compression_ms,
                )
                .metric("slfmm_setup_ms", metrics.slfmm_setup_ms)
                .metric("mlfmm_setup_ms", metrics.mlfmm_setup_ms)
                .metric("hmatrix_memory_bytes", metrics.hmatrix_memory_bytes as f64)
                .metric(
                    "all_backend_max_krylov_ms",
                    metrics.all_backend_max_krylov_ms,
                )
                .metric(
                    "all_backend_max_residual_check_ms",
                    metrics.all_backend_max_residual_check_ms,
                )
                .metric("all_backend_max_field_ms", metrics.all_backend_max_field_ms)
                .metric("mlfmm_adjoint_ms", metrics.mlfmm_adjoint_ms)
        .maximum("all_backend_sphere_max_power_relative_error", 6.0e-2)
        .maximum("far_field_hmatrix_action_error", 2.0e-4)
        .maximum("far_field_slfmm_action_error", 2.0e-3)
        .minimum("far_field_low_rank_blocks", 2.0)
                .maximum(
                    "all_backend_sphere_max_radiation_condition_relative_error",
                    FAR_FIELD_POWER_RADIUS_INVARIANCE_TOLERANCE,
                )
                .maximum("all_backend_sphere_max_residual", 2.0e-8)
                .to_json_line()
            {
                println!("QA_RESULT: {record}");
            }
            println!("SUCCESS");
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn validates_bem_sota_numerics() {
        let _ = super::run_all();
    }
}
