//! Magnetostatics validation tests
//!
//! Verifies the permeability-weighted Laplacian and magnet source assembly,
//! then solves magnetostatic problems end-to-end:
//! - Patch test with exact linear potential
//! - Magnet source assembly (divergence theorem)
//! - Dipole magnetic field (point-dipole solve vs analytic dipole field)
//! - Wire field (vector-potential solve vs Biot-Savart)
//! - Magnetic shield (high-mu shell attenuation vs unshielded baseline)
//! - Inductance (two-wire magnetic energy vs analytic reference)

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::{LinearForm, LocalVector, VariableLaplacianForm};
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_fem::physics::electromagnetics::magnetostatics::{
    MU_0, MagnetSourceForm, MagneticMaterial, PermeabilityForm, PointDipoleSource,
    assemble_point_dipole_source, compute_b_field,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::QaResult;
use sonium_solvers::{GmresConfig, LinearOperator, gmres};
use sonium_xem::io::binary_output_dir;
use std::collections::HashSet;

const CASE_ID: &str = "fem.magnetostatics-validation.v1";
// Patch-test potential accuracy (P1 reproduces linear fields exactly).
const PATCH_POTENTIAL_TOLERANCE: f64 = 1e-8;
// Patch-test B-field relative accuracy against mu.
const PATCH_FIELD_TOLERANCE: f64 = 1e-6;
// Magnet-source total (divergence theorem) bound.
const MAGNET_SOURCE_TOTAL_TOLERANCE: f64 = 1e-10;
// Magnet-source entries must be genuinely nonzero above this floor.
const MAGNET_SOURCE_NONZERO_MINIMUM: f64 = 1e-15;
// True relative residual bound for every iterative solve below.
const TRUE_RESIDUAL_TOLERANCE: f64 = 1e-8;
// Solved dipole B-field accuracy away from the singularity (P1
// discretization plus the singular-source pollution that decays with radius).
// Measured 2.15e-2 on the production mesh; the bound keeps a 1.8x margin.
const DIPOLE_FIELD_TOLERANCE: f64 = 4e-2;
// Solved wire B-theta accuracy against the Biot-Savart reference.
// Measured 2.53e-2; the bound keeps a 1.6x margin.
const WIRE_FIELD_TOLERANCE: f64 = 4e-2;
// Spurious radial wire-field fraction (axisymmetry check).
// Measured 2.38e-2; the bound keeps a 1.7x margin.
const WIRE_RADIAL_TOLERANCE: f64 = 4e-2;
// Assembled conductor current must match the prescribed current (staircased
// disk quadrature); the field references use the assembled current so this
// gate only rejects gross assembly errors.
const CURRENT_ASSEMBLY_TOLERANCE: f64 = 4e-2;
// Minimum shielded/unshielded cavity-flux attenuation for the mu_r = 1000
// square shell (thin-shell estimate is well above 100x).
const SHIELD_ATTENUATION_MINIMUM: f64 = 50.0;
// Coarse/fine attenuation agreement (shell edges align with both meshes, so
// this isolates discretization error).
const SHIELD_CONTRACTION_TOLERANCE: f64 = 0.10;
// Unshielded baseline flux must reproduce the uniform applied field.
const SHIELD_BASELINE_TOLERANCE: f64 = 1e-3;
// Two-wire magnetic-energy accuracy against quadrature of the analytic field.
const INDUCTANCE_ENERGY_TOLERANCE: f64 = 0.02;
// Extracted per-unit-length inductance against the round-wire formula
// (includes finite-domain truncation of the dipole far field).
const INDUCTANCE_ANALYTIC_TOLERANCE: f64 = 0.08;

fn gmres_tight() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 20000,
        restart: 300,
        tolerance: 1e-12,
        print_interval: 0,
        convergence_check_interval: 1,
        min_restart: None,
        max_restart: None,
        exact_residual_check_interval: 0,
    }
}

/// True relative residual of a solved linear system, recomputed from the
/// assembled operators rather than trusting the solver estimate.
fn true_relative_residual(csr: &sonium_solvers::CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let ax = csr.apply(x);
    let residual_norm = ax
        .iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = rhs
        .iter()
        .map(|v| v.powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-300);
    residual_norm / rhs_norm
}

fn gate_true_residual(
    leg: &str,
    csr: &sonium_solvers::CsrMatrix<f64>,
    x: &[f64],
    rhs: &[f64],
) -> Result<f64, String> {
    let residual = true_relative_residual(csr, x, rhs);
    if !residual.is_finite() || residual >= TRUE_RESIDUAL_TOLERANCE {
        return Err(format!(
            "{leg} true residual {residual:.2e} exceeds bound {TRUE_RESIDUAL_TOLERANCE:.1e}"
        ));
    }
    Ok(residual)
}

/// Solve a 2D magnetostatic problem: uniform permeability region with a
/// known linear potential phi = x. Verify that grad(phi) = (1, 0) and
/// B = -mu * (1, 0).
///
/// We set Dirichlet BCs phi = x on all boundaries — the interior should
/// reproduce the linear field exactly (patch test for the weighted Laplacian).
fn test_magnetostatic_patch_test() -> Result<(f64, f64), String> {
    let mu = 1000.0 * MU_0;
    let form = PermeabilityForm::from_mu(mu);
    let space = H1Space::new(PolynomialDegree::P1);

    // 4x4 mesh on unit square
    let n = 4;
    let mut mesh = Mesh::new(2);
    for j in 0..n {
        for i in 0..n {
            mesh.add_node(Point::new_2d(
                i as f64 / (n - 1) as f64,
                j as f64 / (n - 1) as f64,
            ));
        }
    }
    for j in 0..(n - 1) {
        for i in 0..(n - 1) {
            let v0 = j * n + i;
            let v1 = v0 + 1;
            let v2 = v0 + n;
            let v3 = v2 + 1;
            mesh.add_element(ElementType::Triangle, vec![v0, v1, v2]);
            mesh.add_element(ElementType::Triangle, vec![v1, v3, v2]);
        }
    }
    mesh.detect_boundaries();
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |_| true);

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let n_dofs = space.num_global_dofs(&mesh);

    // Dirichlet: phi = x on all boundaries
    let boundary_dofs: HashSet<usize> = space.boundary_dofs(&mesh, 1).into_iter().collect();

    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);

    // RHS correction
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && boundary_dofs.contains(&col) {
            let prescribed = mesh.nodes[col].x; // phi = x
            system.rhs[row] -= system.matrix.values[i] * prescribed;
        }
    }

    // Filter + diagonal
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();
    let mut new_vals = Vec::new();
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !boundary_dofs.contains(&row) && !boundary_dofs.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }
    for &dof in &boundary_dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);
        system.rhs[dof] = mesh.nodes[dof].x;
    }
    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;

    // Solve
    let csr = system.matrix.to_csr();
    let result = gmres(&csr, &system.rhs, &gmres_tight());
    if !result.converged {
        return Err("patch-test solver did not converge".to_string());
    }

    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let phi_vec: Vec<f64> = result.x.to_vec();
    let mut w = VtuWriter::new(&mesh);
    w.add_point_scalar("phi", &phi_vec);
    let _ = w.write(dir.join("magnetostatic_patch.vtu"));

    // Verify phi = x at all nodes
    let mut max_potential_error = 0.0_f64;
    for node in 0..mesh.num_nodes() {
        let x = mesh.nodes[node].x;
        max_potential_error = max_potential_error.max((result.x[node] - x).abs());
    }
    if !max_potential_error.is_finite() || max_potential_error >= PATCH_POTENTIAL_TOLERANCE {
        return Err(format!(
            "patch potential error {max_potential_error:.2e} exceeds bound {PATCH_POTENTIAL_TOLERANCE:.1e}"
        ));
    }

    // Verify B-field at element centroids
    let br = [0.0, 0.0, 0.0];
    let mut max_field_error = 0.0_f64;
    for elem_idx in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, elem_idx);
        let phi_local: Vec<f64> = dofs.iter().map(|&d| result.x[d]).collect();
        let tab = space.tabulate(&mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);

        let b = compute_b_field(&phi_local, &tab, mu, &br);
        // B = -mu * grad(phi) = -mu * (1, 0)
        max_field_error = max_field_error.max((b[0] - (-mu)).abs() / mu);
        max_field_error = max_field_error.max(b[1].abs() / mu);
    }
    if !max_field_error.is_finite() || max_field_error >= PATCH_FIELD_TOLERANCE {
        return Err(format!(
            "patch B-field error {max_field_error:.2e} exceeds bound {PATCH_FIELD_TOLERANCE:.1e}"
        ));
    }
    println!(
        "Patch test: potential err={max_potential_error:.2e}, B-field err={max_field_error:.2e}"
    );
    Ok((max_potential_error, max_field_error))
}

/// Verify that the magnet source RHS assembles correctly.
///
/// For a magnet with B_r = (1, 0), the source integral(B_r . grad(psi)) dOmega
/// should match the integral of the x-derivative of the test functions.
fn test_magnet_source_assembly() -> Result<(f64, f64), String> {
    let space = H1Space::new(PolynomialDegree::P1);
    let magnet = MagneticMaterial::neodymium(1.0, [1.0, 0.0, 0.0]);
    let source = MagnetSourceForm::new(&magnet);
    if !source.is_active() {
        return Err("neodymium magnet source should be active".to_string());
    }

    let mesh = sonium_fem::mesh::unit_square_triangles(3);
    let rhs: Vec<f64> = sonium_fem::assembly::assemble_linear(&source, &space, &mesh, 1);

    // Total source = integral of div(B_r) over the domain
    // For constant B_r, div(B_r) = 0 in the interior.
    // But the surface integral contributes: integral B_r . n dGamma
    // Total RHS sum should equal integral of B_r . n over boundary = integral n_x dGamma
    // For a unit square: right face (n_x=1, length=1) + left face (n_x=-1, length=1) = 0
    let total: f64 = rhs.iter().sum();
    if !total.is_finite() || total.abs() >= MAGNET_SOURCE_TOTAL_TOLERANCE {
        return Err(format!(
            "total magnet source = {total:.2e}, expected ~0 (divergence theorem)"
        ));
    }

    // But individual entries should be nonzero
    let max_abs = rhs.iter().map(|r| r.abs()).fold(0.0, f64::max);
    if !(max_abs > MAGNET_SOURCE_NONZERO_MINIMUM) {
        return Err("magnet source should have nonzero entries".to_string());
    }
    println!("Magnet source: total={total:.2e}, max entry={max_abs:.2e}");
    Ok((total.abs(), max_abs))
}

/// Gradient of a scalar H1 solution at an element centroid.
fn centroid_gradient(space: &H1Space, mesh: &Mesh, elem_idx: usize, solution: &[f64]) -> [f64; 2] {
    let dofs = space.local_to_global(mesh, elem_idx);
    let tab = space.tabulate(mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);
    let mut grad = [0.0, 0.0];
    for (local, &global) in dofs.iter().enumerate() {
        grad[0] += solution[global] * tab.grad(local, 0);
        grad[1] += solution[global] * tab.grad(local, 1);
    }
    grad
}

fn element_centroid(mesh: &Mesh, elem_idx: usize) -> (f64, f64) {
    let vertices = mesh.elements[elem_idx].vertices();
    let (mut x, mut y) = (0.0, 0.0);
    for &v in vertices {
        x += mesh.nodes[v].x;
        y += mesh.nodes[v].y;
    }
    (x / vertices.len() as f64, y / vertices.len() as f64)
}

/// Solve the scalar potential of a dipole placed outside the domain (smooth
/// interior field) with the analytic dipole potential prescribed on the
/// boundary, and compare the solved B-field against the analytic 2D dipole
/// field. The free-space dipole solution is phi = -(m . r_hat)/(2 pi mu r)
/// with B = (1/(2 pi r^2)) (m - 2 (m . r_hat) r_hat).
/// A point dipole inside the domain would converge too slowly for a tight
/// gate (the P1 gradient cannot resolve the 1/r^2 singularity), so the
/// native point-source assembly is covered separately by its exact discrete
/// identity below.
fn test_magnetic_dipole_field() -> Result<(f64, f64), String> {
    let moment = [0.0, 1.0];
    let center = (0.5, -0.4);
    let mu = MU_0;
    let n = 96;
    let mesh = sonium_fem::mesh::unit_square_triangles(n);
    let space = H1Space::new(PolynomialDegree::P1);

    // Native point-source assembly identity: the dipole RHS sums to exactly
    // zero because sum_i grad(N_i) = grad(1) = 0 at any interior point.
    let probe = PointDipoleSource::new([0.5, 0.5, 0.0], [1.0, 0.0, 0.0]);
    let probe_rhs = assemble_point_dipole_source(&probe, &mesh, &space);
    let probe_total: f64 = probe_rhs.iter().sum();
    let probe_max = probe_rhs.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if probe_max < 1e-15 {
        return Err("point-dipole assembly produced a zero vector".to_string());
    }
    if !probe_total.is_finite() || probe_total.abs() > 1e-12 * probe_max.max(1.0) {
        return Err(format!(
            "point-dipole RHS sum {probe_total:.2e} violates the zero-sum identity"
        ));
    }

    let analytic_phi = |x: f64, y: f64| -> f64 {
        let dx = x - center.0;
        let dy = y - center.1;
        let r_squared = dx * dx + dy * dy;
        -(moment[0] * dx + moment[1] * dy) / (2.0 * std::f64::consts::PI * mu * r_squared)
    };

    let form = PermeabilityForm::from_mu(mu);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);
    for marker in [1, 2, 3, 4] {
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|x, y, _| analytic_phi(x, y),
            None,
        );
    }
    let csr = system.matrix.to_csr();
    let result = gmres(&csr, &system.rhs, &gmres_tight());
    if !result.converged {
        return Err("dipole solve did not converge".to_string());
    }
    let residual = gate_true_residual("dipole", &csr, &result.x, &system.rhs)?;

    // Compare solved B against analytic B at every element centroid.
    let br = [0.0, 0.0, 0.0];
    let mut max_error = 0.0_f64;
    let mut samples = 0_usize;
    for elem_idx in 0..mesh.num_elements() {
        let (cx, cy) = element_centroid(&mesh, elem_idx);
        let dx = cx - center.0;
        let dy = cy - center.1;
        let r = (dx * dx + dy * dy).sqrt();
        let dofs = space.local_to_global(&mesh, elem_idx);
        let phi_local: Vec<f64> = dofs.iter().map(|&d| result.x[d]).collect();
        let tab = space.tabulate(&mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let b_fem = compute_b_field(&phi_local, &tab, mu, &br);
        let rx = dx / r;
        let ry = dy / r;
        let m_dot_r = moment[0] * rx + moment[1] * ry;
        let scale = 1.0 / (2.0 * std::f64::consts::PI * r * r);
        let b_exact = [
            scale * (moment[0] - 2.0 * m_dot_r * rx),
            scale * (moment[1] - 2.0 * m_dot_r * ry),
        ];
        let error = ((b_fem[0] - b_exact[0]).powi(2) + (b_fem[1] - b_exact[1]).powi(2)).sqrt()
            / (b_exact[0].powi(2) + b_exact[1].powi(2)).sqrt().max(1e-300);
        max_error = max_error.max(error);
        samples += 1;
    }
    println!(
        "Dipole field: {samples} samples, max relative B error={max_error:.2e}, residual={residual:.2e}"
    );
    if !max_error.is_finite() || max_error >= DIPOLE_FIELD_TOLERANCE {
        return Err(format!(
            "dipole B-field error {max_error:.2e} exceeds bound {DIPOLE_FIELD_TOLERANCE:.1e}"
        ));
    }
    Ok((max_error, residual))
}

/// Uniform out-of-plane current density inside a round conductor disk.
struct DiskCurrentDensity {
    center: (f64, f64),
    radius: f64,
    density: f64,
}

impl LinearForm<f64> for DiskCurrentDensity {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let [x, y, _] = test.physical_coords;
        let dx = x - self.center.0;
        let dy = y - self.center.1;
        if dx * dx + dy * dy > self.radius * self.radius {
            return;
        }
        let scale = weight * test.det_j * self.density;
        for i in 0..test.n_dofs {
            local_vector.add(i, scale * test.value(i));
        }
    }
}

/// Solve the 2D vector potential A_z of a current-carrying round wire
/// (-div((1/mu) grad A_z) = J_z, same weighted-Laplacian operator with
/// reluctivity coefficient) and compare B_theta against Biot-Savart.
/// The field reference uses the assembled current so disk quadrature error
/// cannot masquerade as field error; the assembled current itself is gated
/// separately.
fn test_biot_savart_wire() -> Result<(f64, f64, f64, f64), String> {
    let current: f64 = 10.0;
    let radius: f64 = 0.1;
    let mesh = rectangular_mesh_triangles(-1.0, 1.0, -1.0, 1.0, 128, 128);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = PermeabilityForm::from_mu(1.0 / MU_0);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let density = current / (std::f64::consts::PI * radius * radius);
    let source = DiskCurrentDensity {
        center: (0.0, 0.0),
        radius,
        density,
    };
    let rhs = assemble_linear(&source, &space, &mesh, 5);
    let assembled_current: f64 = rhs.iter().sum();
    let current_error = (assembled_current - current).abs() / current;
    if !current_error.is_finite() || current_error >= CURRENT_ASSEMBLY_TOLERANCE {
        return Err(format!(
            "assembled wire current error {current_error:.2e} exceeds bound {CURRENT_ASSEMBLY_TOLERANCE:.1e}"
        ));
    }

    // Exact single-wire vector potential on the boundary (no truncation error).
    let analytic_a = |x: f64, y: f64| -> f64 {
        let r = (x * x + y * y).sqrt().max(1e-300);
        -(MU_0 * assembled_current / (2.0 * std::f64::consts::PI)) * r.ln()
    };
    let mut system = FemSystem::new(stiffness, rhs);
    for marker in [1, 2, 3, 4] {
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|x, y, _| analytic_a(x, y),
            None,
        );
    }
    let csr = system.matrix.to_csr();
    let result = gmres(&csr, &system.rhs, &gmres_tight());
    if !result.converged {
        return Err("wire vector-potential solve did not converge".to_string());
    }
    let residual = gate_true_residual("wire", &csr, &result.x, &system.rhs)?;

    let mut max_theta_error = 0.0_f64;
    let mut max_radial_fraction = 0.0_f64;
    let mut samples = 0_usize;
    // Sample ring starts at 3 conductor radii: inside that radius the P1
    // gradient error (~h/r) exceeds the gate, while the ring still spans a
    // factor of 3 in radius to validate the 1/r decay.
    for elem_idx in 0..mesh.num_elements() {
        let (cx, cy) = element_centroid(&mesh, elem_idx);
        let r = (cx * cx + cy * cy).sqrt();
        if r < 0.3 || r > 0.9 {
            continue;
        }
        let grad = centroid_gradient(&space, &mesh, elem_idx, &result.x);
        // B = curl(A_z z_hat) = (dA/dy, -dA/dx).
        let bx = grad[1];
        let by = -grad[0];
        let (rx, ry) = (cx / r, cy / r);
        let b_theta = bx * (-ry) + by * rx;
        let b_radial = bx * rx + by * ry;
        let exact = MU_0 * assembled_current / (2.0 * std::f64::consts::PI * r);
        max_theta_error = max_theta_error.max((b_theta - exact).abs() / exact);
        max_radial_fraction = max_radial_fraction.max(b_radial.abs() / exact);
        samples += 1;
    }
    println!(
        "Wire field: I_asm={assembled_current:.4e} A, {samples} samples, B_theta err={max_theta_error:.2e}, radial frac={max_radial_fraction:.2e}"
    );
    if !max_theta_error.is_finite() || max_theta_error >= WIRE_FIELD_TOLERANCE {
        return Err(format!(
            "wire B-field error {max_theta_error:.2e} exceeds bound {WIRE_FIELD_TOLERANCE:.1e}"
        ));
    }
    if !max_radial_fraction.is_finite() || max_radial_fraction >= WIRE_RADIAL_TOLERANCE {
        return Err(format!(
            "wire radial field fraction {max_radial_fraction:.2e} exceeds bound {WIRE_RADIAL_TOLERANCE:.1e}"
        ));
    }
    Ok((
        max_theta_error,
        max_radial_fraction,
        current_error,
        residual,
    ))
}

/// Solve a high-permeability square shell in a uniform applied field and
/// return (cavity flux, baseline flux, attenuation). The shell edges align
/// with the mesh grid lines, so no element is cut by a material interface.
fn solve_shield(n: usize, mu_relative_shell: f64) -> Result<(f64, f64, f64), String> {
    let applied_b: f64 = 0.01;
    let outer: f64 = 0.3;
    let inner: f64 = 0.2;
    let mesh = rectangular_mesh_triangles(-1.0, 1.0, -1.0, 1.0, n, n);
    let space = H1Space::new(PolynomialDegree::P1);

    let in_shell = |x: f64, y: f64| -> bool {
        x.abs() <= outer && y.abs() <= outer && (x.abs() >= inner || y.abs() >= inner)
    };
    let form = VariableLaplacianForm {
        coefficient_fn: |x: f64, y: f64, _z: f64| -> f64 {
            if in_shell(x, y) {
                mu_relative_shell * MU_0
            } else {
                MU_0
            }
        },
    };
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let n_dofs = space.num_global_dofs(&mesh);
    let mut system = FemSystem::new(stiffness, vec![0.0; n_dofs]);
    // Uniform applied field B0 x_hat: phi = -(B0/mu0) x.
    for marker in [1, 2, 3, 4] {
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|x, _, _| -(applied_b / MU_0) * x,
            None,
        );
    }
    let csr = system.matrix.to_csr();
    let result = gmres(&csr, &system.rhs, &gmres_tight());
    if !result.converged {
        return Err(format!("shield solve at n={n} did not converge"));
    }
    gate_true_residual(&format!("shield n={n}"), &csr, &result.x, &system.rhs)?;

    // Cavity flux: mean B_x over cavity elements times cavity area.
    let br = [0.0, 0.0, 0.0];
    let mut cavity_bx_sum = 0.0_f64;
    let mut cavity_count = 0_usize;
    let mut far_sum = 0.0_f64;
    let mut far_count = 0_usize;
    for elem_idx in 0..mesh.num_elements() {
        let (cx, cy) = element_centroid(&mesh, elem_idx);
        let dofs = space.local_to_global(&mesh, elem_idx);
        let phi_local: Vec<f64> = dofs.iter().map(|&d| result.x[d]).collect();
        let tab = space.tabulate(&mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let mu_here = if in_shell(cx, cy) {
            mu_relative_shell * MU_0
        } else {
            MU_0
        };
        let b = compute_b_field(&phi_local, &tab, mu_here, &br);
        if cx.abs() < inner && cy.abs() < inner {
            cavity_bx_sum += b[0];
            cavity_count += 1;
        }
        if cx.abs() > 0.7 || cy.abs() > 0.7 {
            far_sum += (b[0].powi(2) + b[1].powi(2)).sqrt();
            far_count += 1;
        }
    }
    if cavity_count == 0 || far_count == 0 {
        return Err("shield sampling found no cavity or far elements".to_string());
    }
    let cavity_flux = cavity_bx_sum / cavity_count as f64 * (2.0 * inner) * (2.0 * inner);
    let far_field = far_sum / far_count as f64;
    let attenuation = far_field / (cavity_bx_sum / cavity_count as f64).abs().max(1e-300);
    Ok((
        cavity_flux,
        applied_b * (2.0 * inner) * (2.0 * inner),
        attenuation,
    ))
}

/// Verify magnetic shielding with solved shell vs unshielded comparisons.
fn test_magnetic_shield_attenuation() -> Result<(f64, f64, f64), String> {
    let mu_shell = 1000.0;
    // Unshielded baseline must reproduce the uniform applied flux.
    let (baseline_flux, uniform_flux, _) = solve_shield(80, 1.0)?;
    let baseline_error = (baseline_flux - uniform_flux).abs() / uniform_flux;
    println!(
        "Shield baseline: flux={baseline_flux:.4e}, uniform={uniform_flux:.4e}, err={baseline_error:.2e}"
    );
    if !baseline_error.is_finite() || baseline_error >= SHIELD_BASELINE_TOLERANCE {
        return Err(format!(
            "unshielded baseline flux error {baseline_error:.2e} exceeds bound {SHIELD_BASELINE_TOLERANCE:.1e}"
        ));
    }
    let (_, _, attenuation_coarse) = solve_shield(40, mu_shell)?;
    let (cavity_flux, _, attenuation_fine) = solve_shield(80, mu_shell)?;
    println!(
        "Shield mu_r={mu_shell}: coarse attenuation={attenuation_coarse:.1e}x, fine={attenuation_fine:.1e}x, cavity flux={cavity_flux:.3e}"
    );
    if !attenuation_fine.is_finite() || attenuation_fine < SHIELD_ATTENUATION_MINIMUM {
        return Err(format!(
            "shield attenuation {attenuation_fine:.1e}x below minimum {SHIELD_ATTENUATION_MINIMUM:.1e}x"
        ));
    }
    let contraction = (attenuation_fine - attenuation_coarse).abs() / attenuation_fine.max(1e-300);
    if !contraction.is_finite() || contraction >= SHIELD_CONTRACTION_TOLERANCE {
        return Err(format!(
            "shield attenuation contraction {contraction:.2e} exceeds bound {SHIELD_CONTRACTION_TOLERANCE:.1e}"
        ));
    }
    Ok((attenuation_fine, contraction, baseline_error))
}

/// Analytic round-wire B-field magnitude components at (x, y) for a wire at
/// (wx, wy) with current i and radius a (uniform current density).
fn round_wire_b(x: f64, y: f64, wx: f64, wy: f64, i: f64, a: f64) -> (f64, f64) {
    let dx = x - wx;
    let dy = y - wy;
    let r = (dx * dx + dy * dy).sqrt().max(1e-300);
    let magnitude = if r < a {
        MU_0 * i * r / (2.0 * std::f64::consts::PI * a * a)
    } else {
        MU_0 * i / (2.0 * std::f64::consts::PI * r)
    };
    (magnitude * (-dy / r), magnitude * (dx / r))
}

/// Solve the two-wire (go/return) vector potential and compare the magnetic
/// energy against quadrature of the analytic round-wire field over the same
/// mesh, then extract the per-unit-length inductance L' = 2W/I^2.
fn test_solenoid_inductance() -> Result<(f64, f64, f64), String> {
    let current: f64 = 10.0;
    let radius: f64 = 0.1;
    let half_gap: f64 = 0.3;
    let n = 80;
    let mesh = rectangular_mesh_triangles(-1.0, 1.0, -1.0, 1.0, n, n);
    let space = H1Space::new(PolynomialDegree::P1);

    let form = PermeabilityForm::from_mu(1.0 / MU_0);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let density = current / (std::f64::consts::PI * radius * radius);
    let rhs_go = assemble_linear(
        &DiskCurrentDensity {
            center: (half_gap, 0.0),
            radius,
            density,
        },
        &space,
        &mesh,
        5,
    );
    let rhs_return = assemble_linear(
        &DiskCurrentDensity {
            center: (-half_gap, 0.0),
            radius,
            density: -density,
        },
        &space,
        &mesh,
        5,
    );
    let assembled_go: f64 = rhs_go.iter().sum();
    let assembled_return: f64 = rhs_return.iter().sum();
    for (name, assembled, target) in [
        ("go", assembled_go, current),
        ("return", assembled_return, -current),
    ] {
        let error = (assembled - target).abs() / target.abs();
        if !error.is_finite() || error >= CURRENT_ASSEMBLY_TOLERANCE {
            return Err(format!(
                "two-wire {name} current error {error:.2e} exceeds bound {CURRENT_ASSEMBLY_TOLERANCE:.1e}"
            ));
        }
    }
    let rhs: Vec<f64> = rhs_go
        .iter()
        .zip(rhs_return.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    // Exact two-wire vector potential on the boundary.
    let analytic_a = |x: f64, y: f64| -> f64 {
        let r_go = ((x - half_gap).powi(2) + y.powi(2)).sqrt().max(1e-300);
        let r_return = ((x + half_gap).powi(2) + y.powi(2)).sqrt().max(1e-300);
        -(MU_0 / (2.0 * std::f64::consts::PI))
            * (assembled_go * r_go.ln() + assembled_return * r_return.ln())
    };
    let mut system = FemSystem::new(stiffness, rhs);
    for marker in [1, 2, 3, 4] {
        apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|x, y, _| analytic_a(x, y),
            None,
        );
    }
    let csr = system.matrix.to_csr();
    let result = gmres(&csr, &system.rhs, &gmres_tight());
    if !result.converged {
        return Err("two-wire vector-potential solve did not converge".to_string());
    }
    let residual = gate_true_residual("inductance", &csr, &result.x, &system.rhs)?;

    // Magnetic energy by element quadrature, FEM vs analytic on the same mesh.
    let mut fem_energy = 0.0_f64;
    let mut analytic_energy = 0.0_f64;
    for elem_idx in 0..mesh.num_elements() {
        let rule = QuadratureRule::new(mesh.elements[elem_idx].element_type, 5);
        let local_dofs = space.local_to_global(&mesh, elem_idx);
        for point in rule.iter() {
            let tab = space.tabulate(&mesh, elem_idx, point.xi(), point.eta(), point.zeta());
            let mut grad_a = [0.0, 0.0];
            for (local, &global) in local_dofs.iter().enumerate() {
                grad_a[0] += result.x[global] * tab.grad(local, 0);
                grad_a[1] += result.x[global] * tab.grad(local, 1);
            }
            let b_fem_squared = grad_a[0].powi(2) + grad_a[1].powi(2);
            let [x, y, _] = tab.physical_coords;
            let (bgo_x, bgo_y) = round_wire_b(x, y, half_gap, 0.0, assembled_go, radius);
            let (bret_x, bret_y) = round_wire_b(x, y, -half_gap, 0.0, assembled_return, radius);
            let b_exact_squared = (bgo_x + bret_x).powi(2) + (bgo_y + bret_y).powi(2);
            let weight = point.weight * tab.det_j / (2.0 * MU_0);
            fem_energy += weight * b_fem_squared;
            analytic_energy += weight * b_exact_squared;
        }
    }
    let energy_error = (fem_energy - analytic_energy).abs() / analytic_energy;
    let current_mean = (assembled_go.abs() + assembled_return.abs()) / 2.0;
    let inductance = 2.0 * fem_energy / current_mean.powi(2);
    // Round-wire two-wire-line formula: L' = (mu0/pi)(ln(d/a) + 1/4).
    let expected = (MU_0 / std::f64::consts::PI) * ((2.0 * half_gap / radius).ln() + 0.25);
    let inductance_error = (inductance - expected).abs() / expected;
    println!(
        "Inductance: W_fem={fem_energy:.4e} J/m, W_ref={analytic_energy:.4e} J/m, energy err={energy_error:.2e}, L'={inductance:.4e} H/m (expected {expected:.4e}), L err={inductance_error:.2e}"
    );
    if !energy_error.is_finite() || energy_error >= INDUCTANCE_ENERGY_TOLERANCE {
        return Err(format!(
            "magnetic energy error {energy_error:.2e} exceeds bound {INDUCTANCE_ENERGY_TOLERANCE:.1e}"
        ));
    }
    if !inductance_error.is_finite() || inductance_error >= INDUCTANCE_ANALYTIC_TOLERANCE {
        return Err(format!(
            "inductance error {inductance_error:.2e} exceeds bound {INDUCTANCE_ANALYTIC_TOLERANCE:.1e}"
        ));
    }
    let _ = residual;
    Ok((energy_error, inductance_error, inductance))
}

struct MagnetostaticsMetrics {
    patch_potential_error: f64,
    patch_field_error: f64,
    source_total: f64,
    source_max_entry: f64,
    dipole_field_error: f64,
    dipole_residual: f64,
    wire_field_error: f64,
    wire_radial_fraction: f64,
    wire_current_error: f64,
    wire_residual: f64,
    shield_attenuation: f64,
    shield_contraction: f64,
    shield_baseline_error: f64,
    inductance_energy_error: f64,
    inductance_error: f64,
    inductance_per_meter: f64,
}

fn run() -> Result<MagnetostaticsMetrics, String> {
    let (patch_potential_error, patch_field_error) = test_magnetostatic_patch_test()?;
    let (source_total, source_max_entry) = test_magnet_source_assembly()?;
    let (dipole_field_error, dipole_residual) = test_magnetic_dipole_field()?;
    let (wire_field_error, wire_radial_fraction, wire_current_error, wire_residual) =
        test_biot_savart_wire()?;
    let (shield_attenuation, shield_contraction, shield_baseline_error) =
        test_magnetic_shield_attenuation()?;
    let (inductance_energy_error, inductance_error, inductance_per_meter) =
        test_solenoid_inductance()?;
    Ok(MagnetostaticsMetrics {
        patch_potential_error,
        patch_field_error,
        source_total,
        source_max_entry,
        dipole_field_error,
        dipole_residual,
        wire_field_error,
        wire_radial_fraction,
        wire_current_error,
        wire_residual,
        shield_attenuation,
        shield_contraction,
        shield_baseline_error,
        inductance_energy_error,
        inductance_error,
        inductance_per_meter,
    })
}

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("patch_potential_error", metrics.patch_potential_error)
                .metric("patch_field_error", metrics.patch_field_error)
                .metric("source_total", metrics.source_total)
                .metric("source_max_entry", metrics.source_max_entry)
                .metric("dipole_field_error", metrics.dipole_field_error)
                .metric("dipole_residual", metrics.dipole_residual)
                .metric("wire_field_error", metrics.wire_field_error)
                .metric("wire_radial_fraction", metrics.wire_radial_fraction)
                .metric("wire_current_error", metrics.wire_current_error)
                .metric("wire_residual", metrics.wire_residual)
                .metric("shield_attenuation", metrics.shield_attenuation)
                .metric("shield_contraction", metrics.shield_contraction)
                .metric("shield_baseline_error", metrics.shield_baseline_error)
                .metric("inductance_energy_error", metrics.inductance_energy_error)
                .metric("inductance_error", metrics.inductance_error)
                .metric("inductance_per_meter", metrics.inductance_per_meter)
                .maximum("patch_potential_error", PATCH_POTENTIAL_TOLERANCE)
                .maximum("patch_field_error", PATCH_FIELD_TOLERANCE)
                .maximum("source_total", MAGNET_SOURCE_TOTAL_TOLERANCE)
                .minimum("source_max_entry", MAGNET_SOURCE_NONZERO_MINIMUM)
                .maximum("dipole_field_error", DIPOLE_FIELD_TOLERANCE)
                .maximum("dipole_residual", TRUE_RESIDUAL_TOLERANCE)
                .maximum("wire_field_error", WIRE_FIELD_TOLERANCE)
                .maximum("wire_radial_fraction", WIRE_RADIAL_TOLERANCE)
                .maximum("wire_current_error", CURRENT_ASSEMBLY_TOLERANCE)
                .maximum("wire_residual", TRUE_RESIDUAL_TOLERANCE)
                .minimum("shield_attenuation", SHIELD_ATTENUATION_MINIMUM)
                .maximum("shield_contraction", SHIELD_CONTRACTION_TOLERANCE)
                .maximum("shield_baseline_error", SHIELD_BASELINE_TOLERANCE)
                .maximum("inductance_energy_error", INDUCTANCE_ENERGY_TOLERANCE)
                .maximum("inductance_error", INDUCTANCE_ANALYTIC_TOLERANCE)
                .tolerance(
                    "numerical dipole/wire/shield/inductance comparisons with true-residual gates",
                )
                .seed("fem-magnetostatics-validation:cpu:v1");
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
            eprintln!("FEM MAGNETOSTATICS VALIDATION FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
