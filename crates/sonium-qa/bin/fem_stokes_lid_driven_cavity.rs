//! Stokes flow — 2D Lid-Driven Cavity with Taylor-Hood P2-P1
//!
//! Solves steady Stokes on [0,1]^2:
//!   - Top wall (y=1): u = (1, 0)  (lid moves right)
//!   - Other walls:    u = (0, 0)  (no-slip)
//!   - Pressure pinned at one point (bottom-left corner)
//!
//! For Stokes (Re → 0), the primary vortex center is at approximately
//! (x, y) ≈ (0.5, 0.76) and the flow is symmetric about x = 0.5.
//! Centerline velocities are checked against independently refined clamped-
//! biharmonic finite differences; bulk divergence is checked against zero.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use sonium_solvers::{
    AmgConfig, BlockPreconditionerConfig, FgmresConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, fgmres_preconditioned,
};
use sonium_xem::io::binary_output_dir;
use std::collections::HashMap;

fn solve_cavity(n: usize) -> Result<CavityMetrics, String> {
    println!("=== Stokes Lid-Driven Cavity (Taylor-Hood P2-P1) ===\n");

    let mu = 1.0;

    // --- 1. Mesh ---
    let mesh = unit_square_triangles(n);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // --- 2. Function spaces ---
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let n_total = n_v + n_p;
    println!("Velocity P2: {} DOFs", n_v);
    println!("Pressure P1: {} DOFs", n_p);
    println!("Total:       {} DOFs\n", n_total);

    // --- 3. Assemble ---
    println!("Assembling Stokes system...");
    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 4);
    let mono = system.to_monolithic();

    // Pressure mass for Schur complement
    let mp = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);
    let mut schur = TripletMatrix::<f64>::with_capacity(mp.num_rows, mp.num_cols, mp.nnz());
    for i in 0..mp.nnz() {
        schur.push(mp.rows[i], mp.cols[i], mp.values[i] / mu);
    }

    // --- 4. DOF coordinates ---
    let topo = MeshTopology::build(&mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;

    let mut dof_x = vec![0.0; n_scalar];
    let mut dof_y = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dof_x[i] = mesh.nodes[i].x;
        dof_y[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dof_x[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dof_y[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }

    // --- 5. Boundary conditions ---
    println!("Applying boundary conditions...");
    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);
    let tol = 1e-10;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dof_x[sd];
        let y = dof_y[sd];
        let on_boundary =
            x.abs() < tol || (x - 1.0).abs() < tol || y.abs() < tol || (y - 1.0).abs() < tol;

        if on_boundary {
            if (y - 1.0).abs() < tol && x > tol && x < 1.0 - tol {
                // Lid: u_x = 1, u_y = 0 (exclude corners for compatibility)
                bc_map.insert(sd * 2, 1.0);
            } else {
                // No-slip walls: u_x = 0
                bc_map.insert(sd * 2, 0.0);
            }
            // u_y = 0 on all walls
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }

    // Pin pressure at bottom-left corner (node 0) to remove nullspace
    bc_map.insert(n_v, 0.0);

    println!("  {} constrained DOFs", bc_map.len());

    // Apply BCs
    apply_dirichlet(&mut fem, &bc_map);

    // --- 6. Solve ---
    println!("\nSolving with FGMRES + block preconditioner...");
    let a_csr = fem.matrix.to_csr();
    let schur_csr = schur.to_csr();

    let sp_config = SaddlePointConfig {
        block1_size: n_v,
        block1_config: BlockPreconditionerConfig::Amg(AmgConfig::for_fem()),
        block2_config: BlockPreconditionerConfig::Amg(AmgConfig::for_fem()),
        schur_strategy: SchurStrategy::External,
        upper_triangular: true,
    };
    let precond =
        SaddlePointPreconditioner::<f64>::from_matrix_with_schur(&a_csr, &schur_csr, sp_config);

    let rhs = fem.rhs.clone();
    let config = FgmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-10,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };
    let result = fgmres_preconditioned(&a_csr, &precond, &rhs, None, &config);

    println!(
        "  {} (iters: {}, residual: {:.2e})",
        if result.converged {
            "Converged"
        } else {
            "NOT converged"
        },
        result.iterations,
        result.residual,
    );

    let true_residual = sonium_qa::relative_residual_norm(&a_csr, &result.x, &rhs);
    if !result.converged || !true_residual.is_finite() || true_residual > 1e-8 {
        return Err(format!(
            "cavity n={n} did not converge: true residual={true_residual:.3e}, converged={}",
            result.converged
        ));
    }
    let mut centerlines = Vec::new();
    for horizontal in [false, true] {
        for i in 1..8 {
            let t = i as f64 / 8.0;
            let (x, y, component) = if horizontal { (t, 0.5, 1) } else { (0.5, t, 0) };
            let sd = (0..n_scalar)
                .find(|&i| (dof_x[i] - x).abs() < 1e-12 && (dof_y[i] - y).abs() < 1e-12)
                .ok_or_else(|| format!("missing exact centerline sample ({x},{y})"))?;
            centerlines.push(result.x[2 * sd + component]);
        }
    }
    // The discontinuous lid at the corners gives singular velocity gradients.
    // Integrate divergence on the fixed bulk square [1/8,7/8]^2, independent of
    // refinement. The full-domain weak continuity equations remain in Ax-b.
    let scalar_space = H1Space::new(PolynomialDegree::P2);
    let quadrature =
        sonium_fem::quadrature::QuadratureRule::new(sonium_fem::mesh::ElementType::Triangle, 2);
    let mut divergence_energy = 0.0;
    let mut bulk_area = 0.0;
    for (ei, element) in mesh.elements.iter().enumerate() {
        if element.vertices().iter().any(|&i| {
            let p = mesh.nodes[i];
            p.x < 0.125 - 1e-12 || p.x > 0.875 + 1e-12 || p.y < 0.125 - 1e-12 || p.y > 0.875 + 1e-12
        }) {
            continue;
        }
        let dofs = scalar_space.local_to_global(&mesh, ei);
        for point in quadrature.iter() {
            let tab = scalar_space.tabulate(&mesh, ei, point.coords[0], point.coords[1], 0.0);
            let mut divergence = 0.0;
            for (i, &dof) in dofs.iter().enumerate() {
                divergence +=
                    tab.grad(i, 0) * result.x[2 * dof] + tab.grad(i, 1) * result.x[2 * dof + 1];
            }
            let weight = tab.det_j * point.weight;
            divergence_energy += weight * divergence * divergence;
            bulk_area += weight;
        }
    }
    let bulk_divergence_rms = (divergence_energy / bulk_area).sqrt();
    let u_sol = &result.x[..n_v];
    let mut writer = sonium_fem::io::vtk::VtuWriter::new(&mesh);
    let ux = (0..n_nodes).map(|i| u_sol[2 * i]).collect::<Vec<_>>();
    let uy = (0..n_nodes).map(|i| u_sol[2 * i + 1]).collect::<Vec<_>>();
    writer.add_point_scalar("ux", &ux);
    writer.add_point_scalar("uy", &uy);
    writer.add_point_scalar("pressure", &result.x[n_v..]);
    let output_path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join(format!("stokes_cavity_n{n}.vtu"));
    writer
        .write(&output_path)
        .map_err(|error| format!("write cavity VTU: {error}"))?;
    println!(
        "n={n}: centerlines={centerlines:?}; bulk divergence={bulk_divergence_rms:.3e}; true residual={true_residual:.3e}"
    );
    Ok(CavityMetrics {
        centerlines,
        bulk_divergence_rms,
        true_residual,
    })
}

#[derive(Debug)]
struct CavityMetrics {
    centerlines: Vec<f64>,
    bulk_divergence_rms: f64,
    true_residual: f64,
}

const CASE_ID: &str = "fem.stokes-cavity-independent-centerlines.v1";
const REFERENCE_BYTES: &[u8] = include_bytes!("../src/fixtures/stokes/cavity_centerlines.json");

fn centerline_errors(actual: &[f64], reference: &[f64]) -> Result<(f64, f64), String> {
    if actual.len() != 14
        || reference.len() != 14
        || actual.iter().chain(reference).any(|v| !v.is_finite())
    {
        return Err("invalid cavity centerline samples".into());
    }
    let error = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let scale = reference.iter().map(|v| v * v).sum::<f64>();
    let maximum = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    Ok(((error / scale).sqrt(), maximum))
}

fn run() -> Result<sonium_qa::QaResult, String> {
    use sha2::{Digest, Sha256};
    let reference: serde_json::Value =
        serde_json::from_slice(REFERENCE_BYTES).map_err(|e| e.to_string())?;
    let levels = reference["levels"]
        .as_array()
        .ok_or("missing independent reference levels")?;
    let fine_reference = levels.last().ok_or("empty independent reference levels")?;
    if fine_reference["subdivisions"] != 512
        || reference["mms_refinement_rate"].as_f64().unwrap_or(0.0) < 1.8
    {
        return Err("unresolved independent cavity reference".into());
    }
    let reference_change = reference["maximum_adjacent_centerline_changes"][1]
        .as_f64()
        .ok_or("missing reference refinement")?;
    if !reference_change.is_finite() || reference_change > 0.0002 {
        return Err("independent cavity reference uncertainty exceeds budget".into());
    }
    let reference_samples = reference["extrapolated_centerlines"]
        .as_array()
        .ok_or("missing extrapolated reference")?
        .iter()
        .map(|v| v.as_f64().ok_or("invalid extrapolated reference sample"))
        .collect::<Result<Vec<_>, _>>()?;
    let extrapolation_change = reference["maximum_extrapolation_change"]
        .as_f64()
        .ok_or("missing extrapolation refinement")?;
    if !extrapolation_change.is_finite() || extrapolation_change > 2e-6 {
        return Err("independent cavity extrapolation exceeds uncertainty budget".into());
    }
    let mut errors = Vec::new();
    let mut maxima = Vec::new();
    let mut divergences = Vec::new();
    let mut residuals = Vec::new();
    for n in [16, 32, 64] {
        let metrics = solve_cavity(n)?;
        let (error, maximum) = centerline_errors(&metrics.centerlines, &reference_samples)?;
        println!(
            "n={n}: independent centerline relative L2={error:.3e}, maximum absolute error={maximum:.3e}"
        );
        errors.push(error);
        maxima.push(maximum);
        divergences.push(metrics.bulk_divergence_rms);
        residuals.push(metrics.true_residual);
    }
    let contraction = (errors[1] / errors[0]).max(errors[2] / errors[1]);
    let divergence_contraction =
        (divergences[1] / divergences[0]).max(divergences[2] / divergences[1]);
    let checksum = format!("sha256:{:x}", Sha256::digest(REFERENCE_BYTES));
    let record=sonium_qa::QaResult::passed(CASE_ID)
        .requested_backend("cpu").actual_backend("cpu")
        .seed("unit-square:unit-top-lid:corner-zero:P2-P1:n16-n32-n64:bulk-1/8-7/8")
        .reference("independent clamped-biharmonic finite differences n128/256/512 with verified second-order Richardson extrapolation; velocity from streamfunction; bulk divergence zero")
        .reference_checksum(checksum)
        .formulation("native mixed Taylor-Hood Stokes solve; fourteen interior centerline components; quadrature-integrated bulk divergence")
        .metric("centerline_relative_l2",errors[2]).maximum("centerline_relative_l2",0.001)
        .metric("centerline_maximum_absolute_error",maxima[2]).maximum("centerline_maximum_absolute_error",0.0001)
        .metric("centerline_refinement_contraction",contraction).maximum("centerline_refinement_contraction",0.8)
        .metric("bulk_divergence_rms",divergences[2]).maximum("bulk_divergence_rms",0.005)
        .metric("bulk_divergence_refinement_contraction",divergence_contraction).maximum("bulk_divergence_refinement_contraction",0.8)
        .metric("reference_maximum_extrapolation_change",extrapolation_change).maximum("reference_maximum_extrapolation_change",2e-6)
        .metric("reference_maximum_adjacent_change",reference_change).maximum("reference_maximum_adjacent_change",0.0002)
        .metric("maximum_true_residual",residuals.into_iter().fold(0.0_f64,f64::max)).maximum("maximum_true_residual",1e-8)
        .metric("numerical_solve_executed",3.0).minimum("numerical_solve_executed",3.0);
    record.release_eligible()?;
    Ok(record)
}

fn apply_dirichlet(system: &mut FemSystem<f64>, bc_map: &HashMap<usize, f64>) {
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row)
            && let Some(&val) = bc_map.get(&col)
        {
            system.rhs[row] -= system.matrix.values[i] * val;
        }
    }
    let mut nr = Vec::with_capacity(system.matrix.nnz());
    let mut nc = Vec::with_capacity(system.matrix.nnz());
    let mut nv = Vec::with_capacity(system.matrix.nnz());
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !bc_map.contains_key(&row) && !bc_map.contains_key(&col) {
            nr.push(row);
            nc.push(col);
            nv.push(system.matrix.values[i]);
        }
    }
    for (&dof, &val) in bc_map {
        nr.push(dof);
        nc.push(dof);
        nv.push(1.0);
        system.rhs[dof] = val;
    }
    system.matrix.rows = nr;
    system.matrix.cols = nc;
    system.matrix.values = nv;
}

fn main() {
    env_logger::init();
    match run().and_then(|record| record.to_json_line()) {
        Ok(line) => {
            println!("QA_RESULT: {line}");
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(line) = sonium_qa::QaResult::failed(CASE_ID, &error).to_json_line() {
                println!("QA_RESULT: {line}");
            }
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prescribed_lid_without_interior_flow_fails_reference_accuracy() {
        let reference: serde_json::Value = serde_json::from_slice(REFERENCE_BYTES).unwrap();
        let mut samples = Vec::new();
        for key in ["vertical_ux", "horizontal_uy"] {
            samples.extend(
                reference["levels"][2][key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap()),
            );
        }
        let (relative, maximum) = centerline_errors(&[0.0; 14], &samples).unwrap();
        assert!(relative > 0.01 && maximum > 0.003);
        assert!(centerline_errors(&[f64::NAN; 14], &samples).is_err());
    }
}
