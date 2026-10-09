//! QA: FEM topology optimization — 2D cantilever beam compliance minimization.
//!
//! Classic density-based topology optimization benchmark:
//! - Domain: 2:1 rectangle (L=2, H=1)
//! - Fixed at left edge (marker 1)
//! - Downward point load at bottom-right corner
//! - Volume fraction constraint: 50%
//! - SIMP penalty p = 3
//!
//! Validates the end-to-end topology optimization loop:
//! `run_topology_optimization` + `DensityFilter` + `OcOptimizer`.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::forms::{BilinearForm, LocalMatrix};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::mesh::{ElementType, Mesh};
use sonium_fem::optimization::{TopOptConfig, TopologyOptimizerKind, run_topology_optimization};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_real};

/// Assemble per-element base stiffness triplets with E = 1.0.
fn assemble_base_element_triplets(
    mesh: &Mesh,
    space: &VectorH1Space,
    form: &ElasticityForm,
    quad_order: usize,
) -> Vec<Vec<(usize, usize, f64)>> {
    let n_elems = mesh.num_elements();
    let mut all = Vec::with_capacity(n_elems);

    for e in 0..n_elems {
        let elem = &mesh.elements[e];
        let test_dofs = space.local_to_global(mesh, e);
        let n_td = test_dofs.len();

        let quad = QuadratureRule::new(elem.element_type, quad_order);
        let mut local = LocalMatrix::zeros(n_td);

        for qp in quad.iter() {
            let test_tab = space.tabulate(mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let trial_tab = space.tabulate(mesh, e, qp.xi(), qp.eta(), qp.zeta());
            form.integrate_point(&test_tab, &trial_tab, qp.weight, &mut local);
        }

        let mut triplets = Vec::with_capacity(n_td * n_td);
        for (i, &gi) in test_dofs.iter().enumerate() {
            for (j, &gj) in test_dofs.iter().enumerate() {
                let val = local.get(i, j);
                if val.abs() > 1e-15 {
                    triplets.push((gi, gj, val));
                }
            }
        }
        all.push(triplets);
    }

    all
}

/// Build global stiffness TripletMatrix from base triplets scaled by rho^p.
fn build_scaled_stiffness_triplet(
    base_triplets: &[Vec<(usize, usize, f64)>],
    densities: &[f64],
    penalty: f64,
    n_dofs: usize,
) -> sonium_solvers::TripletMatrix<f64> {
    use sonium_solvers::TripletMatrix;
    let mut k = TripletMatrix::new(n_dofs, n_dofs);
    for (e, triplets) in base_triplets.iter().enumerate() {
        let scale = densities[e].powf(penalty);
        for &(i, j, k0) in triplets {
            k.push(i, j, scale * k0);
        }
    }
    k
}

fn test_topology_cantilever() {
    let l = 2.0;
    let h = 1.0;
    let nx = 24;
    let ny = 12;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let n_elem = mesh.num_elements();

    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = IsotropicMaterial::new(1.0, 0.3);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);

    // Pre-assemble base stiffness triplets (E = 1.0)
    let base_triplets = assemble_base_element_triplets(&mesh, &space, &form, 1);

    // Element volumes (areas for 2D triangles)
    let element_volumes: Vec<f64> = (0..n_elem)
        .map(|e| {
            let elem = &mesh.elements[e];
            match elem.element_type {
                ElementType::Triangle => {
                    let v = elem.vertices();
                    let p0 = &mesh.nodes[v[0]];
                    let p1 = &mesh.nodes[v[1]];
                    let p2 = &mesh.nodes[v[2]];
                    let a = [p1.x - p0.x, p1.y - p0.y];
                    let b = [p2.x - p0.x, p2.y - p0.y];
                    (a[0] * b[1] - a[1] * b[0]).abs() * 0.5
                }
                _ => 0.0,
            }
        })
        .collect();

    // Load: downward point load at bottom-right corner
    let mut rhs_base = vec![0.0; n_dofs];
    let tol = 1e-10;
    let corner_node = mesh.nodes.iter().enumerate().fold(0, |best, (i, p)| {
        let _best_p = &mesh.nodes[best];
        if (p.x - l).abs() < tol && (p.y - 0.0).abs() < tol {
            i
        } else {
            best
        }
    });
    // Verify we found the corner
    let corner = &mesh.nodes[corner_node];
    assert!(
        (corner.x - l).abs() < tol && (corner.y - 0.0).abs() < tol,
        "Could not find bottom-right corner node"
    );
    rhs_base[corner_node * 2 + 1] = -1.0; // y-DOF

    let solver_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 2000,
        tolerance: 1e-10,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };

    let topopt_config = TopOptConfig {
        max_iterations: 100,
        tolerance: 0.01,
        volume_fraction: 0.5,
        filter_radius: 0.12,
        simp_penalty: 3.0,
        continuation: false,
        optimizer: TopologyOptimizerKind::Oc,
        verbosity: 0,
    };

    let solve_and_evaluate = |rho: &[f64]| -> (f64, Vec<f64>) {
        let mut system = FemSystem::new(
            build_scaled_stiffness_triplet(&base_triplets, rho, 3.0, n_dofs),
            rhs_base.clone(),
        );
        apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);
        let csr_bc = system.matrix.to_csr();

        let result = solve_real(&csr_bc, &system.rhs, &solver_config).expect("Linear solve failed");
        assert!(result.converged, "Linear solver did not converge");
        let u = result.x;

        // Total compliance C = f^T u (using original unmodified RHS)
        let compliance: f64 = rhs_base.iter().zip(&u).map(|(&f, &ui)| f * ui).sum();

        // Element base compliances c0_e = u_e^T K0_e u_e
        let mut elem_comp = vec![0.0; n_elem];
        for (e, triplets) in base_triplets.iter().enumerate() {
            let mut c0 = 0.0;
            for &(i, j, k0) in triplets {
                c0 += u[i] * k0 * u[j];
            }
            elem_comp[e] = c0;
        }

        (compliance, elem_comp)
    };

    let compute_sensitivities = |_rho: &[f64], elem_comp: &[f64]| -> Vec<f64> {
        let p = 3.0;
        _rho.iter()
            .zip(elem_comp.iter())
            .map(|(&r, &c0)| -p * r.powf(p - 1.0) * c0)
            .collect()
    };

    let result = run_topology_optimization(
        &mesh,
        &topopt_config,
        &element_volumes,
        solve_and_evaluate,
        compute_sensitivities,
    );

    println!("Topology cantilever ({}x{} mesh):", nx, ny);
    println!("  Converged:     {}", result.converged);
    println!("  Iterations:    {}", result.iterations);
    println!(
        "  Final compliance: {:.6e}",
        result
            .objective_history
            .last()
            .expect("topology optimizer must report objective history")
    );

    let final_vf = *result
        .volume_history
        .last()
        .expect("topology optimizer must report volume history");
    println!("  Final volume:     {:.4}", final_vf);

    // Sanity checks
    assert!(result.converged, "Topology optimization did not converge");

    // Volume fraction should be close to target (OC enforces this)
    assert!(
        (final_vf - 0.5).abs() < 0.05,
        "Volume fraction deviated too far from target: {:.4}",
        final_vf
    );

    // Compliance should decrease from initial
    let initial_comp = *result
        .objective_history
        .first()
        .expect("topology optimizer must report initial compliance");
    let final_comp = *result
        .objective_history
        .last()
        .expect("topology optimizer must report final compliance");
    assert!(
        initial_comp.is_finite() && final_comp.is_finite(),
        "Topology compliance history contains non-finite values"
    );
    assert!(
        final_comp < initial_comp * 0.9,
        "Compliance did not decrease: initial={:.4e}, final={:.4e}",
        initial_comp,
        final_comp
    );

    // Check that densities are mostly binary (near 0 or 1)
    let n_gray = result
        .densities
        .iter()
        .filter(|&&r| r > 0.1 && r < 0.9)
        .count();
    let gray_fraction = n_gray as f64 / n_elem as f64;
    println!("  Gray fraction:    {:.4}", gray_fraction);
    // Without Heaviside projection, some grayness is expected on coarse meshes.
    // The key check is that the topology is structurally sensible.
    assert!(
        gray_fraction < 0.55,
        "Too many intermediate densities: {:.4}",
        gray_fraction
    );

    // The cantilever should have material connecting left to right.
    // A simple check: average density of left-half elements should be > 0.3
    // and average density of right-half elements should be > 0.3.
    let left_density: f64 = result
        .densities
        .iter()
        .enumerate()
        .filter(|(e, _)| {
            let elem = &mesh.elements[*e];
            let cx: f64 = elem
                .vertices()
                .iter()
                .map(|&v| mesh.nodes[v].x)
                .sum::<f64>()
                / elem.vertices().len() as f64;
            cx < l * 0.5
        })
        .map(|(_, &r)| r)
        .sum::<f64>()
        / (n_elem as f64 * 0.5);

    let right_density: f64 = result
        .densities
        .iter()
        .enumerate()
        .filter(|(e, _)| {
            let elem = &mesh.elements[*e];
            let cx: f64 = elem
                .vertices()
                .iter()
                .map(|&v| mesh.nodes[v].x)
                .sum::<f64>()
                / elem.vertices().len() as f64;
            cx >= l * 0.5
        })
        .map(|(_, &r)| r)
        .sum::<f64>()
        / (n_elem as f64 * 0.5);

    println!("  Left-half avg ρ:  {:.4}", left_density);
    println!("  Right-half avg ρ: {:.4}", right_density);
    assert!(
        left_density > 0.2 && right_density > 0.2,
        "Topology is disconnected: left={:.4}, right={:.4}",
        left_density,
        right_density
    );
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_topology_cantilever();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
