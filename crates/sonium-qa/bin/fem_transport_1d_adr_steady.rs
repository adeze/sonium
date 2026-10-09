use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::forms::BilinearForm;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::transport::{
    LinearReactionNetwork, TransportAdvectionForm, TransportDiffusionForm, TransportMaterial,
    TransportReactionForm, TransportReactionJacobianForm, TransportSupgForm,
};
use sonium_fem::spaces::BasisTabulation;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn exact_adr_1d(x: f64, l: f64, pe: f64, reaction: f64) -> f64 {
    if reaction.abs() < 1e-12 {
        // Pure advection-diffusion
        if pe.abs() < 1e-10 {
            return x / l;
        }
        (1.0 - (pe * x / l).exp()) / (1.0 - pe.exp())
    } else {
        // Advection-diffusion-reaction
        let d = 1.0;
        let v = pe * d / l;
        let disc = (v * v + 4.0 * d * reaction).sqrt();
        let lambda1 = (v + disc) / (2.0 * d);
        let lambda2 = (v - disc) / (2.0 * d);
        let e1l = (lambda1 * l).exp();
        let e2l = (lambda2 * l).exp();
        let e1x = (lambda1 * x).exp();
        let e2x = (lambda2 * x).exp();
        (e1x - e2x) / (e1l - e2l)
    }
}

fn solve_adr_steady(l: f64, nx: usize, diffusivity: f64, velocity: f64, reaction: f64) -> f64 {
    let h = 0.1;
    let ny = 2;
    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let pe = velocity * l / diffusivity;
    let material =
        TransportMaterial::with_capacity(diffusivity, [velocity, 0.0, 0.0], reaction, 1.0);

    let diff_matrix = assemble_bilinear(
        &TransportDiffusionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    );
    let adv_matrix = assemble_bilinear(
        &TransportAdvectionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    );
    let reac_matrix = assemble_bilinear(
        &TransportReactionForm::new(&material),
        &space,
        &space,
        &mesh,
        2,
    );

    // Combine matrices into one system
    let mut system = FemSystem::new(diff_matrix, vec![0.0; n_dofs]);
    for i in 0..adv_matrix.nnz() {
        system
            .matrix
            .push(adv_matrix.rows[i], adv_matrix.cols[i], adv_matrix.values[i]);
    }
    for i in 0..reac_matrix.nnz() {
        system.matrix.push(
            reac_matrix.rows[i],
            reac_matrix.cols[i],
            reac_matrix.values[i],
        );
    }

    // Dirichlet BCs: u(0) = 0, u(L) = 1
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| 0.0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| 1.0, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES did not converge");

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let exact = exact_adr_1d(node.x, l, pe, reaction);
        let err = (result.x[i] - exact).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

fn test_adr_diffusion_dominated() {
    let l = 1.0;
    let diffusivity = 2.0;
    let velocity = 1.0;
    let reaction = 0.0;
    let pe = velocity * l / diffusivity;

    let max_err = solve_adr_steady(l, 40, diffusivity, velocity, reaction);
    println!(
        "ADR diffusion-dominated (Pe={:.1}): max_err={:.4e}",
        pe, max_err
    );
    assert!(
        max_err < 0.01,
        "Diffusion-dominated error too large: {:.4e}",
        max_err
    );
}

fn test_adr_moderate_pe() {
    let l = 1.0;
    let diffusivity = 1.0;
    let velocity = 5.0;
    let reaction = 0.0;
    let pe = velocity * l / diffusivity;

    let max_err = solve_adr_steady(l, 100, diffusivity, velocity, reaction);
    println!("ADR moderate Pe (Pe={:.0}): max_err={:.4e}", pe, max_err);
    assert!(
        max_err < 0.001,
        "Moderate Pe error too large: {:.4e}",
        max_err
    );
}

fn test_adr_with_reaction() {
    let l = 1.0;
    let diffusivity = 1.0;
    let velocity = 2.0;
    let reaction = 1.0;
    let pe = velocity * l / diffusivity;

    let max_err = solve_adr_steady(l, 80, diffusivity, velocity, reaction);
    println!(
        "ADR with reaction (Pe={:.0}, k={:.0}): max_err={:.4e}",
        pe, reaction, max_err
    );
    assert!(max_err < 0.001, "Reaction error too large: {:.4e}", max_err);
}

fn test_adr_refinement() {
    let l = 1.0;
    let diffusivity = 1.0;
    let velocity = 2.0;
    let reaction = 0.0;

    println!("ADR refinement study:");
    let mut prev_err = 0.0;
    let mut prev_nx = 0.0;
    for nx in [20, 40, 80, 160] {
        let max_err = solve_adr_steady(l, nx, diffusivity, velocity, reaction);
        println!("  nx={:3} max_err={:.4e}", nx, max_err);
        if prev_err > 0.0 {
            let rate = (prev_err / max_err).ln() / ((prev_nx / nx as f64).ln());
            println!("    rate={:.2}", rate);
        }
        prev_err = max_err;
        prev_nx = nx as f64;
    }
}

fn test_supg_and_multispecies_reaction_blocks() {
    let material = TransportMaterial::with_capacity(1.0e-3, [2.0, 0.0, 0.0], 0.5, 1.0);
    let tab = BasisTabulation {
        n_dofs: 2,
        spatial_dim: 1,
        values: vec![0.25, 0.75],
        derivatives: vec![1.0, -1.0],
        hessians: vec![],
        det_j: 0.5,
        physical_coords: [0.0, 0.0, 0.0],
    };
    let mut supg = sonium_fem::forms::LocalMatrix::zeros(2);
    TransportSupgForm {
        material,
        element_length_scale: Some(1.0),
        dt: None,
    }
    .integrate_point(&tab, &tab, 2.0, &mut supg);

    let network = LinearReactionNetwork {
        jacobian: vec![vec![-1.0, 0.25], vec![1.0, -0.5]],
        source: vec![0.0, 0.0],
    };
    let mut block = sonium_fem::forms::LocalMatrix::zeros(2);
    TransportReactionJacobianForm {
        network: &network,
        target_species: 0,
        source_species: 1,
    }
    .integrate_point(&tab, &tab, 1.0, &mut block);

    println!(
        "SUPG/reaction blocks: supg_00={:.4e}, reaction_01={:.4e}",
        supg.get(0, 0),
        block.get(0, 1)
    );
    assert!(supg.get(0, 0) > 0.0);
    assert!(block.get(0, 1) > 0.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: transport 1D ADR steady ===");
        test_adr_diffusion_dominated();
        test_adr_moderate_pe();
        test_adr_with_reaction();
        test_adr_refinement();
        test_supg_and_multispecies_reaction_blocks();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
