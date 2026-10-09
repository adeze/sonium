use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_homogeneous_dirichlet};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::solid::composites::{
    Laminate, LaminateMembraneForm, OrthotropicPly, OrthotropicStrength,
    build_composite_bending_assembler,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{GmresConfig, IluPreconditioner, gmres_preconditioned};

fn test_composites_unidirectional_tension() {
    let l = 1.0;
    let h = 0.1;
    let nx = 20;
    let ny = 2;

    let e1 = 140.0e9;
    let e2 = 10.0e9;
    let g12 = 5.0e9;
    let nu12 = 0.25;
    let thickness = 0.001;
    let total_force = 1.0e3;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);

    let ply = OrthotropicPly {
        e1,
        e2,
        g12,
        nu12,
        thickness,
        angle_rad: 0.0,
    };
    let laminate = Laminate { plies: vec![ply] };
    let form = LaminateMembraneForm::new(&laminate);

    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);

    let tol = 1e-10;
    let tip_nodes: Vec<usize> = (0..mesh.num_nodes())
        .filter(|&i| (mesh.nodes[i].x - l).abs() < tol)
        .collect();

    let mut rhs = vec![0.0; n_dofs];
    let force_per_node = total_force / tip_nodes.len() as f64;
    for &node in &tip_nodes {
        let u_dof = node * 2;
        rhs[u_dof] = force_per_node;
    }

    let mut system = FemSystem::new(stiffness, rhs);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);

    let csr = system.matrix.to_csr();
    let precond = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(
        &csr,
        &precond,
        &system.rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES+ILU should converge for composites");

    // For a [0] laminate under uniaxial tension, the effective stiffness is A11/h (N/m^2)
    // where h is the strip width. The strain is epsilon = F / (A11 * h / L * L) = F / (A11 * h)
    // Wait, that's not right. Let me think again.
    //
    // For a 1D bar analogy: K = A11 * h / L, and u = F / K = F * L / (A11 * h)
    // But A11 has units N/m, h has units m, so A11*h has units N.
    // Then u = F*L/(A11*h) has units m, which is correct.
    let a_matrix = laminate.membrane_stiffness();
    let a11 = a_matrix[0][0];
    let tip_exact = total_force * l / (a11 * h);

    let mut max_err: f64 = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let x = node.x;
        let u_exact_x = tip_exact * (x / l);
        let u_fem_x = result.x[i * 2];
        max_err = max_err.max((u_fem_x - u_exact_x).abs());
    }

    let tip_fem = result.x[tip_nodes[0] * 2];
    println!(
        "Composites tension: a11={:.4e}, max_err={:.4e}, tip_exact={:.6e}, tip_fem={:.6e}",
        a11, max_err, tip_exact, tip_fem
    );

    // Use a loose tolerance since point-load distribution on coarse mesh introduces error
    assert!(
        max_err < 1e-5,
        "Composites tension error too large: {:.4e}",
        max_err
    );
}

fn test_laminate_abd_and_failure_indices() {
    let ply0 = OrthotropicPly {
        e1: 140.0e9,
        e2: 10.0e9,
        g12: 5.0e9,
        nu12: 0.25,
        thickness: 0.125e-3,
        angle_rad: 0.0,
    };
    let ply90 = OrthotropicPly {
        angle_rad: std::f64::consts::FRAC_PI_2,
        ..ply0
    };
    let symmetric = Laminate {
        plies: vec![ply0, ply90, ply90, ply0],
    };
    let unsymmetric = Laminate {
        plies: vec![ply0, ply90],
    };
    let abd_symmetric = symmetric.stiffness_abd();
    let abd_unsymmetric = unsymmetric.stiffness_abd();
    let strength = OrthotropicStrength {
        x_tension: 1500.0e6,
        x_compression: 1200.0e6,
        y_tension: 40.0e6,
        y_compression: 200.0e6,
        shear: 80.0e6,
    };
    let index = strength.tsai_hill_index([300.0e6, 10.0e6, 5.0e6]);
    let max_stress = strength.max_stress_index([300.0e6, 10.0e6, 5.0e6]);

    println!(
        "Laminate ABD: symmetric B11={:.4e}, unsymmetric B11={:.4e}, failure={:.4e}/{:.4e}",
        abd_symmetric.b[0][0], abd_unsymmetric.b[0][0], index, max_stress
    );
    assert!(abd_symmetric.b[0][0].abs() < 1.0e-8);
    assert!(abd_unsymmetric.b[0][0].abs() > 1.0);
    assert!(index > 0.0 && index < 1.0);
    assert!(max_stress > 0.0 && max_stress < 1.0);
}

fn test_composite_bending_assembler_builds() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.1, 2, 1);
    let laminate = Laminate {
        plies: vec![OrthotropicPly {
            e1: 10.0,
            e2: 10.0,
            g12: 4.0,
            nu12: 0.25,
            thickness: 0.01,
            angle_rad: 0.0,
        }],
    };
    let assembler = build_composite_bending_assembler(&mesh, PolynomialDegree::P2, &laminate);
    println!("Composite bending assembler nnz={}", assembler.nnz());
    assert!(assembler.nnz() > 0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: composites tension ===");
        test_composites_unidirectional_tension();
        test_laminate_abd_and_failure_indices();
        test_composite_bending_assembler_builds();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
