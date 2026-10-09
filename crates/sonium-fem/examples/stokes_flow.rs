//! Stokes flow solver with CLI — supports built-in benchmarks and external Gmsh meshes
//!
//! Usage:
//!   cargo run -p sonium-fem --features cli --example stokes_flow -- --help
//!   cargo run -p sonium-fem --features cli --example stokes_flow -- poiseuille -n 16
//!   cargo run -p sonium-fem --features cli --example stokes_flow -- cavity -n 16
//!   cargo run -p sonium-fem --features cli --example stokes_flow -- couette -n 8 --n-angular 32
//!   cargo run -p sonium-fem --features cli --example stokes_flow -- mesh -f channel.msh --bc-walls 1,2 --bc-inlet 3 --bc-outlet 4

use clap::{Parser, Subcommand};
use std::collections::HashMap;

use sonium_fem::assembly;
use sonium_fem::assembly::TripletMatrix;
use sonium_fem::assembly::mixed::assemble_mixed;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::FemSystem;
use sonium_fem::forms::MassForm;
use sonium_fem::mesh::Mesh;
use sonium_fem::mesh::generators::{
    annular_mesh_triangles, rectangular_mesh_triangles, unit_square_triangles,
};
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::fluid::stokes::forms::{StokesDivergenceForm, StokesViscousForm};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use sonium_solvers::{
    AmgConfig, BlockPreconditionerConfig, FgmresConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, fgmres_preconditioned,
};

#[derive(Parser)]
#[command(name = "stokes_flow", about = "Stokes flow solver (Taylor-Hood P2-P1)")]
struct Cli {
    #[command(subcommand)]
    problem: Problem,

    /// Viscosity
    #[arg(long, default_value = "1.0", global = true)]
    mu: f64,

    /// Output VTK file path
    #[arg(short, long, global = true)]
    output: Option<String>,
}

#[derive(Subcommand)]
enum Problem {
    /// 2D Poiseuille channel flow (analytical: parabolic velocity, linear pressure)
    Poiseuille {
        /// Mesh resolution (n x n grid)
        #[arg(short, long, default_value = "16")]
        n: usize,
    },
    /// 2D lid-driven cavity (top wall moves at u=1)
    Cavity {
        /// Mesh resolution
        #[arg(short, long, default_value = "16")]
        n: usize,
    },
    /// 2D Taylor-Couette flow between concentric cylinders
    Couette {
        /// Number of radial layers
        #[arg(short, long, default_value = "8")]
        n: usize,
        /// Number of angular divisions
        #[arg(long, default_value = "32")]
        n_angular: usize,
        /// Inner radius
        #[arg(long, default_value = "1.0")]
        r1: f64,
        /// Outer radius
        #[arg(long, default_value = "2.0")]
        r2: f64,
        /// Angular velocity of inner cylinder
        #[arg(long, default_value = "1.0")]
        omega: f64,
    },
    /// Load external Gmsh .msh file
    Mesh {
        /// Path to .msh file
        #[arg(short, long)]
        file: String,
        /// Boundary markers for no-slip walls (comma-separated)
        #[arg(long, value_delimiter = ',')]
        bc_walls: Vec<i32>,
        /// Boundary marker for inlet (prescribed parabolic velocity)
        #[arg(long)]
        bc_inlet: Option<i32>,
        /// Inlet velocity direction component (0=x, 1=y)
        #[arg(long, default_value = "0")]
        inlet_component: usize,
        /// Maximum inlet velocity
        #[arg(long, default_value = "1.0")]
        inlet_max_vel: f64,
        /// Boundary marker for outlet (p=0)
        #[arg(long)]
        bc_outlet: Option<i32>,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let mu = cli.mu;

    let (mesh, bc_map, n_v) = match &cli.problem {
        Problem::Poiseuille { n } => setup_poiseuille(*n, mu),
        Problem::Cavity { n } => setup_cavity(*n),
        Problem::Couette {
            n,
            n_angular,
            r1,
            r2,
            omega,
        } => setup_couette(*n, *n_angular, *r1, *r2, *omega),
        Problem::Mesh {
            file,
            bc_walls,
            bc_inlet,
            inlet_component,
            inlet_max_vel,
            bc_outlet,
        } => setup_from_gmsh(
            file,
            bc_walls,
            *bc_inlet,
            *inlet_component,
            *inlet_max_vel,
            *bc_outlet,
        ),
    };

    // --- Assemble and solve ---
    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v_computed = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let n_total = n_v_computed + n_p;
    assert_eq!(n_v, n_v_computed);

    println!("Velocity P2: {} DOFs", n_v);
    println!("Pressure P1: {} DOFs", n_p);
    println!("Total:       {} DOFs", n_total);
    println!("Constrained: {} DOFs\n", bc_map.len());

    println!("Assembling...");
    let viscous = StokesViscousForm::new(mu);
    let div_form = StokesDivergenceForm;
    let system = assemble_mixed(&viscous, &div_form, &v_space, &p_space, &mesh, 4);
    let mono = system.to_monolithic();

    let mp = assembly::assemble_bilinear(&MassForm, &p_space, &p_space, &mesh, 2);
    let mut schur = TripletMatrix::<f64>::with_capacity(mp.num_rows, mp.num_cols, mp.nnz());
    for i in 0..mp.nnz() {
        schur.push(mp.rows[i], mp.cols[i], mp.values[i] / mu);
    }

    let mut fem = FemSystem::new(mono, vec![0.0; n_total]);
    apply_dirichlet(&mut fem, &bc_map);

    println!("Solving...");
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

    let rhs: Vec<f64> = fem.rhs.clone();
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

    // --- VTK output ---
    let output_path = cli
        .output
        .unwrap_or_else(|| "stokes_output.vtk".to_string());
    let n_nodes = mesh.num_nodes();
    let u_sol: Vec<f64> = result.x.iter().take(n_v).copied().collect();
    let p_sol: Vec<f64> = result.x.iter().skip(n_v).copied().collect();

    let mut node_ux = vec![0.0; n_nodes];
    let mut node_uy = vec![0.0; n_nodes];
    let mut node_p = vec![0.0; n_nodes];
    for i in 0..n_nodes {
        node_ux[i] = u_sol[i * 2];
        node_uy[i] = u_sol[i * 2 + 1];
        if i < p_sol.len() {
            node_p[i] = p_sol[i];
        }
    }

    let speed: Vec<f64> = (0..n_nodes)
        .map(|i| (node_ux[i].powi(2) + node_uy[i].powi(2)).sqrt())
        .collect();

    let mut writer = sonium_fem::io::vtk::VtuWriter::new(&mesh);
    writer.add_point_scalar("ux", &node_ux);
    writer.add_point_scalar("uy", &node_uy);
    writer.add_point_scalar("pressure", &node_p);
    writer.add_point_scalar("speed", &speed);

    match writer.write_ascii(&output_path) {
        Ok(_) => println!("Wrote {output_path}"),
        Err(e) => println!("Failed to write VTK: {e}"),
    }

    // PyVista output (numpy .npy files)
    let pv_prefix = output_path
        .trim_end_matches(".vtu")
        .trim_end_matches(".vtk");
    let mut pv_writer = sonium_fem::io::pyvista::PyVistaWriter::new(&mesh);
    pv_writer.add_point_scalar("ux", &node_ux);
    pv_writer.add_point_scalar("uy", &node_uy);
    pv_writer.add_point_scalar("pressure", &node_p);
    pv_writer.add_point_scalar("speed", &speed);
    let uz = vec![0.0; n_nodes];
    pv_writer.add_point_vector("velocity", &node_ux, &node_uy, &uz);

    match pv_writer.write(pv_prefix) {
        Ok(_) => println!("Wrote {pv_prefix}_load.py (run with: python {pv_prefix}_load.py)"),
        Err(e) => println!("Failed to write PyVista files: {e}"),
    }
}

// =====================================================================
// Problem setup functions
// =====================================================================

/// Compute P2 DOF coordinates for a mesh.
fn dof_coordinates(mesh: &Mesh) -> (Vec<f64>, Vec<f64>, usize) {
    let topo = MeshTopology::build(mesh);
    let n_nodes = mesh.num_nodes();
    let n_edges = topo.num_edges();
    let n_scalar = n_nodes + n_edges;

    let mut dx = vec![0.0; n_scalar];
    let mut dy = vec![0.0; n_scalar];
    for i in 0..n_nodes {
        dx[i] = mesh.nodes[i].x;
        dy[i] = mesh.nodes[i].y;
    }
    for (ei, &(a, b)) in topo.edges.iter().enumerate() {
        dx[n_nodes + ei] = 0.5 * (mesh.nodes[a].x + mesh.nodes[b].x);
        dy[n_nodes + ei] = 0.5 * (mesh.nodes[a].y + mesh.nodes[b].y);
    }
    (dx, dy, n_scalar)
}

fn setup_poiseuille(n: usize, mu: f64) -> (Mesh, HashMap<usize, f64>, usize) {
    println!("=== Stokes Poiseuille Flow ===");
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let (dx, dy, n_scalar) = dof_coordinates(&mesh);
    let n_v = n_scalar * 2;
    let tol = 1e-10;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dx[sd];
        let y = dy[sd];
        if y.abs() < tol || (y - 1.0).abs() < tol {
            bc_map.insert(sd * 2, 0.0);
            bc_map.insert(sd * 2 + 1, 0.0);
        } else if x.abs() < tol {
            bc_map.insert(sd * 2, 4.0 * y * (1.0 - y));
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }
    for node in 0..mesh.num_nodes() {
        if (mesh.nodes[node].x - 1.0).abs() < tol {
            bc_map.insert(n_v + node, 0.0);
        }
    }

    // Print analytical reference
    println!("Analytical: u_x = 4y(1-y), p = 8*mu*(1-x)");
    let _ = mu;

    (mesh, bc_map, n_v)
}

fn setup_cavity(n: usize) -> (Mesh, HashMap<usize, f64>, usize) {
    println!("=== Stokes Lid-Driven Cavity ===");
    let mesh = unit_square_triangles(n);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let (dx, dy, n_scalar) = dof_coordinates(&mesh);
    let n_v = n_scalar * 2;
    let tol = 1e-10;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dx[sd];
        let y = dy[sd];
        let on_boundary =
            x.abs() < tol || (x - 1.0).abs() < tol || y.abs() < tol || (y - 1.0).abs() < tol;
        if on_boundary {
            if (y - 1.0).abs() < tol && x > tol && x < 1.0 - tol {
                bc_map.insert(sd * 2, 1.0);
            } else {
                bc_map.insert(sd * 2, 0.0);
            }
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }
    bc_map.insert(n_v, 0.0); // pin pressure

    (mesh, bc_map, n_v)
}

fn setup_couette(
    n_radial: usize,
    n_angular: usize,
    r1: f64,
    r2: f64,
    omega: f64,
) -> (Mesh, HashMap<usize, f64>, usize) {
    println!("=== Stokes Taylor-Couette ===");
    let mesh = annular_mesh_triangles(0.0, 0.0, r1, r2, n_radial, n_angular);
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let (dx, dy, n_scalar) = dof_coordinates(&mesh);
    let n_v = n_scalar * 2;
    let tol_inner = r1 * 0.02;
    let tol_outer = r2 * 0.02;
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    for sd in 0..n_scalar {
        let x = dx[sd];
        let y = dy[sd];
        let r = (x * x + y * y).sqrt();
        if (r - r1).abs() < tol_inner {
            bc_map.insert(sd * 2, -omega * y);
            bc_map.insert(sd * 2 + 1, omega * x);
        } else if (r - r2).abs() < tol_outer {
            bc_map.insert(sd * 2, 0.0);
            bc_map.insert(sd * 2 + 1, 0.0);
        }
    }
    for node in 0..mesh.num_nodes() {
        let r = (mesh.nodes[node].x.powi(2) + mesh.nodes[node].y.powi(2)).sqrt();
        if (r - r2).abs() < tol_outer {
            bc_map.insert(n_v + node, 0.0);
            break;
        }
    }

    println!(
        "Analytical: u_theta(r) = {:.4}*r + {:.4}/r",
        -omega * r1 * r1 / (r2 * r2 - r1 * r1),
        omega * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1)
    );

    (mesh, bc_map, n_v)
}

fn setup_from_gmsh(
    path: &str,
    bc_walls: &[i32],
    bc_inlet: Option<i32>,
    inlet_component: usize,
    inlet_max_vel: f64,
    bc_outlet: Option<i32>,
) -> (Mesh, HashMap<usize, f64>, usize) {
    println!("=== Stokes Flow — External Mesh ===");
    println!("Loading {path}...");

    let mesh = sonium_fem::mesh::gmsh::load_gmsh(path)
        .unwrap_or_else(|e| panic!("Failed to load mesh: {e}"));
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let (dx, dy, n_scalar) = dof_coordinates(&mesh);
    let n_v = n_scalar * 2;
    let n_nodes = mesh.num_nodes();

    let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let mut bc_map: HashMap<usize, f64> = HashMap::new();

    // No-slip walls
    for &marker in bc_walls {
        let v_dofs = v_space.boundary_dofs(&mesh, marker);
        for dof in v_dofs {
            bc_map.insert(dof, 0.0);
        }
    }

    // Inlet: parabolic profile along the inlet boundary
    if let Some(marker) = bc_inlet {
        let v_dofs = v_space.boundary_dofs(&mesh, marker);
        // Find inlet extent to compute parabolic profile
        let inlet_scalar: Vec<usize> = p_space.boundary_dofs(&mesh, marker);
        if !inlet_scalar.is_empty() {
            // Get perpendicular coordinate range for parabolic profile
            let perp_component = 1 - inlet_component;
            let coords: Vec<f64> = inlet_scalar
                .iter()
                .filter(|&&sd| sd < n_nodes)
                .map(|&sd| if perp_component == 0 { dx[sd] } else { dy[sd] })
                .collect();
            let y_min = coords.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let h = y_max - y_min;

            for &dof in &v_dofs {
                let sd = dof / 2;
                let comp = dof % 2;
                let perp = if perp_component == 0 { dx[sd] } else { dy[sd] };
                let t = (perp - y_min) / h.max(1e-15);
                if comp == inlet_component {
                    bc_map.insert(dof, inlet_max_vel * 4.0 * t * (1.0 - t));
                } else {
                    bc_map.insert(dof, 0.0);
                }
            }
        }
    }

    // Outlet: p = 0
    if let Some(marker) = bc_outlet {
        let p_dofs = p_space.boundary_dofs(&mesh, marker);
        for dof in p_dofs {
            bc_map.insert(n_v + dof, 0.0);
        }
    } else {
        // Pin one pressure DOF to remove nullspace
        bc_map.insert(n_v, 0.0);
    }

    (mesh, bc_map, n_v)
}

// =====================================================================
// Shared utilities
// =====================================================================

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
