//! Shell + edge-beam shape optimisation via adjoint Lagrangian
//!
//! A flat rectangular plate (DKT shell elements) with 4 H-beam stiffeners
//! along its edges. Clamped on the left side, point load at the far corner.
//!
//! ```text
//!     ==/-------------------/
//!    ==/                   /
//!   ==/                   /
//!  ==/                 * /     * = corner load
//! ==--------------------/
//! ```
//!
//! Finds the optimal H-section (b, h, t) for the edge beams that minimises
//! the corner deflection subject to a total beam cross-sectional area budget.
//!
//! Usage:
//!   cargo run -p sonium-fem --features cli --example shell_beam_shape_opt
//!   cargo run -p sonium-fem --features cli --example shell_beam_shape_opt -- --help
//!   cargo run -p sonium-fem --features cli --example shell_beam_shape_opt -- --tip-load 5000

use clap::Parser;
use std::collections::HashSet;

use sonium_fem::assembly::TripletMatrix;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection, HSection};
use sonium_fem::physics::solid::shell::{ShellElement, ShellSection, assemble_shell_stiffness};

use sonium_solvers::{
    AugmentedLagrangianConfig, Bound, Constraint, ConstraintKind, CsrMatrix, InnerSolver,
    LinearSolveConfig, LinearSolverType, OptimResult, augmented_lagrangian, dense::DenseMatrix,
};

// ── CLI ─────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "shell-beam-shape-opt")]
#[command(about = "Optimise H-beam edge stiffeners on a shell plate")]
struct Cli {
    /// Plate length along X (m)
    #[arg(long, default_value_t = 1.0)]
    lx: f64,
    /// Plate width along Y (m)
    #[arg(long, default_value_t = 0.6)]
    ly: f64,
    /// Mesh divisions along X
    #[arg(long, default_value_t = 16)]
    nx: usize,
    /// Mesh divisions along Y
    #[arg(long, default_value_t = 10)]
    ny: usize,

    /// Shell thickness (m)
    #[arg(long, default_value_t = 0.002)]
    shell_thickness: f64,

    /// Young's modulus for both shell and beams (Pa)
    #[arg(long, default_value_t = 200e9)]
    youngs_modulus: f64,
    /// Poisson's ratio for shell
    #[arg(long, default_value_t = 0.3)]
    poisson_ratio: f64,
    /// Density (kg/m³)
    #[arg(long, default_value_t = 7800.0)]
    density: f64,

    /// Tip load at far corner, positive downward (N)
    #[arg(long, default_value_t = 1000.0)]
    tip_load: f64,

    /// Maximum total beam cross-section area (m²) — applies to each beam
    #[arg(long, default_value_t = 0.001)]
    area_max: f64,

    // Design variable bounds
    /// Min flange width (m)
    #[arg(long, default_value_t = 0.005)]
    b_min: f64,
    /// Max flange width (m)
    #[arg(long, default_value_t = 0.06)]
    b_max: f64,
    /// Min web height (m)
    #[arg(long, default_value_t = 0.01)]
    h_min: f64,
    /// Max web height (m)
    #[arg(long, default_value_t = 0.12)]
    h_max: f64,
    /// Min thickness (m)
    #[arg(long, default_value_t = 0.001)]
    t_min: f64,
    /// Max thickness (m)
    #[arg(long, default_value_t = 0.01)]
    t_max: f64,

    /// Output directory for VTU files and pyvista script
    #[arg(long, default_value = "viz_output")]
    output_dir: String,

    // Initial guess
    /// Initial flange width (m)
    #[arg(long, default_value_t = 0.02)]
    b0: f64,
    /// Initial web height (m)
    #[arg(long, default_value_t = 0.04)]
    h0: f64,
    /// Initial thickness (m)
    #[arg(long, default_value_t = 0.003)]
    t0: f64,
}

// ── Problem definition ──────────────────────────────────────────────────

/// Plate geometry: structured grid of (nx+1)*(ny+1) nodes.
/// Node (i,j) has index j*(nx+1)+i.
struct PlateShellBeamProblem {
    lx: f64,
    ly: f64,
    nx: usize,
    ny: usize,
    shell_section: ShellSection,
    youngs_modulus: f64,
    density: f64,
    tip_load: f64,
    area_max: f64,
    bounds: [(f64, f64); 3],
}

/// 6 DOFs per node: (u, v, w, θx, θy, θz).
const DOFS_PER_NODE: usize = 6;

impl PlateShellBeamProblem {
    fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    fn n_dofs(&self) -> usize {
        self.n_nodes() * DOFS_PER_NODE
    }

    /// Node index for grid position (i, j).
    fn node_id(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }

    /// Node positions as 3D points (flat plate in XY plane, z=0).
    fn node_points(&self) -> Vec<Point> {
        let dx = self.lx / self.nx as f64;
        let dy = self.ly / self.ny as f64;
        let mut pts = Vec::with_capacity(self.n_nodes());
        for j in 0..=self.ny {
            for i in 0..=self.nx {
                pts.push(Point::new_3d(i as f64 * dx, j as f64 * dy, 0.0));
            }
        }
        pts
    }

    /// Build DKT shell elements for the plate.
    fn build_shell_elements(&self, pts: &[Point]) -> Vec<ShellElement> {
        let mut elems = Vec::with_capacity(self.nx * self.ny * 2);
        for j in 0..self.ny {
            for i in 0..self.nx {
                let n00 = self.node_id(i, j);
                let n10 = self.node_id(i + 1, j);
                let n01 = self.node_id(i, j + 1);
                let n11 = self.node_id(i + 1, j + 1);
                // Two triangles per cell (same diagonal split as rectangular_mesh_triangles)
                elems.push(ShellElement::from_points(
                    [n00, n10, n11],
                    &pts[n00],
                    &pts[n10],
                    &pts[n11],
                    self.shell_section,
                ));
                elems.push(ShellElement::from_points(
                    [n00, n11, n01],
                    &pts[n00],
                    &pts[n11],
                    &pts[n01],
                    self.shell_section,
                ));
            }
        }
        elems
    }

    /// Build beam elements along the 4 edges for a given BeamSection.
    /// Returns (elements, dof_map) where dof_map[i] maps beam element DOFs
    /// to global shell DOFs.
    fn build_edge_beams(&self, section: BeamSection) -> (Vec<BeamElement2D>, Vec<[usize; 6]>) {
        let dx = self.lx / self.nx as f64;
        let dy = self.ly / self.ny as f64;
        let mut elems = Vec::new();
        let mut dof_maps = Vec::new();

        // Bottom edge: y=0, nodes (i,0) for i=0..nx, beam along X
        for i in 0..self.nx {
            let n0 = self.node_id(i, 0);
            let n1 = self.node_id(i + 1, 0);
            elems.push(BeamElement2D::new(
                [n0, n1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            ));
            // Beam along X: u→u(0), w→w(2), θ→θy(4)
            dof_maps.push(dof_map_x(n0, n1));
        }

        // Top edge: y=ly, nodes (i, ny), beam along X
        for i in 0..self.nx {
            let n0 = self.node_id(i, self.ny);
            let n1 = self.node_id(i + 1, self.ny);
            elems.push(BeamElement2D::new(
                [n0, n1],
                i as f64 * dx,
                0.0,
                (i + 1) as f64 * dx,
                0.0,
                section,
            ));
            dof_maps.push(dof_map_x(n0, n1));
        }

        // Left edge: x=0, nodes (0, j), beam along Y
        for j in 0..self.ny {
            let n0 = self.node_id(0, j);
            let n1 = self.node_id(0, j + 1);
            elems.push(BeamElement2D::new(
                [n0, n1],
                0.0,
                j as f64 * dy,
                0.0,
                (j + 1) as f64 * dy,
                section,
            ));
            // Beam along Y: u→v(1), w→w(2), θ→θx(3)
            dof_maps.push(dof_map_y(n0, n1));
        }

        // Right edge: x=lx, nodes (nx, j), beam along Y
        for j in 0..self.ny {
            let n0 = self.node_id(self.nx, j);
            let n1 = self.node_id(self.nx, j + 1);
            elems.push(BeamElement2D::new(
                [n0, n1],
                0.0,
                j as f64 * dy,
                0.0,
                (j + 1) as f64 * dy,
                section,
            ));
            dof_maps.push(dof_map_y(n0, n1));
        }

        (elems, dof_maps)
    }

    /// Assemble beam stiffness contributions into the 6-DOF-per-node system.
    fn assemble_beam_stiffness(
        &self,
        elems: &[BeamElement2D],
        dof_maps: &[[usize; 6]],
    ) -> TripletMatrix<f64> {
        let n = self.n_dofs();
        let mut triplet = TripletMatrix::new(n, n);
        for (elem, dmap) in elems.iter().zip(dof_maps) {
            let k = elem.stiffness_global();
            for (i, &gi) in dmap.iter().enumerate() {
                for (j, &gj) in dmap.iter().enumerate() {
                    let val = k[i * 6 + j];
                    if val.abs() > 1e-15 {
                        triplet.push(gi, gj, val);
                    }
                }
            }
        }
        triplet
    }

    /// Clamped DOFs: all 6 DOFs at x=0 nodes.
    fn clamped_dofs(&self) -> HashSet<usize> {
        let mut clamped = HashSet::new();
        for j in 0..=self.ny {
            let node = self.node_id(0, j);
            for d in 0..DOFS_PER_NODE {
                clamped.insert(node * DOFS_PER_NODE + d);
            }
        }
        clamped
    }

    /// DOF index of the w-displacement at the loaded corner (x=lx, y=0).
    fn corner_w_dof(&self) -> usize {
        let node = self.node_id(self.nx, 0); // bottom-right corner
        node * DOFS_PER_NODE + 2 // w DOF
    }

    /// Assemble the full system K = K_shell + K_beams, apply Dirichlet, return CSR.
    fn assemble_system(&self, h_section: &HSection) -> (CsrMatrix<f64>, Vec<f64>) {
        let pts = self.node_points();
        let n = self.n_dofs();

        // Shell stiffness
        let shell_elems = self.build_shell_elements(&pts);
        let mut triplet = assemble_shell_stiffness(&shell_elems, n);

        // Beam stiffness
        let beam_section = h_section.to_beam_section(self.youngs_modulus, self.density);
        let (beam_elems, dof_maps) = self.build_edge_beams(beam_section);
        let k_beams = self.assemble_beam_stiffness(&beam_elems, &dof_maps);
        // Merge beam triplets into the shell triplet
        for i in 0..k_beams.nnz() {
            triplet.push(k_beams.rows[i], k_beams.cols[i], k_beams.values[i]);
        }

        // RHS: corner load
        let mut rhs = vec![0.0; n];
        rhs[self.corner_w_dof()] = -self.tip_load; // downward

        // Dirichlet BC
        let clamped = self.clamped_dofs();
        apply_dirichlet(&mut triplet, &mut rhs, &clamped);

        (triplet.to_csr(), rhs)
    }

    /// Assemble bending-only and axial-only beam contributions separately (for sensitivity).
    fn assemble_beam_bending_and_axial(
        &self,
        h_section: &HSection,
    ) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
        let beam_section = h_section.to_beam_section(self.youngs_modulus, self.density);
        let bending_section = BeamSection::euler_bernoulli(
            self.youngs_modulus,
            beam_section.moment_of_inertia,
            0.0,
            0.0,
        );
        let axial_section =
            BeamSection::euler_bernoulli(self.youngs_modulus, 0.0, beam_section.area, 0.0);

        let n = self.n_dofs();
        let clamped = self.clamped_dofs();

        let assemble_beams_with = |sec: BeamSection| -> CsrMatrix<f64> {
            let (elems, dof_maps) = self.build_edge_beams(sec);
            let mut trip = self.assemble_beam_stiffness(&elems, &dof_maps);
            let mut dummy_rhs = vec![0.0; n];
            apply_dirichlet(&mut trip, &mut dummy_rhs, &clamped);
            trip.to_csr()
        };

        (
            assemble_beams_with(bending_section),
            assemble_beams_with(axial_section),
        )
    }

    /// Forward solve: K u = F.
    fn forward_solve(&self, csr: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
        let config = LinearSolveConfig {
            solver: LinearSolverType::GmresIlu,
            max_iterations: 5000,
            tolerance: 1e-12,
            ..Default::default()
        };
        let result = sonium_solvers::solve_real(csr, rhs, &config).expect("forward solve failed");
        assert!(
            result.converged,
            "forward solve did not converge in {} iterations",
            result.iterations
        );
        result.x
    }

    /// Evaluate objective f(p) = |w_corner| and gradient via the adjoint method.
    fn evaluate(&self, params: &[f64; 3]) -> (f64, [f64; 3]) {
        let section = HSection::new(params[0], params[1], params[2]);
        let (csr, rhs) = self.assemble_system(&section);

        // Forward solve
        let u = self.forward_solve(&csr, &rhs);
        let tip_defl = -u[self.corner_w_dof()]; // positive downward

        // Adjoint solve: Kᵀλ = −∂f/∂u, with f = −u[corner_w]
        let n = self.n_dofs();
        let mut adj_rhs = vec![0.0_f64; n];
        adj_rhs[self.corner_w_dof()] = 1.0; // −(−1) = +1
        let lambda = self.forward_solve(&csr, &adj_rhs);

        // Sensitivity: λᵀ (∂K/∂p_i) u
        // Only the beam part depends on p. Shell is constant.
        // K_beams = I·K_bend_unit + A·K_axial_unit
        // ∂K/∂p_i = (∂I/∂p_i / I) K_bend + (∂A/∂p_i / A) K_axial
        let (k_bend, k_axial) = self.assemble_beam_bending_and_axial(&section);
        let k_bend_u = k_bend.matvec(&u);
        let k_axial_u = k_axial.matvec(&u);
        let lam_kb_u = lambda
            .iter()
            .zip(k_bend_u.iter())
            .map(|(l, ku)| l * ku)
            .sum::<f64>();
        let lam_ka_u = lambda
            .iter()
            .zip(k_axial_u.iter())
            .map(|(l, ku)| l * ku)
            .sum::<f64>();

        let ii = section.moment_of_inertia();
        let aa = section.area();
        assert!(
            ii > 0.0,
            "moment of inertia is zero — check HSection parameters"
        );
        assert!(
            aa > 0.0,
            "cross-sectional area is zero — check HSection parameters"
        );
        let di = section.inertia_derivatives();
        let da = section.area_derivatives();

        let mut grad = [0.0; 3];
        for i in 0..3 {
            grad[i] = (di[i] / ii) * lam_kb_u + (da[i] / aa) * lam_ka_u;
        }

        (tip_defl, grad)
    }

    /// Build a sonium Mesh for VTU output.
    fn build_mesh(&self) -> Mesh {
        let mut mesh = Mesh::new(3);
        let dx = self.lx / self.nx as f64;
        let dy = self.ly / self.ny as f64;
        for j in 0..=self.ny {
            for i in 0..=self.nx {
                mesh.add_node(Point::new_3d(i as f64 * dx, j as f64 * dy, 0.0));
            }
        }
        for j in 0..self.ny {
            for i in 0..self.nx {
                let n00 = self.node_id(i, j);
                let n10 = self.node_id(i + 1, j);
                let n01 = self.node_id(i, j + 1);
                let n11 = self.node_id(i + 1, j + 1);
                mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
                mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
            }
        }
        mesh
    }

    /// Extract per-node displacement fields (u, v, w) from the 6-DOF solution.
    fn extract_displacements(&self, sol: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nn = self.n_nodes();
        let mut ux = vec![0.0; nn];
        let mut uy = vec![0.0; nn];
        let mut uz = vec![0.0; nn];
        for i in 0..nn {
            ux[i] = sol[i * DOFS_PER_NODE];
            uy[i] = sol[i * DOFS_PER_NODE + 1];
            uz[i] = sol[i * DOFS_PER_NODE + 2];
        }
        (ux, uy, uz)
    }

    /// Compute membrane von Mises stress per element (CST: constant per triangle).
    ///
    /// ε = B·u_mem,  σ = D·ε,  σ_vm = √(σxx² − σxx·σyy + σyy² + 3τxy²)
    fn compute_von_mises(&self, sol: &[f64]) -> Vec<f64> {
        let pts = self.node_points();
        let shell_elems = self.build_shell_elements(&pts);
        let d = self.shell_section.membrane_constitutive();

        shell_elems
            .iter()
            .map(|elem| {
                let [(x1, y1), (x2, y2), (x3, y3)] = elem.local_coords;
                let a = elem.area;
                let inv2a = 1.0 / (2.0 * a);

                let y23 = y2 - y3;
                let y31 = y3 - y1;
                let y12 = y1 - y2;
                let x32 = x3 - x2;
                let x13 = x1 - x3;
                let x21 = x2 - x1;

                // Gather membrane DOFs (u, v per node) from global solution.
                // Shell DOFs: node i → u at 6i+0, v at 6i+1
                let ns = &elem.nodes;
                let u_mem = [
                    sol[ns[0] * DOFS_PER_NODE],
                    sol[ns[0] * DOFS_PER_NODE + 1],
                    sol[ns[1] * DOFS_PER_NODE],
                    sol[ns[1] * DOFS_PER_NODE + 1],
                    sol[ns[2] * DOFS_PER_NODE],
                    sol[ns[2] * DOFS_PER_NODE + 1],
                ];

                // ε = B · u_mem
                let exx = inv2a * (y23 * u_mem[0] + y31 * u_mem[2] + y12 * u_mem[4]);
                let eyy = inv2a * (x32 * u_mem[1] + x13 * u_mem[3] + x21 * u_mem[5]);
                let exy = inv2a
                    * (x32 * u_mem[0]
                        + y23 * u_mem[1]
                        + x13 * u_mem[2]
                        + y31 * u_mem[3]
                        + x21 * u_mem[4]
                        + y12 * u_mem[5]);

                // σ = D · ε
                let sxx = d[0] * exx + d[1] * eyy + d[2] * exy;
                let syy = d[3] * exx + d[4] * eyy + d[5] * exy;
                let sxy = d[6] * exx + d[7] * eyy + d[8] * exy;

                // Von Mises
                (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy).sqrt()
            })
            .collect()
    }

    /// Solve for a given section and write a VTU file with all fields.
    fn solve_and_write_vtu(&self, h_section: &HSection, path: &str) -> f64 {
        let (csr, rhs) = self.assemble_system(h_section);
        let u = self.forward_solve(&csr, &rhs);
        let tip_defl = -u[self.corner_w_dof()];

        let mesh = self.build_mesh();
        let (ux, uy, uz) = self.extract_displacements(&u);
        let von_mises = self.compute_von_mises(&u);

        // Displacement magnitude per node
        let disp_mag: Vec<f64> = ux
            .iter()
            .zip(uy.iter())
            .zip(uz.iter())
            .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
            .collect();

        let mut writer = VtuWriter::new(&mesh);
        writer.add_point_vector("displacement", &ux, &uy, &uz);
        writer.add_point_scalar("displacement_magnitude", &disp_mag);
        writer.add_point_scalar("deflection_w", &uz);
        writer.add_cell_scalar("von_mises_membrane", &von_mises);

        writer.write(path).expect("failed to write VTU");
        tip_defl
    }

    /// Run the optimisation.
    fn optimise(&self, initial: &HSection) -> ShapeOptResult {
        let x0 = vec![initial.b, initial.h, initial.t];

        let area_max = self.area_max;
        let area_constraint = Constraint {
            kind: ConstraintKind::Inequality,
            dim: 1,
            fun: Box::new(move |x: &[f64]| {
                let sec = HSection::new(x[0], x[1], x[2]);
                vec![area_max - sec.area()]
            }),
            jac: Some(Box::new(move |x: &[f64]| {
                let sec = HSection::new(x[0], x[1], x[2]);
                let da = sec.area_derivatives();
                DenseMatrix::from_row_major_vec(1, 3, vec![-da[0], -da[1], -da[2]])
            })),
        };

        let bounds = vec![
            Bound::both(self.bounds[0].0, self.bounds[0].1),
            Bound::both(self.bounds[1].0, self.bounds[1].1),
            Bound::both(self.bounds[2].0, self.bounds[2].1),
        ];

        let objective = |x: &[f64]| -> f64 { self.evaluate(&[x[0], x[1], x[2]]).0 };
        let gradient = |x: &[f64]| -> Vec<f64> { self.evaluate(&[x[0], x[1], x[2]]).1.to_vec() };

        let config = AugmentedLagrangianConfig {
            max_outer_iterations: 50,
            max_inner_iterations: 200,
            grad_tolerance: 1e-8,
            constraint_tolerance: 1e-8,
            inner_solver: InnerSolver::Lbfgsb,
            bounds,
            ..Default::default()
        };

        let result = augmented_lagrangian(&objective, &gradient, &x0, &[area_constraint], &config)
            .expect("optimisation failed");

        let opt = HSection::new(result.x[0], result.x[1], result.x[2]);
        ShapeOptResult {
            section: opt,
            corner_deflection: result.fun,
            area: opt.area(),
            optim: result,
        }
    }
}

struct ShapeOptResult {
    section: HSection,
    corner_deflection: f64,
    area: f64,
    optim: OptimResult<f64>,
}

// ── DOF mapping helpers ─────────────────────────────────────────────────

/// Map beam element DOFs to shell DOFs for a beam along the X axis.
/// Beam local: [u1, w1, θ1, u2, w2, θ2]
/// Shell:      [u,  w,  θy]  for each node
fn dof_map_x(n0: usize, n1: usize) -> [usize; 6] {
    [
        n0 * DOFS_PER_NODE,     // u (axial along X)
        n0 * DOFS_PER_NODE + 2, // w (transverse, out-of-plane)
        n0 * DOFS_PER_NODE + 4, // θy (bending rotation for XZ plane)
        n1 * DOFS_PER_NODE,
        n1 * DOFS_PER_NODE + 2,
        n1 * DOFS_PER_NODE + 4,
    ]
}

/// Map beam element DOFs to shell DOFs for a beam along the Y axis.
/// Beam local: [u1, w1, θ1, u2, w2, θ2]
/// Shell:      [v,  w,  θx]  for each node
fn dof_map_y(n0: usize, n1: usize) -> [usize; 6] {
    [
        n0 * DOFS_PER_NODE + 1, // v (axial along Y)
        n0 * DOFS_PER_NODE + 2, // w (transverse, out-of-plane)
        n0 * DOFS_PER_NODE + 3, // θx (bending rotation for YZ plane)
        n1 * DOFS_PER_NODE + 1,
        n1 * DOFS_PER_NODE + 2,
        n1 * DOFS_PER_NODE + 3,
    ]
}

// ── Dirichlet BC helper ─────────────────────────────────────────────────

fn apply_dirichlet(triplet: &mut TripletMatrix<f64>, rhs: &mut [f64], clamped: &HashSet<usize>) {
    let mut new_r = Vec::new();
    let mut new_c = Vec::new();
    let mut new_v = Vec::new();

    for i in 0..triplet.nnz() {
        let r = triplet.rows[i];
        let c = triplet.cols[i];
        if !clamped.contains(&r) && !clamped.contains(&c) {
            new_r.push(r);
            new_c.push(c);
            new_v.push(triplet.values[i]);
        }
    }
    for &d in clamped {
        new_r.push(d);
        new_c.push(d);
        new_v.push(1.0);
        rhs[d] = 0.0;
    }
    triplet.rows = new_r;
    triplet.cols = new_c;
    triplet.values = new_v;
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    println!("=== Shell + Edge-Beam Shape Optimisation ===");
    println!();
    println!(
        "Plate:    {:.3} x {:.3} m,  {} x {} mesh,  t_shell = {:.4} m",
        cli.lx, cli.ly, cli.nx, cli.ny, cli.shell_thickness
    );
    println!(
        "Material: E = {:.1e} Pa,  nu = {:.2},  rho = {:.0} kg/m³",
        cli.youngs_modulus, cli.poisson_ratio, cli.density
    );
    println!(
        "Corner load: P = {:.1} N (downward at x={:.3}, y=0)",
        cli.tip_load, cli.lx
    );
    println!("Beam area budget: A_max = {:.6} m² per beam", cli.area_max);
    println!();
    println!("Design bounds:");
    println!("  b (flange width): [{:.4}, {:.4}] m", cli.b_min, cli.b_max);
    println!("  h (web height):   [{:.4}, {:.4}] m", cli.h_min, cli.h_max);
    println!("  t (thickness):    [{:.4}, {:.4}] m", cli.t_min, cli.t_max);
    println!();

    let problem = PlateShellBeamProblem {
        lx: cli.lx,
        ly: cli.ly,
        nx: cli.nx,
        ny: cli.ny,
        shell_section: ShellSection::new(
            cli.youngs_modulus,
            cli.poisson_ratio,
            cli.shell_thickness,
            cli.density,
        ),
        youngs_modulus: cli.youngs_modulus,
        density: cli.density,
        tip_load: cli.tip_load,
        area_max: cli.area_max,
        bounds: [
            (cli.b_min, cli.b_max),
            (cli.h_min, cli.h_max),
            (cli.t_min, cli.t_max),
        ],
    };

    let initial = HSection::new(cli.b0, cli.h0, cli.t0);
    println!(
        "Mesh:    {} nodes,  {} shell elements,  {} beam elements",
        problem.n_nodes(),
        problem.nx * problem.ny * 2,
        2 * problem.nx + 2 * problem.ny
    );
    println!("DOFs:    {}", problem.n_dofs());
    println!();

    println!(
        "Initial beam section: b = {:.4}, h = {:.4}, t = {:.4}",
        initial.b, initial.h, initial.t
    );
    println!(
        "  A = {:.6} m²,  I = {:.6e} m⁴",
        initial.area(),
        initial.moment_of_inertia()
    );

    let (init_defl, init_grad) = problem.evaluate(&[initial.b, initial.h, initial.t]);
    println!("  corner deflection = {:.6e} m", init_defl);
    println!(
        "  grad = [{:.4e}, {:.4e}, {:.4e}]",
        init_grad[0], init_grad[1], init_grad[2]
    );
    println!();

    // Verify gradient with finite differences
    let eps = 1e-7;
    println!("Gradient check (central FD, eps={eps:.0e}):");
    let p0 = [initial.b, initial.h, initial.t];
    for i in 0..3 {
        let mut pp = p0;
        let mut pm = p0;
        pp[i] += eps;
        pm[i] -= eps;
        let fd = (problem.evaluate(&pp).0 - problem.evaluate(&pm).0) / (2.0 * eps);
        let rel = (init_grad[i] - fd).abs() / fd.abs().max(1e-15);
        let label = ["b", "h", "t"][i];
        println!(
            "  d/d{label}: adjoint={:.6e}  fd={:.6e}  rel_err={rel:.2e}",
            init_grad[i], fd
        );
    }
    println!();

    println!("--- Running adjoint-based optimisation ---");
    let result = problem.optimise(&initial);

    println!();
    println!("=== Result ===");
    println!("Converged:   {}", result.optim.converged);
    println!("Iterations:  {}", result.optim.iterations);
    println!("Fun evals:   {}", result.optim.fun_evals);
    println!("Grad evals:  {}", result.optim.grad_evals);
    println!();
    println!(
        "Optimal beam: b = {:.6}, h = {:.6}, t = {:.6}",
        result.section.b, result.section.h, result.section.t
    );
    println!("  A = {:.6} m²  (budget: {:.6})", result.area, cli.area_max);
    println!("  I = {:.6e} m⁴", result.section.moment_of_inertia());
    println!("  corner deflection = {:.6e} m", result.corner_deflection);
    println!();

    let improvement = (1.0 - result.corner_deflection / init_defl) * 100.0;
    println!("Deflection reduction: {:.1}% vs initial", improvement);
    println!();

    // ASCII cross-section
    print_cross_section(&result.section);

    // ── Write VTU output ────────────────────────────────────────────────
    std::fs::create_dir_all(&cli.output_dir).expect("failed to create output dir");

    let path_initial = format!("{}/initial.vtu", cli.output_dir);
    let path_optimised = format!("{}/optimised.vtu", cli.output_dir);

    println!();
    print!("Writing {path_initial} ...");
    problem.solve_and_write_vtu(&initial, &path_initial);
    println!(" done");

    print!("Writing {path_optimised} ...");
    problem.solve_and_write_vtu(&result.section, &path_optimised);
    println!(" done");

    // ── Write pyvista script ────────────────────────────────────────────
    let script_path = format!("{}/plot.py", cli.output_dir);
    write_pyvista_script(&script_path, &cli.output_dir);
    println!("Written {script_path}");
    println!();
    println!("Visualise with:  python {script_path}");
}

fn print_cross_section(sec: &HSection) {
    let scale = 40.0 / sec.b;
    let total_h = sec.h + 2.0 * sec.t;
    let flange_chars = (sec.b * scale).round() as usize;
    let web_chars = (sec.t * scale).round().max(1.0) as usize;
    let flange_rows = (sec.t / total_h * 10.0).round().max(1.0) as usize;
    let web_rows = (sec.h / total_h * 10.0).round().max(1.0) as usize;
    let pad = (flange_chars.saturating_sub(web_chars)) / 2;

    println!(
        "Optimal beam cross-section (b={:.4}, h={:.4}, t={:.4}):",
        sec.b, sec.h, sec.t
    );
    for _ in 0..flange_rows {
        println!("  {}", "=".repeat(flange_chars));
    }
    for _ in 0..web_rows {
        println!("  {}{}", " ".repeat(pad), "|".repeat(web_chars.max(1)));
    }
    for _ in 0..flange_rows {
        println!("  {}", "=".repeat(flange_chars));
    }
}

fn write_pyvista_script(path: &str, output_dir: &str) {
    let script = format!(
        r#""""
Visualise shell+beam shape optimisation results.

Requires: pip install pyvista
"""
import pyvista as pv

pv.set_plot_theme("document")

initial = pv.read("{output_dir}/initial.vtu")
optimised = pv.read("{output_dir}/optimised.vtu")

# Warp by displacement to show deformed shape (scale for visibility)
w_max_init = max(abs(initial["deflection_w"].min()), abs(initial["deflection_w"].max()))
w_max_opt = max(abs(optimised["deflection_w"].min()), abs(optimised["deflection_w"].max()))
warp_scale = 0.3 * initial.length / max(w_max_init, 1e-15)

init_warped = initial.warp_by_vector("displacement", factor=warp_scale)
opt_warped = optimised.warp_by_vector("displacement", factor=warp_scale)

# Side-by-side comparison
pl = pv.Plotter(shape=(2, 2))

# Row 0: deflection (w) on deformed mesh
pl.subplot(0, 0)
pl.add_text("Initial - deflection w", font_size=10)
pl.add_mesh(init_warped, scalars="deflection_w", cmap="coolwarm", show_edges=True)

pl.subplot(0, 1)
pl.add_text("Optimised - deflection w", font_size=10)
pl.add_mesh(opt_warped, scalars="deflection_w", cmap="coolwarm", show_edges=True)

# Row 1: von Mises stress on deformed mesh
pl.subplot(1, 0)
pl.add_text("Initial - von Mises (membrane)", font_size=10)
pl.add_mesh(init_warped, scalars="von_mises_membrane", cmap="hot", show_edges=True)

pl.subplot(1, 1)
pl.add_text("Optimised - von Mises (membrane)", font_size=10)
pl.add_mesh(opt_warped, scalars="von_mises_membrane", cmap="hot", show_edges=True)

pl.link_views()
pl.show()
"#
    );
    std::fs::write(path, script).expect("failed to write pyvista script");
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_problem() -> PlateShellBeamProblem {
        PlateShellBeamProblem {
            lx: 1.0,
            ly: 0.6,
            nx: 8,
            ny: 5,
            shell_section: ShellSection::new(200e9, 0.3, 0.002, 7800.0),
            youngs_modulus: 200e9,
            density: 7800.0,
            tip_load: 1000.0,
            area_max: 0.003,
            bounds: [(0.005, 0.06), (0.01, 0.12), (0.001, 0.01)],
        }
    }

    #[test]
    fn test_adjoint_gradient_vs_fd() {
        let problem = test_problem();
        let p0 = [0.02, 0.04, 0.003];
        let (_, grad) = problem.evaluate(&p0);
        let eps = 1e-7;

        for i in 0..3 {
            let mut pp = p0;
            let mut pm = p0;
            pp[i] += eps;
            pm[i] -= eps;
            let fd = (problem.evaluate(&pp).0 - problem.evaluate(&pm).0) / (2.0 * eps);
            let rel = (grad[i] - fd).abs() / fd.abs().max(1e-15);
            assert!(
                rel < 1e-3,
                "d/dp[{i}]: adjoint={:.6e}, fd={:.6e}, rel={rel:.2e}",
                grad[i],
                fd
            );
        }
    }

    #[test]
    fn test_optimisation_converges() {
        let problem = test_problem();
        let initial = HSection::new(0.015, 0.03, 0.003);
        assert!(initial.area() <= problem.area_max + 1e-10);

        let result = problem.optimise(&initial);
        let (init_defl, _) = problem.evaluate(&[initial.b, initial.h, initial.t]);

        assert!(
            result.corner_deflection < init_defl,
            "optimised ({:.6e}) should beat initial ({:.6e})",
            result.corner_deflection,
            init_defl,
        );
        assert!(
            result.area <= problem.area_max + 1e-6,
            "area constraint violated: {} > {}",
            result.area,
            problem.area_max,
        );
    }
}
