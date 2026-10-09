//! Contact demo: two blocks compressed into each other
//!
//! Two 3D elastic blocks are stacked vertically with a gap between them.
//! The bottom block is fixed, and the top block is pushed downward until
//! it contacts the bottom block. The augmented Lagrangian contact solver
//! prevents penetration.
//!
//! This demonstrates:
//! - Hyperelastic (Neo-Hookean) large-deformation analysis
//! - Contact detection via AABB bounding volume hierarchy
//! - Augmented Lagrangian (Uzawa) contact enforcement
//! - Adaptive load stepping (bisection on failure, growth on success)
//! - `fem_newton_solve_contact` solver entry point
//! - PyVista visualization script generation
//!
//! # Geometry
//!
//! ```text
//!       z
//!       ^
//!       |  +--------+   top block: z in [1.05, 2.05], pushed down
//!       |  |        |
//!       |  +--------+   <- gap = 0.05
//!       |  +--------+   bottom block: z in [0, 1.0], fixed at z=0
//!       |  |        |
//!       +--+--------+--> x
//! ```
//!
//! # Run
//!
//! ```bash
//! cargo run -p sonium-fem --release --example contact_beam_selfcontact
//! python3 visualize_contact.py   # requires: pip install pyvista numpy
//! ```

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::box_mesh_tetrahedra;
use sonium_fem::mesh::{Element, Mesh};
use sonium_fem::physics::solid::contact::{ContactConfig, ContactManager};
use sonium_fem::physics::solid::nonlinear_solid::{HyperelasticForm, NeoHookean};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_contact};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::NewtonConfig;
use std::io::Write;

/// Snapshot of a converged load step for visualization.
struct Snapshot {
    load_frac: f64,
    displacement: Vec<f64>,
    active_pairs: usize,
    max_penetration: f64,
}

fn main() {
    println!("=== Contact Demo: Two Blocks ===\n");

    // ── 1. Build two-block mesh ──────────────────────────────────────
    let n = 6;
    let gap = 0.05;
    let mesh = two_block_mesh(n, gap);

    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let n_dofs = space.num_global_dofs(&mesh);

    println!(
        "Mesh: {} nodes, {} tets, {} DOFs",
        mesh.num_nodes(),
        mesh.num_elements(),
        n_dofs,
    );

    // ── 2. Material ──────────────────────────────────────────────────
    let material = NeoHookean::from_young(1e4, 0.3, 3);
    let form = HyperelasticForm::new(material);

    // ── 3. Dirichlet BCs (at full load) ──────────────────────────────
    let z_top = 1.0 + gap + 1.0;
    let prescribed_dz = -0.10;
    let geom_tol = 1e-8;

    let mut bc_full: Vec<(usize, f64)> = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.z < geom_tol {
            for d in 0..3 {
                bc_full.push((i * 3 + d, 0.0));
            }
        } else if (z_top - node.z).abs() < geom_tol {
            bc_full.push((i * 3, 0.0));
            bc_full.push((i * 3 + 1, 0.0));
            bc_full.push((i * 3 + 2, prescribed_dz));
        }
    }

    println!(
        "Push top block down by {prescribed_dz:.2} (gap={gap}, penetration={:.2})",
        -(prescribed_dz + gap)
    );

    // ── 4. Contact surface ───────────────────────────────────────────
    let boundary_faces = extract_boundary_triangles(&mesh);
    let ref_coords: Vec<f64> = mesh.nodes.iter().flat_map(|p| [p.x, p.y, p.z]).collect();

    // Scale penalty by mesh size: finer meshes need softer initial penalty
    let h = 1.0 / n as f64; // element size
    let contact_config = ContactConfig {
        epsilon: 1e3 / h, // ~6e3 for n=6, ~3e3 for n=3
        uzawa_tolerance: 1e-3,
        max_uzawa_iterations: 50,
        search_radius: 3.0 * gap,
        bvh_margin: 0.005,
        friction_coefficient: 0.0,
        two_body: false,
    };

    let mut contact_mgr = ContactManager::new(&boundary_faces, ref_coords, contact_config);
    println!("Contact surface: {} triangles\n", boundary_faces.len());

    // ── 5. Solver config ─────────────────────────────────────────────
    let fem_config = FemNewtonConfig {
        newton: NewtonConfig {
            max_iterations: 100,
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-8,
            damping: 1.0,
            line_search: sonium_solvers::LineSearch::Backtracking {
                c1: 1e-4,
                max_halvings: 8,
            },
            verbosity: 0,
        },
        quad_order: 2,
        inner_solver: sonium_solvers::LinearSolveConfig {
            solver: sonium_solvers::LinearSolverType::GmresIlu,
            max_iterations: 2000,
            tolerance: 1e-8,
            restart: 50,
            ..Default::default()
        },
    };

    // ── 6. Adaptive incremental loading ──────────────────────────────
    let initial_dfrac: f64 = 0.10;
    let min_dfrac: f64 = 0.001;
    let max_dfrac: f64 = 0.15;
    let growth: f64 = 1.5;
    let max_cuts = 12;

    let mut u = vec![0.0; n_dofs];
    let mut u_converged = u.clone();
    let mut frac = 0.0;
    let mut dfrac = initial_dfrac;
    let mut step = 0;
    let mut cuts = 0;

    // Store snapshots for visualization
    let mut snapshots: Vec<Snapshot> = Vec::new();
    snapshots.push(Snapshot {
        load_frac: 0.0,
        displacement: u.clone(),
        active_pairs: 0,
        max_penetration: 0.0,
    });

    println!("--- Adaptive loading (target: 100%) ---\n");

    // First step: linearly interpolate initial guess
    {
        let target = dfrac.min(1.0 - frac);
        for (i, node) in mesh.nodes.iter().enumerate() {
            let t = node.z / z_top;
            u[3 * i + 2] = t * prescribed_dz * target;
        }
    }

    while frac < 1.0 - 1e-12 {
        let target = (frac + dfrac).min(1.0);

        let bc_step: Vec<(usize, f64)> = bc_full
            .iter()
            .map(|&(dof, val)| (dof, val * target))
            .collect();

        match fem_newton_solve_contact(
            &form,
            &space,
            &mesh,
            &mut contact_mgr,
            &bc_step,
            &u,
            &fem_config,
        ) {
            Ok((solution, cr)) => {
                step += 1;
                frac = target;
                u = solution.x.to_vec();
                u_converged.clone_from(&u);
                cuts = 0;

                let actual_gap = compute_min_gap(&mesh, &u, 1.0, 1.0 + gap);
                println!(
                    "Step {step:>3} | load={:5.1}% | dfrac={dfrac:.3} | uzawa={:>2} | pairs={:>3} | pen={:.2e} | gap={actual_gap:+.4}",
                    frac * 100.0,
                    cr.uzawa_iterations,
                    cr.active_pairs,
                    cr.max_penetration,
                );

                snapshots.push(Snapshot {
                    load_frac: frac,
                    displacement: u.clone(),
                    active_pairs: cr.active_pairs,
                    max_penetration: cr.max_penetration,
                });

                dfrac = (dfrac * growth).min(max_dfrac);
            }
            Err(_) => {
                cuts += 1;
                if cuts > max_cuts {
                    println!(
                        "  ** Giving up after {max_cuts} consecutive cuts at load={:.1}%",
                        frac * 100.0
                    );
                    break;
                }
                dfrac /= 2.0;
                if dfrac < min_dfrac {
                    dfrac = min_dfrac;
                }
                u.clone_from(&u_converged);
                contact_mgr.reset_multipliers();
                println!(
                    "       | cut #{cuts}: halved step to dfrac={dfrac:.4} (retry from load={:.1}%)",
                    frac * 100.0
                );
            }
        }
    }

    // ── 7. Final summary ─────────────────────────────────────────────
    println!("\n--- Final state ---");
    let max_disp = u.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let actual_gap = compute_min_gap(&mesh, &u, 1.0, 1.0 + gap);
    println!("Load reached:       {:.1}%", frac * 100.0);
    println!("Steps taken:        {step}");
    println!("Max displacement:   {max_disp:.4}");
    println!("Min block gap:      {actual_gap:.4}");
    println!("Active contact:     {}", contact_mgr.count_active_pairs(&u));

    // ── 8. Write PyVista visualization ───────────────────────────────
    let script_path = "visualize_contact.py";
    write_pyvista_script(&mesh, &snapshots, script_path);
    println!("\nVisualization:      python3 {script_path}");
    println!("\n=== Done ===");
}

// ─── Helper functions ────────────────────────────────────────────────

/// Create a mesh with two separate blocks stacked vertically.
fn two_block_mesh(n: usize, gap: f64) -> Mesh {
    let bottom = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, n, n);
    let mut top = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 1.0 + gap, 2.0 + gap, n, n, n);

    let n_bottom_nodes = bottom.nodes.len();
    let n_bottom_elems = bottom.elements.len();

    let mut nodes = bottom.nodes.clone();
    nodes.extend_from_slice(&top.nodes);

    let mut elements = bottom.elements.clone();
    for elem in &mut top.elements {
        let new_nodes: Vec<usize> = elem
            .vertices()
            .iter()
            .map(|&n| n + n_bottom_nodes)
            .collect();
        let new_id = elem.id + n_bottom_elems;
        elements.push(Element::new(elem.element_type, new_nodes, new_id));
    }

    let mut mesh = Mesh::new(3);
    mesh.nodes = nodes;
    mesh.elements = elements;
    mesh.detect_boundaries();
    mesh
}

/// Compute minimum vertical gap between the two blocks.
fn compute_min_gap(mesh: &Mesh, u: &[f64], z_top_bottom: f64, z_bot_top: f64) -> f64 {
    let tol = 1e-6;
    let mut max_z_bottom = f64::MIN;
    for (i, node) in mesh.nodes.iter().enumerate() {
        if (node.z - z_top_bottom).abs() < tol {
            max_z_bottom = max_z_bottom.max(node.z + u[3 * i + 2]);
        }
    }
    let mut min_z_top = f64::MAX;
    for (i, node) in mesh.nodes.iter().enumerate() {
        if (node.z - z_bot_top).abs() < tol {
            min_z_top = min_z_top.min(node.z + u[3 * i + 2]);
        }
    }
    min_z_top - max_z_bottom
}

/// Extract boundary triangle faces from a 3D tet mesh.
fn extract_boundary_triangles(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.boundaries
        .iter()
        .filter(|bf| bf.nodes.len() == 3)
        .map(|bf| [bf.nodes[0], bf.nodes[1], bf.nodes[2]])
        .collect()
}

// ─── PyVista script generation ───────────────────────────────────────

/// Write a self-contained PyVista Python script with embedded mesh + displacement data.
fn write_pyvista_script(mesh: &Mesh, snapshots: &[Snapshot], path: &str) {
    let mut f = std::fs::File::create(path).expect("Failed to create visualization script");

    // ── Header ───────────────────────────────────────────────────────
    writeln!(
        f,
        r#"#!/usr/bin/env python3
"""
PyVista visualization of the contact simulation results.

Generated by: cargo run -p sonium-fem --release --example contact_beam_selfcontact

Usage:
    python3 {path}               # interactive slider
    python3 {path} --animate     # animated GIF (saves contact_animation.gif)

Requirements:
    pip install pyvista numpy
"""
import sys
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista not installed. Run: pip install pyvista numpy")
    sys.exit(1)
"#
    )
    .unwrap();

    // ── Embedded mesh data ───────────────────────────────────────────
    // Nodes
    writeln!(f, "# Reference node coordinates (n_nodes x 3)").unwrap();
    writeln!(f, "nodes = np.array([").unwrap();
    for node in &mesh.nodes {
        writeln!(
            f,
            "    [{:.10e}, {:.10e}, {:.10e}],",
            node.x, node.y, node.z
        )
        .unwrap();
    }
    writeln!(f, "], dtype=np.float64)\n").unwrap();

    // Tetrahedra connectivity (VTK format: [4, n0, n1, n2, n3, ...])
    writeln!(f, "# Tetrahedron connectivity (VTK cell array format)").unwrap();
    writeln!(f, "cells = np.array([").unwrap();
    for elem in &mesh.elements {
        let v = elem.vertices();
        writeln!(f, "    4, {}, {}, {}, {},", v[0], v[1], v[2], v[3]).unwrap();
    }
    writeln!(f, "], dtype=np.int64)\n").unwrap();

    writeln!(
        f,
        "cell_types = np.full({}, pv.CellType.TETRA, dtype=np.uint8)\n",
        mesh.num_elements()
    )
    .unwrap();

    // ── Snapshot displacement data ───────────────────────────────────
    writeln!(f, "# Per-step displacement fields (n_steps x n_nodes x 3)").unwrap();
    writeln!(f, "load_fractions = [").unwrap();
    for snap in snapshots {
        writeln!(f, "    {:.6},", snap.load_frac).unwrap();
    }
    writeln!(f, "]\n").unwrap();

    writeln!(f, "active_pairs = [").unwrap();
    for snap in snapshots {
        writeln!(f, "    {},", snap.active_pairs).unwrap();
    }
    writeln!(f, "]\n").unwrap();

    writeln!(f, "max_penetration = [").unwrap();
    for snap in snapshots {
        writeln!(f, "    {:.6e},", snap.max_penetration).unwrap();
    }
    writeln!(f, "]\n").unwrap();

    writeln!(f, "displacements = [").unwrap();
    for snap in snapshots {
        writeln!(f, "    np.array([").unwrap();
        let n_nodes = mesh.num_nodes();
        for i in 0..n_nodes {
            writeln!(
                f,
                "        [{:.10e}, {:.10e}, {:.10e}],",
                snap.displacement[3 * i],
                snap.displacement[3 * i + 1],
                snap.displacement[3 * i + 2]
            )
            .unwrap();
        }
        writeln!(f, "    ], dtype=np.float64),").unwrap();
    }
    writeln!(f, "]\n").unwrap();

    // ── Visualization code ───────────────────────────────────────────
    write!(
        f,
        r##"
n_steps = len(load_fractions)

def make_mesh(step_idx):
    """Build a PyVista UnstructuredGrid for a given step."""
    disp = displacements[step_idx]
    deformed = nodes + disp
    grid = pv.UnstructuredGrid(cells, cell_types, deformed)
    mag = np.linalg.norm(disp, axis=1)
    grid.point_data["displacement_magnitude"] = mag
    grid.point_data["displacement_z"] = disp[:, 2]
    return grid, mag.max()


def interactive_slider():
    """Show an interactive slider to scrub through load steps."""
    pl = pv.Plotter()
    pl.set_background("white")

    grid, _ = make_mesh(0)
    actor = pl.add_mesh(
        grid,
        scalars="displacement_magnitude",
        cmap="turbo",
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
        clim=[0, np.linalg.norm(displacements[-1], axis=1).max()],
        scalar_bar_args={{"title": "Displacement magnitude"}},
    )

    title = pl.add_text(
        "Step 0 | Load: 0.0% | Contact pairs: 0",
        position="upper_left",
        font_size=10,
        color="black",
    )

    def update(value):
        idx = int(round(value))
        idx = max(0, min(idx, n_steps - 1))
        grid_new, _ = make_mesh(idx)
        actor.mapper.SetInputData(grid_new)
        frac = load_fractions[idx]
        pairs = active_pairs[idx]
        pen = max_penetration[idx]
        title.SetText(
            0,
            f"Step {{idx}} | Load: {{frac*100:.1f}}% | "
            f"Contact pairs: {{pairs}} | Max pen: {{pen:.2e}}",
        )

    pl.add_slider_widget(
        update,
        rng=[0, n_steps - 1],
        value=0,
        title="Load Step",
        pointa=(0.25, 0.07),
        pointb=(0.75, 0.07),
        style="modern",
        fmt="%.0f",
    )

    pl.camera_position = "xz"
    pl.show()


def animate_gif(filename="contact_animation.gif"):
    """Render an animated GIF cycling through all load steps."""
    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    pl.set_background("white")

    clim_max = np.linalg.norm(displacements[-1], axis=1).max()

    pl.open_gif(filename)

    for idx in range(n_steps):
        pl.clear()
        grid, _ = make_mesh(idx)
        pl.add_mesh(
            grid,
            scalars="displacement_magnitude",
            cmap="turbo",
            show_edges=True,
            edge_color="gray",
            line_width=0.5,
            clim=[0, clim_max],
            scalar_bar_args={{"title": "Displacement mag."}},
        )
        frac = load_fractions[idx]
        pairs = active_pairs[idx]
        pl.add_text(
            f"Step {{idx}} | Load: {{frac*100:.1f}}% | Contact: {{pairs}} pairs",
            position="upper_left",
            font_size=10,
            color="black",
        )
        pl.camera_position = "xz"
        pl.write_frame()

    # Hold last frame a few extra times
    for _ in range(5):
        pl.write_frame()

    pl.close()
    print(f"Saved animation to {{filename}}")


if __name__ == "__main__":
    if "--animate" in sys.argv:
        animate_gif()
    else:
        interactive_slider()
"##
    )
    .unwrap();

    println!(
        "Wrote {path} ({} steps, {} nodes, {} tets)",
        snapshots.len(),
        mesh.num_nodes(),
        mesh.num_elements()
    );
}
