//! Contact demo: two half-cylinder shells compressed together
//!
//! Two semicircular shell arches face each other with a gap. They are
//! pushed together until the shells contact, demonstrating:
//! - Corotational shell elements with J2 elastoplasticity
//! - Through-thickness integration (5-point Simpson)
//! - Self-contact detection and augmented Lagrangian enforcement
//! - Adaptive load stepping
//! - PyVista visualization with plasticity data
//!
//! # Geometry
//!
//! ```text
//!           z
//!           ^
//!        /--+--\     top arch (pushed down)
//!       /   |   \
//!      |    |    |   gap
//!       \   |   /
//!        \--+--/     bottom arch (pushed up)
//!           +------> x
//! ```
//!
//! # Run
//!
//! ```bash
//! cargo run -p sonium-fem --release --example contact_shell_tube
//! python3 visualize_shell_contact.py
//! ```

use sonium_fem::mesh::{Mesh, Point};
use sonium_fem::physics::solid::contact::{ContactConfig, ContactManager};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
use sonium_fem::physics::solid::shell::{ElastoplasticShellForm, ShellElement, ShellSection};
use sonium_fem::solver::{FemNewtonConfig, fem_newton_solve_element_contact};
use sonium_solvers::NewtonConfig;
use std::io::Write;

const DOFS_PER_NODE: usize = 6;

struct Snapshot {
    load_frac: f64,
    displacement: Vec<f64>,
    active_pairs: usize,
    max_penetration: f64,
    yielding_elements: usize,
    max_plastic_strain: f64,
}

fn main() {
    println!("=== Shell Contact Demo: Two Arches ===\n");

    // ── 1. Geometry ──────────────────────────────────────────────────
    let radius = 0.3;
    let depth = 0.5; // extrusion depth along y
    let gap = 0.08; // gap between the two arches
    let thickness = 0.005;
    let n_circum = 12; // nodes along each arch
    let n_depth = 4; // nodes along y

    let (mesh, tri_nodes, n_top_arch_nodes) = two_arch_mesh(radius, depth, gap, n_circum, n_depth);
    let n_nodes = mesh.num_nodes();
    let n_dofs = n_nodes * DOFS_PER_NODE;

    println!(
        "Mesh: {} nodes, {} triangles, {} DOFs (6/node)",
        n_nodes,
        tri_nodes.len(),
        n_dofs,
    );

    // ── 2. Shell elements + J2 plasticity ────────────────────────────
    let section = ShellSection::new(210e9, 0.3, thickness, 7800.0);
    let elements: Vec<ShellElement> = tri_nodes
        .iter()
        .map(|&[n0, n1, n2]| ShellElement::from_mesh([n0, n1, n2], &mesh, section))
        .collect();

    let j2_material = J2Material::new(
        210e9,
        0.3,
        250e6,
        HardeningLaw::LinearIsotropic { modulus: 10e9 },
    );
    let mut form = ElastoplasticShellForm::new(elements, n_dofs, j2_material);
    println!("Material: E=210 GPa, sigma_y=250 MPa, H=10 GPa, t={thickness} m");

    // ── 3. Dirichlet BCs ─────────────────────────────────────────────
    // Straight edges of each arch (at x extremes) are fixed.
    // Top arch: push down. Bottom arch: push up.
    let prescribed_dz = 0.15; // each arch moves this much toward the other
    let geom_tol = 1e-8;

    let mut bc_full: Vec<(usize, f64)> = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        let is_top = i < n_top_arch_nodes;
        let at_left_edge = (node.x - (-radius)).abs() < geom_tol;
        let at_right_edge = (node.x - radius).abs() < geom_tol;

        if at_left_edge || at_right_edge {
            // Fix x, y, and rotations at arch endpoints
            bc_full.push((i * DOFS_PER_NODE, 0.0)); // u_x
            bc_full.push((i * DOFS_PER_NODE + 1, 0.0)); // u_y
            bc_full.push((i * DOFS_PER_NODE + 3, 0.0)); // θx
            bc_full.push((i * DOFS_PER_NODE + 4, 0.0)); // θy
            bc_full.push((i * DOFS_PER_NODE + 5, 0.0)); // θz
            // z: push arches toward each other
            if is_top {
                bc_full.push((i * DOFS_PER_NODE + 2, -prescribed_dz));
            } else {
                bc_full.push((i * DOFS_PER_NODE + 2, prescribed_dz));
            }
        }
    }

    println!(
        "Gap: {gap}, push each arch by {prescribed_dz:.2} (closes {:.2} of gap)",
        2.0 * prescribed_dz / gap
    );

    // ── 4. Contact surface ───────────────────────────────────────────
    let boundary_faces: Vec<[usize; 3]> = tri_nodes.clone();
    let ref_coords: Vec<f64> = mesh.nodes.iter().flat_map(|p| [p.x, p.y, p.z]).collect();

    let contact_config = ContactConfig {
        epsilon: 5e5,
        uzawa_tolerance: 1e-3,
        max_uzawa_iterations: 50,
        search_radius: gap + 0.1 * radius,
        bvh_margin: 0.01,
        friction_coefficient: 0.0,
        two_body: false,
    };

    let mut contact_mgr = ContactManager::new(&boundary_faces, ref_coords, contact_config);
    println!("Contact surface: {} triangles\n", boundary_faces.len());

    // ── 5. Solver config ─────────────────────────────────────────────
    let fem_config = FemNewtonConfig {
        newton: NewtonConfig {
            max_iterations: 100,
            absolute_tolerance: 1e-4,
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
            solver: sonium_solvers::LinearSolverType::GmresJacobi,
            max_iterations: 5000,
            tolerance: 1e-8,
            restart: 80,
            ..Default::default()
        },
    };

    // ── 6. Adaptive loading ──────────────────────────────────────────
    let initial_dfrac: f64 = 0.05;
    let min_dfrac: f64 = 0.001;
    let max_dfrac: f64 = 0.10;
    let growth: f64 = 1.3;
    let max_cuts = 10;

    let mut u = vec![0.0; n_dofs];
    let mut u_converged = u.clone();
    let mut frac = 0.0;
    let mut dfrac = initial_dfrac;
    let mut step = 0;
    let mut cuts = 0;

    let mut snapshots: Vec<Snapshot> = vec![Snapshot {
        load_frac: 0.0,
        displacement: u.clone(),
        active_pairs: 0,
        max_penetration: 0.0,
        yielding_elements: 0,
        max_plastic_strain: 0.0,
    }];

    println!("--- Adaptive loading (target: 100%) ---\n");

    // First step: linear interpolation
    {
        let target = dfrac.min(1.0 - frac);
        for (i, node) in mesh.nodes.iter().enumerate() {
            let is_top = i < n_top_arch_nodes;
            let sign = if is_top { -1.0 } else { 1.0 };
            // Linear interpolation: move arch endpoints' z proportionally
            let x_frac = 1.0 - (node.x.abs() / radius).min(1.0);
            u[DOFS_PER_NODE * i + 2] = sign * prescribed_dz * target * x_frac;
        }
    }

    while frac < 1.0 - 1e-12 {
        let target = (frac + dfrac).min(1.0);

        let bc_step: Vec<(usize, f64)> = bc_full
            .iter()
            .map(|&(dof, val)| (dof, val * target))
            .collect();

        match fem_newton_solve_element_contact(
            &form,
            &mut contact_mgr,
            DOFS_PER_NODE,
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

                form.commit_state(&u);

                let n_yield = form.yielding_count();
                let max_ep = form.max_equiv_plastic_strain();

                // Measure minimum gap between arches
                let min_gap = measure_arch_gap(&mesh, &u, n_top_arch_nodes);

                println!(
                    "Step {step:>3} | load={:5.1}% | uzawa={:>2} | pairs={:>3} | pen={:.2e} | yield={n_yield:>3}/{} | ep={max_ep:.3e} | gap={min_gap:.4}",
                    frac * 100.0,
                    cr.uzawa_iterations,
                    cr.active_pairs,
                    cr.max_penetration,
                    tri_nodes.len(),
                );

                snapshots.push(Snapshot {
                    load_frac: frac,
                    displacement: u.clone(),
                    active_pairs: cr.active_pairs,
                    max_penetration: cr.max_penetration,
                    yielding_elements: n_yield,
                    max_plastic_strain: max_ep,
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
                    "       | cut #{cuts}: dfrac={dfrac:.4} (retry from {:.1}%)",
                    frac * 100.0
                );
            }
        }
    }

    // ── 7. Summary ───────────────────────────────────────────────────
    println!("\n--- Final state ---");
    println!("Load reached:       {:.1}%", frac * 100.0);
    println!("Steps taken:        {step}");
    println!(
        "Yielding:           {}/{}",
        form.yielding_count(),
        tri_nodes.len()
    );
    println!(
        "Max plastic strain: {:.4e}",
        form.max_equiv_plastic_strain()
    );
    println!("Active contact:     {}", contact_mgr.count_active_pairs(&u));

    // Build BC arrow data for visualization: one arrow per constrained node
    // showing the prescribed z-displacement direction.
    let mut bc_arrows: Vec<BcArrow> = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        let is_top = i < n_top_arch_nodes;
        let at_left_edge = (node.x - (-radius)).abs() < geom_tol;
        let at_right_edge = (node.x - radius).abs() < geom_tol;

        if at_left_edge || at_right_edge {
            let (dir_z, mag) = if is_top {
                (-1.0, prescribed_dz)
            } else {
                (1.0, prescribed_dz)
            };
            bc_arrows.push(BcArrow {
                node: i,
                direction: [0.0, 0.0, dir_z],
                magnitude: mag,
            });
        }
    }

    let script_path = "visualize_shell_contact.py";
    write_pyvista_script(&mesh, &tri_nodes, &snapshots, &bc_arrows, script_path);
    println!("Visualization:      python3 {script_path}");
    println!("\n=== Done ===");
}

// ─── Mesh generation ─────────────────────────────────────────────────

/// Generate two semicircular arches facing each other.
///
/// Top arch: semicircle opening downward at z = gap/2 + radius
/// Bottom arch: semicircle opening upward at z = -(gap/2 + radius)
///
/// Returns (mesh, triangles, n_top_nodes).
fn two_arch_mesh(
    radius: f64,
    depth: f64,
    gap: f64,
    n_circum: usize,
    n_depth: usize,
) -> (Mesh, Vec<[usize; 3]>, usize) {
    let mut nodes = Vec::new();
    let mut triangles = Vec::new();

    let half_gap = gap / 2.0;

    // Top arch: semicircle from angle 0 to π (opening downward, center at z = half_gap + radius)
    let top_center_z = half_gap + radius;
    let top_offset = 0;
    for j in 0..=n_depth {
        let y = depth * j as f64 / n_depth as f64;
        for i in 0..=n_circum {
            let theta = std::f64::consts::PI * i as f64 / n_circum as f64;
            let x = -radius * theta.cos(); // -R to +R
            let z = top_center_z - radius * theta.sin(); // starts at half_gap+R, dips to half_gap
            nodes.push(Point { x, y, z });
        }
    }
    let n_top_nodes = nodes.len();

    // Top arch triangles
    let row_len = n_circum + 1;
    for j in 0..n_depth {
        for i in 0..n_circum {
            let n00 = top_offset + j * row_len + i;
            let n10 = top_offset + (j + 1) * row_len + i;
            let n01 = top_offset + j * row_len + i + 1;
            let n11 = top_offset + (j + 1) * row_len + i + 1;
            triangles.push([n00, n10, n01]);
            triangles.push([n10, n11, n01]);
        }
    }

    // Bottom arch: semicircle from angle 0 to π (opening upward, center at z = -(half_gap + radius))
    let bot_center_z = -(half_gap + radius);
    let bot_offset = nodes.len();
    for j in 0..=n_depth {
        let y = depth * j as f64 / n_depth as f64;
        for i in 0..=n_circum {
            let theta = std::f64::consts::PI * i as f64 / n_circum as f64;
            let x = -radius * theta.cos();
            let z = bot_center_z + radius * theta.sin(); // starts at -(half_gap+R), rises to -half_gap
            nodes.push(Point { x, y, z });
        }
    }

    // Bottom arch triangles (reversed winding so normals point outward/downward)
    for j in 0..n_depth {
        for i in 0..n_circum {
            let n00 = bot_offset + j * row_len + i;
            let n10 = bot_offset + (j + 1) * row_len + i;
            let n01 = bot_offset + j * row_len + i + 1;
            let n11 = bot_offset + (j + 1) * row_len + i + 1;
            triangles.push([n00, n01, n10]);
            triangles.push([n10, n01, n11]);
        }
    }

    let mut mesh = Mesh::new(3);
    mesh.nodes = nodes;
    (mesh, triangles, n_top_nodes)
}

/// Measure the minimum vertical distance between top arch (moving down)
/// and bottom arch (moving up) in the deformed configuration.
fn measure_arch_gap(mesh: &Mesh, u: &[f64], n_top_nodes: usize) -> f64 {
    // Find the lowest point of the top arch
    let mut min_z_top = f64::MAX;
    for i in 0..n_top_nodes {
        let z_def = mesh.nodes[i].z + u[DOFS_PER_NODE * i + 2];
        min_z_top = min_z_top.min(z_def);
    }
    // Find the highest point of the bottom arch
    let mut max_z_bot = f64::MIN;
    for i in n_top_nodes..mesh.num_nodes() {
        let z_def = mesh.nodes[i].z + u[DOFS_PER_NODE * i + 2];
        max_z_bot = max_z_bot.max(z_def);
    }
    min_z_top - max_z_bot
}

// ─── PyVista ─────────────────────────────────────────────────────────

/// Info about a prescribed displacement BC for visualization.
struct BcArrow {
    node: usize,
    direction: [f64; 3], // unit direction of the applied displacement
    magnitude: f64,      // full-load magnitude
}

fn write_pyvista_script(
    mesh: &Mesh,
    tri_nodes: &[[usize; 3]],
    snapshots: &[Snapshot],
    bc_arrows: &[BcArrow],
    path: &str,
) {
    let mut f = std::fs::File::create(path).expect("Failed to create visualization script");

    writeln!(
        f,
        r#"#!/usr/bin/env python3
"""PyVista visualization of two-arch shell contact with plasticity.
Usage:
    python3 {path}               # interactive slider
    python3 {path} --animate     # saves shell_contact_animation.gif
"""
import sys, numpy as np
try:
    import pyvista as pv
except ImportError:
    print("pip install pyvista numpy"); sys.exit(1)
"#
    )
    .unwrap();

    let n_nodes = mesh.num_nodes();
    writeln!(f, "nodes = np.array([").unwrap();
    for node in &mesh.nodes {
        writeln!(f, " [{:.10e},{:.10e},{:.10e}],", node.x, node.y, node.z).unwrap();
    }
    writeln!(f, "],dtype=np.float64)\n").unwrap();

    writeln!(f, "faces = np.array([").unwrap();
    for tri in tri_nodes {
        writeln!(f, " 3,{},{},{},", tri[0], tri[1], tri[2]).unwrap();
    }
    writeln!(f, "],dtype=np.int64)\n").unwrap();

    writeln!(
        f,
        "load_fractions = {:?}",
        snapshots.iter().map(|s| s.load_frac).collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(
        f,
        "active_pairs = {:?}",
        snapshots.iter().map(|s| s.active_pairs).collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(
        f,
        "yielding_elements = {:?}",
        snapshots
            .iter()
            .map(|s| s.yielding_elements)
            .collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(
        f,
        "max_plastic_strain = {:?}",
        snapshots
            .iter()
            .map(|s| s.max_plastic_strain)
            .collect::<Vec<_>>()
    )
    .unwrap();
    writeln!(
        f,
        "max_penetration = {:?}\n",
        snapshots
            .iter()
            .map(|s| s.max_penetration)
            .collect::<Vec<_>>()
    )
    .unwrap();

    // BC arrow data: node indices, direction vectors, magnitude
    writeln!(f, "bc_nodes = np.array([").unwrap();
    for a in bc_arrows {
        writeln!(f, " {},", a.node).unwrap();
    }
    writeln!(f, "],dtype=np.int64)\n").unwrap();

    writeln!(f, "bc_directions = np.array([").unwrap();
    for a in bc_arrows {
        writeln!(
            f,
            " [{:.6e},{:.6e},{:.6e}],",
            a.direction[0], a.direction[1], a.direction[2]
        )
        .unwrap();
    }
    writeln!(f, "],dtype=np.float64)\n").unwrap();

    writeln!(
        f,
        "bc_magnitude = {:.10e}\n",
        if bc_arrows.is_empty() {
            0.0
        } else {
            bc_arrows[0].magnitude
        }
    )
    .unwrap();

    writeln!(f, "displacements = [").unwrap();
    for snap in snapshots {
        writeln!(f, " np.array([").unwrap();
        for i in 0..n_nodes {
            writeln!(
                f,
                "  [{:.10e},{:.10e},{:.10e}],",
                snap.displacement[DOFS_PER_NODE * i],
                snap.displacement[DOFS_PER_NODE * i + 1],
                snap.displacement[DOFS_PER_NODE * i + 2],
            )
            .unwrap();
        }
        writeln!(f, " ],dtype=np.float64),").unwrap();
    }
    writeln!(f, "]\n").unwrap();

    write!(
        f,
        r##"
n_steps = len(load_fractions)

# Arrow scale: make arrows visible relative to the geometry
_geom_size = np.ptp(nodes, axis=0).max()  # bounding box diagonal
ARROW_SCALE = 0.3 * _geom_size / max(bc_magnitude, 1e-30)

def make_mesh(idx):
    """Build shell PolyData for a given step."""
    d = displacements[idx]
    grid = pv.PolyData(nodes + d, faces)
    grid.point_data["disp_mag"] = np.linalg.norm(d, axis=1)
    grid.point_data["disp_z"] = d[:, 2]
    return grid

def make_arrows(idx):
    """Build arrow glyphs showing applied displacements at BC nodes."""
    frac = load_fractions[idx]
    d = displacements[idx]
    pts = nodes[bc_nodes] + d[bc_nodes]
    vecs = bc_directions * bc_magnitude * frac * ARROW_SCALE
    arrow_pd = pv.PolyData(pts)
    arrow_pd["vectors"] = vecs
    arrow_pd.set_active_vectors("vectors")
    return arrow_pd

def interactive_slider():
    pl = pv.Plotter()
    pl.set_background("white")
    grid = make_mesh(0)
    clim = [0, max(np.linalg.norm(displacements[-1], axis=1).max(), 1e-10)]
    actor = pl.add_mesh(grid, scalars="disp_mag", cmap="turbo", show_edges=True,
                        edge_color="gray", line_width=0.3, clim=clim,
                        scalar_bar_args={{"title": "Displacement"}})

    # Initial arrows
    arrows_pd = make_arrows(0)
    arrow_glyphs = arrows_pd.glyph(orient="vectors", scale=False, factor=1.0)
    arrow_actor = pl.add_mesh(arrow_glyphs, color="red", label="Applied disp.")

    title = pl.add_text("Step 0", position="upper_left", font_size=10, color="black")

    def update(value):
        idx = int(round(value))
        idx = max(0, min(idx, n_steps - 1))
        g = make_mesh(idx)
        actor.mapper.SetInputData(g)

        # Update arrows
        apd = make_arrows(idx)
        new_glyphs = apd.glyph(orient="vectors", scale=False, factor=1.0)
        arrow_actor.mapper.SetInputData(new_glyphs)

        frac = load_fractions[idx]
        ny = yielding_elements[idx]
        ep = max_plastic_strain[idx]
        pairs = active_pairs[idx]
        pen = max_penetration[idx]
        title.SetText(0,
            f"Step {{idx}} | Load: {{frac*100:.1f}}% | "
            f"Yield: {{ny}} | ep={{ep:.2e}} | "
            f"Contact: {{pairs}} | Pen: {{pen:.2e}}")

    pl.add_slider_widget(update, rng=[0, n_steps-1], value=0, title="Step",
                         pointa=(0.25, 0.07), pointb=(0.75, 0.07), style="modern", fmt="%.0f")
    pl.camera_position = "xz"
    pl.show()

def animate_gif(filename="shell_contact_animation.gif"):
    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    pl.set_background("white")
    clim = [0, max(np.linalg.norm(displacements[-1], axis=1).max(), 1e-10)]
    pl.open_gif(filename)
    for idx in range(n_steps):
        pl.clear()
        g = make_mesh(idx)
        pl.add_mesh(g, scalars="disp_mag", cmap="turbo", show_edges=True,
                    edge_color="gray", line_width=0.3, clim=clim,
                    scalar_bar_args={{"title": "Displacement"}})
        # Arrows
        apd = make_arrows(idx)
        gl = apd.glyph(orient="vectors", scale=False, factor=1.0)
        pl.add_mesh(gl, color="red")

        frac = load_fractions[idx]
        ny = yielding_elements[idx]
        pairs = active_pairs[idx]
        pl.add_text(f"Step {{idx}} | Load: {{frac*100:.1f}}% | Yield: {{ny}} | Contact: {{pairs}}",
                    position="upper_left", font_size=10, color="black")
        pl.camera_position = "xz"
        pl.write_frame()
    for _ in range(5):
        pl.write_frame()
    pl.close()
    print(f"Saved {{filename}}")

if __name__ == "__main__":
    if "--animate" in sys.argv:
        animate_gif()
    else:
        interactive_slider()
"##
    )
    .unwrap();

    println!(
        "Wrote {path} ({} steps, {} nodes, {} tris)",
        snapshots.len(),
        n_nodes,
        tri_nodes.len(),
    );
}
