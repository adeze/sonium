//! Website demo drivers: run small showcase simulations and dump VTU
//! frame series (plus a JSON manifest) for the website movie pipeline.
//!
//! Each subcommand writes `frame_XXXX.vtu` files and `manifest.json`
//! into the output directory given as the second argument. The Python
//! pipeline (`scripts/website_demos.py`) renders and encodes them.
//!
//! These are demonstrations, not QA gates: physics parameters favour
//! short, legible movies. Where a demo mirrors a QA case (Taylor-Green),
//! the setup follows the gated configuration.

use std::path::{Path, PathBuf};

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::{CellKind, MeshView, VtuWriter};
use sonium_fem::mesh::Mesh;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    LinearSolverCpuPreconditionerKind, NavierStokesConfig, solve_bdf_step_cached,
};
use sonium_fem::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
use sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::analytical::ns_taylor_green_vortex;

/// In-memory mesh exposed to the VTU writer.
struct SimpleMesh {
    points: Vec<[f64; 3]>,
    cells: Vec<(CellKind, Vec<u64>)>,
}

impl MeshView for SimpleMesh {
    fn num_nodes(&self) -> usize {
        self.points.len()
    }

    fn num_cells(&self) -> usize {
        self.cells.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.points.iter().copied())
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.cells.iter().cloned())
    }
}

/// Scalar/vector point fields for one frame.
struct FrameFields {
    point_scalars: Vec<(&'static str, Vec<f64>)>,
    point_vectors: Vec<(&'static str, Vec<[f64; 3]>)>,
    cell_scalars: Vec<(&'static str, Vec<f64>)>,
}

/// Write one VTU frame and return its file name.
fn write_frame(
    out_dir: &Path,
    stem: &str,
    mesh: &SimpleMesh,
    fields: &FrameFields,
) -> Result<String, String> {
    let mut writer = VtuWriter::new(mesh);
    for (name, values) in &fields.point_scalars {
        writer
            .try_add_point_scalar(name, values)
            .map_err(|e| format!("frame {stem}: point scalar {name}: {e}"))?;
    }
    for (name, values) in &fields.point_vectors {
        let mut vx = Vec::with_capacity(values.len());
        let mut vy = Vec::with_capacity(values.len());
        let mut vz = Vec::with_capacity(values.len());
        for value in values {
            vx.push(value[0]);
            vy.push(value[1]);
            vz.push(value[2]);
        }
        writer
            .try_add_point_vector(name, &vx, &vy, &vz)
            .map_err(|e| format!("frame {stem}: point vector {name}: {e}"))?;
    }
    for (name, values) in &fields.cell_scalars {
        writer
            .try_add_cell_scalar(name, values)
            .map_err(|e| format!("frame {stem}: cell scalar {name}: {e}"))?;
    }
    let file_name = format!("{stem}.vtu");
    writer
        .write(out_dir.join(&file_name))
        .map_err(|e| format!("frame {stem}: write: {e}"))?;
    Ok(file_name)
}

/// Escape a JSON string literal (frame/demo metadata is ASCII-only by
/// construction, but keep the writer honest).
fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Write the demo manifest consumed by the render pipeline.
fn write_manifest(
    out_dir: &Path,
    demo: &str,
    title: &str,
    frames: &[String],
    times: &[f64],
    field: &str,
    field_range: (f64, f64),
) -> Result<(), String> {
    let frame_list: Vec<String> = frames
        .iter()
        .map(|name| format!("\"{}\"", json_escape(name)))
        .collect();
    let time_list: Vec<String> = times.iter().map(|t| format!("{t:.6}")).collect();
    let body = format!(
        "{{\n  \"demo\": \"{}\",\n  \"title\": \"{}\",\n  \"frames\": [{}],\n  \"times\": [{}],\n  \"field\": \"{}\",\n  \"field_range\": [{:.6}, {:.6}]\n}}\n",
        json_escape(demo),
        json_escape(title),
        frame_list.join(", "),
        time_list.join(", "),
        json_escape(field),
        field_range.0,
        field_range.1,
    );
    std::fs::write(out_dir.join("manifest.json"), body)
        .map_err(|e| format!("manifest write: {e}"))
}

/// Build a `SimpleMesh` from a FEM mesh (triangles and tetrahedra only).
fn simple_mesh_from_fem(mesh: &Mesh) -> Result<SimpleMesh, String> {
    let points: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|node| [node.x, node.y, node.z])
        .collect();
    let mut cells = Vec::with_capacity(mesh.elements.len());
    for element in &mesh.elements {
        let kind = match element.nodes.len() {
            3 => CellKind::Triangle,
            4 => CellKind::Tetra,
            other => return Err(format!("unsupported demo cell size {other}")),
        };
        let indices: Vec<u64> = element
            .nodes
            .iter()
            .map(|&index| u64::try_from(index).map_err(|_| "node index overflow".to_string()))
            .collect::<Result<_, _>>()?;
        cells.push((kind, indices));
    }
    Ok(SimpleMesh { points, cells })
}

/// Taylor-Green vortex decay (mirrors the gated NS benchmark setup).
fn taylor_green(out_dir: &Path) -> Result<(), String> {
    // Demo parameters: coarser than the QA gate, larger viscosity so the
    // decay is visible inside a short movie.
    let mesh_n = 40usize;
    let nu = 0.04f64;
    let dt = 0.08f64;
    let steps = 75usize;
    let period = std::f64::consts::TAU;

    let mut mesh = unit_square_triangles(mesh_n);
    for node in mesh.nodes.iter_mut() {
        node.x *= period;
        node.y *= period;
    }
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let geom_tol = 1e-9;
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < geom_tol || (period - x) < geom_tol || y < geom_tol || (period - y) < geom_tol
    });
    let config = NavierStokesConfig {
        mu: nu,
        rho: 1.0,
        use_direct_solver: false,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: 1e-8,
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::IlutSchur,
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(dt, steps);
    let mut workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
    let gauge = mesh
        .nodes
        .iter()
        .position(|node| node.x < geom_tol && node.y < geom_tol)
        .ok_or("Taylor-Green mesh has no origin corner node")?;
    workspace.set_pressure_gauge(gauge, 0.0);
    let velocity_coords = v_space
        .dof_coordinates(&mesh)
        .ok_or("P1 velocity space did not expose nodal coordinates")?;

    let simple = simple_mesh_from_fem(&mesh)?;
    let n_nodes = mesh.nodes.len();
    // Velocity DOFs are nodal pairs (u then v) but the pair order follows
    // the space's DOF numbering, not the mesh node order: map each pair
    // back to its node by coordinates.
    if velocity_coords.len() != 2 * n_nodes {
        return Err("velocity dof layout is not nodal pairs".to_string());
    }
    let mut dof_pair_node = vec![usize::MAX; n_nodes];
    for pair in 0..n_nodes {
        let point = velocity_coords[2 * pair];
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9 && (candidate.y - point[1]).abs() < 1e-9
            })
            .ok_or("velocity dof pair has no matching mesh node")?;
        dof_pair_node[pair] = node;
    }

    let bc = |x: f64, y: f64, _z: f64| {
        let (u, v, _) = ns_taylor_green_vortex(x, y, 0.0, nu);
        vec![u, v]
    };
    let mut state = BdfState::from_rest(n_v, n_p);
    for (dof, point) in velocity_coords.iter().enumerate() {
        let (u, v, _) = ns_taylor_green_vortex(point[0], point[1], 0.0, nu);
        state.velocity[dof] = if dof % 2 == 0 { u } else { v };
        state.velocity_prev[dof] = state.velocity[dof];
    }

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_speed: f64 = 0.0;
    let dump = |state: &BdfState, step: usize, _time: f64, max_speed: &mut f64| {
        let mut velocity = vec![[0.0f64; 3]; n_nodes];
        let mut speed = vec![0.0f64; n_nodes];
        for pair in 0..n_nodes {
            let node = dof_pair_node[pair];
            let u = state.velocity[2 * pair];
            let v = state.velocity[2 * pair + 1];
            velocity[node] = [u, v, 0.0];
            let magnitude = (u * u + v * v).sqrt();
            *max_speed = max_speed.max(magnitude);
            speed[node] = magnitude;
        }
        let fields = FrameFields {
            point_scalars: vec![("speed", speed)],
            point_vectors: vec![("velocity", velocity)],
            cell_scalars: Vec::new(),
        };
        write_frame(
            out_dir,
            &format!("frame_{step:04}"),
            &simple,
            &fields,
        )
    };

    frames.push(dump(&state, 0, 0.0, &mut max_speed)?);
    times.push(0.0);
    for step in 0..steps {
        if !solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &bdf,
            &mut state,
            &bc,
            &bc_dofs,
            &[],
            &workspace,
        ) {
            return Err(format!("Taylor-Green BDF solve failed at step {step}"));
        }
        let time = dt * (step + 1) as f64;
        frames.push(dump(&state, step + 1, time, &mut max_speed)?);
        times.push(time);
    }
    write_manifest(
        out_dir,
        "taylor-green",
        "Taylor-Green vortex decay",
        &frames,
        &times,
        "speed",
        (0.0, max_speed),
    )
}

/// Per-element membrane strain intensity (constant-strain triangle in
/// the element local frame) averaged onto nodes for colouring.
fn shell_strain_field(
    elements: &[sonium_fem::physics::solid::shell::ShellElement],
    u: &[f64],
    total_nodes: usize,
) -> Vec<f64> {
    let mut nodal = vec![0.0f64; total_nodes];
    let mut weight = vec![0.0f64; total_nodes];
    for element in elements {
        let nodes = element.nodes;
        let mut local_u = [(0.0f64, 0.0f64); 3];
        for (k, &node) in nodes.iter().enumerate() {
            let g = [u[node * 6], u[node * 6 + 1], u[node * 6 + 2]];
            let r = &element.rotation;
            local_u[k] = (
                r[0] * g[0] + r[3] * g[1] + r[6] * g[2],
                r[1] * g[0] + r[4] * g[1] + r[7] * g[2],
            );
        }
        let [(x1, y1), (x2, y2), (x3, y3)] = element.local_coords;
        let two_a = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        if two_a.abs() < 1e-30 {
            continue;
        }
        let (u1, v1) = local_u[0];
        let (u2, v2) = local_u[1];
        let (u3, v3) = local_u[2];
        let ex = ((y2 - y3) * u1 + (y3 - y1) * u2 + (y1 - y2) * u3) / two_a;
        let ey = ((x3 - x2) * v1 + (x1 - x3) * v2 + (x2 - x1) * v3) / two_a;
        let gam = ((x3 - x2) * u1 + (x1 - x3) * u2 + (x2 - x1) * u3
            + (y2 - y3) * v1
            + (y3 - y1) * v2
            + (y1 - y2) * v3)
            / two_a;
        let vm = (ex * ex - ex * ey + ey * ey + 0.75 * gam * gam).sqrt();
        for &node in &nodes {
            nodal[node] += vm * element.area;
            weight[node] += element.area;
        }
    }
    for node in 0..total_nodes {
        if weight[node] > 0.0 {
            nodal[node] /= weight[node];
        }
    }
    nodal
}

/// Thin-walled tube crushed by prescribed end-shortening against a
/// rigid wall (Q4 crash shells, J2 plasticity; driven top ring).
fn crush(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::Point;
    use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
    use sonium_fem::physics::solid::shell::crash::{
        q4_crash_forces, DefShell, ElasticShearHook, HourglassParams, LayerStates, Q4CrashGeom,
    };
    use sonium_fem::physics::solid::shell::layered::{J2LayerState, SmallStrainJ2Model};
    use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};

    // Aluminium tube: 90 mm diameter, 160 mm tall, 1.2 mm wall.
    let radius = 0.045f64;
    let height = 0.16f64;
    let z_base = 0.002f64;
    let n_around = 20usize;
    let n_axial = 12usize;
    let section = ShellSection::new(70.0e9, 0.33, 0.0012, 2700.0);
    let node_id = |level: usize, around: usize| level * n_around + (around % n_around);
    let point = |level: usize, around: usize| {
        let theta = std::f64::consts::TAU * around as f64 / n_around as f64;
        let z = z_base + height * level as f64 / n_axial as f64;
        // Slight six-lobed imperfection (0.4%) to seed diamond folds.
        let r = radius
            * (1.0
                + 0.004
                    * (6.0 * theta).cos()
                    * (std::f64::consts::PI * (z - z_base) / height).sin());
        Point::new_3d(r * theta.cos(), r * theta.sin(), z)
    };

    let tube_nodes = (n_axial + 1) * n_around;
    let wall_start = tube_nodes;
    let total_nodes = tube_nodes + 4;

    // Solver mesh: one Q4 crash quad per cell. Colouring mesh: the
    // same cells split into triangles (shell_strain_field input).
    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut elements: Vec<ShellElement> = Vec::new();
    let mut tube_faces: Vec<[usize; 3]> = Vec::new();
    for level in 0..n_axial {
        for around in 0..n_around {
            let a = node_id(level, around);
            let b = node_id(level, around + 1);
            let c = node_id(level + 1, around + 1);
            let d = node_id(level + 1, around);
            quads.push([a, b, c, d]);
            for (nodes, pa, pb, pc) in [
                ([a, b, c], point(level, around), point(level, around + 1), point(level + 1, around + 1)),
                ([a, c, d], point(level, around), point(level + 1, around + 1), point(level + 1, around)),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
                tube_faces.push(nodes);
            }
        }
    }
    // Deformed coordinates of every node (tube + wall corners).
    let initial: Vec<[f64; 3]> = (0..tube_nodes)
        .map(|id| {
            let p = point(id / n_around, id % n_around);
            [p.x, p.y, p.z]
        })
        .chain([
            [-0.10, -0.10, 0.0],
            [0.10, -0.10, 0.0],
            [0.10, 0.10, 0.0],
            [-0.10, 0.10, 0.0],
        ])
        .collect();
    let wall_faces: Vec<[usize; 3]> = vec![
        [wall_start, wall_start + 1, wall_start + 2],
        [wall_start, wall_start + 2, wall_start + 3],
    ];

    // Q4 crash integrator (Belytschko-Tsay one-point quads,
    // hourglass control, velocity-increment J2 layers), driven by
    // prescribed end-shortening: the bottom ring sits on the wall
    // (fixed) and the top ring is driven down at a constant rate.
    // The corotational ShellExplicitModel path tried first diverged
    // for this tube in every regime (free flight and driven,
    // dt = 0.08-0.85 x wave_dt, two mesh densities, elastic and
    // plastic, damped and not; steps 10191-41091): its strain
    // measure breaks once folds rotate elements, while the crash
    // kernel's rate form is the gated production path for this
    // load case (see the fem_crash_shells QA driver).
    let density = 2700.0f64;
    let thickness = 0.0012f64;
    let material = SmallStrainJ2Model {
        material: J2Material::new(
            70.0e9,
            0.33,
            250.0e6,
            HardeningLaw::LinearIsotropic { modulus: 600.0e6 },
        ),
    };
    let def_shell = DefShell::explicit_defaults();
    let shear_modulus = 70.0e9 / (2.0 * (1.0 + 0.33));
    let shear_hook = ElasticShearHook::new(shear_modulus);
    let sound = (70.0e9 / density).sqrt();
    let mut layers: Vec<LayerStates<J2LayerState>> = quads
        .iter()
        .map(|_| LayerStates::new(J2LayerState::new(), 5, thickness).map_err(|e| e.message))
        .collect::<Result<_, _>>()?;
    let bottom_ring: Vec<usize> = (0..n_around).map(|a| node_id(0, a)).collect();
    let top_ring: Vec<usize> = (0..n_around).map(|a| node_id(n_axial, a)).collect();
    let mut fixed = vec![false; total_nodes];
    for &node in bottom_ring.iter().chain(top_ring.iter()) {
        fixed[node] = true;
    }
    for wall_node in 0..4 {
        fixed[wall_start + wall_node] = true;
    }
    // Lumped masses plus the original explicit-shell rotary
    // stabilization (m_node * A / 9, as in the crash-box QA driver).
    let mut masses = vec![[0.0f64; 6]; total_nodes];
    let mut dt_min = f64::INFINITY;
    for quad in &quads {
        let geom = Q4CrashGeom::from_corners(quad.map(|n| initial[n]), thickness)
            .map_err(|e| e.message)?;
        dt_min = dt_min.min(geom.critical_step(sound));
        let mut lumped = geom.lumped_masses(density);
        for node in 0..4 {
            for d in 3..6 {
                lumped[6 * node + d] += lumped[6 * node] * geom.area / 9.0;
            }
        }
        for (slot, &node) in quad.iter().enumerate() {
            for d in 0..6 {
                masses[node][d] += lumped[6 * slot + d];
            }
        }
    }
    let dt = 0.5 * dt_min;
    let end_time = 0.010f64;
    // Total prescribed end-shortening in m (34% of the tube height).
    let crush_distance = 0.055f64;
    let drive_rate = crush_distance / end_time;
    let total_steps = (end_time / dt).ceil() as usize;
    let frame_count = 72usize;
    let frame_every = (total_steps / frame_count).max(1);

    // Render mesh: tube triangles + wall, deformed positions per frame.
    let mut render_cells: Vec<(CellKind, Vec<u64>)> = tube_faces
        .iter()
        .map(|face| {
            (
                CellKind::Triangle,
                face.iter().map(|&n| n as u64).collect::<Vec<u64>>(),
            )
        })
        .collect();
    for face in &wall_faces {
        render_cells.push((
            CellKind::Triangle,
            face.iter().map(|&n| n as u64).collect::<Vec<u64>>(),
        ));
    }

    // Per-element membrane strain intensity (constant-strain triangle in
    // the element local frame) averaged onto nodes for colouring.
    let element_strain = |u: &[f64]| -> Vec<f64> { shell_strain_field(&elements, u, total_nodes) };

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_strain: f64 = 0.0;
    let dump = |points: &[[f64; 3]],
                frame_index: usize,
                max_strain: &mut f64|
     -> Result<String, String> {
        // Colouring strain from the triangle split of the quads.
        let mut u6 = vec![0.0f64; total_nodes * 6];
        for node in 0..total_nodes {
            for c in 0..3 {
                u6[node * 6 + c] = points[node][c] - initial[node][c];
            }
        }
        let strain = element_strain(&u6);
        *max_strain = (*max_strain).max(strain.iter().copied().fold(0.0, f64::max));
        let mesh = SimpleMesh {
            points: points.to_vec(),
            cells: render_cells.clone(),
        };
        let fields = FrameFields {
            point_scalars: vec![("strain", strain)],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &mesh, &fields)
    };

    let mut points = initial.clone();
    let mut velocity = vec![[0.0f64; 6]; total_nodes];
    frames.push(dump(&points, 0, &mut max_strain)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    for step in 1..=total_steps {
        // Prescribed top-ring end-shortening at a constant rate.
        let drive = crush_distance * (step as f64 * dt / end_time).min(1.0);
        for &node in &top_ring {
            points[node][2] = initial[node][2] - drive;
            velocity[node] = [0.0, 0.0, -drive_rate, 0.0, 0.0, 0.0];
        }
        let mut force = vec![[0.0f64; 6]; total_nodes];
        for (qi, quad) in quads.iter().enumerate() {
            let geom = Q4CrashGeom::from_corners(quad.map(|n| points[n]), thickness)
                .map_err(|e| e.message)?;
            let hg = HourglassParams::for_shell(
                def_shell.formulation,
                density,
                sound,
                shear_modulus,
                geom.area,
                thickness,
                geom.char_length,
                false,
            )
            .map_err(|e| e.message)?;
            let to_local = |w: [f64; 3]| {
                [
                    geom.ex[0] * w[0] + geom.ex[1] * w[1] + geom.ex[2] * w[2],
                    geom.ey[0] * w[0] + geom.ey[1] * w[1] + geom.ey[2] * w[2],
                    geom.ez[0] * w[0] + geom.ez[1] * w[1] + geom.ez[2] * w[2],
                ]
            };
            let from_local = |w: [f64; 3]| {
                [
                    geom.ex[0] * w[0] + geom.ey[0] * w[1] + geom.ez[0] * w[2],
                    geom.ex[1] * w[0] + geom.ey[1] * w[1] + geom.ez[1] * w[2],
                    geom.ex[2] * w[0] + geom.ey[2] * w[1] + geom.ez[2] * w[2],
                ]
            };
            let mut u = [0.0f64; 24];
            let mut v = [0.0f64; 24];
            for (slot, &node) in quad.iter().enumerate() {
                let disp = [
                    points[node][0] - initial[node][0],
                    points[node][1] - initial[node][1],
                    points[node][2] - initial[node][2],
                ];
                let lu = to_local(disp);
                let lv = to_local([velocity[node][0], velocity[node][1], velocity[node][2]]);
                let lw = to_local([velocity[node][3], velocity[node][4], velocity[node][5]]);
                for c in 0..3 {
                    u[6 * slot + c] = lu[c];
                    v[6 * slot + c] = lv[c];
                    v[6 * slot + 3 + c] = lw[c];
                }
            }
            let out = q4_crash_forces(
                &geom,
                &mut layers[qi],
                &material,
                &def_shell,
                &u,
                &v,
                dt,
                &hg,
                &shear_hook,
            )
            .map_err(|e| e.message)?;
            for (slot, &node) in quad.iter().enumerate() {
                let ft = from_local([
                    out.forces[6 * slot],
                    out.forces[6 * slot + 1],
                    out.forces[6 * slot + 2],
                ]);
                let fr = from_local([
                    out.forces[6 * slot + 3],
                    out.forces[6 * slot + 4],
                    out.forces[6 * slot + 5],
                ]);
                // q4_crash_forces returns internal forces; the
                // applied nodal force is their negation (as in the
                // crash-box QA driver).
                for c in 0..3 {
                    force[node][c] -= ft[c];
                    force[node][3 + c] -= fr[c];
                }
            }
        }
        for node in 0..tube_nodes {
            if fixed[node] {
                continue;
            }
            for d in 0..6 {
                velocity[node][d] += dt * force[node][d] / masses[node][d];
            }
            for c in 0..3 {
                points[node][c] += dt * velocity[node][c];
            }
        }
        if !points.iter().flatten().all(|value| value.is_finite()) {
            return Err(format!("crush diverged at step {step}"));
        }
        if step % frame_every == 0 || step == total_steps {
            frames.push(dump(&points, frame_index, &mut max_strain)?);
            times.push(step as f64 * dt);
            frame_index += 1;
        }
    }
    println!(
        "crush: {total_steps} steps, dt={dt:.3e}, max strain {max_strain:.3}, final time {:.4}",
        total_steps as f64 * dt
    );
    write_manifest(
        out_dir,
        "crush",
        "Thin-walled tube crush",
        &frames,
        &times,
        "strain",
        (0.0, max_strain),
    )
}

/// Necking bar: finite-strain J2 slender bar (same formulation as the
/// gated `fem.necking-bar-spatial` case), rendered as a surface of
/// revolution so the diffuse neck localises in 3D.
fn necking(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::physics::solid::plasticity::{
        FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, StressUpdateModel,
    };

    const E_MOD: f64 = 200.0e9;
    const NU: f64 = 0.3;
    const YIELD: f64 = 250.0e6;
    const Q_HARD: f64 = 300.0e6;
    const B_HARD: f64 = 5.0;
    const AREA: f64 = 1.0e-4;
    const GAUSS: [f64; 2] = [-0.577_350_269_189_625_8, 0.577_350_269_189_625_8];

    #[derive(Clone)]
    struct PointState {
        material: FiniteStrainState,
        axial: f64,
        lateral: f64,
    }
    impl Default for PointState {
        fn default() -> Self {
            Self {
                material: FiniteStrainState::new(),
                axial: 1.0,
                lateral: 1.0,
            }
        }
    }
    struct PointTrial {
        state: PointState,
        force: f64,
        tangent: f64,
    }
    fn initial_area(x: f64, imperfection: f64) -> f64 {
        AREA * (1.0 - imperfection * (-0.5 * ((x - 0.5) / 0.1).powi(2)).exp())
    }
    fn point_trial(
        old: &PointState,
        axial: f64,
        a0: f64,
        mat: &FiniteStrainJ2Material,
    ) -> Result<PointTrial, String> {
        if axial <= 0.0 || !axial.is_finite() {
            return Err("bar inverted during Newton trial".into());
        }
        let dx = (axial / old.axial).ln();
        let mut dt = -0.5 * dx;
        for _ in 0..30 {
            let increment = [
                dx.exp(),
                0.0,
                0.0,
                0.0,
                dt.exp(),
                0.0,
                0.0,
                0.0,
                dt.exp(),
            ];
            let update = mat.update(&old.material, &increment);
            let residual = 0.5 * (update.stress[1] + update.stress[2]);
            let c = &update.tangent;
            let transverse = c[7] + c[8];
            if !transverse.is_finite() || transverse <= 0.0 {
                return Err("invalid transverse tangent".into());
            }
            if residual.abs() / YIELD < 1e-12 {
                let lateral = old.lateral * dt.exp();
                let slope = -c[6] / transverse;
                let condensed = c[0] + (c[1] + c[2]) * slope;
                let area = a0 * lateral * lateral;
                let mut state = old.material.clone();
                mat.commit(&mut state, &update);
                return Ok(PointTrial {
                    state: PointState {
                        material: state,
                        axial,
                        lateral,
                    },
                    force: area * update.stress[0],
                    tangent: area / axial * (condensed + 2.0 * update.stress[0] * slope),
                });
            }
            dt -= residual / transverse;
        }
        Err("bar traction-free transverse solve failed".into())
    }
    struct Assembly {
        residual: Vec<f64>,
        diagonal: Vec<f64>,
        upper: Vec<f64>,
        states: Vec<PointState>,
    }
    fn assemble(
        u: &[f64],
        old: &[PointState],
        imperfection: f64,
        mat: &FiniteStrainJ2Material,
    ) -> Result<Assembly, String> {
        let n = u.len() - 1;
        let h = 1.0 / n as f64;
        let mut out = Assembly {
            residual: vec![0.0; n + 1],
            diagonal: vec![0.0; n + 1],
            upper: vec![0.0; n],
            states: Vec::with_capacity(2 * n),
        };
        for e in 0..n {
            let axial = 1.0 + (u[e + 1] - u[e]) / h;
            for (q, xi) in GAUSS.into_iter().enumerate() {
                let x = (e as f64 + 0.5 * (1.0 + xi)) * h;
                let p = point_trial(&old[2 * e + q], axial, initial_area(x, imperfection), mat)?;
                let force = 0.5 * p.force;
                let stiffness = 0.5 * p.tangent / h;
                out.residual[e] -= force;
                out.residual[e + 1] += force;
                out.diagonal[e] += stiffness;
                out.diagonal[e + 1] += stiffness;
                out.upper[e] -= stiffness;
                out.states.push(p.state);
            }
        }
        Ok(out)
    }
    fn solve_tridiagonal(diagonal: &[f64], upper: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
        let mut d = diagonal.to_vec();
        let mut r = rhs.to_vec();
        for i in 1..d.len() {
            if d[i - 1].abs() < 1e-20 || !d[i - 1].is_finite() {
                return Err("singular bar tangent".into());
            }
            let factor = upper[i - 1] / d[i - 1];
            d[i] -= factor * upper[i - 1];
            r[i] -= factor * r[i - 1];
        }
        let mut x = vec![0.0; d.len()];
        for i in (0..d.len()).rev() {
            x[i] = (r[i]
                - if i + 1 < d.len() {
                    upper[i] * x[i + 1]
                } else {
                    0.0
                })
                / d[i];
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite bar correction".into());
        }
        Ok(x)
    }

    let n = 48usize;
    let steps = 200usize;
    let imperfection = 0.01f64;
    let end_extension = 0.32f64;
    let mat = FiniteStrainJ2Material::new(
        E_MOD,
        NU,
        YIELD,
        HardeningLaw::VoceSaturation {
            q: Q_HARD,
            b: B_HARD,
        },
    );
    let mut u = vec![0.0f64; n + 1];
    let mut states = vec![PointState::default(); 2 * n];

    // Surface of revolution: one ring per bar node.
    let n_around = 20usize;
    let radius0 = (AREA / std::f64::consts::PI).sqrt() * 6.0; // exaggerated for legibility
    let mut cells: Vec<(CellKind, Vec<u64>)> = Vec::new();
    for e in 0..n {
        for k in 0..n_around {
            let a = (e * n_around + k) as u64;
            let b = (e * n_around + (k + 1) % n_around) as u64;
            let c = ((e + 1) * n_around + (k + 1) % n_around) as u64;
            let d = ((e + 1) * n_around + k) as u64;
            cells.push((CellKind::Quad, vec![a, b, c, d]));
        }
    }

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_strain: f64 = 0.0;
    let dump = |u: &[f64],
                states: &[PointState],
                frame_index: usize,
                max_strain: &mut f64|
     -> Result<String, String> {
        // Nodal lateral stretch and axial strain from adjacent points.
        let mut points = Vec::with_capacity((n + 1) * n_around);
        let mut strain = Vec::with_capacity((n + 1) * n_around);
        for node in 0..=n {
            let (lateral, axial_strain) = if node == 0 {
                (states[0].lateral, states[0].axial.ln())
            } else if node == n {
                (states[2 * n - 1].lateral, states[2 * n - 1].axial.ln())
            } else {
                let left = &states[2 * node - 1];
                let right = &states[2 * node];
                (
                    0.5 * (left.lateral + right.lateral),
                    0.5 * (left.axial.ln() + right.axial.ln()),
                )
            };
            *max_strain = (*max_strain).max(axial_strain);
            let x = node as f64 / n as f64 + u[node];
            let r = radius0 * lateral;
            for k in 0..n_around {
                let theta = std::f64::consts::TAU * k as f64 / n_around as f64;
                points.push([x, r * theta.cos(), r * theta.sin()]);
                strain.push(axial_strain);
            }
        }
        let mesh = SimpleMesh {
            points,
            cells: cells.clone(),
        };
        let fields = FrameFields {
            point_scalars: vec![("strain", strain)],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &mesh, &fields)
    };

    frames.push(dump(&u, &states, 0, &mut max_strain)?);
    times.push(0.0);
    // Adaptive march: near localization the equilibrium path stiffens,
    // so halve the increment on a stalled Newton and stop gracefully at
    // the smallest useful increment (the movie ends at the neck).
    let mut extension_done = 0.0f64;
    let mut step_size = end_extension / steps as f64;
    let mut frame_index = 0usize;
    while extension_done < end_extension - 1e-12 && frame_index < 400 {
        let target = (extension_done + step_size).min(end_extension);
        let saved_u = u.clone();
        let saved_states = states.clone();
        let increment = target - u[n];
        for (i, value) in u.iter_mut().enumerate() {
            *value += increment * i as f64 / n as f64;
        }
        u[0] = 0.0;
        u[n] = target;
        let mut converged = None;
        let mut stalled = false;
        for _ in 0..200 {
            let trial = assemble(&u, &states, imperfection, &mat)?;
            let norm = trial.residual[1..n]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                / (AREA * YIELD);
            if norm < 1e-9 {
                converged = Some(trial);
                break;
            }
            let rhs: Vec<_> = trial.residual[1..n].iter().map(|v| -v).collect();
            let correction = solve_tridiagonal(&trial.diagonal[1..n], &trial.upper[1..n - 1], &rhs)?;
            let mut accepted = false;
            let mut alpha = 1.0;
            for _ in 0..16 {
                let mut candidate = u.clone();
                for (v, du) in candidate[1..n].iter_mut().zip(&correction) {
                    *v += alpha * du;
                }
                if let Ok(next) = assemble(&candidate, &states, imperfection, &mat) {
                    let next_norm = next.residual[1..n]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                        / (AREA * YIELD);
                    if next_norm < norm {
                        u = candidate;
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if !accepted {
                stalled = true;
                break;
            }
        }
        if stalled || converged.is_none() {
            u = saved_u;
            states = saved_states;
            step_size *= 0.5;
            if step_size < 1.0e-5 {
                println!("necking: stopping at extension {extension_done:.3} (localization limit)");
                break;
            }
            continue;
        }
        let trial = converged.expect("converged Newton trial");
        states = trial.states;
        extension_done = target;
        frame_index += 1;
        frames.push(dump(&u, &states, frame_index, &mut max_strain)?);
        times.push(extension_done);
    }
    // Explicit-dynamics continuation through localization: the
    // quasi-static Newton stalls at the turning point, so march the
    // same residual with lumped masses (mass-scaled, still
    // quasi-static rates) until the neck is fully developed.
    if extension_done > 0.15 {
        let rho_scaled = 7800.0 * 25.0;
        let nodal_mass = AREA * rho_scaled / n as f64;
        let wave_speed = (E_MOD / rho_scaled).sqrt();
        let dyn_dt = 0.7 * (1.0 / n as f64) / wave_speed;
        let end_velocity = 0.45f64;
        let target_extension = 0.42f64;
        let mut velocity = vec![0.0f64; n + 1];
        let mut dyn_time = 0.0f64;
        let mut dyn_step = 0usize;
        while u[n] < target_extension && dyn_step < 200_000 {
            let trial = assemble(&u, &states, imperfection, &mat)?;
            states = trial.states;
            let ramp = (dyn_time / 0.02).min(1.0);
            for i in 1..n {
                let accel = -trial.residual[i] / nodal_mass - 30.0 * velocity[i];
                velocity[i] += accel * dyn_dt;
            }
            for i in 1..n {
                u[i] += velocity[i] * dyn_dt;
            }
            u[0] = 0.0;
            u[n] += end_velocity * ramp * dyn_dt;
            dyn_time += dyn_dt;
            dyn_step += 1;
            if dyn_step % 60 == 0 {
                frame_index += 1;
                frames.push(dump(&u, &states, frame_index, &mut max_strain)?);
                times.push(u[n]);
            }
        }
        println!(
            "necking: dynamic continuation {dyn_step} steps to extension {:.3}",
            u[n]
        );
    }
    println!("necking: max axial strain {max_strain:.3}");
    write_manifest(
        out_dir,
        "necking",
        "Necking bar",
        &frames,
        &times,
        "strain",
        (0.0, max_strain),
    )
}

/// Vibrating plate: first shell eigenmodes (generalized eigenproblem
/// on the assembled stiffness/lumped-mass pair, LOBPCG), animated over
/// one period per mode.
fn plate_modes(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::Point;
    use sonium_fem::physics::solid::shell::explicit::lumped_masses;
    use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};
    use sonium_solvers::{LobpcgConfig, lobpcg_generalized_preconditioned};

    struct DenseOp {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    }
    impl LinearOperator<f64> for DenseOp {
        fn num_rows(&self) -> usize {
            self.rows
        }
        fn num_cols(&self) -> usize {
            self.cols
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            for (row, out) in y.iter_mut().enumerate() {
                *out = self.data[row * self.cols..(row + 1) * self.cols]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            // The assembled plate stiffness is symmetric.
            LinearOperator::apply_into(self, x, y);
        }
    }
    struct DiagOp {
        diag: Vec<f64>,
    }
    impl LinearOperator<f64> for DiagOp {
        fn num_rows(&self) -> usize {
            self.diag.len()
        }
        fn num_cols(&self) -> usize {
            self.diag.len()
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            for (out, (d, value)) in y.iter_mut().zip(self.diag.iter().zip(x)) {
                *out = d * value;
            }
        }
        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            LinearOperator::apply_into(self, x, y);
        }
    }
    impl Preconditioner<f64> for DiagOp {
        fn apply(&self, r: &[f64]) -> Vec<f64> {
            r.iter().zip(&self.diag).map(|(value, d)| value * d).collect()
        }
        fn apply_into(&self, r: &[f64], y: &mut [f64]) {
            LinearOperator::apply_into(self, r, y);
        }
    }

    // Clamped square steel plate, 260 mm across, 3 mm thick.
    let side = 0.26f64;
    let grid = 12usize; // elements per side
    let nodes_per_side = grid + 1;
    let n_nodes = nodes_per_side * nodes_per_side;
    let n_dofs = n_nodes * 6;
    let section = ShellSection::new(200.0e9, 0.3, 0.003, 7800.0);
    let node_point = |i: usize, j: usize| {
        Point::new_3d(
            side * i as f64 / grid as f64,
            side * j as f64 / grid as f64,
            0.0,
        )
    };
    let node_id = |i: usize, j: usize| j * nodes_per_side + i;
    let mut elements: Vec<ShellElement> = Vec::new();
    let mut cells: Vec<(CellKind, Vec<u64>)> = Vec::new();
    for j in 0..grid {
        for i in 0..grid {
            let a = node_id(i, j);
            let b = node_id(i + 1, j);
            let c = node_id(i + 1, j + 1);
            let d = node_id(i, j + 1);
            for (nodes, pa, pb, pc) in [
                ([a, b, c], node_point(i, j), node_point(i + 1, j), node_point(i + 1, j + 1)),
                ([a, c, d], node_point(i, j), node_point(i + 1, j + 1), node_point(i, j + 1)),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
                cells.push((
                    CellKind::Triangle,
                    nodes.iter().map(|&n| n as u64).collect(),
                ));
            }
        }
    }
    let masses = lumped_masses(&elements, n_dofs).map_err(|e| e.message)?;
    let base_points: Vec<[f64; 3]> = (0..n_nodes)
        .map(|id| {
            let p = node_point(id % nodes_per_side, id / nodes_per_side);
            [p.x, p.y, p.z]
        })
        .collect();

    // Free DOFs: everything except boundary nodes (clamped edges).
    let is_boundary = |node: usize| {
        let (i, j) = (node % nodes_per_side, node / nodes_per_side);
        i == 0 || j == 0 || i == grid || j == grid
    };
    let mut free_index = vec![usize::MAX; n_dofs];
    let mut n_free = 0usize;
    for dof in 0..n_dofs {
        if !is_boundary(dof / 6) {
            free_index[dof] = n_free;
            n_free += 1;
        }
    }
    let mut k_data = vec![0.0f64; n_free * n_free];
    for element in &elements {
        let ke = element.stiffness_global();
        let mut gdofs = [0usize; 18];
        for (slot, &node) in element.nodes.iter().enumerate() {
            for c in 0..6 {
                gdofs[slot * 6 + c] = node * 6 + c;
            }
        }
        for li in 0..18 {
            let fi = free_index[gdofs[li]];
            if fi == usize::MAX {
                continue;
            }
            for lj in 0..18 {
                let fj = free_index[gdofs[lj]];
                if fj == usize::MAX {
                    continue;
                }
                k_data[fi * n_free + fj] += ke[li * 18 + lj];
            }
        }
    }
    let m_free: Vec<f64> = (0..n_dofs)
        .filter(|&dof| free_index[dof] != usize::MAX)
        .map(|dof| masses[dof])
        .collect();
    if m_free.iter().any(|&m| m <= 0.0 || !m.is_finite()) {
        return Err("plate mass matrix has a non-positive free mass".into());
    }
    let k_op = DenseOp {
        rows: n_free,
        cols: n_free,
        data: k_data,
    };
    let m_op = DiagOp { diag: m_free.clone() };
    let precond = DiagOp {
        diag: (0..n_free)
            .map(|i| {
                let d = k_op.data[i * n_free + i];
                if d > 0.0 { 1.0 / d } else { 0.0 }
            })
            .collect(),
    };
    let config = LobpcgConfig {
        num_eigenvalues: 4,
        max_iterations: 600,
        tolerance: 1e-8,
        block_padding: 4,
    };
    let result = lobpcg_generalized_preconditioned(&k_op, &m_op, &precond, &config)
        .map_err(|e| format!("plate LOBPCG failed: {e}"))?;
    if !result.converged {
        return Err(format!(
            "plate LOBPCG did not converge (residuals {:?})",
            result.residual_norms
        ));
    }
    let frequencies: Vec<f64> = result
        .eigenvalues
        .iter()
        .map(|lambda| lambda.max(0.0).sqrt() / std::f64::consts::TAU)
        .collect();
    println!("plate modes (Hz): {frequencies:.1?}");

    // Animate modes 0..3 sequentially, one period each.
    let frames_per_mode = 18usize;
    let amplitude = 0.02f64;
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut frame_index = 0usize;
    for (mode, &lambda) in result.eigenvalues.iter().enumerate() {
        // Eigenvector column -> full DOF displacement shape.
        let mut shape = vec![0.0f64; n_dofs];
        for dof in 0..n_dofs {
            let fi = free_index[dof];
            if fi != usize::MAX {
                shape[dof] = result.eigenvectors[[fi, mode]];
            }
        }
        let max_z = (0..n_nodes)
            .map(|node| shape[node * 6 + 2].abs())
            .fold(0.0, f64::max);
        if max_z <= 0.0 {
            return Err(format!("plate mode {mode} has no transverse shape"));
        }
        let scale = amplitude / max_z;
        for frame in 0..frames_per_mode {
            let phase = std::f64::consts::TAU * frame as f64 / frames_per_mode as f64;
            let sway = phase.sin();
            let points: Vec<[f64; 3]> = (0..n_nodes)
                .map(|node| {
                    [
                        base_points[node][0] + scale * sway * shape[node * 6],
                        base_points[node][1] + scale * sway * shape[node * 6 + 1],
                        base_points[node][2] + scale * sway * shape[node * 6 + 2],
                    ]
                })
                .collect();
            let disp: Vec<f64> = (0..n_nodes)
                .map(|node| scale * sway * shape[node * 6 + 2])
                .collect();
            let mesh = SimpleMesh {
                points,
                cells: cells.clone(),
            };
            let fields = FrameFields {
                point_scalars: vec![("disp", disp)],
                point_vectors: Vec::new(),
                cell_scalars: Vec::new(),
            };
            frames.push(write_frame(
                out_dir,
                &format!("frame_{frame_index:04}"),
                &mesh,
                &fields,
            )?);
            times.push(mode as f64 + frame as f64 / frames_per_mode as f64);
            frame_index += 1;
        }
        let _ = lambda;
    }
    write_manifest(
        out_dir,
        "plate-modes",
        "Plate vibration modes",
        &frames,
        &times,
        "disp",
        (-amplitude, amplitude),
    )
}

/// Room impulse: acoustic wave equation in a 3D room (P1 tetrahedra,
/// lumped mass, central differences; rigid-wall Neumann boundaries).
fn room_wave(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::generators::box_mesh_tetrahedra;

    let (lx, ly, lz) = (3.6f64, 2.8f64, 2.4f64);
    let mesh = box_mesh_tetrahedra(0.0, lx, 0.0, ly, 0.0, lz, 24, 18, 16);
    let n = mesh.nodes.len();
    let c = 343.0f64;
    let source = [1.8, 1.4, 1.2];
    let sigma = 0.22f64;

    // P1 tet element stiffness (unit coefficient) and lumped masses.
    struct Tet {
        nodes: [usize; 4],
        ke: [[f64; 4]; 4],
    }
    let mut tets: Vec<Tet> = Vec::new();
    let mut lumped = vec![0.0f64; n];
    for element in &mesh.elements {
        if element.nodes.len() != 4 {
            return Err("room mesh is not tetrahedral".into());
        }
        let idx = [
            element.nodes[0],
            element.nodes[1],
            element.nodes[2],
            element.nodes[3],
        ];
        let p: [[f64; 3]; 4] = [
            [mesh.nodes[idx[0]].x, mesh.nodes[idx[0]].y, mesh.nodes[idx[0]].z],
            [mesh.nodes[idx[1]].x, mesh.nodes[idx[1]].y, mesh.nodes[idx[1]].z],
            [mesh.nodes[idx[2]].x, mesh.nodes[idx[2]].y, mesh.nodes[idx[2]].z],
            [mesh.nodes[idx[3]].x, mesh.nodes[idx[3]].y, mesh.nodes[idx[3]].z],
        ];
        // Jacobian of the affine map from the reference tet.
        let j = [
            [p[1][0] - p[0][0], p[2][0] - p[0][0], p[3][0] - p[0][0]],
            [p[1][1] - p[0][1], p[2][1] - p[0][1], p[3][1] - p[0][1]],
            [p[1][2] - p[0][2], p[2][2] - p[0][2], p[3][2] - p[0][2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        if det.abs() < 1e-300 {
            return Err("degenerate tetrahedron in room mesh".into());
        }
        let volume = det.abs() / 6.0;
        // Inverse Jacobian (rows), for reference gradients.
        let inv = [
            [
                (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det,
                (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det,
                (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det,
            ],
            [
                (j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det,
                (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det,
                (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det,
            ],
            [
                (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det,
                (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det,
                (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det,
            ],
        ];
        // Reference gradients: N0=(-1,-1,-1), N1=(1,0,0), N2=(0,1,0), N3=(0,0,1).
        let ref_grads = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Physical gradients: grad N = J^{-T} grad_ref N.
        let mut grads = [[0.0f64; 3]; 4];
        for a in 0..4 {
            for component in 0..3 {
                grads[a][component] = inv[0][component] * ref_grads[a][0]
                    + inv[1][component] * ref_grads[a][1]
                    + inv[2][component] * ref_grads[a][2];
            }
        }
        let mut ke = [[0.0f64; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                ke[a][b] = volume
                    * (grads[a][0] * grads[b][0]
                        + grads[a][1] * grads[b][1]
                        + grads[a][2] * grads[b][2]);
            }
        }
        for &node in &idx {
            lumped[node] += volume / 4.0;
        }
        tets.push(Tet { nodes: idx, ke });
    }
    if lumped.iter().any(|&m| m <= 0.0) {
        return Err("room mesh has a node without mass".into());
    }

    let simple = simple_mesh_from_fem(&mesh)?;
    let mut pressure: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|node| {
            let dx = node.x - source[0];
            let dy = node.y - source[1];
            let dz = node.z - source[2];
            (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let mut previous = pressure.clone();
    let h = lx / 24.0;
    let dt = 0.45 * h / (c * 3.0f64.sqrt());
    let total_time = 0.012f64;
    let steps = (total_time / dt) as usize;
    let frame_every = (steps / 72).max(1);

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_pressure: f64 = 0.0;
    let dump = |pressure: &[f64],
                frame_index: usize,
                max_pressure: &mut f64|
     -> Result<String, String> {
        *max_pressure = (*max_pressure).max(pressure.iter().fold(0.0, |acc: f64, v| acc.max(v.abs())));
        let fields = FrameFields {
            point_scalars: vec![("pressure", pressure.to_vec())],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(
            out_dir,
            &format!("frame_{frame_index:04}"),
            &simple,
            &fields,
        )
    };

    frames.push(dump(&pressure, 0, &mut max_pressure)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    let mut kp = vec![0.0f64; n];
    for step in 1..=steps {
        kp.iter_mut().for_each(|v| *v = 0.0);
        for tet in &tets {
            for (a, &na) in tet.nodes.iter().enumerate() {
                let mut sum = 0.0;
                for (b, &nb) in tet.nodes.iter().enumerate() {
                    sum += tet.ke[a][b] * pressure[nb];
                }
                kp[na] += sum;
            }
        }
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            next[i] = 2.0 * pressure[i] - previous[i] - dt * dt * c * c * kp[i] / lumped[i];
        }
        previous = std::mem::replace(&mut pressure, next);
        if step % frame_every == 0 || step == steps {
            frames.push(dump(&pressure, frame_index, &mut max_pressure)?);
            times.push(step as f64 * dt);
            frame_index += 1;
        }
    }
    println!("room: {steps} steps, max pressure {max_pressure:.3}");
    write_manifest(
        out_dir,
        "room-wave",
        "Room impulse response",
        &frames,
        &times,
        "pressure",
        (-max_pressure, max_pressure),
    )
}

/// Shallow arch snap-through: a curved steel strip whose crown is
/// driven down quasi-statically (explicit corotational shells with a
/// near-elastic J2 law) until it snaps to the inverted equilibrium.
fn arch(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::Point;
    use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
    use sonium_fem::physics::solid::shell::explicit::{
        ShellExplicitModel, ShellExplicitState,
    };
    use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};

    let span = 1.0f64;
    let rise = 0.09f64;
    let width = 0.14f64;
    let n_along = 24usize;
    let n_across = 3usize;
    let radius = ((span / 2.0).powi(2) + rise.powi(2)) / (2.0 * rise);
    let phi_max = (span / 2.0 / radius).asin();
    let section = ShellSection::new(200.0e9, 0.3, 0.004, 7800.0);
    let node_id = |i: usize, j: usize| i * (n_across + 1) + j;
    let point = |i: usize, j: usize| {
        let phi = -phi_max + 2.0 * phi_max * i as f64 / n_along as f64;
        Point::new_3d(
            radius * phi.sin(),
            -width / 2.0 + width * j as f64 / n_across as f64,
            radius * phi.cos() - (radius - rise),
        )
    };
    let total_nodes = (n_along + 1) * (n_across + 1);
    let n_dofs = total_nodes * 6;
    let mut elements: Vec<ShellElement> = Vec::new();
    let mut cells: Vec<(CellKind, Vec<u64>)> = Vec::new();
    for i in 0..n_along {
        for j in 0..n_across {
            let a = node_id(i, j);
            let b = node_id(i + 1, j);
            let c = node_id(i + 1, j + 1);
            let d = node_id(i, j + 1);
            for (nodes, pa, pb, pc) in [
                ([a, b, c], point(i, j), point(i + 1, j), point(i + 1, j + 1)),
                ([a, c, d], point(i, j), point(i + 1, j + 1), point(i, j + 1)),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
                cells.push((
                    CellKind::Triangle,
                    nodes.iter().map(|&n| n as u64).collect(),
                ));
            }
        }
    }
    // Near-elastic J2 law on the corotational path (geometric
    // nonlinearity without plasticity): the snap is elastic.
    let material = J2Material::new(
        200.0e9,
        0.3,
        250.0e6,
        HardeningLaw::LinearIsotropic { modulus: 0.0 },
    );
    let mut model = ShellExplicitModel::elastoplastic(elements.clone(), n_dofs, material)
        .map_err(|e| e.message)?;
    for j in 0..=n_across {
        model.fix_node(node_id(0, j));
        model.fix_node(node_id(n_along, j));
    }
    let initial: Vec<[f64; 3]> = (0..total_nodes)
        .map(|id| {
            let p = point(id / (n_across + 1), id % (n_across + 1));
            [p.x, p.y, p.z]
        })
        .collect();
    let crown_nodes: Vec<usize> = (0..=n_across).map(|j| node_id(n_along / 2, j)).collect();

    let mut state = ShellExplicitState::new(n_dofs);
    let dt = 0.85 * model.wave_dt();
    let total_time = 1.15f64;
    let total_steps = (total_time / dt).ceil() as usize;
    let frame_every = (total_steps / 80).max(1);
    let drive_time = 0.85f64;
    let drive_depth = 0.15f64;

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_strain: f64 = 0.0;
    let dump = |state: &ShellExplicitState,
                frame_index: usize,
                max_strain: &mut f64|
     -> Result<String, String> {
        let points: Vec<[f64; 3]> = (0..total_nodes)
            .map(|node| {
                [
                    initial[node][0] + state.u[node * 6],
                    initial[node][1] + state.u[node * 6 + 1],
                    initial[node][2] + state.u[node * 6 + 2],
                ]
            })
            .collect();
        let strain = shell_strain_field(&elements, &state.u, total_nodes);
        *max_strain = (*max_strain).max(strain.iter().copied().fold(0.0, f64::max));
        let mesh = SimpleMesh {
            points,
            cells: cells.clone(),
        };
        let fields = FrameFields {
            point_scalars: vec![("strain", strain)],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &mesh, &fields)
    };

    frames.push(dump(&state, 0, &mut max_strain)?);
    times.push(0.0);
    let zero_force = vec![0.0f64; n_dofs];
    let mut frame_index = 1usize;
    for step in 1..=total_steps {
        model
            .step(&mut state, &zero_force, dt)
            .map_err(|e| format!("arch step {step}: {}", e.message))?;
        // Prescribed crown displacement (smooth ramp) + light damping.
        let t = step as f64 * dt;
        let driven = drive_depth * (t / drive_time).min(1.0);
        for &node in &crown_nodes {
            state.u[node * 6 + 2] = -driven;
            state.v[node * 6 + 2] = 0.0;
        }
        let damp = (-1.2 * dt).exp();
        for value in state.v.iter_mut() {
            *value *= damp;
        }
        if step % frame_every == 0 || step == total_steps {
            frames.push(dump(&state, frame_index, &mut max_strain)?);
            times.push(t);
            frame_index += 1;
        }
    }
    let crown_z = initial[crown_nodes[0]][2] + state.u[crown_nodes[0] * 6 + 2];
    println!(
        "arch: {total_steps} steps, crown final z {crown_z:.3} (started {rise:.3}), max strain {max_strain:.4}"
    );
    write_manifest(
        out_dir,
        "arch",
        "Arch snap-through",
        &frames,
        &times,
        "strain",
        (0.0, max_strain),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: qa-website-demos <taylor-green> <out-dir>");
        std::process::exit(2);
    }
    let out_dir = PathBuf::from(&args[2]);
    if let Err(error) = std::fs::create_dir_all(&out_dir) {
        eprintln!("cannot create {}: {error}", out_dir.display());
        std::process::exit(1);
    }
    let result = match args[1].as_str() {
        "taylor-green" => taylor_green(&out_dir),
        "crush" => crush(&out_dir),
        "necking" => necking(&out_dir),
        "plate-modes" => plate_modes(&out_dir),
        "room-wave" => room_wave(&out_dir),
        "arch" => arch(&out_dir),
        other => Err(format!("unknown demo {other:?}")),
    };
    match result {
        Ok(()) => println!("SUCCESS: qa-website-demos {}", args[1]),
        Err(error) => {
            eprintln!("FAILED: qa-website-demos {}: {error}", args[1]);
            std::process::exit(1);
        }
    }
}
