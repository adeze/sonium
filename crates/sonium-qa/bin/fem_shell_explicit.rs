//! Phase 7 lane 7B leg: native explicit shell solve, clamped plate.
//!
//! Runs a NATIVE Sonium shell dynamics solve (central difference on
//! the T3 CST/DKT kernels via `shell::explicit`, no engine) on a
//! clamped square plate under uniform pressure, settling the static
//! answer by dynamic relaxation with kinetic damping. This fixes the
//! Phase 3 3A deviation (integrity-only crash-shell leg): the plate
//! center deflection is compared against the Kirchhoff closed form
//! AND against a dense static solve of the same stiffness, closing
//! both the element-correctness and the dynamics-convergence loops.
//!
//! Plate: 1 m square, t = 0.02 m steel (E = 200 GPa, nu = 0.3,
//! rho = 7800 kg/m^3), uniform 1000 Pa downward, all edges clamped
//! (all 6 DOFs pinned on boundary nodes), 10x10 grid of split-square
//! T3 elements. Closed form (Timoshenko and Woinowsky-Krieger,
//! "Theory of Plates and Shells", 2nd ed., Table 8: clamped square,
//! nu = 0.3): `w = -0.00126 q a^4 / D` with
//! `D = E t^3 / (12 (1 - nu^2))`.
//!
//! Probes (exit 0 = all pass, exit 1 = any failure):
//!  `relax-static`: relaxed center deflection vs dense static solve,
//!   rel tol 2% (dynamics convergence; observed 0.001% on the gate
//!   mesh once the residual tolerance below is met).
//!  `static-analytic`: static center deflection vs Timoshenko closed
//!   form, rel tol 8% (mesh error of 10x10 DKT triangles; a 4/6/10/14
//!   convergence study at lane time showed 10.1/5.5/2.4/1.4% settling
//!   monotonically toward the closed form, so 8% holds 3x margin at
//!   the gate mesh without hiding a kernel regression).
//!  `stability`: relaxation converged within the step cap on
//!   dt = 0.5 x power-iteration critical step, with finite state.
//!   (Relaxation keeps half the free-vibration bound: kinetic-damping
//!   restarts re-kick the top mode, so near-limit steps ratchet
//!   energy upward instead of settling; see `DynamicRelaxation`.)
//! Offline: no engine, no network. Deterministic: fixed mesh, fixed
//! power-iteration count, no RNG.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::explicit::{
    DynamicRelaxation, ShellExplicitModel, lumped_masses,
};
use sonium_fem::physics::solid::shell::{ShellElement, ShellSection, assemble_shell_load};

/// Plate side in m.
const SIDE: f64 = 1.0;
/// Plate thickness in m.
const THICK: f64 = 0.02;
/// Young's modulus in Pa.
const YOUNG: f64 = 200.0e9;
/// Poisson's ratio.
const POISSON: f64 = 0.3;
/// Density in kg/m^3.
const DENSITY: f64 = 7800.0;
/// Uniform pressure in Pa (positive downward).
const PRESSURE: f64 = 1000.0;
/// Grid squares per side (center node exists at even counts).
const GRID: usize = 10;
/// Timoshenko clamped-square coefficient (nu = 0.3).
const TIMOSHENKO_ALPHA: f64 = 0.00126;
/// Relax-vs-static relative tolerance.
const RELAX_STATIC_TOL: f64 = 0.02;
/// Static-vs-analytic relative tolerance (mesh error band).
const STATIC_ANALYTIC_TOL: f64 = 0.08;
/// Relaxation residual tolerance.
const RELAX_TOL: f64 = 1.0e-4;
/// Relaxation step cap (stall guard).
const MAX_STEPS: usize = 200_000;
/// Power-iteration cap for the critical step (early-stop on convergence).
const POWER_ITERATIONS: usize = 2000;
/// Critical-step safety margin (relaxation guidance: half the bound).
const DT_MARGIN: f64 = 0.5;

/// One probe outcome.
struct Probe {
    name: &'static str,
    pass: bool,
    detail: String,
}

fn main() {
    if let Err(error) = run_leg() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_leg() -> Result<(), String> {
    let section = ShellSection::new(YOUNG, POISSON, THICK, DENSITY);
    let (elements, n_nodes) = plate_mesh(GRID, section);
    let n_dofs = n_nodes * 6;
    println!(
        "mesh: {n_nodes} nodes, {} T3 elements, {n_dofs} DOFs",
        elements.len()
    );

    // Mass audit: translational lumps must recover rho A t exactly.
    let masses =
        lumped_masses(&elements, n_dofs).map_err(|error| format!("mass audit failed: {error}"))?;
    let mut transl = 0.0;
    for node in 0..n_nodes {
        transl += masses[node * 6] + masses[node * 6 + 1] + masses[node * 6 + 2];
    }
    let expected_mass = 3.0 * DENSITY * SIDE * SIDE * THICK;
    println!("mass: lumped translational {transl:.6e} kg, exact {expected_mass:.6e} kg");

    let mut model = ShellExplicitModel::elastic(elements.clone(), n_dofs)
        .map_err(|error| format!("model build failed: {error}"))?;
    for j in 0..=GRID {
        for i in 0..=GRID {
            if i == 0 || j == 0 || i == GRID || j == GRID {
                model.fix_node(j * (GRID + 1) + i);
            }
        }
    }
    let load = assemble_shell_load(&elements, n_dofs, |_| -PRESSURE);
    let wave = model.wave_dt();
    let critical = model
        .elastic_critical_dt(POWER_ITERATIONS)
        .map_err(|error| format!("power iteration failed: {error}"))?;
    let dt = DT_MARGIN * critical;
    println!("dt: wave {wave:.6e} s, power-critical {critical:.6e} s, used {dt:.6e} s");

    let driver = DynamicRelaxation::new(dt, RELAX_TOL, MAX_STEPS)
        .map_err(|error| format!("relaxation config rejected: {error}"))?;
    let out = driver
        .relax(&mut model, &load)
        .map_err(|error| format!("relaxation failed: {error}"))?;
    println!(
        "relax: converged = {}, steps = {}, residual {:.3e} / load {:.3e}",
        out.converged, out.steps, out.residual_norm, out.load_norm
    );
    let answer = model
        .static_solve(&load)
        .map_err(|error| format!("static solve failed: {error}"))?;

    let center = (GRID / 2) * (GRID + 1) + GRID / 2;
    let w_relax = out.displacement[center * 6 + 2];
    let w_static = answer[center * 6 + 2];
    let rigidity = YOUNG * THICK.powi(3) / (12.0 * (1.0 - POISSON.powi(2)));
    let w_analytic = -TIMOSHENKO_ALPHA * PRESSURE * SIDE.powi(4) / rigidity;
    println!(
        "center: relaxed {w_relax:.6e} m, static {w_static:.6e} m, analytic {w_analytic:.6e} m"
    );

    let mut probes = Vec::new();
    let close = |actual: f64, expected: f64, rel: f64| {
        (actual - expected).abs() <= rel * expected.abs().max(1.0e-300)
    };
    probes.push(Probe {
        name: "relax-static",
        pass: close(w_relax, w_static, RELAX_STATIC_TOL),
        detail: format!("relaxed {w_relax:.6e} vs static {w_static:.6e} (tol {RELAX_STATIC_TOL})"),
    });
    probes.push(Probe {
        name: "static-analytic",
        pass: close(w_static, w_analytic, STATIC_ANALYTIC_TOL),
        detail: format!(
            "static {w_static:.6e} vs analytic {w_analytic:.6e} (tol {STATIC_ANALYTIC_TOL})"
        ),
    });
    probes.push(Probe {
        name: "stability",
        pass: out.converged
            && out.displacement.iter().all(|x| x.is_finite())
            && (transl - expected_mass).abs() / expected_mass < 1e-12,
        detail: format!(
            "converged = {}, steps = {} (cap {MAX_STEPS}), mass exact",
            out.converged, out.steps
        ),
    });

    let mut failed = false;
    for probe in &probes {
        println!(
            "probe {}: {} ({})",
            probe.name,
            if probe.pass { "pass" } else { "FAIL" },
            probe.detail
        );
        failed |= !probe.pass;
    }
    if failed {
        println!("FAILED: qa-fem-shell-explicit");
        return Err("FAIL: qa-fem-shell-explicit".to_string());
    }
    println!("PASS: qa-fem-shell-explicit");
    println!("SUCCESS: qa-fem-shell-explicit");
    Ok(())
}

/// Square plate mesh of split-square T3 elements.
///
/// Node `(i, j)` sits at id `j * (grid + 1) + i`; every square splits
/// along the rising diagonal with CCW (+z normal) winding on both
/// triangles. Returns the elements and the node count.
fn plate_mesh(grid: usize, section: ShellSection) -> (Vec<ShellElement>, usize) {
    let step = SIDE / grid as f64;
    let point = |i: usize, j: usize| Point::new_3d(i as f64 * step, j as f64 * step, 0.0);
    let id = |i: usize, j: usize| j * (grid + 1) + i;
    let mut elements = Vec::with_capacity(2 * grid * grid);
    for j in 0..grid {
        for i in 0..grid {
            elements.push(ShellElement::from_points(
                [id(i, j), id(i + 1, j), id(i + 1, j + 1)],
                &point(i, j),
                &point(i + 1, j),
                &point(i + 1, j + 1),
                section,
            ));
            elements.push(ShellElement::from_points(
                [id(i, j), id(i + 1, j + 1), id(i, j + 1)],
                &point(i, j),
                &point(i + 1, j + 1),
                &point(i, j + 1),
                section,
            ));
        }
    }
    (elements, (grid + 1) * (grid + 1))
}
